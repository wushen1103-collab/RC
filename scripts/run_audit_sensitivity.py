from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rankcover.core import (  # noqa: E402
    _finite_sample_quantile,
    evaluate_sets,
    make_bins,
    rank_scores,
    rank_sets,
    risk_proxy,
    stability_score,
)
from rankcover.data import load_datasets  # noqa: E402


def _load_run_benchmark():
    script = ROOT / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("rankcover_run_benchmark", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_run_benchmark()


AUDIT_DESIGNS: Dict[str, Dict[str, float]] = {
    "main_35_055_025_020": {
        "audit_frac": 0.35,
        "entropy_weight": 0.55,
        "margin_weight": 0.25,
        "stability_weight": 0.20,
    },
    "audit_25_055_025_020": {
        "audit_frac": 0.25,
        "entropy_weight": 0.55,
        "margin_weight": 0.25,
        "stability_weight": 0.20,
    },
    "audit_50_055_025_020": {
        "audit_frac": 0.50,
        "entropy_weight": 0.55,
        "margin_weight": 0.25,
        "stability_weight": 0.20,
    },
    "equal_weights_35": {
        "audit_frac": 0.35,
        "entropy_weight": 1.0 / 3.0,
        "margin_weight": 1.0 / 3.0,
        "stability_weight": 1.0 / 3.0,
    },
    "entropy_only_35": {
        "audit_frac": 0.35,
        "entropy_weight": 1.0,
        "margin_weight": 0.0,
        "stability_weight": 0.0,
    },
}


DISPLAY_LABELS = {
    "main_35_055_025_020": "Main: 35%, 0.55/0.25/0.20",
    "audit_25_055_025_020": "Audit fraction 25%",
    "audit_50_055_025_020": "Audit fraction 50%",
    "equal_weights_35": "Equal risk weights",
    "entropy_only_35": "Entropy only",
}


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _dataset_to_load_spec(dataset_name: str) -> str:
    match = re.match(r"^openml_(\d+)(?:_|$)", dataset_name)
    if match:
        return f"openml:{match.group(1)}"
    return dataset_name


def _rank_sets_with_floor(proba: np.ndarray, q: float, floor_k: int, min_classes: int) -> np.ndarray:
    sets = rank_sets(proba, q)
    if floor_k > 0 and proba.shape[1] >= min_classes:
        sets = np.logical_or(sets, rank_sets(proba, float(floor_k)))
    return sets


def _retarget_violation(result, alpha: float) -> None:
    target = 1.0 - alpha
    result.worst_bin_violation = float(max(0.0, target - result.worst_bin_coverage))
    result.extra["target_coverage"] = float(target)


def _theorem_nested_rank_audit(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_risk: np.ndarray,
    test_risk: np.ndarray,
    seed: int,
    alpha: float,
    safe_alpha: float,
    ultrasafe_alpha: float,
    bins: int,
    audit_frac: float,
    min_audit_sscs: float,
    max_audit_violation: float,
    consistency_tolerance: float,
    rank_floor_k: int,
    rank_floor_min_classes: int,
) -> Tuple[object, Dict[str, float]]:
    idx = np.arange(cal_y.size)
    score_idx, audit_idx = train_test_split(
        idx,
        test_size=audit_frac,
        random_state=seed + 31415,
        stratify=cal_y,
    )

    score_bins, audit_bins, _ = make_bins(cal_risk[score_idx], cal_risk[audit_idx], n_bins=bins)
    test_bins = make_bins(cal_risk, test_risk, n_bins=bins)[1]

    score_scores = rank_scores(cal_proba[score_idx], cal_y[score_idx])
    full_scores = rank_scores(cal_proba, cal_y)
    q_safe_score = _finite_sample_quantile(score_scores, safe_alpha)
    q_safe_full = _finite_sample_quantile(full_scores, safe_alpha)
    q_safe_final = max(q_safe_score, q_safe_full)
    q_ultra_full = _finite_sample_quantile(full_scores, ultrasafe_alpha)
    q_ultra_final = max(q_safe_final, q_ultra_full)

    audit_sets = _rank_sets_with_floor(
        cal_proba[audit_idx],
        q_safe_score,
        rank_floor_k,
        rank_floor_min_classes,
    )
    audit_safe = evaluate_sets(audit_sets, cal_y[audit_idx], audit_bins, target=1.0 - alpha)
    _retarget_violation(audit_safe, alpha)
    audit_sscs = float(audit_safe.extra.get("sscs", 0.0))
    audit_pass = audit_sscs >= min_audit_sscs and audit_safe.worst_bin_violation <= max_audit_violation

    q_selected = q_safe_final if audit_pass else q_ultra_final
    route = "audit_pass_safe" if audit_pass else "audit_fail_ultrasafe"
    consistency_pass = True
    if audit_pass:
        safe_sets = _rank_sets_with_floor(test_proba, q_safe_final, rank_floor_k, rank_floor_min_classes)
        safe_result = evaluate_sets(safe_sets, test_y, test_bins, target=1.0 - alpha)
        _retarget_violation(safe_result, alpha)
        consistency_pass = safe_result.avg_size + consistency_tolerance >= audit_safe.avg_size
        if not consistency_pass:
            q_selected = q_ultra_final
            route = "audit_pass_consistency_fallback"

    final_sets = _rank_sets_with_floor(test_proba, q_selected, rank_floor_k, rank_floor_min_classes)
    result = evaluate_sets(final_sets, test_y, test_bins, target=1.0 - alpha)
    _retarget_violation(result, alpha)

    meta = {
        "q_safe_score": float(q_safe_score),
        "q_safe_full": float(q_safe_full),
        "q_safe_final": float(q_safe_final),
        "q_ultra_final": float(q_ultra_final),
        "audit_sscs": float(audit_sscs),
        "audit_worst_bin_violation": float(audit_safe.worst_bin_violation),
        "audit_size": float(audit_safe.avg_size),
        "audit_pass": bool(audit_pass),
        "consistency_pass": bool(consistency_pass),
        "route": route,
        "score_size": int(score_idx.size),
        "audit_size_n": int(audit_idx.size),
    }
    return result, meta


def _run_unit(payload: Dict[str, object]) -> List[Dict[str, object]]:
    dataset_name = str(payload["dataset"])
    model_name = str(payload["model"])
    seed = int(payload["seed"])
    args = payload["args"]
    t0 = time.time()
    try:
        load_spec = _dataset_to_load_spec(dataset_name)
        bundles = list(
            load_datasets(
                load_spec,
                openml_cache=str(args["openml_cache"]) if args["openml_cache"] else None,
                max_rows=int(args["max_rows"]),
                seed=seed,
            )
        )
        if len(bundles) != 1:
            raise RuntimeError(f"{dataset_name} produced {len(bundles)} bundles")
        bundle = bundles[0]

        X_train, X_cal, X_test, y_train, y_cal, y_test = BENCHMARK._split(bundle, seed)
        n_classes = int(np.unique(bundle.y).size)
        proba_cal, classes = BENCHMARK._fit_predict(
            model_name,
            X_train,
            y_train,
            X_cal,
            seed=seed,
            n_jobs=int(args["n_jobs"]),
            tabicl_estimators=1,
            tabpfn_estimators=1,
        )
        proba_test, classes_test = BENCHMARK._fit_predict(
            model_name,
            X_train,
            y_train,
            X_test,
            seed=seed,
            n_jobs=int(args["n_jobs"]),
            tabicl_estimators=1,
            tabpfn_estimators=1,
        )
        proba_cal = BENCHMARK._align_proba(proba_cal, classes, n_classes)
        proba_test = BENCHMARK._align_proba(proba_test, classes_test, n_classes)

        cal_stab_runs, test_stab_runs = BENCHMARK._stability_predictions(
            model_name,
            X_train,
            y_train,
            X_cal,
            X_test,
            seed=seed,
            n_jobs=int(args["n_jobs"]),
            n_classes=n_classes,
            k=int(args["stability_k"]),
            train_subsample=int(args["stability_train_subsample"]),
        )
        cal_stab = stability_score(cal_stab_runs)
        test_stab = stability_score(test_stab_runs)

        rows: List[Dict[str, object]] = []
        for design_name, design in AUDIT_DESIGNS.items():
            cal_risk = risk_proxy(
                proba_cal,
                stability=cal_stab,
                ref_proba=proba_cal,
                ref_stability=cal_stab,
                entropy_weight=float(design["entropy_weight"]),
                margin_weight=float(design["margin_weight"]),
                stability_weight=float(design["stability_weight"]),
            )
            test_risk = risk_proxy(
                proba_test,
                stability=test_stab,
                ref_proba=proba_cal,
                ref_stability=cal_stab,
                entropy_weight=float(design["entropy_weight"]),
                margin_weight=float(design["margin_weight"]),
                stability_weight=float(design["stability_weight"]),
            )
            result, meta = _theorem_nested_rank_audit(
                cal_proba=proba_cal,
                cal_y=y_cal,
                test_proba=proba_test,
                test_y=y_test,
                cal_risk=cal_risk,
                test_risk=test_risk,
                seed=seed,
                alpha=float(args["alpha"]),
                safe_alpha=float(args["compact_score_alpha"]),
                ultrasafe_alpha=float(args["rank_ultrasafe_alpha"]),
                bins=int(args["bins"]),
                audit_frac=float(design["audit_frac"]),
                min_audit_sscs=float(args["rank_audit_min_sscs"]),
                max_audit_violation=float(args["rank_audit_max_violation"]),
                consistency_tolerance=float(args["rank_audit_consistency_tol"]),
                rank_floor_k=int(args["rank_floor_k"]),
                rank_floor_min_classes=int(args["rank_floor_min_classes"]),
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "classes": n_classes,
                    "n_train": int(y_train.size),
                    "n_cal": int(y_cal.size),
                    "n_test": int(y_test.size),
                    "audit_design": design_name,
                    "audit_design_label": DISPLAY_LABELS[design_name],
                    "audit_frac": float(design["audit_frac"]),
                    "entropy_weight": float(design["entropy_weight"]),
                    "margin_weight": float(design["margin_weight"]),
                    "stability_weight": float(design["stability_weight"]),
                    "coverage": float(result.coverage),
                    "sscs": float(result.extra.get("sscs", 0.0)),
                    "worst_bin_coverage": float(result.worst_bin_coverage),
                    "worst_bin_violation": float(result.worst_bin_violation),
                    "violation_run": int(result.worst_bin_violation > 1e-12),
                    "avg_size": float(result.avg_size),
                    "elapsed_unit_seconds": float(time.time() - t0),
                    **meta,
                }
            )
        return rows
    except Exception as exc:
        return [
            {
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        ]


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw.get("error").isna()] if "error" in raw.columns else raw
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "audit_design",
                "audit_design_label",
                "runs",
                "mean_coverage",
                "mean_sscs",
                "min_sscs",
                "violation_runs",
                "mean_size",
                "mean_worst_bin_violation",
                "audit_pass_rate",
            ]
        )
    grouped = ok.groupby(["audit_design", "audit_design_label"], sort=False)
    out = grouped.agg(
        runs=("violation_run", "size"),
        mean_coverage=("coverage", "mean"),
        mean_sscs=("sscs", "mean"),
        min_sscs=("sscs", "min"),
        violation_runs=("violation_run", "sum"),
        mean_size=("avg_size", "mean"),
        mean_worst_bin_violation=("worst_bin_violation", "mean"),
        audit_pass_rate=("audit_pass", "mean"),
    ).reset_index()
    order = {k: i for i, k in enumerate(AUDIT_DESIGNS)}
    out["order"] = out["audit_design"].map(order)
    out = out.sort_values("order").drop(columns=["order"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", default="rf,lgbm")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=1800)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stability-k", type=int, default=3)
    parser.add_argument("--stability-train-subsample", type=int, default=512)
    parser.add_argument("--compact-score-alpha", type=float, default=0.025)
    parser.add_argument("--rank-ultrasafe-alpha", type=float, default=0.01)
    parser.add_argument("--rank-floor-k", type=int, default=3)
    parser.add_argument("--rank-floor-min-classes", type=int, default=5)
    parser.add_argument("--rank-audit-min-sscs", type=float, default=1.0)
    parser.add_argument("--rank-audit-max-violation", type=float, default=0.0)
    parser.add_argument("--rank-audit-consistency-tol", type=float, default=0.0)
    parser.add_argument("--openml-cache", default="")
    parser.add_argument("--outdir", default="results/audit_sensitivity")
    parser.add_argument("--precache-only", action="store_true")
    parsed = parser.parse_args()

    datasets = _parse_csv_list(parsed.datasets)
    models = _parse_csv_list(parsed.models)
    seeds = _parse_int_list(parsed.seeds)
    outdir = Path(parsed.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    args = vars(parsed)
    if parsed.precache_only:
        for dataset in datasets:
            load_spec = _dataset_to_load_spec(dataset)
            bundles = list(
                load_datasets(
                    load_spec,
                    openml_cache=str(parsed.openml_cache) if parsed.openml_cache else None,
                    max_rows=int(parsed.max_rows),
                    seed=0,
                )
            )
            if len(bundles) != 1:
                raise RuntimeError(f"{dataset} produced {len(bundles)} bundles during precache")
            print(f"cached {dataset} via {load_spec}: n={bundles[0].y.size}", flush=True)
        return

    payloads = [
        {"dataset": dataset, "model": model, "seed": seed, "args": args}
        for dataset in datasets
        for model in models
        for seed in seeds
    ]
    metadata = {
        "created_by": "run_audit_sensitivity.py",
        "audit_designs": AUDIT_DESIGNS,
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "args": args,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: List[Dict[str, object]] = []
    workers = max(1, int(parsed.workers))
    if workers == 1:
        for i, payload in enumerate(payloads, start=1):
            rows.extend(_run_unit(payload))
            pd.DataFrame(rows).to_csv(outdir / "audit_design_sensitivity_raw.partial.csv", index=False)
            print(f"[{i}/{len(payloads)}] {payload['dataset']} {payload['model']} seed={payload['seed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_payload = {executor.submit(_run_unit, payload): payload for payload in payloads}
            for i, future in enumerate(as_completed(future_to_payload), start=1):
                payload = future_to_payload[future]
                rows.extend(future.result())
                pd.DataFrame(rows).to_csv(outdir / "audit_design_sensitivity_raw.partial.csv", index=False)
                print(f"[{i}/{len(payloads)}] {payload['dataset']} {payload['model']} seed={payload['seed']}", flush=True)

    raw = pd.DataFrame(rows)
    raw_path = outdir / "audit_design_sensitivity_raw.csv"
    summary_path = outdir / "audit_design_sensitivity_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary = _summarize(raw)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    if "error" in raw.columns and raw["error"].notna().any():
        error_path = outdir / "audit_design_sensitivity_errors.csv"
        raw[raw["error"].notna()].to_csv(error_path, index=False)
        raise SystemExit(f"completed with {raw['error'].notna().sum()} failed rows; see {error_path}")


if __name__ == "__main__":
    main()
