from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts import run_benchmark
from rankcover.core import (
    _finite_sample_quantile,
    aps_scores,
    aps_sets,
    class_conditional_conformal,
    evaluate_sets,
    make_bins,
    raps_scores,
    raps_sets,
    rank_scores,
    rank_sets,
    result_to_row,
    risk_proxy,
    saps_scores,
    saps_sets,
)
from rankcover.data import load_datasets


def retarget(result, target_alpha: float) -> None:
    result.extra["score_alpha"] = result.extra.get("score_alpha", target_alpha)
    result.extra["eval_alpha"] = target_alpha
    result.worst_bin_violation = max(0.0, (1.0 - target_alpha) - result.worst_bin_coverage)


def topk_floor_sets(base_sets: np.ndarray, proba: np.ndarray, floor_k: int, min_classes: int) -> np.ndarray:
    out = np.array(base_sets, dtype=bool, copy=True)
    n_classes = proba.shape[1]
    if floor_k <= 0 or n_classes < min_classes:
        return out
    k = min(int(floor_k), n_classes)
    top_idx = np.argsort(-proba, axis=1)[:, :k]
    rows = np.arange(proba.shape[0])[:, None]
    out[rows, top_idx] = True
    return out


def rank_scores_all(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(proba.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, proba.shape[1] + 1, dtype=float)[None, :]
    return ranks


def generic_split(
    *,
    name: str,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    test_bins: np.ndarray,
    alpha: float,
    score_fn: Callable,
    set_fn: Callable,
    set_kwargs: dict | None = None,
    floor_k: int = 0,
    floor_min_classes: int = 5,
):
    set_kwargs = {} if set_kwargs is None else dict(set_kwargs)
    scores = score_fn(cal_proba, cal_y)
    q = _finite_sample_quantile(scores, alpha)
    pred_sets = set_fn(test_proba, q, **set_kwargs)
    pred_sets = topk_floor_sets(pred_sets, test_proba, floor_k, floor_min_classes)
    result = evaluate_sets(pred_sets, test_y, test_bins, 1.0 - alpha)
    result.name = name
    result.extra.update(
        {
            "q": q,
            "score_alpha": alpha,
            "eval_alpha": alpha,
            "generic_floor_k": int(floor_k),
            "generic_floor_min_classes": int(floor_min_classes),
        }
    )
    return result


def score_saps_lam(lam: float):
    return lambda p, y: saps_scores(p, y, lam=lam)


def set_saps_lam(lam: float):
    return lambda p, q: saps_sets(p, q, lam=lam)


def score_raps_lam(lam: float, k_reg: int):
    return lambda p, y: raps_scores(p, y, lam=lam, k_reg=k_reg)


def set_raps_lam(lam: float, k_reg: int):
    return lambda p, q: raps_sets(p, q, lam=lam, k_reg=k_reg)


def build_size_matched(
    *,
    family: str,
    ref_size: float,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    test_bins: np.ndarray,
    eval_alpha: float,
    alpha_grid: list[float],
    raps_lambda: float,
    raps_k_reg: int,
    saps_lambda: float,
):
    if family == "aps":
        score_fn, set_fn, kwargs = aps_scores, aps_sets, {}
    elif family == "rank":
        score_fn, set_fn, kwargs = rank_scores, rank_sets, {}
    elif family == "raps":
        score_fn, set_fn, kwargs = score_raps_lam(raps_lambda, raps_k_reg), set_raps_lam(raps_lambda, raps_k_reg), {}
    elif family == "saps":
        score_fn, set_fn, kwargs = score_saps_lam(saps_lambda), set_saps_lam(saps_lambda), {}
    else:
        raise ValueError(f"unknown family: {family}")
    candidates = []
    for score_alpha in alpha_grid:
        result = generic_split(
            name=f"{family}_oracle_size_matched",
            cal_proba=cal_proba,
            cal_y=cal_y,
            test_proba=test_proba,
            test_y=test_y,
            test_bins=test_bins,
            alpha=score_alpha,
            score_fn=score_fn,
            set_fn=set_fn,
            set_kwargs=kwargs,
        )
        retarget(result, eval_alpha)
        result.extra["size_match_alpha"] = score_alpha
        result.extra["size_match_ref_size"] = ref_size
        result.extra["size_match_abs_gap"] = abs(result.avg_size - ref_size)
        candidates.append(result)
    return min(candidates, key=lambda r: (abs(r.avg_size - ref_size), -r.extra["size_match_alpha"]))


def run_one(args, bundle, model_name: str, seed: int):
    start = time.time()
    X_train, X_cal, X_test, y_train, y_cal, y_test = run_benchmark._split(bundle, seed)
    n_classes = int(np.unique(np.concatenate([y_train, y_cal, y_test])).size)
    proba_cal, classes = run_benchmark._fit_predict(
        model_name,
        X_train,
        y_train,
        X_cal,
        seed,
        args.n_jobs,
        tabicl_estimators=args.tabicl_estimators,
        tabpfn_estimators=args.tabpfn_estimators,
    )
    proba_test, classes_test = run_benchmark._fit_predict(
        model_name,
        X_train,
        y_train,
        X_test,
        seed,
        args.n_jobs,
        tabicl_estimators=args.tabicl_estimators,
        tabpfn_estimators=args.tabpfn_estimators,
    )
    proba_cal = run_benchmark._align_proba(proba_cal, classes, n_classes)
    proba_test = run_benchmark._align_proba(proba_test, classes_test, n_classes)
    cal_risk = risk_proxy(proba_cal, entropy_weight=0.55, margin_weight=0.25, stability_weight=0.0)
    test_risk = risk_proxy(
        proba_test,
        entropy_weight=0.55,
        margin_weight=0.25,
        stability_weight=0.0,
        ref_proba=proba_cal,
    )
    _, test_bins, _ = make_bins(cal_risk, test_risk, n_bins=args.bins)

    safe_alpha = args.compact_score_alpha if args.compact_score_alpha is not None else args.alpha / 4.0
    ultra_alpha = args.rank_ultrasafe_alpha
    reference = run_benchmark._rank_audit_guard(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_risk=cal_risk,
        test_bins=test_bins,
        seed=seed,
        alpha=args.alpha,
        safe_alpha=safe_alpha,
        ultrasafe_alpha=ultra_alpha,
        bins=args.bins,
        audit_frac=args.audit_frac,
        min_bin_cal=args.min_bin_cal,
        min_audit_sscs=args.rank_audit_min_sscs,
        max_audit_violation=args.rank_audit_max_violation,
        consistency_guard=True,
        consistency_tolerance=args.rank_audit_consistency_tol,
        rank_floor_k=args.rank_floor_k,
        rank_floor_min_classes=args.rank_floor_min_classes,
        name=f"rank_audit_consistency_floor{args.rank_floor_k}",
    )
    rows = [result_to_row(bundle.name, model_name, seed, reference)]

    rc3p_rank = class_conditional_conformal(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
        min_class_cal=args.min_class_cal,
        score_all_fn=rank_scores_all,
        name="rc3p_like_rank_global",
    )
    rows.append(result_to_row(bundle.name, model_name, seed, rc3p_rank))
    rc3p_rank_safe = class_conditional_conformal(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=safe_alpha,
        eval_bins=test_bins,
        min_class_cal=args.min_class_cal,
        score_all_fn=rank_scores_all,
        name="rc3p_like_rank_safe",
    )
    retarget(rc3p_rank_safe, args.alpha)
    rows.append(result_to_row(bundle.name, model_name, seed, rc3p_rank_safe))

    controls = [
        (
            "aps_safe_generic_floor3",
            aps_scores,
            aps_sets,
            {},
            safe_alpha,
        ),
        (
            "raps_safe_generic_floor3",
            score_raps_lam(args.raps_lambda, args.raps_k_reg),
            set_raps_lam(args.raps_lambda, args.raps_k_reg),
            {},
            safe_alpha,
        ),
        (
            f"saps_safe_l{str(args.saps_lambda).replace('.', 'p')}_generic_floor3",
            score_saps_lam(args.saps_lambda),
            set_saps_lam(args.saps_lambda),
            {},
            safe_alpha,
        ),
        (
            "rank_safe_generic_floor3",
            rank_scores,
            rank_sets,
            {},
            safe_alpha,
        ),
    ]
    for name, score_fn, set_fn, kwargs, score_alpha in controls:
        result = generic_split(
            name=name,
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            test_bins=test_bins,
            alpha=score_alpha,
            score_fn=score_fn,
            set_fn=set_fn,
            set_kwargs=kwargs,
            floor_k=args.generic_floor_k,
            floor_min_classes=args.generic_floor_min_classes,
        )
        retarget(result, args.alpha)
        rows.append(result_to_row(bundle.name, model_name, seed, result))

    alpha_grid = [float(x) for x in args.size_match_alpha_grid.split(",") if x.strip()]
    for family in ["aps", "raps", "saps", "rank"]:
        result = build_size_matched(
            family=family,
            ref_size=reference.avg_size,
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            test_bins=test_bins,
            eval_alpha=args.alpha,
            alpha_grid=alpha_grid,
            raps_lambda=args.raps_lambda,
            raps_k_reg=args.raps_k_reg,
            saps_lambda=args.saps_lambda,
        )
        rows.append(result_to_row(bundle.name, model_name, seed, result))

    elapsed = time.time() - start
    for row in rows:
        row["seconds"] = elapsed
        row["classes"] = n_classes
        row["train"] = len(y_train)
        row["cal"] = len(y_cal)
        row["test"] = len(y_test)
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    return (
        df.groupby(["method"])
        .agg(
            runs=("dataset", "size"),
            datasets=("dataset", "nunique"),
            models=("model", lambda x: "+".join(sorted(set(map(str, x))))),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_sscs=("sscs", "mean"),
            std_sscs=("sscs", "std"),
            mean_worst_bin_violation=("worst_bin_violation", "mean"),
            violation_runs=("worst_bin_violation", lambda x: int((x > 1e-12).sum())),
            mean_size=("avg_size", "mean"),
            std_size=("avg_size", "std"),
        )
        .reset_index()
        .sort_values(["mean_sscs", "mean_size"], ascending=[False, True])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", default="rf,lgbm,catboost")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=1800)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--tabicl-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-estimators", type=int, default=4)
    parser.add_argument("--compact-score-alpha", type=float, default=0.025)
    parser.add_argument("--rank-ultrasafe-alpha", type=float, default=0.01)
    parser.add_argument("--rank-floor-k", type=int, default=3)
    parser.add_argument("--rank-floor-min-classes", type=int, default=5)
    parser.add_argument("--generic-floor-k", type=int, default=3)
    parser.add_argument("--generic-floor-min-classes", type=int, default=5)
    parser.add_argument("--rank-audit-min-sscs", type=float, default=1.0)
    parser.add_argument("--rank-audit-max-violation", type=float, default=0.0)
    parser.add_argument("--rank-audit-consistency-tol", type=float, default=0.0)
    parser.add_argument("--audit-frac", type=float, default=0.35)
    parser.add_argument("--min-bin-cal", type=int, default=8)
    parser.add_argument("--min-class-cal", type=int, default=20)
    parser.add_argument("--raps-lambda", type=float, default=0.01)
    parser.add_argument("--raps-k-reg", type=int, default=1)
    parser.add_argument("--saps-lambda", type=float, default=0.1)
    parser.add_argument(
        "--size-match-alpha-grid",
        default="0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3",
    )
    parser.add_argument("--openml-cache", default="data/openml")
    parser.add_argument("--outdir", default="results/controls")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dataset_tokens = [d.strip() for d in args.datasets.split(",") if d.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    partial_path = outdir / f"partial_{stamp}.jsonl"
    errors_path = outdir / f"errors_{stamp}.jsonl"
    rows = []
    errors = []

    for seed in seeds:
        for dataset_token in dataset_tokens:
            try:
                bundles = list(load_datasets(dataset_token, args.openml_cache, args.max_rows, seed))
            except Exception as exc:
                err = {"dataset": dataset_token, "model": "*", "seed": seed, "error": repr(exc)}
                errors.append(err)
                print(json.dumps({"ok": False, **err}), flush=True)
                continue
            for bundle in bundles:
                for model in models:
                    try:
                        run_rows = run_one(args, bundle, model, seed)
                        rows.extend(run_rows)
                        with partial_path.open("a", encoding="utf-8") as f:
                            for row in run_rows:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(json.dumps({"ok": True, "dataset": bundle.name, "model": model, "seed": seed, "rows": len(run_rows)}), flush=True)
                    except Exception as exc:
                        err = {"dataset": bundle.name, "model": model, "seed": seed, "error": repr(exc)}
                        errors.append(err)
                        with errors_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(err, ensure_ascii=False) + "\n")
                        print(json.dumps({"ok": False, **err}), flush=True)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"combined_rows_{stamp}.csv", index=False)
        summarize(rows).to_csv(outdir / f"summary_{stamp}.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / f"errors_{stamp}.csv", index=False)


if __name__ == "__main__":
    main()
