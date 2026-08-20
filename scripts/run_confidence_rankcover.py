from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rankcover.core import (  # noqa: E402
    aps_scores_all,
    class_conditional_conformal,
    evaluate_sets,
    lac_scores_all,
    make_bins,
    rank_scores,
    rank_sets,
    raps_scores_all,
    risk_proxy,
    split_aps,
    split_lac,
    split_raps,
    split_saps,
    split_saps_randomized,
    stability_score,
)
from rankcover.data import load_datasets  # noqa: E402


def _load_run_smoke():
    script = ROOT / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("rankcover_run_smoke", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOKE = _load_run_smoke()


@dataclass
class Candidate:
    label: str
    q: float
    score_alpha: float | None
    floor: int
    all_labels: bool


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.replace("\n", ",").split(",") if x.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def read_dataset_specs(args: argparse.Namespace) -> list[str]:
    specs = parse_csv(args.datasets)
    for item in args.dataset_file:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                specs.append(line)
    seen: set[str] = set()
    out = []
    for spec in specs:
        if spec not in seen:
            out.append(spec)
            seen.add(spec)
    return out


def dataset_to_load_spec(dataset_name: str) -> str:
    if dataset_name.startswith(("task:", "openml:")):
        return dataset_name
    if dataset_name.startswith("openml_"):
        return "openml:" + dataset_name.split("_", 2)[1]
    if dataset_name.startswith("task_"):
        return "task:" + dataset_name.split("_", 2)[1]
    return dataset_name


def split_score_audit(
    y_cal: np.ndarray,
    audit_frac: float,
    seed: int,
    *,
    stratify_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(y_cal.size)
    labels, counts = np.unique(y_cal, return_counts=True)
    stratify = y_cal if stratify_labels and labels.size >= 2 and counts.min() >= 2 else None
    score_idx, audit_idx = train_test_split(
        idx,
        test_size=audit_frac,
        random_state=seed + 31415,
        stratify=stratify,
    )
    return np.asarray(score_idx, dtype=int), np.asarray(audit_idx, dtype=int)


def bins_from_reference(ref_risk: np.ndarray, eval_risk: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    ref_bins, eval_bins, _ = make_bins(ref_risk, eval_risk, n_bins=n_bins)
    return ref_bins, eval_bins


def saturated_rank_quantile(scores: np.ndarray, alpha: float, n_classes: int) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float(n_classes)
    ordered = np.sort(scores)
    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if k >= scores.size + 1:
        return float(n_classes)
    k = max(1, k)
    return float(ordered[k - 1])


def rank_sets_with_floor(
    proba: np.ndarray,
    q: float,
    floor: int,
    floor_min_classes: int,
    *,
    apply_floor_gate: bool = True,
) -> np.ndarray:
    if proba.shape[1] <= 0:
        raise ValueError("empty probability matrix")
    if q >= proba.shape[1]:
        sets = np.ones_like(proba, dtype=bool)
    else:
        sets = rank_sets(proba, q)
    if floor > 0 and (not apply_floor_gate or proba.shape[1] >= floor_min_classes):
        sets = np.logical_or(sets, rank_sets(proba, float(min(floor, proba.shape[1]))))
    return sets


def candidate_effective_rank(candidate: Candidate, n_classes: int) -> float:
    if candidate.all_labels:
        return float(n_classes)
    return float(min(n_classes, max(float(candidate.q), float(candidate.floor))))


def fixed_sequence_candidates(candidates: list[Candidate], n_classes: int) -> list[Candidate]:
    """Order unique, informative candidates from most to least conservative.

    Effective rank ``n_classes`` is reserved for the explicit all-label fallback.
    Candidate aliases that induce the same effective rank are collapsed before
    auditing, so pass/fallback semantics cannot depend on a construction label.
    """
    unique: dict[float, Candidate] = {}
    for candidate in candidates:
        if candidate.all_labels:
            continue
        rank = candidate_effective_rank(candidate, n_classes)
        if rank >= float(n_classes):
            continue
        unique.setdefault(rank, candidate)
    return sorted(
        unique.values(),
        key=lambda candidate: (
            candidate_effective_rank(candidate, n_classes),
            int(candidate.floor),
            float(candidate.q),
            candidate.label,
        ),
        reverse=True,
    )


def all_label_sets(proba: np.ndarray) -> np.ndarray:
    return np.ones_like(proba, dtype=bool)


def dmean_from_result(result: Any, target: float) -> float:
    covs = [float(v) for v in result.bin_coverages.values()]
    if not covs:
        return 0.0
    return float(np.mean([max(0.0, target - v) for v in covs]))


def retarget(result: Any, alpha_eval: float) -> Any:
    target = 1.0 - alpha_eval
    result.worst_bin_violation = float(max(0.0, target - result.worst_bin_coverage))
    result.extra["target_coverage"] = float(target)
    return result


def row_from_result(meta: dict[str, Any], result: Any, method: str, role: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    target = 1.0 - float(meta["alpha_eval"])
    row = {
        **meta,
        "method": method,
        "role": role,
        "coverage": float(result.coverage),
        "sscs": float(result.extra.get("sscs", np.nan)),
        "avg_size": float(result.avg_size),
        "worst_bin_coverage": float(result.worst_bin_coverage),
        "dmax": float(result.worst_bin_violation),
        "dmean": dmean_from_result(result, target),
        "violation_run": int(float(result.worst_bin_violation) > 1e-12),
        "bin_coverages": json.dumps(result.bin_coverages, sort_keys=True),
        "bin_sizes": json.dumps(result.bin_sizes, sort_keys=True),
        "size_stratified_coverages": json.dumps(result.extra.get("size_stratified_coverages", {}), sort_keys=True),
        "size_stratified_sizes": json.dumps(result.extra.get("size_stratified_sizes", {}), sort_keys=True),
    }
    for key, value in result.extra.items():
        if key not in row and isinstance(value, (str, int, float, bool, type(None))):
            row[key] = value
    if extra:
        row.update(extra)
    return row


def make_rank_result(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    test_bins: np.ndarray,
    alpha_score: float,
    alpha_eval: float,
    floor: int,
    floor_min_classes: int,
    method_name: str,
) -> Any:
    q = saturated_rank_quantile(rank_scores(cal_proba, cal_y), alpha_score, test_proba.shape[1])
    pred = rank_sets_with_floor(test_proba, q, floor, floor_min_classes)
    result = evaluate_sets(pred, test_y, test_bins, target=1.0 - alpha_eval)
    result.name = method_name
    result.extra.update(
        {
            "q": float(q),
            "score_alpha": float(alpha_score),
            "eval_alpha": float(alpha_eval),
            "rank_floor_k": int(floor),
            "rank_floor_min_classes": int(floor_min_classes),
            "selected_k": int(pred.sum(axis=1).max()) if pred.size else 0,
        }
    )
    return result


def confidence_upper(errors: int, total: int, delta: float) -> float:
    if total <= 0:
        return 1.0
    if errors >= total:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, total - errors))


def audit_candidate(
    candidate: Candidate,
    *,
    audit_proba: np.ndarray,
    audit_y: np.ndarray,
    audit_bins: np.ndarray,
    alpha_eval: float,
    delta: float,
    floor_min_classes: int,
    apply_floor_gate: bool = False,
    certificate_mode: str = "cp_upper",
) -> tuple[Any, dict[str, Any]]:
    if candidate.all_labels:
        pred = all_label_sets(audit_proba)
    else:
        pred = rank_sets_with_floor(
            audit_proba,
            candidate.q,
            candidate.floor,
            floor_min_classes,
            apply_floor_gate=apply_floor_gate,
        )
    result = evaluate_sets(pred, audit_y, audit_bins, target=1.0 - alpha_eval)
    retarget(result, alpha_eval)
    covered = pred[np.arange(audit_y.size), audit_y]
    per_bin: dict[str, dict[str, float]] = {}
    upper_values = []
    unique_bins = sorted(np.unique(audit_bins).astype(int).tolist())
    delta_per_bin = float(delta) / max(1, len(unique_bins))
    for b in unique_bins:
        mask = audit_bins == b
        n = int(mask.sum())
        errors = int((~covered[mask]).sum())
        upper = confidence_upper(errors, n, delta_per_bin)
        per_bin[str(b)] = {"n": n, "errors": errors, "upper": upper}
        upper_values.append(upper)
    max_upper = max(upper_values) if upper_values else 1.0
    total_errors = int((~covered).sum())
    if certificate_mode == "cp_upper":
        certified = bool(max_upper <= alpha_eval)
    elif certificate_mode == "empirical_zero":
        certified = bool(total_errors == 0)
    else:
        raise ValueError(f"unknown certificate_mode={certificate_mode!r}")
    meta = {
        "candidate": candidate.label,
        "candidate_score_alpha": np.nan if candidate.score_alpha is None else float(candidate.score_alpha),
        "candidate_q": float(candidate.q),
        "candidate_floor": int(candidate.floor),
        "candidate_all_labels": bool(candidate.all_labels),
        "audit_coverage": float(result.coverage),
        "audit_sscs": float(result.extra.get("sscs", np.nan)),
        "audit_avg_size": float(result.avg_size),
        "audit_dmax": float(result.worst_bin_violation),
        "audit_dmean": dmean_from_result(result, 1.0 - alpha_eval),
        "certificate_max_upper": float(max_upper),
        "certificate_pass": certified,
        "certificate_mode": certificate_mode,
        "audit_total_errors": total_errors,
        "certificate_bin_json": json.dumps(per_bin, sort_keys=True),
        "candidate_effective_rank": candidate_effective_rank(candidate, audit_proba.shape[1]),
    }
    return result, meta


def candidate_family(
    *,
    score_proba: np.ndarray,
    score_y: np.ndarray,
    alpha_eval: float,
    safe_alpha: float,
    ultra_alpha: float,
    floors: list[int],
    include_base: bool,
    include_ultra: bool,
    n_classes: int,
    family_mode: str = "sparse",
) -> list[Candidate]:
    if family_mode == "dense_rank":
        q_base = saturated_rank_quantile(rank_scores(score_proba, score_y), alpha_eval, n_classes)
        first_rank = max(1, int(math.ceil(q_base)))
        out = [
            Candidate(label=f"rank_{rank}", q=float(rank), score_alpha=None, floor=0, all_labels=False)
            for rank in range(first_rank, n_classes)
        ]
        out.append(Candidate(label="all_labels", q=float(n_classes), score_alpha=None, floor=n_classes, all_labels=True))
        return out
    if family_mode != "sparse":
        raise ValueError(f"unknown family_mode={family_mode!r}")
    levels: list[tuple[str, float]] = []
    if include_base:
        levels.append(("base", alpha_eval))
    levels.append(("safe", safe_alpha))
    if include_ultra:
        levels.append(("ultra", ultra_alpha))
    out: list[Candidate] = []
    seen_ranks: set[float] = set()
    for level_name, alpha_score in levels:
        q = saturated_rank_quantile(rank_scores(score_proba, score_y), alpha_score, n_classes)
        level_floors = [0] if level_name == "base" else floors
        for floor in level_floors:
            label = f"{level_name}_floor{floor}" if floor else level_name
            effective_rank = float(min(n_classes, max(float(q), float(floor))))
            if effective_rank < float(n_classes) and effective_rank not in seen_ranks:
                out.append(Candidate(label=label, q=q, score_alpha=alpha_score, floor=floor, all_labels=False))
                seen_ranks.add(effective_rank)
    out.append(Candidate(label="all_labels", q=float(n_classes), score_alpha=None, floor=n_classes, all_labels=True))
    return out


def confidence_rankcover(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_risk: np.ndarray,
    test_risk: np.ndarray,
    seed: int,
    alpha_eval: float,
    safe_alpha: float,
    ultra_alpha: float,
    delta: float,
    bins: int,
    audit_frac: float,
    floors: list[int],
    floor_min_classes: int,
    include_base: bool,
    include_ultra: bool,
    method_name: str,
    candidate_family_mode: str = "sparse",
    certificate_mode: str = "cp_upper",
    score_idx: np.ndarray | None = None,
    audit_idx: np.ndarray | None = None,
    stratify_audit: bool = True,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], np.ndarray]:
    if score_idx is None or audit_idx is None:
        score_idx, audit_idx = split_score_audit(
            cal_y,
            audit_frac,
            seed,
            stratify_labels=stratify_audit,
        )
    _, audit_bins = bins_from_reference(cal_risk[score_idx], cal_risk[audit_idx], bins)
    _, test_bins = bins_from_reference(cal_risk[score_idx], test_risk, bins)
    candidates = candidate_family(
        score_proba=cal_proba[score_idx],
        score_y=cal_y[score_idx],
        alpha_eval=alpha_eval,
        safe_alpha=safe_alpha,
        ultra_alpha=ultra_alpha,
        floors=floors,
        include_base=include_base,
        include_ultra=include_ultra,
        n_classes=test_proba.shape[1],
        family_mode=candidate_family_mode,
    )
    audit_rows: list[dict[str, Any]] = []
    ordered_candidates = fixed_sequence_candidates(candidates, test_proba.shape[1])
    last_certified: tuple[Candidate, Any, dict[str, Any]] | None = None
    stopped_at: tuple[Candidate, Any, dict[str, Any]] | None = None
    for sequence_index, candidate in enumerate(ordered_candidates, start=1):
        audit_result, audit_meta = audit_candidate(
            candidate,
            audit_proba=cal_proba[audit_idx],
            audit_y=cal_y[audit_idx],
            audit_bins=audit_bins,
            alpha_eval=alpha_eval,
            delta=delta,
            floor_min_classes=floor_min_classes,
            apply_floor_gate=False,
            certificate_mode=certificate_mode,
        )
        audit_meta.update(
            {
                "sequence_index": int(sequence_index),
                "sequence_total": int(len(ordered_candidates)),
                "sequence_visited": True,
            }
        )
        audit_rows.append(audit_meta)
        if bool(audit_meta["certificate_pass"]):
            last_certified = (candidate, audit_result, audit_meta)
            continue
        stopped_at = (candidate, audit_result, audit_meta)
        break

    if last_certified is not None:
        selected_candidate, selected_audit_result, selected_meta = last_certified
        forced_uncertified_fallback = False
    else:
        selected_candidate = next(candidate for candidate in candidates if candidate.all_labels)
        fallback_pred = all_label_sets(cal_proba[audit_idx])
        selected_audit_result = evaluate_sets(fallback_pred, cal_y[audit_idx], audit_bins, target=1.0 - alpha_eval)
        selected_meta = {
            "certificate_pass": False,
            "certificate_max_upper": float(stopped_at[2]["certificate_max_upper"]) if stopped_at is not None else 1.0,
            "audit_coverage": float(selected_audit_result.coverage),
            "audit_sscs": float(selected_audit_result.extra.get("sscs", np.nan)),
            "audit_dmax": float(selected_audit_result.worst_bin_violation),
            "audit_dmean": dmean_from_result(selected_audit_result, 1.0 - alpha_eval),
            "audit_avg_size": float(selected_audit_result.avg_size),
        }
        forced_uncertified_fallback = True

    if selected_candidate.all_labels:
        pred = all_label_sets(test_proba)
    else:
        pred = rank_sets_with_floor(
            test_proba,
            selected_candidate.q,
            selected_candidate.floor,
            floor_min_classes,
            apply_floor_gate=False,
        )
    result = evaluate_sets(pred, test_y, test_bins, target=1.0 - alpha_eval)
    result.name = method_name
    retarget(result, alpha_eval)
    selected_size = pred.sum(axis=1)
    meta = {
        "method_name": method_name,
        "selected_candidate": selected_candidate.label,
        "selected_score_alpha": np.nan if selected_candidate.score_alpha is None else float(selected_candidate.score_alpha),
        "selected_q": float(selected_candidate.q),
        "selected_floor": int(selected_candidate.floor),
        "selected_k_mean": float(np.mean(selected_size)),
        "selected_k_max": int(np.max(selected_size)),
        "certificate_pass": bool(selected_meta["certificate_pass"]),
        "certificate_max_upper": float(selected_meta["certificate_max_upper"]),
        "audit_coverage": float(selected_meta["audit_coverage"]),
        "audit_sscs": float(selected_meta["audit_sscs"]),
        "audit_dmax": float(selected_meta["audit_dmax"]),
        "audit_dmean": float(selected_meta["audit_dmean"]),
        "audit_avg_size": float(selected_meta["audit_avg_size"]),
        "score_size": int(score_idx.size),
        "audit_size_n": int(audit_idx.size),
        "confidence_delta": float(delta),
        "audit_frac": float(audit_frac),
        "forced_uncertified_fallback": bool(forced_uncertified_fallback),
        "selection_rule": "nested_fixed_sequence",
        "candidate_family_mode": candidate_family_mode,
        "certificate_mode": certificate_mode,
        "floor_gate_enabled": False,
        "sequence_total": int(len(ordered_candidates)),
        "sequence_visited": int(len(audit_rows)),
        "sequence_stop_candidate": "" if stopped_at is None else stopped_at[0].label,
        "sequence_stop_upper": np.nan if stopped_at is None else float(stopped_at[2]["certificate_max_upper"]),
    }
    return result, meta, audit_rows, test_bins


def run_unit(payload: dict[str, Any]) -> dict[str, Any]:
    args = payload["args"]
    dataset_spec = payload["dataset"]
    model_name = payload["model"]
    seed = int(payload["seed"])
    t0 = time.time()
    try:
        split_mode = str(args.get("split_mode", "stratified"))
        load_max_rows = 0 if split_mode == "random" else int(args["max_rows"])
        bundles = list(
            load_datasets(
                dataset_to_load_spec(dataset_spec),
                openml_cache=args["openml_cache"] or None,
                max_rows=load_max_rows,
                seed=seed,
            )
        )
        if len(bundles) != 1:
            raise RuntimeError(f"{dataset_spec} produced {len(bundles)} bundles")
        bundle = bundles[0]
        X_all, y_all = bundle.X, bundle.y
        if split_mode == "random" and int(args["max_rows"]) > 0 and y_all.size > int(args["max_rows"]):
            rng = np.random.default_rng(seed + 2718)
            keep = rng.choice(y_all.size, size=int(args["max_rows"]), replace=False)
            X_all, y_all = X_all[keep], y_all[keep]
        if split_mode == "stratified":
            X_train, X_cal, X_test, y_train, y_cal, y_test = SMOKE._split(
                bundle,
                seed,
                train_frac=float(args["train_frac"]),
                cal_frac=float(args["cal_frac"]),
            )
        elif split_mode == "random":
            test_frac = 1.0 - float(args["train_frac"]) - float(args["cal_frac"])
            X_tmp, X_test, y_tmp, y_test = train_test_split(
                X_all,
                y_all,
                test_size=test_frac,
                random_state=seed,
                stratify=None,
            )
            rel_cal = float(args["cal_frac"]) / (float(args["train_frac"]) + float(args["cal_frac"]))
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_tmp,
                y_tmp,
                test_size=rel_cal,
                random_state=seed + 1009,
                stratify=None,
            )
        else:
            raise ValueError(f"unknown split_mode={split_mode!r}")
        n_classes = int(np.unique(y_all).size)
        score_idx, audit_idx = split_score_audit(
            y_cal,
            float(args["audit_frac"]),
            seed,
            stratify_labels=split_mode == "stratified",
        )

        cache_root = str(args.get("prediction_cache_dir", "")).strip()
        cache_path: Path | None = None
        if cache_root:
            cache_payload = {
                "dataset": dataset_spec,
                "model": model_name,
                "seed": seed,
                "split_mode": split_mode,
                "max_rows": int(args["max_rows"]),
                "train_frac": float(args["train_frac"]),
                "cal_frac": float(args["cal_frac"]),
                "tabicl_estimators": int(args["tabicl_estimators"]),
                "tabpfn_estimators": int(args["tabpfn_estimators"]),
            }
            cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
            cache_dir = Path(cache_root)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{cache_key}.npz"
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            proba_cal = np.asarray(cached["proba_cal"], dtype=float)
            proba_test = np.asarray(cached["proba_test"], dtype=float)
            prediction_cache_hit = True
        else:
            proba_cal, classes = SMOKE._fit_predict(
                model_name,
                X_train,
                y_train,
                X_cal,
                seed=seed,
                n_jobs=int(args["n_jobs"]),
                tabicl_estimators=int(args["tabicl_estimators"]),
                tabpfn_estimators=int(args["tabpfn_estimators"]),
            )
            proba_test, classes_test = SMOKE._fit_predict(
                model_name,
                X_train,
                y_train,
                X_test,
                seed=seed,
                n_jobs=int(args["n_jobs"]),
                tabicl_estimators=int(args["tabicl_estimators"]),
                tabpfn_estimators=int(args["tabpfn_estimators"]),
            )
            proba_cal = SMOKE._align_proba(proba_cal, classes, n_classes)
            proba_test = SMOKE._align_proba(proba_test, classes_test, n_classes)
            if cache_path is not None:
                np.savez_compressed(cache_path, proba_cal=proba_cal, proba_test=proba_test)
            prediction_cache_hit = False

        stability_error = ""
        if float(args["stability_weight"]) <= 0.0:
            cal_stab = np.zeros(proba_cal.shape[0], dtype=float)
            test_stab = np.zeros(proba_test.shape[0], dtype=float)
        else:
            try:
                cal_stab_runs, test_stab_runs = SMOKE._stability_predictions(
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
            except Exception as exc:
                stability_error = repr(exc)
                cal_stab = np.zeros(proba_cal.shape[0], dtype=float)
                test_stab = np.zeros(proba_test.shape[0], dtype=float)
        cal_risk = risk_proxy(
            proba_cal,
            stability=cal_stab,
            ref_proba=proba_cal[score_idx],
            ref_stability=cal_stab[score_idx],
            entropy_weight=float(args["entropy_weight"]),
            margin_weight=float(args["margin_weight"]),
            stability_weight=float(args["stability_weight"]),
        )
        test_risk = risk_proxy(
            proba_test,
            stability=test_stab,
            ref_proba=proba_cal[score_idx],
            ref_stability=cal_stab[score_idx],
            entropy_weight=float(args["entropy_weight"]),
            margin_weight=float(args["margin_weight"]),
            stability_weight=float(args["stability_weight"]),
        )
        _, test_bins = bins_from_reference(cal_risk[score_idx], test_risk, int(args["bins"]))

        meta = {
            "benchmark": args["benchmark"],
            "dataset_spec": dataset_spec,
            "dataset": bundle.name,
            "model": model_name,
            "seed": seed,
            "alpha_eval": float(args["alpha_eval"]),
            "n": int(X_all.shape[0]),
            "d": int(X_all.shape[1]),
            "classes": int(n_classes),
            "train": int(y_train.size),
            "cal": int(y_cal.size),
            "test": int(y_test.size),
            "stability_error": stability_error,
            "split_mode": split_mode,
            "risk_reference": "scoring_subset_only",
            "prediction_cache_hit": prediction_cache_hit,
        }

        rows: list[dict[str, Any]] = []
        audit_rows_all: list[dict[str, Any]] = []
        alpha_eval = float(args["alpha_eval"])
        safe_alpha = float(args["safe_alpha"])
        ultra_alpha = float(args["ultra_alpha"])

        baselines: list[tuple[str, str, Any]] = []
        baselines.append(("aps_global", "compact_cp", split_aps(proba_cal, y_cal, proba_test, y_test, alpha=alpha_eval, eval_bins=test_bins)))
        baselines.append(("lac_global", "compact_cp", split_lac(proba_cal, y_cal, proba_test, y_test, alpha=alpha_eval, eval_bins=test_bins)))
        baselines.append(
            (
                "raps_global",
                "compact_cp",
                split_raps(
                    proba_cal,
                    y_cal,
                    proba_test,
                    y_test,
                    alpha=alpha_eval,
                    eval_bins=test_bins,
                    lam=float(args["raps_lambda"]),
                    k_reg=int(args["raps_k_reg"]),
                ),
            )
        )
        aps_safe = split_aps(proba_cal, y_cal, proba_test, y_test, alpha=safe_alpha, eval_bins=test_bins)
        retarget(aps_safe, alpha_eval)
        baselines.append(("aps_safe", "conservative_cp", aps_safe))
        for lam in parse_float_csv(args["saps_lambdas"]):
            saps = split_saps(proba_cal, y_cal, proba_test, y_test, alpha=alpha_eval, eval_bins=test_bins, lam=lam)
            baselines.append((f"saps_l{lam:g}", "compact_cp", saps))
            saps_safe = split_saps(proba_cal, y_cal, proba_test, y_test, alpha=safe_alpha, eval_bins=test_bins, lam=lam)
            retarget(saps_safe, alpha_eval)
            baselines.append((f"saps_safe_l{lam:g}", "conservative_cp", saps_safe))
        for lam in parse_float_csv(args["saps_randomized_lambdas"]):
            saps_r = split_saps_randomized(
                proba_cal,
                y_cal,
                proba_test,
                y_test,
                alpha=alpha_eval,
                eval_bins=test_bins,
                lam=lam,
                random_state=seed * 100003 + int(round(lam * 10000)),
            )
            baselines.append((f"saps_randomized_l{lam:g}", "compact_cp", saps_r))
            saps_r_safe = split_saps_randomized(
                proba_cal,
                y_cal,
                proba_test,
                y_test,
                alpha=safe_alpha,
                eval_bins=test_bins,
                lam=lam,
                random_state=seed * 100003 + int(round(lam * 10000)) + 104729,
            )
            retarget(saps_r_safe, alpha_eval)
            baselines.append((f"saps_randomized_safe_l{lam:g}", "conservative_cp", saps_r_safe))

        baselines.append(
            (
                "class_conditional_aps",
                "stratified_cp",
                class_conditional_conformal(
                    proba_cal,
                    y_cal,
                    proba_test,
                    y_test,
                    alpha=alpha_eval,
                    eval_bins=test_bins,
                    min_class_cal=int(args["min_class_cal"]),
                    score_all_fn=aps_scores_all,
                    name="class_conditional_aps",
                ),
            )
        )
        baselines.append(
            (
                "class_conditional_lac",
                "stratified_cp",
                class_conditional_conformal(
                    proba_cal,
                    y_cal,
                    proba_test,
                    y_test,
                    alpha=alpha_eval,
                    eval_bins=test_bins,
                    min_class_cal=int(args["min_class_cal"]),
                    score_all_fn=lac_scores_all,
                    name="class_conditional_lac",
                ),
            )
        )
        baselines.append(
            (
                "class_conditional_raps",
                "stratified_cp",
                class_conditional_conformal(
                    proba_cal,
                    y_cal,
                    proba_test,
                    y_test,
                    alpha=alpha_eval,
                    eval_bins=test_bins,
                    min_class_cal=int(args["min_class_cal"]),
                    score_all_fn=lambda p: raps_scores_all(p, lam=float(args["raps_lambda"]), k_reg=int(args["raps_k_reg"])),
                    name="class_conditional_raps",
                ),
            )
        )
        for label, alpha_score, floor, role in [
            ("rank_global", alpha_eval, 0, "recent_rank_cp"),
            ("rank_safe", safe_alpha, 0, "conservative_rank_cp"),
            ("rank_ultra", ultra_alpha, 0, "conservative_rank_cp"),
            ("rank_safe_floor3", safe_alpha, 3, "static_repair"),
            ("rank_ultra_floor3", ultra_alpha, 3, "static_repair"),
        ]:
            baselines.append(
                (
                    label,
                    role,
                    make_rank_result(
                        cal_proba=proba_cal,
                        cal_y=y_cal,
                        test_proba=proba_test,
                        test_y=y_test,
                        test_bins=test_bins,
                        alpha_score=alpha_score,
                        alpha_eval=alpha_eval,
                        floor=floor,
                        floor_min_classes=int(args["floor_min_classes"]),
                        method_name=label,
                    ),
                )
            )

        if bool(args["include_old_rankcover"]):
            try:
                old = SMOKE._rank_audit_guard(
                    cal_proba=proba_cal,
                    cal_y=y_cal,
                    test_proba=proba_test,
                    test_y=y_test,
                    cal_risk=cal_risk,
                    test_bins=test_bins,
                    seed=seed,
                    alpha=alpha_eval,
                    safe_alpha=safe_alpha,
                    ultrasafe_alpha=ultra_alpha,
                    bins=int(args["bins"]),
                    audit_frac=float(args["audit_frac"]),
                    min_bin_cal=int(args["min_bin_cal"]),
                    min_audit_sscs=float(args["old_audit_min_sscs"]),
                    max_audit_violation=float(args["old_audit_max_violation"]),
                    consistency_guard=True,
                    consistency_tolerance=0.0,
                    final_calibration="full",
                    rank_floor_k=3,
                    rank_floor_min_classes=int(args["floor_min_classes"]),
                    name="rankcover_empirical_floor3",
                )
                retarget(old, alpha_eval)
                baselines.append(("rankcover_empirical_floor3", "old_rankcover", old))
            except Exception as exc:
                meta["old_rankcover_error"] = repr(exc)

        for method, role, result in baselines:
            rows.append(row_from_result(meta, result, method, role))

        floors = parse_int_csv(args["candidate_floors"])
        new_result, new_meta, audit_rows, _ = confidence_rankcover(
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            cal_risk=cal_risk,
            test_risk=test_risk,
            seed=seed,
            alpha_eval=alpha_eval,
            safe_alpha=safe_alpha,
            ultra_alpha=ultra_alpha,
            delta=float(args["confidence_delta"]),
            bins=int(args["bins"]),
            audit_frac=float(args["audit_frac"]),
            floors=floors,
            floor_min_classes=int(args["floor_min_classes"]),
            include_base=True,
            include_ultra=True,
            method_name="rankcover_confcert_adaptive",
            candidate_family_mode=str(args.get("candidate_family_mode", "sparse")),
            certificate_mode="cp_upper",
            score_idx=score_idx,
            audit_idx=audit_idx,
            stratify_audit=split_mode == "stratified",
        )
        rows.append(row_from_result(meta, new_result, "rankcover_confcert_adaptive", "new_rankcover", new_meta))
        for item in audit_rows:
            audit_rows_all.append({**meta, "method": "rankcover_confcert_adaptive", **item})

        empirical_result, empirical_meta, empirical_audit_rows, _ = confidence_rankcover(
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            cal_risk=cal_risk,
            test_risk=test_risk,
            seed=seed,
            alpha_eval=alpha_eval,
            safe_alpha=safe_alpha,
            ultra_alpha=ultra_alpha,
            delta=float(args["confidence_delta"]),
            bins=int(args["bins"]),
            audit_frac=float(args["audit_frac"]),
            floors=floors,
            floor_min_classes=int(args["floor_min_classes"]),
            include_base=True,
            include_ultra=True,
            method_name="rankcover_empirical_zero_matched",
            candidate_family_mode=str(args.get("candidate_family_mode", "sparse")),
            certificate_mode="empirical_zero",
            score_idx=score_idx,
            audit_idx=audit_idx,
            stratify_audit=split_mode == "stratified",
        )
        rows.append(
            row_from_result(
                meta,
                empirical_result,
                "rankcover_empirical_zero_matched",
                "matched_empirical_ablation",
                empirical_meta,
            )
        )
        for item in empirical_audit_rows:
            audit_rows_all.append({**meta, "method": "rankcover_empirical_zero_matched", **item})

        if bool(args.get("include_dense_ablation", False)):
            dense_result, dense_meta, dense_audit_rows, _ = confidence_rankcover(
                cal_proba=proba_cal,
                cal_y=y_cal,
                test_proba=proba_test,
                test_y=y_test,
                cal_risk=cal_risk,
                test_risk=test_risk,
                seed=seed,
                alpha_eval=alpha_eval,
                safe_alpha=safe_alpha,
                ultra_alpha=ultra_alpha,
                delta=float(args["confidence_delta"]),
                bins=int(args["bins"]),
                audit_frac=float(args["audit_frac"]),
                floors=floors,
                floor_min_classes=int(args["floor_min_classes"]),
                include_base=True,
                include_ultra=True,
                method_name="rankcover_confcert_dense",
                candidate_family_mode="dense_rank",
                certificate_mode="cp_upper",
                score_idx=score_idx,
                audit_idx=audit_idx,
                stratify_audit=split_mode == "stratified",
            )
            rows.append(row_from_result(meta, dense_result, "rankcover_confcert_dense", "candidate_family_ablation", dense_meta))
            for item in dense_audit_rows:
                audit_rows_all.append({**meta, "method": "rankcover_confcert_dense", **item})

        norepair_result, norepair_meta, norepair_audit_rows, _ = confidence_rankcover(
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            cal_risk=cal_risk,
            test_risk=test_risk,
            seed=seed,
            alpha_eval=alpha_eval,
            safe_alpha=safe_alpha,
            ultra_alpha=ultra_alpha,
            delta=float(args["confidence_delta"]),
            bins=int(args["bins"]),
            audit_frac=float(args["audit_frac"]),
            floors=[0],
            floor_min_classes=int(args["floor_min_classes"]),
            include_base=True,
            include_ultra=True,
            method_name="rankcover_confcert_no_repair",
            candidate_family_mode="sparse",
            certificate_mode="cp_upper",
            score_idx=score_idx,
            audit_idx=audit_idx,
            stratify_audit=split_mode == "stratified",
        )
        rows.append(row_from_result(meta, norepair_result, "rankcover_confcert_no_repair", "ablation", norepair_meta))
        for item in norepair_audit_rows:
            audit_rows_all.append({**meta, "method": "rankcover_confcert_no_repair", **item})

        for row in rows:
            row["elapsed_unit_seconds"] = float(time.time() - t0)
        return {"ok": True, "rows": rows, "audit_rows": audit_rows_all}
    except Exception as exc:
        return {
            "ok": False,
            "rows": [],
            "audit_rows": [],
            "error": {
                "benchmark": args["benchmark"],
                "dataset_spec": dataset_spec,
                "model": model_name,
                "seed": seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        }


def cluster_ci(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    dataset_col: str = "dataset",
    n_boot: int = 1000,
    seed: int = 20260818,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for keys, block in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        datasets = block[dataset_col].dropna().unique()
        if len(datasets) == 0:
            continue
        vals = []
        by_dataset = {d: block[block[dataset_col].eq(d)][value_col].to_numpy(dtype=float) for d in datasets}
        for _ in range(n_boot):
            sampled = rng.choice(datasets, size=len(datasets), replace=True)
            pieces = [by_dataset[d] for d in sampled if len(by_dataset[d]) > 0]
            if pieces:
                vals.append(float(np.concatenate(pieces).mean()))
        row = {col: val for col, val in zip(group_cols, keys)}
        if vals:
            row[f"{value_col}_ci_low"] = float(np.quantile(vals, 0.025))
            row[f"{value_col}_ci_high"] = float(np.quantile(vals, 0.975))
        else:
            row[f"{value_col}_ci_low"] = np.nan
            row[f"{value_col}_ci_high"] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def summarize(raw: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = raw[raw["error"].isna()].copy() if "error" in raw.columns else raw.copy()
    if ok.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    group_cols = ["benchmark", "model", "method", "role"]
    summary = (
        ok.groupby(group_cols, dropna=False)
        .agg(
            runs=("method", "size"),
            datasets=("dataset", "nunique"),
            seeds=("seed", "nunique"),
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_sscs=("sscs", "mean"),
            std_sscs=("sscs", "std"),
            min_sscs=("sscs", "min"),
            p05_sscs=("sscs", lambda s: float(np.quantile(s, 0.05))),
            mean_dmax=("dmax", "mean"),
            max_dmax=("dmax", "max"),
            mean_dmean=("dmean", "mean"),
            violation_runs=("violation_run", "sum"),
            violation_rate=("violation_run", "mean"),
            mean_size=("avg_size", "mean"),
            std_size=("avg_size", "std"),
            certificate_pass_rate=("certificate_pass", "mean"),
            forced_uncertified_fallback_rate=("forced_uncertified_fallback", "mean"),
            mean_certificate_upper=("certificate_max_upper", "mean"),
            mean_selected_k=("selected_k_mean", "mean"),
        )
        .reset_index()
    )
    ci_v = cluster_ci(ok, "violation_run", group_cols, n_boot=n_boot, seed=seed)
    ci_dmax = cluster_ci(ok, "dmax", group_cols, n_boot=n_boot, seed=seed + 11)
    summary = summary.merge(ci_v, on=group_cols, how="left").merge(ci_dmax, on=group_cols, how="left")
    dataset_level = (
        ok.groupby(group_cols + ["dataset"], dropna=False)
        .agg(
            dataset_runs=("method", "size"),
            dataset_any_violation=("violation_run", lambda s: int((s > 0).any())),
            dataset_violation_rate=("violation_run", "mean"),
            dataset_mean_size=("avg_size", "mean"),
            dataset_mean_dmax=("dmax", "mean"),
        )
        .reset_index()
    )
    prevalence = (
        dataset_level.groupby(group_cols, dropna=False)
        .agg(dataset_violation_prevalence=("dataset_any_violation", "mean"))
        .reset_index()
    )
    summary = summary.merge(prevalence, on=group_cols, how="left")
    selected = ok[
        ok["method"].isin(
            [
                "rankcover_confcert_adaptive",
                "rankcover_empirical_zero_matched",
                "rankcover_confcert_no_repair",
            ]
        )
    ].copy()
    if selected.empty:
        dist = pd.DataFrame()
    else:
        dist = (
            selected.groupby(group_cols + ["selected_candidate"], dropna=False)
            .agg(
                runs=("method", "size"),
                mean_size=("avg_size", "mean"),
                mean_dmax=("dmax", "mean"),
            )
            .reset_index()
        )
        totals = selected.groupby(group_cols, dropna=False).size().rename("total_runs").reset_index()
        dist = dist.merge(totals, on=group_cols, how="left")
        dist["share"] = dist["runs"] / dist["total_runs"].clip(lower=1)
    return summary, dataset_level, dist


def write_report(outdir: Path, summary: pd.DataFrame, errors: pd.DataFrame) -> None:
    lines = ["# IS revision confidence RankCover results", ""]
    if summary.empty:
        lines.append("No successful rows were produced.")
    else:
        keep = summary[
            summary["method"].isin(
                [
                    "rankcover_confcert_adaptive",
                    "rankcover_empirical_zero_matched",
                    "rankcover_confcert_no_repair",
                    "rankcover_empirical_floor3",
                    "rank_safe_floor3",
                    "rank_ultra_floor3",
                    "aps_safe",
                    "lac_global",
                    "raps_global",
                    "rank_global",
                ]
            )
        ].copy()
        show_cols = [
            "benchmark",
            "model",
            "method",
            "runs",
            "datasets",
            "mean_coverage",
            "mean_sscs",
            "mean_dmax",
            "mean_dmean",
            "violation_runs",
            "violation_rate",
            "violation_run_ci_low",
            "violation_run_ci_high",
            "dataset_violation_prevalence",
            "mean_size",
            "certificate_pass_rate",
            "forced_uncertified_fallback_rate",
        ]
        lines.append(keep[show_cols].to_markdown(index=False, floatfmt=".6f"))
    if not errors.empty:
        lines.extend(["", "## Errors", "", errors.head(50).to_markdown(index=False)])
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="RankCover")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--dataset-file", action="append", default=[])
    parser.add_argument("--models", default="rf,lgbm")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alpha-eval", type=float, default=0.10)
    parser.add_argument("--safe-alpha", type=float, default=0.025)
    parser.add_argument("--ultra-alpha", type=float, default=0.01)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--audit-frac", type=float, default=0.35)
    parser.add_argument("--min-bin-cal", type=int, default=25)
    parser.add_argument("--min-class-cal", type=int, default=20)
    parser.add_argument("--candidate-floors", default="0")
    parser.add_argument("--candidate-family-mode", choices=["sparse", "dense_rank"], default="dense_rank")
    parser.add_argument("--include-dense-ablation", action="store_true")
    parser.add_argument("--floor-min-classes", type=int, default=5)
    parser.add_argument("--entropy-weight", type=float, default=0.50)
    parser.add_argument("--margin-weight", type=float, default=0.50)
    parser.add_argument("--stability-weight", type=float, default=0.0)
    parser.add_argument("--raps-lambda", type=float, default=0.01)
    parser.add_argument("--raps-k-reg", type=int, default=1)
    parser.add_argument("--saps-lambdas", default="0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--saps-randomized-lambdas", default="0.1")
    parser.add_argument("--max-rows", type=int, default=1800)
    parser.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--train-frac", type=float, default=0.45)
    parser.add_argument("--cal-frac", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tabicl-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-estimators", type=int, default=4)
    parser.add_argument("--stability-k", type=int, default=3)
    parser.add_argument("--stability-train-subsample", type=int, default=512)
    parser.add_argument("--old-audit-min-sscs", type=float, default=1.0)
    parser.add_argument("--old-audit-max-violation", type=float, default=0.0)
    parser.add_argument("--include-old-rankcover", action="store_true")
    parser.add_argument("--openml-cache", default="data/openml")
    parser.add_argument("--prediction-cache-dir", default="")
    parser.add_argument("--outdir", default="results/main")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--precache-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    datasets = read_dataset_specs(args)
    models = parse_csv(args.models)
    seeds = parse_int_csv(args.seeds)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.postprocess_only:
        raw = pd.read_csv(outdir / "rankcover_raw.partial.csv")
        audit_df = pd.read_csv(outdir / "rankcover_candidate_audit.partial.csv")
        err_path = outdir / "rankcover_errors.csv"
        err_df = pd.read_csv(err_path) if err_path.exists() else pd.DataFrame()
        raw.to_csv(outdir / "rankcover_raw.csv", index=False)
        audit_df.to_csv(outdir / "rankcover_candidate_audit.csv", index=False)
        summary, dataset_level, distribution = summarize(raw, n_boot=int(args.bootstrap), seed=20260818)
        summary.to_csv(outdir / "rankcover_summary.csv", index=False)
        dataset_level.to_csv(outdir / "rankcover_dataset_level.csv", index=False)
        distribution.to_csv(outdir / "rankcover_candidate_distribution.csv", index=False)
        write_report(outdir, summary, err_df)
        print(
            json.dumps(
                {
                    "outdir": str(outdir),
                    "raw_rows": len(raw),
                    "audit_rows": len(audit_df),
                    "errors": len(err_df),
                    "postprocess_only": True,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    if not datasets:
        raise SystemExit("No datasets were provided.")

    metadata = {
        "created_by": "run_confidence_rankcover.py",
        "args": vars(args),
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "method_note": "New RankCover uses score-split rank candidates and held-out audit labels. A candidate is certified when every audit risk bin has a Clopper-Pearson upper bound on miscoverage at or below alpha_eval; the final candidate is the smallest certified member of a prespecified nested family, with all-label fallback if no candidate is certified.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.precache_only:
        for dataset in datasets:
            bundles = list(load_datasets(dataset_to_load_spec(dataset), args.openml_cache or None, args.max_rows, seed=0))
            print(json.dumps({"cached": dataset, "resolved": bundles[0].name, "n": int(bundles[0].y.size)}, ensure_ascii=False), flush=True)
        return

    payload_args = vars(args).copy()
    payloads = [
        {"dataset": dataset, "model": model, "seed": seed, "args": payload_args}
        for dataset in datasets
        for model in models
        for seed in seeds
    ]

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    partial_raw = outdir / "rankcover_raw.partial.csv"
    partial_audit = outdir / "rankcover_candidate_audit.partial.csv"
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = []
        for payload in payloads:
            iterator.append(run_unit(payload))
    else:
        iterator = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_payload = {executor.submit(run_unit, payload): payload for payload in payloads}
            for i, future in enumerate(as_completed(future_to_payload), start=1):
                result = future.result()
                iterator.append(result)
                payload = future_to_payload[future]
                print(f"[{i}/{len(payloads)}] {payload['dataset']} {payload['model']} seed={payload['seed']} ok={result['ok']}", flush=True)
                if result["ok"]:
                    rows.extend(result["rows"])
                    audit_rows.extend(result["audit_rows"])
                else:
                    errors.append(result["error"])
                pd.DataFrame(rows).to_csv(partial_raw, index=False)
                pd.DataFrame(audit_rows).to_csv(partial_audit, index=False)
        iterator = []

    if workers == 1:
        for i, result in enumerate(iterator, start=1):
            print(f"[{i}/{len(payloads)}] ok={result['ok']}", flush=True)
            if result["ok"]:
                rows.extend(result["rows"])
                audit_rows.extend(result["audit_rows"])
            else:
                errors.append(result["error"])
            pd.DataFrame(rows).to_csv(partial_raw, index=False)
            pd.DataFrame(audit_rows).to_csv(partial_audit, index=False)

    raw = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows)
    err_df = pd.DataFrame(errors)
    raw_path = outdir / "rankcover_raw.csv"
    audit_path = outdir / "rankcover_candidate_audit.csv"
    raw.to_csv(raw_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    if not err_df.empty:
        err_df.to_csv(outdir / "rankcover_errors.csv", index=False)
    summary, dataset_level, distribution = summarize(raw, n_boot=int(args.bootstrap), seed=20260818)
    summary.to_csv(outdir / "rankcover_summary.csv", index=False)
    dataset_level.to_csv(outdir / "rankcover_dataset_level.csv", index=False)
    distribution.to_csv(outdir / "rankcover_candidate_distribution.csv", index=False)
    write_report(outdir, summary, err_df)
    print(json.dumps({"outdir": str(outdir), "raw_rows": len(raw), "audit_rows": len(audit_df), "errors": len(err_df)}, ensure_ascii=False), flush=True)
    if not summary.empty:
        print(summary.head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
