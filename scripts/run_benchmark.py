from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rankcover.core import (
    _finite_sample_quantile,
    aps_scores_all,
    aps_scores,
    aps_sets,
    class_conditional_conformal,
    evaluate_sets,
    lac_scores_all,
    lac_scores,
    lac_sets,
    raps_scores_all,
    make_bins,
    result_to_row,
    risk_proxy,
    split_aps,
    split_lac,
    split_rank,
    split_rank_floor,
    split_raps,
    split_saps,
    split_saps_randomized,
    stability_score,
    rankcover_aps,
    rankcover_lac,
    rankcover_rank,
    rankcover_raps,
)
from rankcover.data import DatasetBundle, load_datasets, stratified_subsample


def _retarget_violation(result, target_alpha: float) -> None:
    result.extra["score_alpha"] = result.extra.get("score_alpha", target_alpha)
    result.extra["eval_alpha"] = target_alpha
    result.worst_bin_violation = max(0.0, (1.0 - target_alpha) - result.worst_bin_coverage)


def _lambda_suffix(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _binned_sets(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    eval_proba: np.ndarray,
    cal_bins: np.ndarray,
    eval_bins: np.ndarray,
    *,
    alpha: float,
    min_bin_cal: int,
    score_fn,
    set_fn,
) -> np.ndarray:
    scores = score_fn(cal_proba, cal_y)
    global_q = _finite_sample_quantile(scores, alpha)
    out = np.zeros_like(eval_proba, dtype=bool)
    for b in sorted(np.unique(eval_bins).tolist()):
        cal_mask = cal_bins == b
        if int(cal_mask.sum()) < min_bin_cal:
            q = global_q
        else:
            q = _finite_sample_quantile(scores[cal_mask], alpha)
        mask = eval_bins == b
        out[mask] = set_fn(eval_proba[mask], q)
    return out


def _ssc_repair_lac(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_risk: np.ndarray,
    test_risk: np.ndarray,
    seed: int,
    alpha: float,
    compact_score_alpha: float,
    bins: int,
    min_bin_cal: int,
    audit_frac: float,
    min_audit_group: int,
):
    idx = np.arange(cal_y.size)
    score_idx, audit_idx = train_test_split(
        idx,
        test_size=audit_frac,
        random_state=seed + 27183,
        stratify=cal_y,
    )
    score_bins, audit_bins, _ = make_bins(cal_risk[score_idx], cal_risk[audit_idx], n_bins=bins)
    _, test_bins, _ = make_bins(cal_risk[score_idx], test_risk, n_bins=bins)

    audit_lac = _binned_sets(
        cal_proba[score_idx],
        cal_y[score_idx],
        cal_proba[audit_idx],
        score_bins,
        audit_bins,
        alpha=compact_score_alpha,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
    )
    test_lac = _binned_sets(
        cal_proba[score_idx],
        cal_y[score_idx],
        test_proba,
        score_bins,
        test_bins,
        alpha=compact_score_alpha,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
    )
    test_aps = _binned_sets(
        cal_proba[score_idx],
        cal_y[score_idx],
        test_proba,
        score_bins,
        test_bins,
        alpha=alpha,
        min_bin_cal=min_bin_cal,
        score_fn=aps_scores,
        set_fn=aps_sets,
    )

    target = 1.0 - alpha
    audit_sizes = audit_lac.sum(axis=1).astype(int)
    audit_covered = audit_lac[np.arange(audit_idx.size), cal_y[audit_idx]]
    audit_result = evaluate_sets(audit_lac, cal_y[audit_idx], audit_bins, target)
    audit_sscs = float(audit_result.extra.get("sscs", 1.0))
    bad_sizes = []
    for size in sorted(np.unique(audit_sizes).tolist()):
        mask = audit_sizes == size
        if int(mask.sum()) < min_audit_group:
            continue
        if float(audit_covered[mask].mean()) < target:
            bad_sizes.append(int(size))

    repaired = test_lac.copy()
    test_sizes = test_lac.sum(axis=1).astype(int)
    repair_mask = np.isin(test_sizes, bad_sizes)
    repaired[repair_mask] = test_aps[repair_mask]
    result = evaluate_sets(repaired, test_y, test_bins, target)
    result.name = "rankcover_ssc_repair_lac"
    result.extra.update(
        {
            "score_alpha": compact_score_alpha,
            "eval_alpha": alpha,
            "audit_frac": audit_frac,
            "min_audit_group": min_audit_group,
            "audit_cov": audit_result.coverage,
            "audit_sscs": audit_sscs,
            "audit_worst_violation": audit_result.worst_bin_violation,
            "audit_size": audit_result.avg_size,
            "bad_sizes": bad_sizes,
            "repair_rate": float(repair_mask.mean()),
        }
    )
    return result


def _rank_audit_guard(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_risk: np.ndarray,
    test_bins: np.ndarray,
    seed: int,
    alpha: float,
    safe_alpha: float,
    ultrasafe_alpha: float,
    bins: int,
    audit_frac: float,
    min_bin_cal: int,
    min_audit_sscs: float,
    max_audit_violation: float,
    consistency_guard: bool = False,
    consistency_tolerance: float = 0.0,
    final_calibration: str = "full",
    rank_floor_k: int = 0,
    rank_floor_min_classes: int = 5,
    name: str = "rank_audit_guard",
):
    def split_rank_variant(cal_p, cal_labels, eval_p, eval_labels, *, score_alpha, eval_bins):
        if rank_floor_k > 0:
            return split_rank_floor(
                cal_p,
                cal_labels,
                eval_p,
                eval_labels,
                alpha=score_alpha,
                eval_bins=eval_bins,
                min_rank_size=rank_floor_k,
                min_classes=rank_floor_min_classes,
            )
        return split_rank(
            cal_p,
            cal_labels,
            eval_p,
            eval_labels,
            alpha=score_alpha,
            eval_bins=eval_bins,
        )

    idx = np.arange(cal_y.size)
    score_idx, audit_idx = train_test_split(
        idx,
        test_size=audit_frac,
        random_state=seed + 31415,
        stratify=cal_y,
    )
    score_bins, audit_bins, _ = make_bins(cal_risk[score_idx], cal_risk[audit_idx], n_bins=bins)
    audit_safe = split_rank_variant(
        cal_proba[score_idx],
        cal_y[score_idx],
        cal_proba[audit_idx],
        cal_y[audit_idx],
        score_alpha=safe_alpha,
        eval_bins=audit_bins,
    )
    audit_sscs = float(audit_safe.extra.get("sscs", 0.0))
    audit_pass = (
        audit_sscs >= min_audit_sscs
        and audit_safe.worst_bin_violation <= max_audit_violation
    )
    if final_calibration not in {"full", "score"}:
        raise ValueError(f"unknown final_calibration={final_calibration!r}")
    final_cal_proba = cal_proba if final_calibration == "full" else cal_proba[score_idx]
    final_cal_y = cal_y if final_calibration == "full" else cal_y[score_idx]
    selected_alpha = safe_alpha if audit_pass else ultrasafe_alpha
    result = split_rank_variant(
        final_cal_proba,
        final_cal_y,
        test_proba,
        test_y,
        score_alpha=selected_alpha,
        eval_bins=test_bins,
    )
    full_safe_size = result.avg_size if audit_pass else float("nan")
    consistency_pass = True
    if consistency_guard and audit_pass:
        consistency_pass = full_safe_size + consistency_tolerance >= audit_safe.avg_size
        if not consistency_pass:
            selected_alpha = ultrasafe_alpha
            result = split_rank_variant(
                final_cal_proba,
                final_cal_y,
                test_proba,
                test_y,
                score_alpha=selected_alpha,
                eval_bins=test_bins,
            )
    result.name = name
    result.extra.update(
        {
            "score_alpha": selected_alpha,
            "eval_alpha": alpha,
            "route": (
                "rank_safe_audit_pass"
                if audit_pass and consistency_pass
                else (
                    "rank_ultrasafe_consistency_fallback"
                    if audit_pass
                    else "rank_ultrasafe_audit_fallback"
                )
            ),
            "rank_safe_alpha": safe_alpha,
            "rank_ultrasafe_alpha": ultrasafe_alpha,
            "rank_audit_frac": audit_frac,
            "rank_audit_min_sscs": min_audit_sscs,
            "rank_audit_max_violation": max_audit_violation,
            "rank_audit_cov": audit_safe.coverage,
            "rank_audit_sscs": audit_sscs,
            "rank_audit_worst_violation": audit_safe.worst_bin_violation,
            "rank_audit_size": audit_safe.avg_size,
            "rank_full_safe_size": full_safe_size,
            "rank_consistency_tolerance": consistency_tolerance if consistency_guard else float("nan"),
            "rank_consistency_margin": (
                full_safe_size - audit_safe.avg_size if audit_pass else float("nan")
            ),
            "rank_consistency_pass": consistency_pass,
            "rank_final_calibration": final_calibration,
            "rank_final_cal_size": int(final_cal_y.size),
            "rank_floor_k": rank_floor_k,
            "rank_floor_min_classes": rank_floor_min_classes,
        }
    )
    if selected_alpha != alpha:
        _retarget_violation(result, alpha)
    return result


def _singleton_or_aps_lac(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    alpha: float,
    compact_score_alpha: float,
    min_bin_cal: int,
):
    lac_pred = _binned_sets(
        cal_proba,
        cal_y,
        test_proba,
        cal_bins,
        test_bins,
        alpha=compact_score_alpha,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
    )
    aps_pred = _binned_sets(
        cal_proba,
        cal_y,
        test_proba,
        cal_bins,
        test_bins,
        alpha=alpha,
        min_bin_cal=min_bin_cal,
        score_fn=aps_scores,
        set_fn=aps_sets,
    )
    sizes = lac_pred.sum(axis=1)
    fallback_mask = sizes > 1
    final = lac_pred.copy()
    final[fallback_mask] = aps_pred[fallback_mask]
    result = evaluate_sets(final, test_y, test_bins, 1.0 - alpha)
    result.name = "rankcover_singleton_or_aps_lac"
    result.extra.update(
        {
            "score_alpha": compact_score_alpha,
            "eval_alpha": alpha,
            "fallback_rate": float(fallback_mask.mean()),
        }
    )
    return result


def _singleton_or_global_aps_lac(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    alpha: float,
    compact_score_alpha: float,
    min_bin_cal: int,
):
    lac_pred = _binned_sets(
        cal_proba,
        cal_y,
        test_proba,
        cal_bins,
        test_bins,
        alpha=compact_score_alpha,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
    )
    global_aps_q = _finite_sample_quantile(aps_scores(cal_proba, cal_y), alpha)
    aps_pred = aps_sets(test_proba, global_aps_q)
    sizes = lac_pred.sum(axis=1)
    fallback_mask = sizes > 1
    final = lac_pred.copy()
    final[fallback_mask] = aps_pred[fallback_mask]
    result = evaluate_sets(final, test_y, test_bins, 1.0 - alpha)
    result.name = "rankcover_singleton_or_global_aps_lac"
    result.extra.update(
        {
            "score_alpha": compact_score_alpha,
            "eval_alpha": alpha,
            "fallback_rate": float(fallback_mask.mean()),
            "global_aps_q": float(global_aps_q),
        }
    )
    return result


def _singleton_or_allset_lac(
    *,
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    alpha: float,
    compact_score_alpha: float,
    min_bin_cal: int,
):
    lac_pred = _binned_sets(
        cal_proba,
        cal_y,
        test_proba,
        cal_bins,
        test_bins,
        alpha=compact_score_alpha,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
    )
    sizes = lac_pred.sum(axis=1)
    fallback_mask = sizes > 1
    final = lac_pred.copy()
    final[fallback_mask] = True
    result = evaluate_sets(final, test_y, test_bins, 1.0 - alpha)
    result.name = "rankcover_singleton_or_allset_lac"
    result.extra.update(
        {
            "score_alpha": compact_score_alpha,
            "eval_alpha": alpha,
            "fallback_rate": float(fallback_mask.mean()),
        }
    )
    return result


def _split(bundle: DatasetBundle, seed: int, train_frac: float = 0.45, cal_frac: float = 0.25):
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=1.0 - train_frac - cal_frac,
        random_state=seed,
        stratify=bundle.y,
    )
    rel_cal = cal_frac / (train_frac + cal_frac)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_cal,
        random_state=seed + 1009,
        stratify=y_tmp,
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def _fit_predict(
    model_name: str,
    X_train,
    y_train,
    X_eval,
    seed: int,
    n_jobs: int,
    tabicl_estimators: int,
    tabpfn_estimators: int,
):
    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=260,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif model_name == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=min(n_jobs, 8), random_state=seed),
        )
    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            class_weight="balanced",
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1.0,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif model_name == "catboost":
        from catboost import CatBoostClassifier

        n_classes = int(np.unique(y_train).size)
        clf = CatBoostClassifier(
            iterations=260,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    elif model_name == "tabicl":
        from tabicl import TabICLClassifier

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None
        clf = TabICLClassifier(
            device=device,
            n_estimators=tabicl_estimators,
            batch_size=8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=False,
        )
    elif model_name == "tabpfn":
        from tabpfn import TabPFNClassifier

        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        clf = TabPFNClassifier(
            n_estimators=tabpfn_estimators,
            device=device,
            random_state=seed,
            n_preprocessing_jobs=min(n_jobs, 8),
            ignore_pretraining_limits=True,
            show_progress_bar=False,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_eval)
    classes = getattr(clf, "classes_", np.arange(np.unique(y_train).size))
    return proba, np.asarray(classes)


def _align_proba(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes.astype(int).tolist()):
        if 0 <= cls < n_classes:
            out[:, cls] = proba[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= 0
    if bad.any():
        out[bad] = 1.0 / n_classes
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def _stability_predictions(
    model_name: str,
    X_train,
    y_train,
    X_cal,
    X_test,
    seed: int,
    n_jobs: int,
    n_classes: int,
    k: int,
    train_subsample: int,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    if k <= 1:
        return None, None
    cal_runs: List[np.ndarray] = []
    test_runs: List[np.ndarray] = []
    for i in range(k):
        X_sub, y_sub = stratified_subsample(X_train, y_train, max_rows=train_subsample, seed=seed * 100 + i)
        proba_cal, classes = _fit_predict(
            model_name,
            X_sub,
            y_sub,
            X_cal,
            seed=seed * 1000 + i,
            n_jobs=n_jobs,
            tabicl_estimators=1,
            tabpfn_estimators=1,
        )
        proba_test, classes_test = _fit_predict(
            model_name,
            X_sub,
            y_sub,
            X_test,
            seed=seed * 1000 + i,
            n_jobs=n_jobs,
            tabicl_estimators=1,
            tabpfn_estimators=1,
        )
        cal_runs.append(_align_proba(proba_cal, classes, n_classes))
        test_runs.append(_align_proba(proba_test, classes_test, n_classes))
    return np.stack(cal_runs, axis=0), np.stack(test_runs, axis=0)


def run_one(args, bundle: DatasetBundle, model_name: str, seed: int) -> List[Dict[str, object]]:
    t0 = time.time()
    X_train, X_cal, X_test, y_train, y_cal, y_test = _split(bundle, seed)
    n_classes = int(np.unique(bundle.y).size)

    proba_cal, classes = _fit_predict(
        model_name,
        X_train,
        y_train,
        X_cal,
        seed=seed,
        n_jobs=args.n_jobs,
        tabicl_estimators=args.tabicl_estimators,
        tabpfn_estimators=args.tabpfn_estimators,
    )
    proba_test, classes_test = _fit_predict(
        model_name,
        X_train,
        y_train,
        X_test,
        seed=seed,
        n_jobs=args.n_jobs,
        tabicl_estimators=args.tabicl_estimators,
        tabpfn_estimators=args.tabpfn_estimators,
    )
    proba_cal = _align_proba(proba_cal, classes, n_classes)
    proba_test = _align_proba(proba_test, classes_test, n_classes)

    cal_stab_runs, test_stab_runs = _stability_predictions(
        model_name,
        X_train,
        y_train,
        X_cal,
        X_test,
        seed=seed,
        n_jobs=args.n_jobs,
        n_classes=n_classes,
        k=args.stability_k,
        train_subsample=args.stability_train_subsample,
    )
    cal_stab = stability_score(cal_stab_runs)
    test_stab = stability_score(test_stab_runs)

    risk_variants = {
        "mondrian_entropy_aps": dict(entropy_weight=1.0, margin_weight=0.0, stability_weight=0.0),
        "mondrian_margin_aps": dict(entropy_weight=0.0, margin_weight=1.0, stability_weight=0.0),
        "mondrian_stability_aps": dict(entropy_weight=0.0, margin_weight=0.0, stability_weight=1.0),
        "rankcover_em_aps": dict(entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0),
        "rankcover_ems_aps": dict(entropy_weight=0.55, margin_weight=0.25, stability_weight=0.20),
    }
    main_risk_kwargs = risk_variants["rankcover_ems_aps"]
    cal_risk = risk_proxy(
        proba_cal,
        stability=cal_stab,
        ref_proba=proba_cal,
        ref_stability=cal_stab,
        **main_risk_kwargs,
    )
    test_risk = risk_proxy(
        proba_test,
        stability=test_stab,
        ref_proba=proba_cal,
        ref_stability=cal_stab,
        **main_risk_kwargs,
    )
    cal_bins, test_bins, cuts = make_bins(cal_risk, test_risk, n_bins=args.bins)

    baseline_aps = split_aps(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
    )
    baseline_lac = split_lac(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
    )
    baseline_raps = split_raps(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
        lam=args.raps_lambda,
        k_reg=args.raps_k_reg,
    )
    baseline_rank = split_rank(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
    )
    label_mondrian_aps = class_conditional_conformal(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
        min_class_cal=args.min_class_cal,
        score_all_fn=aps_scores_all,
        name="mondrian_label_aps",
    )
    label_mondrian_lac = class_conditional_conformal(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
        min_class_cal=args.min_class_cal,
        score_all_fn=lac_scores_all,
        name="mondrian_label_lac",
    )
    label_mondrian_raps = class_conditional_conformal(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.alpha,
        eval_bins=test_bins,
        min_class_cal=args.min_class_cal,
        score_all_fn=lambda p: raps_scores_all(p, lam=args.raps_lambda, k_reg=args.raps_k_reg),
        name="mondrian_label_raps",
    )
    label_mondrian_raps.extra.update({"raps_lambda": args.raps_lambda, "raps_k_reg": args.raps_k_reg})
    repair_alpha = args.compact_score_alpha if args.compact_score_alpha is not None else args.alpha
    baseline_aps_safe = split_aps(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=repair_alpha,
        eval_bins=test_bins,
    )
    baseline_aps_safe.name = "aps_global_safe"
    baseline_aps_safe.extra["score_alpha"] = repair_alpha
    if repair_alpha != args.alpha:
        _retarget_violation(baseline_aps_safe, args.alpha)
    baseline_rank_safe = split_rank(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=repair_alpha,
        eval_bins=test_bins,
    )
    baseline_rank_safe.name = "rank_global_safe"
    baseline_rank_safe.extra["score_alpha"] = repair_alpha
    if repair_alpha != args.alpha:
        _retarget_violation(baseline_rank_safe, args.alpha)
    baseline_rank_ultrasafe = split_rank(
        proba_cal,
        y_cal,
        proba_test,
        y_test,
        alpha=args.rank_ultrasafe_alpha,
        eval_bins=test_bins,
    )
    baseline_rank_ultrasafe.name = "rank_global_ultrasafe"
    baseline_rank_ultrasafe.extra["score_alpha"] = args.rank_ultrasafe_alpha
    if args.rank_ultrasafe_alpha != args.alpha:
        _retarget_violation(baseline_rank_ultrasafe, args.alpha)
    rank_floor_results = []
    if args.rank_floor_k > 0:
        baseline_rank_floor_safe = split_rank_floor(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=repair_alpha,
            eval_bins=test_bins,
            min_rank_size=args.rank_floor_k,
            min_classes=args.rank_floor_min_classes,
        )
        baseline_rank_floor_safe.name = f"rank_global_floor{args.rank_floor_k}_safe"
        baseline_rank_floor_safe.extra["score_alpha"] = repair_alpha
        if repair_alpha != args.alpha:
            _retarget_violation(baseline_rank_floor_safe, args.alpha)
        rank_floor_results.append(baseline_rank_floor_safe)
        baseline_rank_floor_ultrasafe = split_rank_floor(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=args.rank_ultrasafe_alpha,
            eval_bins=test_bins,
            min_rank_size=args.rank_floor_k,
            min_classes=args.rank_floor_min_classes,
        )
        baseline_rank_floor_ultrasafe.name = f"rank_global_floor{args.rank_floor_k}_ultrasafe"
        baseline_rank_floor_ultrasafe.extra["score_alpha"] = args.rank_ultrasafe_alpha
        if args.rank_ultrasafe_alpha != args.alpha:
            _retarget_violation(baseline_rank_floor_ultrasafe, args.alpha)
        rank_floor_results.append(baseline_rank_floor_ultrasafe)
    saps_lambdas = [float(x) for x in args.saps_lambdas.split(",") if x.strip()]
    baseline_saps = []
    for saps_lambda in saps_lambdas:
        suffix = _lambda_suffix(saps_lambda)
        saps = split_saps(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=args.alpha,
            eval_bins=test_bins,
            lam=saps_lambda,
        )
        saps.name = f"saps_global_l{suffix}"
        baseline_saps.append(saps)
        saps_safe = split_saps(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=repair_alpha,
            eval_bins=test_bins,
            lam=saps_lambda,
        )
        saps_safe.name = f"saps_global_safe_l{suffix}"
        saps_safe.extra["score_alpha"] = repair_alpha
        if repair_alpha != args.alpha:
            _retarget_violation(saps_safe, args.alpha)
        baseline_saps.append(saps_safe)
    saps_randomized_lambdas = [float(x) for x in args.saps_randomized_lambdas.split(",") if x.strip()]
    baseline_saps_randomized = []
    for saps_lambda in saps_randomized_lambdas:
        suffix = _lambda_suffix(saps_lambda)
        seed_base = 7919 + seed * 1009 + int(round(saps_lambda * 10000.0)) * 13
        saps_randomized = split_saps_randomized(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=args.alpha,
            eval_bins=test_bins,
            lam=saps_lambda,
            random_state=seed_base,
        )
        saps_randomized.name = f"saps_randomized_l{suffix}"
        baseline_saps_randomized.append(saps_randomized)
        saps_randomized_safe = split_saps_randomized(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=repair_alpha,
            eval_bins=test_bins,
            lam=saps_lambda,
            random_state=seed_base + 104729,
        )
        saps_randomized_safe.name = f"saps_randomized_safe_l{suffix}"
        saps_randomized_safe.extra["score_alpha"] = repair_alpha
        if repair_alpha != args.alpha:
            _retarget_violation(saps_randomized_safe, args.alpha)
        baseline_saps_randomized.append(saps_randomized_safe)
    rank_audit = _rank_audit_guard(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_risk=cal_risk,
        test_bins=test_bins,
        seed=seed,
        alpha=args.alpha,
        safe_alpha=repair_alpha,
        ultrasafe_alpha=args.rank_ultrasafe_alpha,
        bins=args.bins,
        audit_frac=args.audit_frac,
        min_bin_cal=args.min_bin_cal,
        min_audit_sscs=args.rank_audit_min_sscs,
        max_audit_violation=args.rank_audit_max_violation,
    )
    rank_audit_consistency = _rank_audit_guard(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_risk=cal_risk,
        test_bins=test_bins,
        seed=seed,
        alpha=args.alpha,
        safe_alpha=repair_alpha,
        ultrasafe_alpha=args.rank_ultrasafe_alpha,
        bins=args.bins,
        audit_frac=args.audit_frac,
        min_bin_cal=args.min_bin_cal,
        min_audit_sscs=args.rank_audit_min_sscs,
        max_audit_violation=args.rank_audit_max_violation,
        consistency_guard=True,
        consistency_tolerance=args.rank_audit_consistency_tol,
        name="rank_audit_consistency",
    )
    rank_audit_consistency_floor = None
    if args.rank_floor_k > 0:
        rank_audit_consistency_floor = _rank_audit_guard(
            cal_proba=proba_cal,
            cal_y=y_cal,
            test_proba=proba_test,
            test_y=y_test,
            cal_risk=cal_risk,
            test_bins=test_bins,
            seed=seed,
            alpha=args.alpha,
            safe_alpha=repair_alpha,
            ultrasafe_alpha=args.rank_ultrasafe_alpha,
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

    rows = [
        result_to_row(bundle.name, model_name, seed, baseline_aps),
        result_to_row(bundle.name, model_name, seed, baseline_aps_safe, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, baseline_lac, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, baseline_raps, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, baseline_rank, baseline=baseline_aps),
        *[result_to_row(bundle.name, model_name, seed, item, baseline=baseline_aps) for item in baseline_saps],
        *[result_to_row(bundle.name, model_name, seed, item, baseline=baseline_aps) for item in baseline_saps_randomized],
        result_to_row(bundle.name, model_name, seed, baseline_rank_safe, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, baseline_rank_ultrasafe, baseline=baseline_aps),
        *[result_to_row(bundle.name, model_name, seed, item, baseline=baseline_aps) for item in rank_floor_results],
        result_to_row(bundle.name, model_name, seed, rank_audit, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, rank_audit_consistency, baseline=baseline_aps),
        *(
            [result_to_row(bundle.name, model_name, seed, rank_audit_consistency_floor, baseline=baseline_aps)]
            if rank_audit_consistency_floor is not None
            else []
        ),
        result_to_row(bundle.name, model_name, seed, label_mondrian_aps, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, label_mondrian_lac, baseline=baseline_aps),
        result_to_row(bundle.name, model_name, seed, label_mondrian_raps, baseline=baseline_aps),
    ]
    proposed_by_name = {}
    for method_name, risk_kwargs in risk_variants.items():
        cal_r = risk_proxy(
            proba_cal,
            stability=cal_stab,
            ref_proba=proba_cal,
            ref_stability=cal_stab,
            **risk_kwargs,
        )
        test_r = risk_proxy(
            proba_test,
            stability=test_stab,
            ref_proba=proba_cal,
            ref_stability=cal_stab,
            **risk_kwargs,
        )
        cal_b, test_b, _ = make_bins(cal_r, test_r, n_bins=args.bins)
        proposed = rankcover_aps(
            proba_cal,
            y_cal,
            proba_test,
            y_test,
            alpha=args.alpha,
            cal_bins=cal_b,
            test_bins=test_b,
            min_bin_cal=args.min_bin_cal,
        )
        proposed.name = method_name
        proposed_by_name[method_name] = proposed
        rows.append(result_to_row(bundle.name, model_name, seed, proposed, baseline=baseline_aps))
        if method_name in {"mondrian_entropy_aps", "rankcover_em_aps", "rankcover_ems_aps"}:
            prefix = method_name.replace("_aps", "")
            compact_score_alpha = args.compact_score_alpha if args.compact_score_alpha is not None else args.alpha
            proposed_lac = rankcover_lac(
                proba_cal,
                y_cal,
                proba_test,
                y_test,
                alpha=compact_score_alpha,
                cal_bins=cal_b,
                test_bins=test_b,
                min_bin_cal=args.min_bin_cal,
                name=f"{prefix}_lac",
            )
            proposed_lac.extra["score_alpha"] = compact_score_alpha
            if compact_score_alpha != args.alpha:
                _retarget_violation(proposed_lac, args.alpha)
            rows.append(result_to_row(bundle.name, model_name, seed, proposed_lac, baseline=baseline_aps))
            proposed_raps = rankcover_raps(
                proba_cal,
                y_cal,
                proba_test,
                y_test,
                alpha=compact_score_alpha,
                cal_bins=cal_b,
                test_bins=test_b,
                min_bin_cal=args.min_bin_cal,
                lam=args.raps_lambda,
                k_reg=args.raps_k_reg,
                name=f"{prefix}_raps",
            )
            proposed_raps.extra["score_alpha"] = compact_score_alpha
            if compact_score_alpha != args.alpha:
                _retarget_violation(proposed_raps, args.alpha)
            rows.append(result_to_row(bundle.name, model_name, seed, proposed_raps, baseline=baseline_aps))
            proposed_rank = rankcover_rank(
                proba_cal,
                y_cal,
                proba_test,
                y_test,
                alpha=compact_score_alpha,
                cal_bins=cal_b,
                test_bins=test_b,
                min_bin_cal=args.min_bin_cal,
                name=f"{prefix}_rank_safe",
            )
            proposed_rank.extra["score_alpha"] = compact_score_alpha
            if compact_score_alpha != args.alpha:
                _retarget_violation(proposed_rank, args.alpha)
            rows.append(result_to_row(bundle.name, model_name, seed, proposed_rank, baseline=baseline_aps))

    repair = _ssc_repair_lac(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_risk=cal_risk,
        test_risk=test_risk,
        seed=seed,
        alpha=args.alpha,
        compact_score_alpha=repair_alpha,
        bins=args.bins,
        min_bin_cal=args.min_bin_cal,
        audit_frac=args.audit_frac,
        min_audit_group=args.min_audit_group,
    )
    rows.append(result_to_row(bundle.name, model_name, seed, repair, baseline=baseline_aps))

    singleton_fallback = _singleton_or_aps_lac(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_bins=cal_bins,
        test_bins=test_bins,
        alpha=args.alpha,
        compact_score_alpha=repair_alpha,
        min_bin_cal=args.min_bin_cal,
    )
    rows.append(result_to_row(bundle.name, model_name, seed, singleton_fallback, baseline=baseline_aps))

    singleton_global_fallback = _singleton_or_global_aps_lac(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_bins=cal_bins,
        test_bins=test_bins,
        alpha=args.alpha,
        compact_score_alpha=repair_alpha,
        min_bin_cal=args.min_bin_cal,
    )
    rows.append(result_to_row(bundle.name, model_name, seed, singleton_global_fallback, baseline=baseline_aps))

    singleton_allset_fallback = _singleton_or_allset_lac(
        cal_proba=proba_cal,
        cal_y=y_cal,
        test_proba=proba_test,
        test_y=y_test,
        cal_bins=cal_bins,
        test_bins=test_bins,
        alpha=args.alpha,
        compact_score_alpha=repair_alpha,
        min_bin_cal=args.min_bin_cal,
    )
    rows.append(result_to_row(bundle.name, model_name, seed, singleton_allset_fallback, baseline=baseline_aps))

    fallback_rate = float(singleton_fallback.extra.get("fallback_rate", 0.0))
    repair_audit_sscs = float(repair.extra.get("audit_sscs", 0.0))
    repair_mean_size = float(repair.avg_size)
    repair_good = (
        repair_audit_sscs >= args.auto_repair_audit_sscs
        and repair_mean_size <= args.auto_repair_max_size_ratio * baseline_aps_safe.avg_size
    )
    if fallback_rate > args.auto_high_fallback:
        auto = copy.deepcopy(baseline_aps)
        route = "aps_high_fallback"
    elif fallback_rate < args.auto_low_fallback:
        auto = copy.deepcopy(singleton_global_fallback)
        route = "global_low_fallback"
    else:
        auto = copy.deepcopy(singleton_fallback)
        route = "local_mid_fallback"
    auto.name = "autorankcover_route"
    auto.extra.update(
        {
            "route": route,
            "fallback_rate": fallback_rate,
            "auto_low_fallback": args.auto_low_fallback,
            "auto_high_fallback": args.auto_high_fallback,
        }
    )
    rows.append(result_to_row(bundle.name, model_name, seed, auto, baseline=baseline_aps))

    if n_classes <= args.auto_allset_max_classes and fallback_rate > args.auto_allset_small_fallback:
        all_sets = np.ones_like(proba_test, dtype=bool)
        all_guard = evaluate_sets(all_sets, y_test, test_bins, 1.0 - args.alpha)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update(
            {
                "route": "allset_guard",
                "fallback_rate": fallback_rate,
                "auto_allset_high_fallback": args.auto_allset_high_fallback,
                "auto_allset_small_fallback": args.auto_allset_small_fallback,
                "auto_allset_max_classes": args.auto_allset_max_classes,
            }
        )
    elif fallback_rate > args.auto_allset_high_fallback:
        all_guard = copy.deepcopy(baseline_aps_safe)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update({"route": "safe_aps_high_fallback", "fallback_rate": fallback_rate})
    elif (
        n_classes > args.auto_allset_max_classes
        and args.auto_em_mid_low <= fallback_rate < args.auto_em_mid_high
    ):
        all_guard = copy.deepcopy(proposed_by_name["rankcover_em_aps"])
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update(
            {
                "route": "em_aps_mid_fallback",
                "fallback_rate": fallback_rate,
                "auto_em_mid_low": args.auto_em_mid_low,
                "auto_em_mid_high": args.auto_em_mid_high,
            }
        )
    elif (
        (n_classes <= args.auto_allset_max_classes and fallback_rate > args.auto_small_allset_fallback)
        or (n_classes > args.auto_allset_max_classes and fallback_rate <= args.auto_allset_candidate_fallback)
    ):
        all_guard = copy.deepcopy(singleton_allset_fallback)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update(
            {
                "route": "singleton_allset_mid_fallback",
                "fallback_rate": fallback_rate,
                "auto_allset_candidate_fallback": args.auto_allset_candidate_fallback,
                "auto_small_allset_fallback": args.auto_small_allset_fallback,
            }
        )
    elif repair_good:
        all_guard = copy.deepcopy(repair)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update(
            {
                "route": "ssc_repair_audit_pass",
                "fallback_rate": fallback_rate,
                "repair_audit_sscs": repair_audit_sscs,
                "repair_mean_size": repair_mean_size,
                "auto_repair_audit_sscs": args.auto_repair_audit_sscs,
                "auto_repair_max_size_ratio": args.auto_repair_max_size_ratio,
            }
        )
    elif fallback_rate < args.auto_low_fallback:
        all_guard = copy.deepcopy(singleton_global_fallback)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update({"route": "global_low_fallback", "fallback_rate": fallback_rate})
    else:
        all_guard = copy.deepcopy(singleton_fallback)
        all_guard.name = "autorankcover_all_guard"
        all_guard.extra.update({"route": "local_mid_fallback", "fallback_rate": fallback_rate})
    rows.append(result_to_row(bundle.name, model_name, seed, all_guard, baseline=baseline_aps))

    safe_all_source_route = str(all_guard.extra.get("route", ""))
    safe_all_uses_all_guard = (
        safe_all_source_route == "em_aps_mid_fallback"
        or (
            safe_all_source_route == "singleton_allset_mid_fallback"
            and args.auto_safe_all_singleton_low <= fallback_rate <= args.auto_safe_all_singleton_high
        )
    )
    if safe_all_uses_all_guard:
        safe_all_guard = copy.deepcopy(all_guard)
        safe_all_route = f"safe_all_{safe_all_source_route}"
    else:
        safe_all_guard = copy.deepcopy(baseline_aps_safe)
        safe_all_route = "safe_all_safe_default"
    safe_all_guard.name = "autorankcover_safe_all_guard"
    safe_all_guard.extra.update(
        {
            "route": safe_all_route,
            "source_route": safe_all_source_route,
            "fallback_rate": fallback_rate,
            "auto_safe_all_singleton_low": args.auto_safe_all_singleton_low,
            "auto_safe_all_singleton_high": args.auto_safe_all_singleton_high,
        }
    )
    rows.append(result_to_row(bundle.name, model_name, seed, safe_all_guard, baseline=baseline_aps))

    if (
        n_classes > args.auto_allset_max_classes
        and args.auto_pareto_em_low <= fallback_rate < args.auto_pareto_em_high
    ):
        pareto_guard = copy.deepcopy(proposed_by_name["rankcover_em_aps"])
        pareto_guard.name = "autorankcover_pareto_guard"
        pareto_guard.extra.update(
            {
                "route": "pareto_em_mid_fallback",
                "fallback_rate": fallback_rate,
                "auto_pareto_em_low": args.auto_pareto_em_low,
                "auto_pareto_em_high": args.auto_pareto_em_high,
                "auto_allset_max_classes": args.auto_allset_max_classes,
                "auto_pareto_min_classes_for_singleton": args.auto_pareto_min_classes_for_singleton,
                "auto_pareto_singleton_low": args.auto_pareto_singleton_low,
                "auto_pareto_singleton_high": args.auto_pareto_singleton_high,
                "auto_pareto_max_classes_for_singleton": args.auto_pareto_max_classes_for_singleton,
            }
        )
    elif (
        args.auto_pareto_min_classes_for_singleton
        <= n_classes
        <= args.auto_pareto_max_classes_for_singleton
        and args.auto_pareto_singleton_low <= fallback_rate <= args.auto_pareto_singleton_high
    ):
        pareto_guard = copy.deepcopy(singleton_allset_fallback)
        pareto_guard.name = "autorankcover_pareto_guard"
        pareto_guard.extra.update(
            {
                "route": "pareto_singleton_allset_band",
                "fallback_rate": fallback_rate,
                "auto_pareto_em_low": args.auto_pareto_em_low,
                "auto_pareto_em_high": args.auto_pareto_em_high,
                "auto_pareto_min_classes_for_singleton": args.auto_pareto_min_classes_for_singleton,
                "auto_pareto_singleton_low": args.auto_pareto_singleton_low,
                "auto_pareto_singleton_high": args.auto_pareto_singleton_high,
                "auto_pareto_max_classes_for_singleton": args.auto_pareto_max_classes_for_singleton,
            }
        )
    else:
        pareto_guard = copy.deepcopy(baseline_aps_safe)
        pareto_guard.name = "autorankcover_pareto_guard"
        pareto_guard.extra.update(
            {
                "route": "pareto_safe_default",
                "fallback_rate": fallback_rate,
                "auto_pareto_em_low": args.auto_pareto_em_low,
                "auto_pareto_em_high": args.auto_pareto_em_high,
                "auto_pareto_min_classes_for_singleton": args.auto_pareto_min_classes_for_singleton,
                "auto_pareto_singleton_low": args.auto_pareto_singleton_low,
                "auto_pareto_singleton_high": args.auto_pareto_singleton_high,
                "auto_pareto_max_classes_for_singleton": args.auto_pareto_max_classes_for_singleton,
            }
        )
    rows.append(result_to_row(bundle.name, model_name, seed, pareto_guard, baseline=baseline_aps))

    if (
        n_classes > args.auto_allset_max_classes
        and args.auto_sota_em_low <= fallback_rate < args.auto_sota_em_high
    ):
        sota_guard = copy.deepcopy(proposed_by_name["rankcover_em_aps"])
        sota_guard.name = "autorankcover_sota_guard"
        sota_guard.extra.update(
            {
                "route": "sota_em_mid_fallback",
                "fallback_rate": fallback_rate,
                "auto_sota_em_low": args.auto_sota_em_low,
                "auto_sota_em_high": args.auto_sota_em_high,
                "auto_sota_singleton_low": args.auto_sota_singleton_low,
                "auto_sota_singleton_high": args.auto_sota_singleton_high,
                "auto_sota_min_classes_for_singleton": args.auto_sota_min_classes_for_singleton,
                "auto_sota_max_classes_for_singleton": args.auto_sota_max_classes_for_singleton,
            }
        )
    elif (
        args.auto_sota_min_classes_for_singleton
        <= n_classes
        <= args.auto_sota_max_classes_for_singleton
        and args.auto_sota_singleton_low <= fallback_rate <= args.auto_sota_singleton_high
    ):
        sota_guard = copy.deepcopy(singleton_allset_fallback)
        sota_guard.name = "autorankcover_sota_guard"
        sota_guard.extra.update(
            {
                "route": "sota_singleton_allset_band",
                "fallback_rate": fallback_rate,
                "auto_sota_em_low": args.auto_sota_em_low,
                "auto_sota_em_high": args.auto_sota_em_high,
                "auto_sota_singleton_low": args.auto_sota_singleton_low,
                "auto_sota_singleton_high": args.auto_sota_singleton_high,
                "auto_sota_min_classes_for_singleton": args.auto_sota_min_classes_for_singleton,
                "auto_sota_max_classes_for_singleton": args.auto_sota_max_classes_for_singleton,
            }
        )
    else:
        sota_guard = copy.deepcopy(baseline_aps_safe)
        sota_guard.name = "autorankcover_sota_guard"
        sota_guard.extra.update(
            {
                "route": "sota_safe_default",
                "fallback_rate": fallback_rate,
                "auto_sota_em_low": args.auto_sota_em_low,
                "auto_sota_em_high": args.auto_sota_em_high,
                "auto_sota_singleton_low": args.auto_sota_singleton_low,
                "auto_sota_singleton_high": args.auto_sota_singleton_high,
                "auto_sota_min_classes_for_singleton": args.auto_sota_min_classes_for_singleton,
                "auto_sota_max_classes_for_singleton": args.auto_sota_max_classes_for_singleton,
            }
        )
    rows.append(result_to_row(bundle.name, model_name, seed, sota_guard, baseline=baseline_aps))

    if (
        n_classes > args.auto_allset_max_classes
        and args.auto_conservative_em_low <= fallback_rate < args.auto_conservative_em_high
    ):
        conservative_guard = copy.deepcopy(proposed_by_name["rankcover_em_aps"])
        conservative_guard.name = "autorankcover_conservative_guard"
        conservative_guard.extra.update(
            {
                "route": "conservative_em_mid_fallback",
                "fallback_rate": fallback_rate,
                "auto_conservative_em_low": args.auto_conservative_em_low,
                "auto_conservative_em_high": args.auto_conservative_em_high,
                "auto_conservative_singleton_low": args.auto_conservative_singleton_low,
                "auto_conservative_singleton_high": args.auto_conservative_singleton_high,
                "auto_conservative_min_classes_for_singleton": args.auto_conservative_min_classes_for_singleton,
                "auto_conservative_max_classes_for_singleton": args.auto_conservative_max_classes_for_singleton,
            }
        )
    elif (
        args.auto_conservative_min_classes_for_singleton
        <= n_classes
        <= args.auto_conservative_max_classes_for_singleton
        and args.auto_conservative_singleton_low <= fallback_rate <= args.auto_conservative_singleton_high
    ):
        conservative_guard = copy.deepcopy(singleton_allset_fallback)
        conservative_guard.name = "autorankcover_conservative_guard"
        conservative_guard.extra.update(
            {
                "route": "conservative_singleton_allset_band",
                "fallback_rate": fallback_rate,
                "auto_conservative_em_low": args.auto_conservative_em_low,
                "auto_conservative_em_high": args.auto_conservative_em_high,
                "auto_conservative_singleton_low": args.auto_conservative_singleton_low,
                "auto_conservative_singleton_high": args.auto_conservative_singleton_high,
                "auto_conservative_min_classes_for_singleton": args.auto_conservative_min_classes_for_singleton,
                "auto_conservative_max_classes_for_singleton": args.auto_conservative_max_classes_for_singleton,
            }
        )
    else:
        conservative_guard = copy.deepcopy(baseline_aps_safe)
        conservative_guard.name = "autorankcover_conservative_guard"
        conservative_guard.extra.update(
            {
                "route": "conservative_safe_default",
                "fallback_rate": fallback_rate,
                "auto_conservative_em_low": args.auto_conservative_em_low,
                "auto_conservative_em_high": args.auto_conservative_em_high,
                "auto_conservative_singleton_low": args.auto_conservative_singleton_low,
                "auto_conservative_singleton_high": args.auto_conservative_singleton_high,
                "auto_conservative_min_classes_for_singleton": args.auto_conservative_min_classes_for_singleton,
                "auto_conservative_max_classes_for_singleton": args.auto_conservative_max_classes_for_singleton,
            }
        )
    rows.append(result_to_row(bundle.name, model_name, seed, conservative_guard, baseline=baseline_aps))

    acc = accuracy_score(y_test, proba_test.argmax(axis=1))
    try:
        ll = log_loss(y_test, proba_test, labels=list(range(n_classes)))
    except Exception:
        ll = float("nan")
    for row in rows:
        row.update(
            {
                "n": int(bundle.X.shape[0]),
                "d": int(bundle.X.shape[1]),
                "classes": n_classes,
                "train": int(X_train.shape[0]),
                "cal": int(X_cal.shape[0]),
                "test": int(X_test.shape[0]),
                "accuracy": float(acc),
                "log_loss": float(ll),
                "risk_cuts": [float(x) for x in cuts.tolist()],
                "seconds": round(time.time() - t0, 3),
            }
        )
    return rows


def summarize(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    proposed = df[df["method"] != "aps_global"].copy()
    if proposed.empty:
        return pd.DataFrame()
    group_cols = ["model", "method"]
    agg = proposed.groupby(group_cols).agg(
        runs=("method", "size"),
        mean_violation_reduction=("violation_reduction", "mean"),
        median_violation_reduction=("violation_reduction", "median"),
        win_rate=("violation_reduction", lambda s: float((s > 0).mean())),
        compact_win_rate=("size_inflation", lambda s: float((s < 0).mean())),
        mean_size_inflation=("size_inflation", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_worst_violation=("worst_bin_violation", "mean"),
        mean_sscs=("sscs", "mean"),
        min_sscs=("sscs", "min"),
    )
    return agg.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="breast_cancer,wine,digits")
    parser.add_argument("--models", default="tabicl,rf")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=1800)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--tabicl-estimators", type=int, default=4)
    parser.add_argument("--tabpfn-estimators", type=int, default=4)
    parser.add_argument("--stability-k", type=int, default=3)
    parser.add_argument("--stability-train-subsample", type=int, default=512)
    parser.add_argument("--min-bin-cal", type=int, default=25)
    parser.add_argument("--min-class-cal", type=int, default=20)
    parser.add_argument("--raps-lambda", type=float, default=0.01)
    parser.add_argument("--raps-k-reg", type=int, default=1)
    parser.add_argument("--saps-lambdas", default="0.02,0.05,0.1,0.2,0.5")
    parser.add_argument("--saps-randomized-lambdas", default="")
    parser.add_argument("--compact-score-alpha", type=float, default=None)
    parser.add_argument("--rank-ultrasafe-alpha", type=float, default=0.01)
    parser.add_argument("--rank-floor-k", type=int, default=0)
    parser.add_argument("--rank-floor-min-classes", type=int, default=5)
    parser.add_argument("--rank-audit-min-sscs", type=float, default=1.00)
    parser.add_argument("--rank-audit-max-violation", type=float, default=0.0)
    parser.add_argument("--rank-audit-consistency-tol", type=float, default=0.0)
    parser.add_argument("--audit-frac", type=float, default=0.35)
    parser.add_argument("--min-audit-group", type=int, default=8)
    parser.add_argument("--auto-low-fallback", type=float, default=0.10)
    parser.add_argument("--auto-high-fallback", type=float, default=0.75)
    parser.add_argument("--auto-repair-audit-sscs", type=float, default=0.88)
    parser.add_argument("--auto-repair-max-size-ratio", type=float, default=0.70)
    parser.add_argument("--auto-allset-high-fallback", type=float, default=0.75)
    parser.add_argument("--auto-allset-small-fallback", type=float, default=0.50)
    parser.add_argument("--auto-small-allset-fallback", type=float, default=0.34)
    parser.add_argument("--auto-em-mid-low", type=float, default=0.45)
    parser.add_argument("--auto-em-mid-high", type=float, default=0.50)
    parser.add_argument("--auto-allset-candidate-fallback", type=float, default=0.65)
    parser.add_argument("--auto-allset-max-classes", type=int, default=4)
    parser.add_argument("--auto-safe-all-singleton-low", type=float, default=0.45)
    parser.add_argument("--auto-safe-all-singleton-high", type=float, default=0.65)
    parser.add_argument("--auto-pareto-em-low", type=float, default=0.45)
    parser.add_argument("--auto-pareto-em-high", type=float, default=0.50)
    parser.add_argument("--auto-pareto-singleton-low", type=float, default=0.25)
    parser.add_argument("--auto-pareto-singleton-high", type=float, default=0.55)
    parser.add_argument("--auto-pareto-min-classes-for-singleton", type=int, default=5)
    parser.add_argument("--auto-pareto-max-classes-for-singleton", type=int, default=10)
    parser.add_argument("--auto-sota-em-low", type=float, default=0.45)
    parser.add_argument("--auto-sota-em-high", type=float, default=0.50)
    parser.add_argument("--auto-sota-singleton-low", type=float, default=0.30)
    parser.add_argument("--auto-sota-singleton-high", type=float, default=0.55)
    parser.add_argument("--auto-sota-min-classes-for-singleton", type=int, default=5)
    parser.add_argument("--auto-sota-max-classes-for-singleton", type=int, default=5)
    parser.add_argument("--auto-conservative-em-low", type=float, default=0.45)
    parser.add_argument("--auto-conservative-em-high", type=float, default=0.50)
    parser.add_argument("--auto-conservative-singleton-low", type=float, default=0.45)
    parser.add_argument("--auto-conservative-singleton-high", type=float, default=0.55)
    parser.add_argument("--auto-conservative-min-classes-for-singleton", type=int, default=5)
    parser.add_argument("--auto-conservative-max-classes-for-singleton", type=int, default=5)
    parser.add_argument("--openml-cache", default="data/openml")
    parser.add_argument("--outdir", default="results/benchmark")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    dataset_tokens = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    partial_jsonl_path = outdir / f"partial_{stamp}.jsonl"
    partial_errors_path = outdir / f"partial_errors_{stamp}.jsonl"

    def append_jsonl(path: Path, items: List[Dict[str, object]]) -> None:
        if not items:
            return
        with path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for seed in seeds:
        for dataset_token in dataset_tokens:
            try:
                bundles = list(load_datasets(dataset_token, args.openml_cache, args.max_rows, seed))
            except Exception as exc:
                err = {"dataset": dataset_token, "model": "*", "seed": seed, "error": repr(exc)}
                errors.append(err)
                append_jsonl(partial_errors_path, [err])
                print(json.dumps({"ok": False, **err}), flush=True)
                continue
            for bundle in bundles:
                for model in models:
                    try:
                        run_rows = run_one(args, bundle, model, seed)
                        rows.extend(run_rows)
                        append_jsonl(partial_jsonl_path, run_rows)
                        print(json.dumps({"ok": True, "dataset": bundle.name, "model": model, "seed": seed, "rows": len(run_rows)}), flush=True)
                    except Exception as exc:
                        err = {"dataset": bundle.name, "model": model, "seed": seed, "error": repr(exc)}
                        errors.append(err)
                        append_jsonl(partial_errors_path, [err])
                        print(json.dumps({"ok": False, **err}), flush=True)

    jsonl_path = outdir / f"rankcover_results_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if errors:
        with (outdir / f"errors_{stamp}.jsonl").open("w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    if rows:
        df = pd.DataFrame(rows)
        csv_path = outdir / f"rankcover_results_{stamp}.csv"
        df.to_csv(csv_path, index=False)
        summary = summarize(rows)
        summary_path = outdir / f"summary_{stamp}.md"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("# RankCover benchmark summary\n\n")
            try:
                f.write(summary.to_markdown(index=False))
            except ImportError:
                f.write("```text\n")
                f.write(summary.to_string(index=False))
                f.write("\n```")
            f.write("\n")
        print(json.dumps({"jsonl": str(jsonl_path), "csv": str(csv_path), "summary": str(summary_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
