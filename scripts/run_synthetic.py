from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scripts.run_benchmark import _rank_audit_guard, _retarget_violation
from rankcover.core import (
    aps_sets,
    evaluate_sets,
    make_bins,
    rank_sets,
    risk_proxy,
    split_aps,
    split_rank,
    split_rank_floor,
)


def weights_for(k: int, rho: float) -> list[float]:
    raw = np.geomspace(1.0, 1.0 / float(rho), num=k)
    raw = raw / raw.sum()
    return raw.tolist()


def temperature_scale(proba: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0) ** float(gamma)
    return p / p.sum(axis=1, keepdims=True)


def label_ranks(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-proba, axis=1)
    return np.argmax(order == y[:, None], axis=1) + 1


def fixed_topk(name: str, proba: np.ndarray, y: np.ndarray, bins: np.ndarray, k: int, target: float):
    pred = rank_sets(proba, float(k))
    out = evaluate_sets(pred, y, bins, target)
    out.name = name
    out.extra.update({"q": float(k), "fixed_topk": int(k)})
    return out


def row_from_result(meta: dict, result) -> dict:
    row = {
        **meta,
        "method": result.name,
        "coverage": result.coverage,
        "avg_size": result.avg_size,
        "worst_bin_coverage": result.worst_bin_coverage,
        "worst_bin_violation": result.worst_bin_violation,
    }
    row.update(result.extra)
    target = 1.0 - float(meta["alpha"])
    row["sscs"] = float(result.extra.get("sscs", np.nan))
    row["sscv_target_gap"] = max(0.0, target - row["sscs"])
    row["sscs_deficit_from_1"] = 1.0 - row["sscs"]
    return row


def run_one(k: int, rho: float, difficulty: str, gamma: float, seed: int, n_jobs: int, n_samples: int, alpha: float) -> list[dict]:
    sep = {"easy": 2.0, "medium": 1.0, "hard": 0.55}[difficulty]
    n_features = 24
    n_informative = min(18, max(4, 2 * k))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=k,
        n_clusters_per_class=1,
        weights=weights_for(k, rho),
        class_sep=sep,
        flip_y=0.015 if difficulty != "easy" else 0.005,
        random_state=seed + 1000 * k + int(rho) * 17,
    )
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_train, X_cal, y_train, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.357142857, stratify=y_tmp, random_state=seed + 1009)
    clf = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    cal_p = temperature_scale(clf.predict_proba(X_cal), gamma)
    test_p = temperature_scale(clf.predict_proba(X_test), gamma)
    cal_risk = risk_proxy(cal_p, ref_proba=cal_p, entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0)
    test_risk = risk_proxy(test_p, ref_proba=cal_p, entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0)
    _, test_bins, _ = make_bins(cal_risk, test_risk, n_bins=3)
    ranks = label_ranks(test_p, y_test)
    top1_error = float((ranks > 1).mean())
    top2_error = float((ranks > 2).mean())
    top3_error = float((ranks > 3).mean())
    meta = {
        "dataset": f"synthetic_k{k}_rho{rho:g}_{difficulty}_gamma{gamma:g}",
        "k": k,
        "rho": rho,
        "difficulty": difficulty,
        "gamma": gamma,
        "seed": seed,
        "alpha": alpha,
        "n": int(X.shape[0]),
        "train": int(X_train.shape[0]),
        "cal": int(X_cal.shape[0]),
        "test": int(X_test.shape[0]),
        "top1_error": top1_error,
        "top2_error": top2_error,
        "top3_error": top3_error,
        "min_class_count": int(np.bincount(y).min()),
    }
    safe_alpha = alpha / 4.0
    ultra_alpha = alpha / 10.0
    results = []
    aps_safe = split_aps(cal_p, y_cal, test_p, y_test, alpha=safe_alpha, eval_bins=test_bins)
    aps_safe.name = "aps_safe"
    aps_safe.extra["score_alpha"] = safe_alpha
    _retarget_violation(aps_safe, alpha)
    results.append(aps_safe)
    results.append(split_rank(cal_p, y_cal, test_p, y_test, alpha=alpha, eval_bins=test_bins))
    rank_safe = split_rank(cal_p, y_cal, test_p, y_test, alpha=safe_alpha, eval_bins=test_bins)
    rank_safe.name = "rank_safe"
    rank_safe.extra["score_alpha"] = safe_alpha
    _retarget_violation(rank_safe, alpha)
    results.append(rank_safe)
    for floor in [2, 3, 4, 5]:
        rf = split_rank_floor(cal_p, y_cal, test_p, y_test, alpha=safe_alpha, eval_bins=test_bins, min_rank_size=floor, min_classes=1)
        rf.name = f"rank_floor{floor}_safe"
        rf.extra["score_alpha"] = safe_alpha
        _retarget_violation(rf, alpha)
        results.append(rf)
    results.append(fixed_topk("fixed_top2", test_p, y_test, test_bins, 2, 1.0 - alpha))
    results.append(fixed_topk("fixed_top3", test_p, y_test, test_bins, 3, 1.0 - alpha))
    audit = _rank_audit_guard(
        cal_proba=cal_p,
        cal_y=y_cal,
        test_proba=test_p,
        test_y=y_test,
        cal_risk=cal_risk,
        test_bins=test_bins,
        seed=seed,
        alpha=alpha,
        safe_alpha=safe_alpha,
        ultrasafe_alpha=ultra_alpha,
        bins=3,
        audit_frac=0.35,
        min_bin_cal=25,
        min_audit_sscs=1.0,
        max_audit_violation=0.0,
        consistency_guard=True,
        consistency_tolerance=0.0,
        name="rank_audit_consistency",
    )
    results.append(audit)
    audit_floor3 = _rank_audit_guard(
        cal_proba=cal_p,
        cal_y=y_cal,
        test_proba=test_p,
        test_y=y_test,
        cal_risk=cal_risk,
        test_bins=test_bins,
        seed=seed,
        alpha=alpha,
        safe_alpha=safe_alpha,
        ultrasafe_alpha=ultra_alpha,
        bins=3,
        audit_frac=0.35,
        min_bin_cal=25,
        min_audit_sscs=1.0,
        max_audit_violation=0.0,
        consistency_guard=True,
        consistency_tolerance=0.0,
        rank_floor_k=3,
        rank_floor_min_classes=5,
        name="rank_audit_consistency_floor3",
    )
    results.append(audit_floor3)
    return [row_from_result(meta, result) for result in results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/synthetic")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=6000)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    partial = outdir / f"partial_{stamp}.jsonl"
    with partial.open("w", encoding="utf-8") as f:
        for seed in seeds:
            for k in [3, 5, 10, 20]:
                for rho in [1.0, 10.0, 50.0]:
                    for difficulty in ["easy", "medium", "hard"]:
                        for gamma in [0.7, 1.0, 1.8]:
                            try:
                                run_rows = run_one(k, rho, difficulty, gamma, seed, args.n_jobs, args.n_samples, args.alpha)
                                rows.extend(run_rows)
                                for row in run_rows:
                                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                print(json.dumps({"ok": True, "seed": seed, "k": k, "rho": rho, "difficulty": difficulty, "gamma": gamma, "rows": len(run_rows)}), flush=True)
                            except Exception as exc:
                                err = {"ok": False, "seed": seed, "k": k, "rho": rho, "difficulty": difficulty, "gamma": gamma, "error": repr(exc)}
                                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                                print(json.dumps(err), flush=True)
    df = pd.DataFrame(rows)
    csv_path = outdir / f"synthetic_mechanism_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    summary = (
        df.groupby(["k", "rho", "difficulty", "gamma", "method"], dropna=False)
        .agg(
            runs=("method", "size"),
            mean_top2_error=("top2_error", "mean"),
            mean_sscs=("sscs", "mean"),
            mean_sscv_target_gap=("sscv_target_gap", "mean"),
            mean_size=("avg_size", "mean"),
            max_violation=("worst_bin_violation", "max"),
            violation_runs=("worst_bin_violation", lambda s: int((s > 1e-12).sum())),
        )
        .reset_index()
    )
    summary_path = outdir / f"summary_{stamp}.csv"
    summary.to_csv(summary_path, index=False)
    md_path = outdir / f"summary_{stamp}.md"
    keep = summary[summary["method"].isin(["fixed_top2", "fixed_top3", "rank_safe", "rank_floor3_safe", "rank_audit_consistency_floor3", "aps_safe"])]
    md_path.write_text("# RankCover synthetic mechanism summary\n\n" + keep.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), "md": str(md_path), "rows": len(df)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
