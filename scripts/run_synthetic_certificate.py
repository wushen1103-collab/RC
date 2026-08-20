from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_confidence_rankcover import (  # noqa: E402
    Candidate,
    audit_candidate,
    candidate_family,
    confidence_rankcover,
    dmean_from_result,
    fixed_sequence_candidates,
    rank_sets_with_floor,
)
from rankcover.core import evaluate_sets, make_bins, risk_proxy  # noqa: E402


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def weights_for(k: int, rho: float) -> list[float]:
    raw = np.geomspace(1.0, 1.0 / float(rho), num=k)
    raw = raw / raw.sum()
    return raw.tolist()


def temperature_scale(proba: np.ndarray, gamma: float) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0) ** float(gamma)
    return p / p.sum(axis=1, keepdims=True)


def stratified_take(y: np.ndarray, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(y.size)
    labels, counts = np.unique(y, return_counts=True)
    stratify = y if labels.size >= 2 and counts.min() >= 2 else None
    take, rest = train_test_split(idx, train_size=size, random_state=seed, stratify=stratify)
    return np.asarray(take, dtype=int), np.asarray(rest, dtype=int)


def split_synthetic_pool(y: np.ndarray, n_train: int, n_score: int, n_audit: int, n_pop: int, seed: int) -> dict[str, np.ndarray]:
    train_idx, rest = stratified_take(y, n_train, seed)
    score_idx_rel, rest2_rel = stratified_take(y[rest], n_score, seed + 11)
    rest2 = rest[rest2_rel]
    score_idx = rest[score_idx_rel]
    audit_idx_rel, rest3_rel = stratified_take(y[rest2], n_audit, seed + 17)
    audit_idx = rest2[audit_idx_rel]
    rest3 = rest2[rest3_rel]
    if rest3.size < n_pop:
        raise ValueError(f"not enough population rows: have {rest3.size}, need {n_pop}")
    pop_idx_rel, _ = stratified_take(y[rest3], n_pop, seed + 23)
    pop_idx = rest3[pop_idx_rel]
    return {"train": train_idx, "score": score_idx, "audit": audit_idx, "population": pop_idx}


def run_distribution(
    *,
    k: int,
    rho: float,
    difficulty: str,
    gamma: float,
    audit_n: int,
    seed: int,
    args: dict[str, Any],
) -> list[dict[str, Any]]:
    sep = {"easy": 2.0, "medium": 1.0, "hard": 0.55}[difficulty]
    n_train = int(args["n_train"])
    n_score = int(args["n_score"])
    n_pop = int(args["n_population"])
    total = int(math.ceil((n_train + n_score + audit_n + n_pop) * 1.15))
    n_features = int(args["n_features"])
    n_informative = min(n_features - 2, max(4, 2 * k))
    X, y = make_classification(
        n_samples=total,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_repeated=0,
        n_classes=k,
        n_clusters_per_class=1,
        weights=weights_for(k, rho),
        class_sep=sep,
        flip_y=0.02 if difficulty == "hard" else 0.01,
        random_state=seed + 1009 * k + int(rho * 13),
    )
    splits = split_synthetic_pool(y, n_train, n_score, audit_n, n_pop, seed + 4099)
    clf = RandomForestClassifier(
        n_estimators=int(args["trees"]),
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=int(args["n_jobs"]),
        random_state=seed,
    )
    clf.fit(X[splits["train"]], y[splits["train"]])
    score_p = temperature_scale(clf.predict_proba(X[splits["score"]]), gamma)
    audit_p = temperature_scale(clf.predict_proba(X[splits["audit"]]), gamma)
    pop_p = temperature_scale(clf.predict_proba(X[splits["population"]]), gamma)
    y_score = y[splits["score"]]
    y_audit = y[splits["audit"]]
    y_pop = y[splits["population"]]

    score_risk = risk_proxy(score_p, ref_proba=score_p, entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0)
    audit_risk = risk_proxy(audit_p, ref_proba=score_p, entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0)
    pop_risk = risk_proxy(pop_p, ref_proba=score_p, entropy_weight=0.65, margin_weight=0.35, stability_weight=0.0)
    _, audit_bins, _ = make_bins(score_risk, audit_risk, n_bins=int(args["bins"]))
    _, pop_bins, _ = make_bins(score_risk, pop_risk, n_bins=int(args["bins"]))

    base_meta = {
        "dataset": f"synthetic_k{k}_rho{rho:g}_{difficulty}_gamma{gamma:g}",
        "k": k,
        "rho": rho,
        "difficulty": difficulty,
        "gamma": gamma,
        "audit_n": audit_n,
        "seed": seed,
        "alpha_eval": float(args["alpha_eval"]),
        "delta": float(args["confidence_delta"]),
        "n_train": n_train,
        "n_score": n_score,
        "n_population": n_pop,
        "min_class_count": int(np.bincount(y).min()),
    }

    conf_result, conf_meta, conf_audit_rows, _ = confidence_rankcover(
        cal_proba=np.vstack([score_p, audit_p]),
        cal_y=np.concatenate([y_score, y_audit]),
        test_proba=pop_p,
        test_y=y_pop,
        cal_risk=np.concatenate([score_risk, audit_risk]),
        test_risk=pop_risk,
        seed=seed,
        alpha_eval=float(args["alpha_eval"]),
        safe_alpha=float(args["safe_alpha"]),
        ultra_alpha=float(args["ultra_alpha"]),
        delta=float(args["confidence_delta"]),
        bins=int(args["bins"]),
        audit_frac=audit_n / float(n_score + audit_n),
        floors=parse_int_csv(args["candidate_floors"]),
        floor_min_classes=int(args["floor_min_classes"]),
        include_base=True,
        include_ultra=True,
        method_name="confidence_certificate",
        candidate_family_mode=str(args["candidate_family_mode"]),
    )
    true_dmax = float(conf_result.worst_bin_violation)
    true_dmean = dmean_from_result(conf_result, 1.0 - float(args["alpha_eval"]))

    candidates = candidate_family(
        score_proba=score_p,
        score_y=y_score,
        alpha_eval=float(args["alpha_eval"]),
        safe_alpha=float(args["safe_alpha"]),
        ultra_alpha=float(args["ultra_alpha"]),
        floors=parse_int_csv(args["candidate_floors"]),
        include_base=True,
        include_ultra=True,
        n_classes=k,
        family_mode=str(args["candidate_family_mode"]),
    )
    empirical_rows = []
    empirical_last_pass = None
    empirical_stop = None
    ordered_empirical = fixed_sequence_candidates(candidates, k)
    for cand in ordered_empirical:
        audit_result, audit_meta = audit_candidate(
            cand,
            audit_proba=audit_p,
            audit_y=y_audit,
            audit_bins=audit_bins,
            alpha_eval=float(args["alpha_eval"]),
            delta=float(args["confidence_delta"]),
            floor_min_classes=int(args["floor_min_classes"]),
            apply_floor_gate=False,
        )
        empirical_pass = bool(audit_result.worst_bin_violation <= 0.0)
        empirical_rows.append((cand, audit_meta, empirical_pass))
        if empirical_pass:
            empirical_last_pass = (cand, audit_result, audit_meta)
            continue
        empirical_stop = (cand, audit_result, audit_meta)
        break
    if empirical_last_pass is not None:
        emp_cand, _, emp_meta = empirical_last_pass
        emp_forced = False
    else:
        emp_cand = next(candidate for candidate in candidates if candidate.all_labels)
        emp_meta = {
            "certificate_max_upper": float(empirical_stop[2]["certificate_max_upper"]) if empirical_stop is not None else 1.0,
            "audit_coverage": 1.0,
        }
        emp_forced = True
    if emp_cand.all_labels:
        emp_sets = np.ones_like(pop_p, dtype=bool)
    else:
        emp_sets = rank_sets_with_floor(
            pop_p,
            emp_cand.q,
            emp_cand.floor,
            int(args["floor_min_classes"]),
            apply_floor_gate=False,
        )
    emp_result = evaluate_sets(emp_sets, y_pop, pop_bins, target=1.0 - float(args["alpha_eval"]))
    emp_true_dmax = float(emp_result.worst_bin_violation)

    return [
        {
            **base_meta,
            "method": "confidence_certificate",
            "selected_candidate": conf_meta["selected_candidate"],
            "certificate_pass": bool(conf_meta["certificate_pass"]),
            "forced_uncertified_fallback": bool(conf_meta["forced_uncertified_fallback"]),
            "audit_upper": float(conf_meta["certificate_max_upper"]),
            "audit_coverage": float(conf_meta["audit_coverage"]),
            "true_coverage": float(conf_result.coverage),
            "true_sscs": float(conf_result.extra.get("sscs", np.nan)),
            "true_dmax": true_dmax,
            "true_dmean": true_dmean,
            "true_violation": int(true_dmax > 1e-12),
            "false_certificate": int(bool(conf_meta["certificate_pass"]) and true_dmax > 1e-12),
            "avg_size": float(conf_result.avg_size),
        },
        {
            **base_meta,
            "method": "empirical_zero_audit",
            "selected_candidate": emp_cand.label,
            "certificate_pass": bool(not emp_forced),
            "forced_uncertified_fallback": bool(emp_forced),
            "audit_upper": float(emp_meta["certificate_max_upper"]),
            "audit_coverage": float(emp_meta["audit_coverage"]),
            "true_coverage": float(emp_result.coverage),
            "true_sscs": float(emp_result.extra.get("sscs", np.nan)),
            "true_dmax": emp_true_dmax,
            "true_dmean": dmean_from_result(emp_result, 1.0 - float(args["alpha_eval"])),
            "true_violation": int(emp_true_dmax > 1e-12),
            "false_certificate": int((not emp_forced) and emp_true_dmax > 1e-12),
            "avg_size": float(emp_result.avg_size),
        },
    ]


def run_unit(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    try:
        rows = run_distribution(**payload)
        for row in rows:
            row["elapsed_unit_seconds"] = float(time.time() - t0)
        return {"ok": True, "rows": rows}
    except Exception as exc:
        return {
            "ok": False,
            "rows": [],
            "error": {
                "payload": json.dumps({k: v for k, v in payload.items() if k != "args"}, sort_keys=True),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["method", "audit_n"], dropna=False)
        .agg(
            runs=("method", "size"),
            false_certificate_rate=("false_certificate", "mean"),
            true_violation_rate=("true_violation", "mean"),
            mean_true_coverage=("true_coverage", "mean"),
            mean_true_sscs=("true_sscs", "mean"),
            mean_true_dmax=("true_dmax", "mean"),
            mean_true_dmean=("true_dmean", "mean"),
            mean_size=("avg_size", "mean"),
            pass_rate=("certificate_pass", "mean"),
            forced_fallback_rate=("forced_uncertified_fallback", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results/synthetic_certificate")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--k-values", default="3,5,10")
    parser.add_argument("--rho-values", default="1,10,50")
    parser.add_argument("--difficulties", default="medium,hard")
    parser.add_argument("--gammas", default="0.7,1.0,1.8")
    parser.add_argument("--audit-sizes", default="60,120,240,480")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-score", type=int, default=500)
    parser.add_argument("--n-population", type=int, default=24000)
    parser.add_argument("--n-features", type=int, default=24)
    parser.add_argument("--trees", type=int, default=140)
    parser.add_argument("--alpha-eval", type=float, default=0.10)
    parser.add_argument("--safe-alpha", type=float, default=0.025)
    parser.add_argument("--ultra-alpha", type=float, default=0.01)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=3)
    parser.add_argument("--candidate-floors", default="0")
    parser.add_argument("--candidate-family-mode", choices=["sparse", "dense_rank"], default="dense_rank")
    parser.add_argument("--floor-min-classes", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metadata.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    seeds = parse_int_csv(args.seeds)
    payloads = []
    for seed in seeds:
        for k in parse_int_csv(args.k_values):
            for rho in parse_float_csv(args.rho_values):
                for difficulty in parse_csv(args.difficulties):
                    for gamma in parse_float_csv(args.gammas):
                        for audit_n in parse_int_csv(args.audit_sizes):
                            payloads.append(
                                {
                                    "k": k,
                                    "rho": rho,
                                    "difficulty": difficulty,
                                    "gamma": gamma,
                                    "audit_n": audit_n,
                                    "seed": seed,
                                    "args": vars(args),
                                }
                            )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    partial = outdir / "synthetic_certificate_raw.partial.csv"
    workers = max(1, int(args.workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_payload = {executor.submit(run_unit, payload): payload for payload in payloads}
        for i, future in enumerate(as_completed(future_to_payload), start=1):
            result = future.result()
            payload = future_to_payload[future]
            print(
                f"[{i}/{len(payloads)}] k={payload['k']} rho={payload['rho']} {payload['difficulty']} gamma={payload['gamma']} audit={payload['audit_n']} seed={payload['seed']} ok={result['ok']}",
                flush=True,
            )
            if result["ok"]:
                rows.extend(result["rows"])
            else:
                errors.append(result["error"])
            pd.DataFrame(rows).to_csv(partial, index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "synthetic_certificate_raw.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(outdir / "synthetic_certificate_summary.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(outdir / "synthetic_certificate_errors.csv", index=False)
    report = "# Synthetic certificate calibration\n\n"
    if not summary.empty:
        report += summary.to_markdown(index=False, floatfmt=".6f") + "\n"
    if errors:
        report += f"\nErrors: {len(errors)}\n"
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "rows": len(raw), "errors": len(errors)}, ensure_ascii=False), flush=True)


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    main()
