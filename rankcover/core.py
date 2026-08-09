from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np


EPS = 1e-12


@dataclass
class ConformalResult:
    name: str
    coverage: float
    avg_size: float
    worst_bin_coverage: float
    worst_bin_violation: float
    bin_coverages: Dict[str, float]
    bin_sizes: Dict[str, int]
    extra: Dict[str, float]


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("inf")
    scores = np.sort(scores)
    rank = int(np.ceil((scores.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), scores.size - 1)
    return float(scores[rank])


def aps_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1)
    return cumsum[np.arange(proba.shape[0]), ranks]


def aps_sets(proba: np.ndarray, q: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    include_sorted = cumsum <= q
    include_sorted[:, 0] = True

    # Include the first class that crosses the threshold.
    crossing = np.argmax(cumsum >= q, axis=1)
    include_sorted[np.arange(proba.shape[0]), crossing] = True

    sets = np.zeros_like(include_sorted, dtype=bool)
    rows = np.arange(proba.shape[0])[:, None]
    sets[rows, order] = include_sorted
    return sets


def lac_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    return 1.0 - proba[np.arange(y.size), y]


def lac_sets(proba: np.ndarray, q: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    sets = (1.0 - proba) <= q
    sets[np.arange(proba.shape[0]), proba.argmax(axis=1)] = True
    return sets


def raps_scores(proba: np.ndarray, y: np.ndarray, lam: float = 0.01, k_reg: int = 1) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1)
    penalties = lam * np.maximum((ranks + 1) - k_reg, 0)
    return cumsum[np.arange(proba.shape[0]), ranks] + penalties


def saps_scores(proba: np.ndarray, y: np.ndarray, lam: float = 0.10) -> np.ndarray:
    """Sorted adaptive prediction set scores.

    This is the deterministic conservative variant of SAPS: the top label
    receives the maximum probability score, and labels below the top receive
    a rank penalty. It follows the label-ranking SAPS shape while matching the
    non-randomized convention used by the APS/RAPS baselines in this project.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, axis=1)
    p_top = np.take_along_axis(proba, order[:, :1], axis=1)[:, 0]
    ranks = np.argmax(order == y[:, None], axis=1)
    return p_top + lam * ranks.astype(float)


def rank_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1)
    return ranks.astype(float) + 1.0


def rank_sets(proba: np.ndarray, q: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    k = int(np.ceil(q))
    k = min(max(k, 1), proba.shape[1])
    order = np.argsort(-proba, axis=1)
    sets = np.zeros_like(proba, dtype=bool)
    rows = np.arange(proba.shape[0])[:, None]
    sets[rows, order[:, :k]] = True
    return sets


def raps_sets(proba: np.ndarray, q: float, lam: float = 0.01, k_reg: int = 1) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    ranks = np.arange(1, proba.shape[1] + 1, dtype=float)[None, :]
    scores_sorted = cumsum + lam * np.maximum(ranks - k_reg, 0)
    include_sorted = scores_sorted <= q
    include_sorted[:, 0] = True

    sets = np.zeros_like(include_sorted, dtype=bool)
    rows = np.arange(proba.shape[0])[:, None]
    sets[rows, order] = include_sorted
    return sets


def saps_scores_all(proba: np.ndarray, lam: float = 0.10) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    p_top = np.take_along_axis(proba, order[:, :1], axis=1)
    ranks = np.arange(proba.shape[1], dtype=float)[None, :]
    sorted_scores = p_top + lam * ranks
    scores = np.empty_like(proba, dtype=float)
    rows = np.arange(proba.shape[0])[:, None]
    scores[rows, order] = sorted_scores
    return scores


def saps_sets(proba: np.ndarray, q: float, lam: float = 0.10) -> np.ndarray:
    scores = saps_scores_all(proba, lam=lam)
    sets = scores <= q
    sets[np.arange(proba.shape[0]), proba.argmax(axis=1)] = True
    return sets


def saps_randomized_scores(
    proba: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    lam: float = 0.10,
) -> np.ndarray:
    """Randomized SAPS scores from label-ranking conformal prediction.

    For non-top labels this uses ``p_top + lambda * (rank - U)`` with zero-based
    rank; for the top label the score is ``U * p_top``. This matches the SAPS
    randomized score distribution while keeping the implementation NumPy-only.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(-proba, axis=1)
    p_top = np.take_along_axis(proba, order[:, :1], axis=1)[:, 0]
    ranks = np.argmax(order == y[:, None], axis=1).astype(float)
    u = rng.random(proba.shape[0])
    scores = p_top + lam * (ranks - u)
    top = ranks == 0.0
    scores[top] = u[top] * p_top[top]
    return scores


def saps_randomized_scores_all(
    proba: np.ndarray,
    *,
    rng: np.random.Generator,
    lam: float = 0.10,
) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    increments = np.full_like(sorted_p, fill_value=lam, dtype=float)
    increments[:, 0] = sorted_p[:, 0]
    u = rng.random(sorted_p.shape)
    sorted_scores = np.cumsum(increments, axis=1) - increments * u
    scores = np.empty_like(proba, dtype=float)
    rows = np.arange(proba.shape[0])[:, None]
    scores[rows, order] = sorted_scores
    return scores


def saps_randomized_sets(
    proba: np.ndarray,
    q: float,
    *,
    rng: np.random.Generator,
    lam: float = 0.10,
) -> np.ndarray:
    scores = saps_randomized_scores_all(proba, rng=rng, lam=lam)
    sets = scores <= q
    # Keep the same non-empty top-label convention as the deterministic
    # APS/RAPS/SAPS baselines in this repository.
    sets[np.arange(proba.shape[0]), proba.argmax(axis=1)] = True
    return sets


def aps_scores_all(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    sorted_scores = np.cumsum(sorted_p, axis=1)
    scores = np.empty_like(proba, dtype=float)
    rows = np.arange(proba.shape[0])[:, None]
    scores[rows, order] = sorted_scores
    return scores


def lac_scores_all(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return 1.0 - proba


def raps_scores_all(proba: np.ndarray, lam: float = 0.01, k_reg: int = 1) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(-proba, axis=1)
    sorted_p = np.take_along_axis(proba, order, axis=1)
    ranks = np.arange(1, proba.shape[1] + 1, dtype=float)[None, :]
    sorted_scores = np.cumsum(sorted_p, axis=1) + lam * np.maximum(ranks - k_reg, 0)
    scores = np.empty_like(proba, dtype=float)
    rows = np.arange(proba.shape[0])[:, None]
    scores[rows, order] = sorted_scores
    return scores


def entropy_norm(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(proba, dtype=float), EPS, 1.0)
    h = -(proba * np.log(proba)).sum(axis=1)
    denom = np.log(proba.shape[1]) if proba.shape[1] > 1 else 1.0
    return h / max(denom, EPS)


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] == 1:
        return np.zeros(proba.shape[0], dtype=float)
    part = np.sort(proba, axis=1)[:, -2:]
    margin = part[:, 1] - part[:, 0]
    return 1.0 - np.clip(margin, 0.0, 1.0)


def normalize01(values: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    base = values if ref is None else np.asarray(ref, dtype=float)
    lo = float(np.nanpercentile(base, 1))
    hi = float(np.nanpercentile(base, 99))
    if hi <= lo + EPS:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def stability_score(stability_probas: np.ndarray | None) -> np.ndarray | None:
    if stability_probas is None:
        return None
    arr = np.asarray(stability_probas, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 2:
        return None
    return arr.max(axis=2).std(axis=0)


def risk_proxy(
    proba: np.ndarray,
    *,
    stability: np.ndarray | None = None,
    entropy_weight: float = 0.55,
    margin_weight: float = 0.25,
    stability_weight: float = 0.20,
    ref_proba: np.ndarray | None = None,
    ref_stability: np.ndarray | None = None,
) -> np.ndarray:
    ref_proba = proba if ref_proba is None else ref_proba
    ent = normalize01(entropy_norm(proba), entropy_norm(ref_proba))
    mar = normalize01(margin_uncertainty(proba), margin_uncertainty(ref_proba))
    if stability is not None:
        ref_st = stability if ref_stability is None else ref_stability
        sta = normalize01(stability, ref_st)
    else:
        sta = np.zeros_like(ent)
        stability_weight = 0.0
    weights = np.array([entropy_weight, margin_weight, stability_weight], dtype=float)
    if weights.sum() <= EPS:
        weights = np.array([1.0, 0.0, 0.0], dtype=float)
    weights = weights / weights.sum()
    return weights[0] * ent + weights[1] * mar + weights[2] * sta


def make_bins(cal_risk: np.ndarray, test_risk: np.ndarray, n_bins: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cal_risk = np.asarray(cal_risk, dtype=float)
    test_risk = np.asarray(test_risk, dtype=float)
    n_bins = max(1, int(n_bins))
    if n_bins == 1 or np.unique(cal_risk).size < 2:
        return np.zeros_like(cal_risk, dtype=int), np.zeros_like(test_risk, dtype=int), np.array([])
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(cal_risk, qs))
    cal_bins = np.searchsorted(cuts, cal_risk, side="right")
    test_bins = np.searchsorted(cuts, test_risk, side="right")
    return cal_bins, test_bins, cuts


def evaluate_sets(pred_sets: np.ndarray, y: np.ndarray, bins: np.ndarray, target: float) -> ConformalResult:
    y = np.asarray(y, dtype=int)
    covered = pred_sets[np.arange(y.size), y]
    set_sizes = pred_sets.sum(axis=1)
    avg_size = set_sizes.mean()
    bin_coverages: Dict[str, float] = {}
    bin_sizes: Dict[str, int] = {}
    worst = 1.0
    for b in sorted(np.unique(bins).tolist()):
        mask = bins == b
        if mask.sum() == 0:
            continue
        cov = float(covered[mask].mean())
        bin_coverages[str(int(b))] = cov
        bin_sizes[str(int(b))] = int(mask.sum())
        worst = min(worst, cov)
    size_coverages: Dict[str, float] = {}
    size_bin_sizes: Dict[str, int] = {}
    sscs = 1.0
    for k in sorted(np.unique(set_sizes).astype(int).tolist()):
        mask = set_sizes == k
        if mask.sum() == 0:
            continue
        cov = float(covered[mask].mean())
        size_coverages[str(int(k))] = cov
        size_bin_sizes[str(int(k))] = int(mask.sum())
        sscs = min(sscs, cov)
    violation = max(0.0, target - worst)
    return ConformalResult(
        name="",
        coverage=float(covered.mean()),
        avg_size=float(avg_size),
        worst_bin_coverage=float(worst),
        worst_bin_violation=float(violation),
        bin_coverages=bin_coverages,
        bin_sizes=bin_sizes,
        extra={
            "sscs": float(sscs),
            "size_stratified_coverages": size_coverages,
            "size_stratified_sizes": size_bin_sizes,
        },
    )


def split_aps(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
) -> ConformalResult:
    q = _finite_sample_quantile(aps_scores(cal_proba, cal_y), alpha)
    pred_sets = aps_sets(test_proba, q)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "aps_global"
    out.extra.update({"q": q})
    return out


def split_lac(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
) -> ConformalResult:
    q = _finite_sample_quantile(lac_scores(cal_proba, cal_y), alpha)
    pred_sets = lac_sets(test_proba, q)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "lac_global"
    out.extra.update({"q": q})
    return out


def split_raps(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
    lam: float = 0.01,
    k_reg: int = 1,
) -> ConformalResult:
    q = _finite_sample_quantile(raps_scores(cal_proba, cal_y, lam=lam, k_reg=k_reg), alpha)
    pred_sets = raps_sets(test_proba, q, lam=lam, k_reg=k_reg)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "raps_global"
    out.extra.update({"q": q, "raps_lambda": lam, "raps_k_reg": k_reg})
    return out


def split_saps(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
    lam: float = 0.10,
) -> ConformalResult:
    q = _finite_sample_quantile(saps_scores(cal_proba, cal_y, lam=lam), alpha)
    pred_sets = saps_sets(test_proba, q, lam=lam)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "saps_global"
    out.extra.update({"q": q, "saps_lambda": lam})
    return out


def split_saps_randomized(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
    lam: float = 0.10,
    random_state: int | None = None,
) -> ConformalResult:
    rng = np.random.default_rng(random_state)
    q = _finite_sample_quantile(saps_randomized_scores(cal_proba, cal_y, rng=rng, lam=lam), alpha)
    pred_sets = saps_randomized_sets(test_proba, q, rng=rng, lam=lam)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "saps_randomized_global"
    out.extra.update({"q": q, "saps_lambda": lam, "saps_random_state": -1 if random_state is None else int(random_state)})
    return out


def split_rank(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
) -> ConformalResult:
    q = _finite_sample_quantile(rank_scores(cal_proba, cal_y), alpha)
    pred_sets = rank_sets(test_proba, q)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "rank_global"
    out.extra.update({"q": q})
    return out


def split_rank_floor(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
    min_rank_size: int = 3,
    min_classes: int = 5,
) -> ConformalResult:
    raw_q = _finite_sample_quantile(rank_scores(cal_proba, cal_y), alpha)
    q = raw_q
    floor_applied = int(test_proba.shape[1] >= min_classes and min_rank_size > 0 and q < min_rank_size)
    if floor_applied:
        q = float(min(min_rank_size, test_proba.shape[1]))
    pred_sets = rank_sets(test_proba, q)
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = "rank_global_floor"
    out.extra.update(
        {
            "q": q,
            "raw_q": raw_q,
            "rank_floor_k": int(min_rank_size),
            "rank_floor_min_classes": int(min_classes),
            "rank_floor_applied": floor_applied,
        }
    )
    return out


def class_conditional_conformal(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    eval_bins: np.ndarray,
    min_class_cal: int = 20,
    score_all_fn: Callable[[np.ndarray], np.ndarray],
    name: str,
) -> ConformalResult:
    cal_scores_all = score_all_fn(cal_proba)
    test_scores_all = score_all_fn(test_proba)
    cal_scores = cal_scores_all[np.arange(cal_y.size), cal_y]
    global_q = _finite_sample_quantile(cal_scores, alpha)
    q_by_class: Dict[str, float] = {}
    q_values = np.full(test_proba.shape[1], global_q, dtype=float)
    for k in range(test_proba.shape[1]):
        mask = cal_y == k
        if int(mask.sum()) >= min_class_cal:
            q_values[k] = _finite_sample_quantile(cal_scores[mask], alpha)
        q_by_class[str(k)] = float(q_values[k])
    pred_sets = test_scores_all <= q_values[None, :]
    pred_sets[np.arange(test_proba.shape[0]), test_proba.argmax(axis=1)] = True
    out = evaluate_sets(pred_sets, test_y, eval_bins, 1.0 - alpha)
    out.name = name
    out.extra.update(
        {
            "global_q": global_q,
            "min_class_cal": min_class_cal,
            **{f"q_label_{k}": v for k, v in q_by_class.items()},
        }
    )
    return out


def rankcover_aps(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    min_bin_cal: int = 25,
) -> ConformalResult:
    scores = aps_scores(cal_proba, cal_y)
    global_q = _finite_sample_quantile(scores, alpha)
    pred_sets = np.zeros_like(test_proba, dtype=bool)
    q_by_bin: Dict[str, float] = {}
    for b in sorted(np.unique(test_bins).tolist()):
        cal_mask = cal_bins == b
        if int(cal_mask.sum()) < min_bin_cal:
            q = global_q
        else:
            q = _finite_sample_quantile(scores[cal_mask], alpha)
        q_by_bin[str(int(b))] = q
        test_mask = test_bins == b
        pred_sets[test_mask] = aps_sets(test_proba[test_mask], q)
    out = evaluate_sets(pred_sets, test_y, test_bins, 1.0 - alpha)
    out.name = "rankcover_bin_aps"
    out.extra.update({"global_q": global_q, **{f"q_bin_{k}": v for k, v in q_by_bin.items()}})
    return out


def _binned_conformal(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    min_bin_cal: int,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    set_fn: Callable[[np.ndarray, float], np.ndarray],
    name: str,
) -> ConformalResult:
    scores = score_fn(cal_proba, cal_y)
    global_q = _finite_sample_quantile(scores, alpha)
    pred_sets = np.zeros_like(test_proba, dtype=bool)
    q_by_bin: Dict[str, float] = {}
    for b in sorted(np.unique(test_bins).tolist()):
        cal_mask = cal_bins == b
        if int(cal_mask.sum()) < min_bin_cal:
            q = global_q
        else:
            q = _finite_sample_quantile(scores[cal_mask], alpha)
        q_by_bin[str(int(b))] = q
        test_mask = test_bins == b
        pred_sets[test_mask] = set_fn(test_proba[test_mask], q)
    out = evaluate_sets(pred_sets, test_y, test_bins, 1.0 - alpha)
    out.name = name
    out.extra.update({"global_q": global_q, **{f"q_bin_{k}": v for k, v in q_by_bin.items()}})
    return out


def rankcover_lac(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    min_bin_cal: int = 25,
    name: str = "rankcover_bin_lac",
) -> ConformalResult:
    return _binned_conformal(
        cal_proba,
        cal_y,
        test_proba,
        test_y,
        alpha=alpha,
        cal_bins=cal_bins,
        test_bins=test_bins,
        min_bin_cal=min_bin_cal,
        score_fn=lac_scores,
        set_fn=lac_sets,
        name=name,
    )


def rankcover_raps(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    min_bin_cal: int = 25,
    lam: float = 0.01,
    k_reg: int = 1,
    name: str = "rankcover_bin_raps",
) -> ConformalResult:
    return _binned_conformal(
        cal_proba,
        cal_y,
        test_proba,
        test_y,
        alpha=alpha,
        cal_bins=cal_bins,
        test_bins=test_bins,
        min_bin_cal=min_bin_cal,
        score_fn=lambda p, y: raps_scores(p, y, lam=lam, k_reg=k_reg),
        set_fn=lambda p, q: raps_sets(p, q, lam=lam, k_reg=k_reg),
        name=name,
    )


def rankcover_rank(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    test_proba: np.ndarray,
    test_y: np.ndarray,
    *,
    alpha: float,
    cal_bins: np.ndarray,
    test_bins: np.ndarray,
    min_bin_cal: int = 25,
    name: str = "rankcover_bin_rank",
) -> ConformalResult:
    return _binned_conformal(
        cal_proba,
        cal_y,
        test_proba,
        test_y,
        alpha=alpha,
        cal_bins=cal_bins,
        test_bins=test_bins,
        min_bin_cal=min_bin_cal,
        score_fn=rank_scores,
        set_fn=rank_sets,
        name=name,
    )


def result_to_row(
    dataset: str,
    model: str,
    seed: int,
    result: ConformalResult,
    *,
    baseline: ConformalResult | None = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "method": result.name,
        "coverage": result.coverage,
        "avg_size": result.avg_size,
        "worst_bin_coverage": result.worst_bin_coverage,
        "worst_bin_violation": result.worst_bin_violation,
        "bin_coverages": result.bin_coverages,
        "bin_sizes": result.bin_sizes,
    }
    if baseline is not None:
        base_violation = baseline.worst_bin_violation
        if base_violation > EPS:
            row["violation_reduction"] = (base_violation - result.worst_bin_violation) / base_violation
        else:
            row["violation_reduction"] = 0.0
        row["size_inflation"] = (result.avg_size - baseline.avg_size) / max(baseline.avg_size, EPS)
    row.update(result.extra)
    return row
