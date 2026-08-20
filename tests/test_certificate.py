import numpy as np

from scripts.run_confidence_rankcover import (
    Candidate,
    candidate_family,
    confidence_upper,
    fixed_sequence_candidates,
    saturated_rank_quantile,
)


def test_saturated_rank_quantile_returns_all_labels_at_boundary() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    assert saturated_rank_quantile(scores, alpha=0.01, n_classes=5) == 5.0


def test_fixed_sequence_is_conservative_to_compact() -> None:
    candidates = [
        Candidate("rank_1", 1.0, None, 0, False),
        Candidate("rank_3", 3.0, None, 0, False),
        Candidate("all_labels", 5.0, None, 5, True),
        Candidate("rank_2", 2.0, None, 0, False),
    ]
    ordered = fixed_sequence_candidates(candidates, n_classes=5)
    assert [candidate.q for candidate in ordered] == [3.0, 2.0, 1.0]


def test_dense_family_reserves_k_for_fallback() -> None:
    score_proba = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.75, 0.15],
            [0.10, 0.20, 0.70],
            [0.60, 0.25, 0.15],
        ]
    )
    score_y = np.array([0, 1, 2, 0])
    family = candidate_family(
        score_proba=score_proba,
        score_y=score_y,
        alpha_eval=0.10,
        safe_alpha=0.025,
        ultra_alpha=0.01,
        floors=[0],
        include_base=True,
        include_ultra=True,
        n_classes=3,
        family_mode="dense_rank",
    )
    assert family[-1].all_labels
    assert family[-1].q == 3.0
    assert all(candidate.q < 3.0 for candidate in family[:-1])


def test_zero_errors_still_has_positive_upper_bound() -> None:
    assert 0.0 < confidence_upper(errors=0, total=60, delta=0.05 / 3.0) < 1.0
