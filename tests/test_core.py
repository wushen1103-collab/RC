import numpy as np

from rankcover.core import evaluate_sets, make_bins, rank_scores, rank_sets


def test_rank_scores_and_sets_follow_probability_order() -> None:
    proba = np.array([[0.70, 0.20, 0.10], [0.10, 0.30, 0.60]])
    labels = np.array([1, 2])
    np.testing.assert_array_equal(rank_scores(proba, labels), np.array([2.0, 1.0]))
    np.testing.assert_array_equal(
        rank_sets(proba, q=2),
        np.array([[True, True, False], [False, True, True]]),
    )


def test_evaluation_reports_coverage_size_and_supported_sscs() -> None:
    prediction_sets = np.array(
        [
            [True, False, False],
            [True, True, False],
            [False, True, False],
            [False, True, True],
        ]
    )
    labels = np.array([0, 1, 2, 1])
    bins = np.array([0, 0, 1, 1])
    result = evaluate_sets(prediction_sets, labels, bins, target=0.75)
    assert result.coverage == 0.75
    assert result.avg_size == 1.5
    assert result.extra["sscs"] == 0.5
    assert result.worst_bin_violation == 0.25


def test_risk_bins_are_fitted_on_calibration_values() -> None:
    calibration_risk = np.array([0.0, 0.1, 0.2, 0.7, 0.8, 0.9])
    test_risk = np.array([0.05, 0.50, 0.95])
    calibration_bins, test_bins, edges = make_bins(calibration_risk, test_risk, n_bins=3)
    assert calibration_bins.shape == calibration_risk.shape
    assert test_bins.tolist() == [0, 1, 2]
    assert len(edges) == 2
