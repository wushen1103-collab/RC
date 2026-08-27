# Reproducibility guide

## Final RankCover policy

For each task--model--seed run, the data are split into training, calibration, and test partitions. The calibration partition is divided again into disjoint scoring and audit subsets. The scoring subset supplies the finite-sample rank threshold `q0`; the audit subset alone determines whether a candidate is certified. Test labels are accessed only after selection.

The final candidate family is the dense nested path

```text
q0, q0 + 1, ..., K - 1
```

with rank `K` reserved for all-label fallback. Unique informative candidates are inspected from `K - 1` toward `q0`. For candidate `q` and prespecified risk bin `b`, let `e[q,b]` be the number of audit errors among `n[q,b]` audit examples. RankCover computes the one-sided Clopper--Pearson upper bound

```text
U[q,b] = BetaQuantile(1 - delta / B; e[q,b] + 1, n[q,b] - e[q,b])
```

with the standard boundary value `U = 1` when every audit example is an error. Candidate `q` passes when `max_b U[q,b] <= alpha_eval`. The fixed sequence stops at the first failure and returns the smallest previously certified set. Failure of the first candidate returns all labels. Pointwise nesting preserves the scoring-split conformal marginal guarantee, while the fixed unsafe boundary controls erroneous adaptive deployment without candidate-wise correction.

The matched empirical-zero comparator uses the identical dense family, order, and stopping rule and replaces only the confidence-bound acceptance predicate with zero observed audit errors.

## Main protocol

| Quantity | Main value |
|---|---:|
| Evaluation miscoverage, `alpha_eval` | 0.10 |
| Certificate error level, `delta` | 0.05 |
| Train/calibration/test fractions | 0.45/0.25/0.30 |
| Audit share of calibration | 0.35 |
| Candidate family | all integer ranks from `q0` to `K - 1` |
| Risk statistic | 0.5 predictive entropy + 0.5 top-label margin uncertainty |
| Risk bins | 3 equal-frequency bins fitted from calibration evidence |
| Seeds | 0, 1, 2 |
| Explicit fallback | all labels |

Small calibration sets use the saturated conformal rank quantile: when the finite-sample order-statistic index reaches `n + 1`, the threshold is set to `K`, which returns all labels.

## Benchmark roles

- `benchmarks/hard20.txt`: 20 prespecified multiclass stress tasks, evaluated with five conventional backbones and three seeds (300 runs). These tasks also occur in TabICL119, so Hard20 is a backbone-stress module rather than an independent task holdout.
- `benchmarks/tabicl119.txt`: 119 public OpenML classification tasks evaluated with the common TabICL prediction source and three seeds (357 runs).
- `benchmarks/external20.txt`: 20 retained tasks from a source list frozen before RankCover evaluation. Loader failures, exact/reuploaded tasks, within-pool duplicates, and documented parent-source relatives were removed without consulting RankCover outcomes and without replacement (300 runs).

OpenML inputs are encoded, missing numeric values are median-imputed, categorical features are ordinal-encoded with an explicit unknown category, and low-support classes are removed according to the loader rules. Each main run records the dataset specification, model, seed, split, audit counts, confidence bounds, candidate order, stopping boundary, selected rank, fallback state, and test metrics.

## Metrics

- **Coverage:** fraction of test labels contained in their prediction sets.
- **Mean set size:** average prediction-set cardinality.
- **SSCS:** minimum empirical coverage across supported prediction-set-size strata. For a fixed-rank predictor, cardinality is constant within a run and SSCS equals empirical coverage.
- **Maximum risk-bin deficit (`Dmax`):** largest positive shortfall from `1 - alpha_eval` across risk bins.
- **Mean risk-bin deficit (`Dmean`):** mean positive shortfall across risk bins.
- **Violation run:** a run with `Dmax > 0`.
- **False certificate:** a synthetic condition in which a non-fallback candidate is certified although its fixed reference population has `Dmax > 0`.

RankCover does not optimize unconditional set-size minimization. It selects the smallest candidate supported by the prespecified certificate; set size and fallback rate quantify the efficiency cost of certification.

## Synthetic reference populations

The final grid crosses three seeds, class counts 3/5/10, imbalance ratios 1/10/50, two difficulty levels, and three probability temperatures, producing 162 conditions per audit size. Audit sizes are 60, 120, 240, and 480. Each selected policy is evaluated on a fixed 24,000-observation reference population.

## Compact result mapping

| File under `results/final/` | Evidence |
|---|---|
| `main_method_aggregate.csv` | All main methods on External20, Hard20, and TabICL119 |
| `synthetic_certificate_summary.csv` | Confidence-certificate and empirical-zero false-certificate results |
| `pass_utility.csv` | Conditional utility among certified non-fallback decisions |
| `selected_rank_distribution*.csv` | Selected informative ranks and all-label fallback |
| `sensitivity_summary.csv` | Calibration, audit, binning, risk, and nominal-level analyses |
| `class_cardinality_summary.csv` | Class-cardinality behavior |
| `random_split_summary.csv` | Theory-aligned random-split check |
| `paired_task_blocked_comparisons.csv` | Task-blocked paired inference |
| `paired_source_family_blocked_comparisons.csv` | Source-family-blocked paired inference |
| `source_family_manifest.csv` | Source-family grouping used in blocked inference |
| `implementation_audit.json` | Candidate-order and stopping-rule validation counts |

In the paper, `synthetic_certificate_summary.csv` supplies Table 4 and Figure 2; `main_method_aggregate.csv` supplies the aggregate values in Tables 5--6 and Figure 3; `pass_utility.csv` supplies Table 7; `selected_rank_distribution*.csv` supplies Figure 4; and `sensitivity_summary.csv` supplies Table 8 and the corresponding Supplementary sensitivity tables. The remaining files map to the explicitly named Supplementary trace, class-cardinality, random-split, and blocked-inference analyses.

Only compact aggregate evidence is versioned. Full reruns regenerate raw rows, candidate-audit traces, dataset-level summaries, reports, and metadata in the requested output directories.

## Determinism, caching, and hardware

Every reported run carries an explicit seed. CPU estimators receive `--n-jobs` when supported. TabICL and other optional accelerators may depend on library and device implementations; record the resolved environment with `python -m pip freeze`. OpenML data and prediction caches are intentionally unversioned and can be redirected with `--openml-cache` and `--prediction-cache-dir`. No local machine path, account name, or server-specific assumption is embedded in the release.

## Historical ablations

The generic baseline scripts retain construction-level safe, ultra-safe, and floor-based rules because they are reported as historical ablations. They are not components of the final dense RankCover procedure.
