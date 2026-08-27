# RankCover

RankCover is a confidence-certified operating-point selector for conformal classification. A scoring split constructs a nested dense family of integer-rank prediction sets, a disjoint audit split supplies bin-wise error counts, and one-sided Clopper--Pearson bounds certify candidates in a fixed conservative-to-compact sequence. If the largest informative candidate cannot be certified, RankCover returns the all-label set.

This repository contains the final method implementation, public benchmark identifiers, experiment drivers, tests, and compact aggregate evidence used in the paper. Dataset caches, prediction caches, logs, and row-level intermediate files are intentionally excluded.

## Repository map

| Path | Purpose |
|---|---|
| `rankcover/core.py` | Conformal scores, set constructors, risk strata, and evaluation metrics |
| `rankcover/data.py` | Built-in and OpenML data loading and preprocessing |
| `scripts/run_confidence_rankcover.py` | Final dense confidence-certified RankCover and matched baselines |
| `scripts/run_synthetic_certificate.py` | Fixed-reference-population certificate validation |
| `scripts/aggregate_final.py` | Main, decision, blocked-inference, and implementation summaries |
| `scripts/aggregate_sensitivity.py` | Calibration, audit, binning, risk, and nominal-level summaries |
| `benchmarks/` | Hard20, TabICL119, and External20 task identifiers |
| `results/final/` | Lightweight publication-facing aggregate tables |
| `tests/` | Unit tests for conformal and certificate logic |

The older generic benchmark and control drivers remain available for baseline and historical-ablation reproduction. They are not components of the final RankCover policy.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install optional tree or foundation-model dependencies only when needed:

```bash
python -m pip install -e ".[trees]"
python -m pip install -e ".[foundation]"
```

## Quick check

This command uses a bundled scikit-learn dataset and requires no download:

```bash
python scripts/run_confidence_rankcover.py \
  --datasets wine \
  --models rf \
  --seeds 0 \
  --workers 1 \
  --n-jobs 2 \
  --bootstrap 50 \
  --outdir results/quickcheck
```

## Main benchmark commands

Hard20:

```bash
python scripts/run_confidence_rankcover.py \
  --benchmark Hard20 \
  --dataset-file benchmarks/hard20.txt \
  --models rf,lgbm,xgb,catboost,logreg \
  --seeds 0,1,2 \
  --workers 8 --n-jobs 2 \
  --bootstrap 200 \
  --outdir results/hard20
```

External20:

```bash
python scripts/run_confidence_rankcover.py \
  --benchmark External20 \
  --dataset-file benchmarks/external20.txt \
  --models rf,lgbm,xgb,catboost,logreg \
  --seeds 0,1,2 \
  --workers 8 --n-jobs 2 \
  --bootstrap 200 \
  --outdir results/external20
```

TabICL119:

```bash
python scripts/run_confidence_rankcover.py \
  --benchmark TabICL119 \
  --dataset-file benchmarks/tabicl119.txt \
  --models tabicl \
  --seeds 0,1,2 \
  --tabicl-estimators 4 \
  --workers 1 --n-jobs 2 \
  --bootstrap 200 \
  --outdir results/tabicl119
```

The defaults reproduce the final policy: evaluation miscoverage 0.10, confidence error level 0.05, three risk bins, a 35% audit share within calibration, equal entropy--margin risk weighting, and the dense integer-rank family.

## Synthetic certificate experiment

The default synthetic command reproduces the evaluated 162 conditions at each audit size:

```bash
python scripts/run_synthetic_certificate.py \
  --audit-sizes 60,120,240,480 \
  --workers 10 --n-jobs 2 \
  --outdir results/synthetic_certificate
```

## Verification

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/verify_aggregates.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact statistical rule, splits, metrics, task roles, result mapping, and environment notes.

## Paper result mapping

| Paper item | Reproducibility output |
|---|---|
| Table 4 and Figure 2 (synthetic certificate validation) | `results/final/synthetic_certificate_summary.csv` |
| Table 5 and Figure 3 (three-module comparison) | `results/final/main_method_aggregate.csv` |
| Table 6 (certificate and candidate-family ablations) | `results/final/main_method_aggregate.csv` and the rerun-level output of `scripts/run_confidence_rankcover.py` |
| Table 7 (utility conditional on certificate pass) | `results/final/pass_utility.csv` |
| Figure 4 (selected-rank distribution) | `results/final/selected_rank_distribution.csv` and `selected_rank_distribution_exact.csv` |
| Table 8 and Supplementary sensitivity tables | `results/final/sensitivity_summary.csv` |
| Supplementary trace, cardinality, split, and blocked-inference tables | `implementation_audit.json`, `class_cardinality_summary.csv`, `random_split_summary.csv`, and the two `paired_*_comparisons.csv` files |

## License

Released under the MIT License.
