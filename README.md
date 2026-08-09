# RankCover

RankCover is a reliability-prioritized conformal classifier for multiclass tabular prediction. It constructs nested rank-based prediction sets, audits a compact candidate on held-out calibration evidence, and switches at run level to a more conservative operating point when the audit fails. A class-count floor is activated only for sufficiently multiclass tasks.

This repository contains the implementation used for the experiments, the fixed benchmark identifiers, executable experiment drivers, and compact aggregate evidence. Large caches and row-level intermediate results are deliberately excluded.

## Repository map

| Path | Purpose |
|---|---|
| `rankcover/core.py` | Conformal scores, prediction sets, risk stratification, and evaluation metrics |
| `rankcover/data.py` | Built-in and OpenML loading, encoding, filtering, and stratified subsampling |
| `scripts/run_benchmark.py` | Main benchmark, baselines, operating points, and ablations |
| `scripts/run_custom_split.py` | Alternative train/calibration/test split experiments |
| `scripts/run_controls.py` | Generic-floor, size-matched, and paired control comparisons |
| `scripts/run_synthetic.py` | Controlled multiclass mechanism study |
| `scripts/run_audit_sensitivity.py` | Prespecified audit-design sensitivity study |
| `benchmarks/` | Fixed Hard20 and 119-task OpenML identifiers |
| `results/aggregates/` | Lightweight aggregate tables reported in the manuscript and supplement |

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

The base installation supports the random-forest and logistic-regression routes. Install optional estimators only when they are needed:

```bash
python -m pip install -e ".[trees]"
python -m pip install -e ".[foundation]"
```

## Quick check

The following command uses a bundled scikit-learn dataset and requires no network download:

```bash
python scripts/run_benchmark.py \
  --datasets wine \
  --models rf \
  --seeds 0 \
  --max-rows 500 \
  --n-jobs 2 \
  --stability-k 2 \
  --compact-score-alpha 0.025 \
  --rank-ultrasafe-alpha 0.01 \
  --rank-floor-k 3 \
  --rank-floor-min-classes 5 \
  --outdir results/quickcheck
```

## Reproducing the reported protocols

Dataset arguments are comma-separated. The committed benchmark files use `openml:<id>` tokens accepted by the loaders.

Linux/macOS:

```bash
HARD20=$(paste -sd, benchmarks/hard20.txt)
python scripts/run_benchmark.py \
  --datasets "$HARD20" \
  --models rf,lgbm,catboost,tabicl \
  --seeds 0,1,2 \
  --alpha 0.10 \
  --compact-score-alpha 0.025 \
  --rank-ultrasafe-alpha 0.01 \
  --rank-floor-k 3 \
  --rank-floor-min-classes 5 \
  --audit-frac 0.35 \
  --max-rows 1800 \
  --outdir results/benchmark
```

PowerShell:

```powershell
$hard20 = (Get-Content benchmarks/hard20.txt) -join ','
python scripts/run_benchmark.py `
  --datasets $hard20 `
  --models rf,lgbm,catboost,tabicl `
  --seeds 0,1,2 `
  --alpha 0.10 `
  --compact-score-alpha 0.025 `
  --rank-ultrasafe-alpha 0.01 `
  --rank-floor-k 3 `
  --rank-floor-min-classes 5 `
  --audit-frac 0.35 `
  --max-rows 1800 `
  --outdir results/benchmark
```

The same list can be passed to the control and audit-design drivers:

```bash
python scripts/run_controls.py --datasets "$HARD20" --outdir results/controls
python scripts/run_audit_sensitivity.py --datasets "$HARD20" --outdir results/audit_sensitivity
python scripts/run_synthetic.py --seeds 0,1,2,3,4 --n-samples 6000 --outdir results/synthetic
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the data split, operating points, metrics, experiment-to-table mapping, and environment notes.

## Aggregate evidence

The files under `results/aggregates/` are publication-facing summaries rather than row-level predictions. They cover the main comparisons, paired controls, repeated-split deficits, audit counterfactuals, floor and trigger sensitivity, multiple evaluation targets, calibration-size sensitivity, synthetic mechanism tests, support-threshold checks, and audit-design sensitivity. Each file is small enough to inspect directly and is protected by a committed SHA-256 manifest.

```bash
python scripts/verify_aggregates.py
```

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```

## License

Released under the MIT License.
