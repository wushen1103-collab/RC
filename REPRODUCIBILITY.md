# Reproducibility guide

## Experimental unit and split

An experimental run is a task-model-seed tuple. The standard protocol uses a stratified 45%/25%/30% train/calibration/test split. OpenML inputs are encoded, missing numeric values are median-imputed, categorical features are ordinal-encoded with an explicit unknown category, classes with fewer than eight observations are removed, and each task is capped at the requested maximum row count through stratified subsampling.

The two committed task collections are:

- `benchmarks/hard20.txt`: 20 prespecified multiclass stress tasks.
- `benchmarks/tabicl119.txt`: 119 public OpenML classification tasks.

## Prespecified RankCover operating points

The reported principal protocol uses:

| Quantity | Value | Role |
|---|---:|---|
| Evaluation target, `alpha` | 0.10 | Coverage and audit target of 0.90 |
| Safe construction level | 0.025 | Compact rank candidate |
| Ultra-safe construction level | 0.01 | Conservative fallback candidate |
| Calibration audit fraction | 0.35 | Held-out audit decision layer |
| Minimum audit SSCS | 1.00 | Audit pass requirement |
| Maximum audit risk-bin deficit | 0.00 | Audit pass requirement |
| Rank floor | 3 | Minimum set size when activated |
| Floor activation threshold | 5 classes | Restricts the floor to sufficiently multiclass tasks |
| Risk strata | 3 | Low, medium, and high model-risk groups |

For a fitted classifier, instability and class ambiguity form the model-risk proxy used to create the risk strata. The compact and fallback candidates are nested. The held-out calibration audit selects the compact candidate only when its size-stratified coverage and worst risk-bin deficit meet the thresholds; otherwise the fallback candidate is deployed. The consistency guard preserves the nested conservative route if the held-out size comparison is not maintained.

## Metrics

- **Coverage:** fraction of test labels contained in their prediction sets.
- **Mean set size:** average prediction-set cardinality; smaller is more efficient at comparable reliability.
- **SSCS:** minimum empirical coverage across supported prediction-set-size strata. For a fixed-threshold rank predictor, set cardinality is constant within a run, so SSCS equals empirical coverage for that run.
- **Worst risk-bin deficit:** largest positive shortfall from the evaluation coverage target across model-risk strata.
- **Mean risk-bin deficit:** support-weighted average positive shortfall across model-risk strata.
- **Violation run:** a run with a positive worst risk-bin deficit.

## Experiment drivers

### Main benchmark

`scripts/run_benchmark.py` contains the main routes, conformal baselines, rank operating points, ablations, risk estimation, and aggregation. Its important command-line parameters are explicit in `python scripts/run_benchmark.py --help`.

### Paired controls

`scripts/run_controls.py` compares RankCover against generic floors and size-matched APS, RAPS, SAPS, and rank controls under the same task-model-seed blocks.

### Custom split and calibration-size study

`scripts/run_custom_split.py` reuses the benchmark implementation while replacing only the train/calibration/test proportions. This keeps the estimator, score construction, audit, and evaluation code fixed across split settings.

### Synthetic mechanism study

`scripts/run_synthetic.py` varies class count, class overlap, difficulty, and class imbalance under controlled data generation. The default uses five seeds and 6,000 samples per condition.

### Audit-design sensitivity

`scripts/run_audit_sensitivity.py` evaluates the prespecified design alongside changes to the audit fraction and risk-proxy fusion weights. It uses theorem-nested safe and ultra-safe thresholds in every design.

## Aggregate-file mapping

| File | Evidence represented |
|---|---|
| `published_rank.csv` | Main grouped comparison on Hard20 and the 119-task benchmark |
| `controls.csv` | Generic-floor and size-matched controls |
| `controls_paired.csv` | Paired RankCover-minus-control comparisons |
| `dataset_blocked_paired_repeated_split.csv` | Dataset-blocked paired inference across repeated splits |
| `deficit_magnitude_repeated_split.csv` | Repeated-split SSCS, maximum deficit, mean deficit, and violation counts |
| `audit_counterfactual.csv` | Compact-versus-fallback counterfactuals conditioned on audit decisions |
| `audit_paired.csv` | Paired audit counterfactual differences |
| `floor_k.csv` | Rank-floor sensitivity |
| `trigger.csv` | Floor activation-threshold sensitivity |
| `multi_alpha.csv` | Multiple evaluation-target analysis |
| `calibration.csv` | Calibration-size sensitivity |
| `calibration_audit.csv` | Audit behavior across calibration sizes |
| `stability.csv` | Estimator- and seed-level stability |
| `synthetic.csv` | Controlled mechanism study |
| `support_aware_sscs.csv` | Support-threshold SSCS sanity check |
| `audit_design_sensitivity.csv` | Audit-fraction and risk-fusion sensitivity |

## Data access and caching

Built-in datasets require no download. OpenML tasks are downloaded through the `openml` Python client and cached under `data/openml/` by default. To use another cache location, pass `--openml-cache <path>`. Dataset files and caches are intentionally not versioned.

The benchmark scripts record command-line parameters and write raw, summary, and error files into the selected output directory. Preserve those generated parameter files when conducting a new full run.

## Determinism and hardware

Every reported run carries an explicit seed. CPU estimators receive the requested `--n-jobs` value when supported. Foundation-model and tree-library behavior can also depend on library version and device implementation; record the resolved Python environment with `python -m pip freeze` for an independent rerun. No server-specific path or hardware assumption is embedded in the repository.
