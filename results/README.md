# Compact final evidence

The `final/` directory contains publication-facing aggregates for the dense confidence-certified RankCover study. It excludes prediction caches, logs, partial files, duplicate raw outputs, and row-level test predictions.

Run the following command from the repository root to verify every committed aggregate:

```bash
python scripts/verify_aggregates.py
```

The benchmark and synthetic drivers regenerate detailed raw rows, candidate-audit traces, metadata, and reports in user-selected output directories.
