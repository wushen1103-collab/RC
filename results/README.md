# Result artifacts

`aggregates/` contains the compact publication-facing evidence released with RankCover. These CSV files are derived summaries: they do not contain individual samples, predictions, downloaded datasets, model checkpoints, or machine-specific metadata.

The experimental drivers write new outputs to separate subdirectories under `results/`. Generated raw and partial files are ignored by Git because full benchmark runs can be large. Run `python scripts/verify_aggregates.py` to validate every committed CSV against `aggregates.sha256`.
