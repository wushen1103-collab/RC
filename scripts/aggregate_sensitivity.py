#!/usr/bin/env python3
"""Aggregate final dense-sequence Hard20 sensitivity runs."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def summarize(directory: str, family: str, setting: str) -> dict:
    raw = pd.read_csv(RESULTS / directory / "rankcover_raw.csv")
    raw = raw.loc[raw.method.eq("rankcover_confcert_adaptive")]
    return {
        "family": family,
        "setting": setting,
        "runs": len(raw),
        "tasks": raw.dataset_spec.nunique(),
        "coverage": raw.coverage.mean(),
        "sscs": raw.sscs.mean(),
        "violations": int(raw.violation_run.sum()),
        "mean_dmax": raw.dmax.mean(),
        "mean_dmean": raw.dmean.mean(),
        "mean_size": raw.avg_size.mean(),
        "pass_rate": raw.certificate_pass.mean(),
        "fallback_rate": raw.forced_uncertified_fallback.mean(),
    }


rows = []
for tag, setting in [("05", "0.05"), ("10", "0.10"), ("20", "0.20"), ("25", "0.25"), ("50", "0.50")]:
    rows.append(summarize(f"sensitivity/calibration_{tag}", "Calibration fraction", setting))
for tag, setting in [("20", "0.20"), ("35", "0.35"), ("50", "0.50")]:
    rows.append(summarize(f"sensitivity/audit_{tag}", "Audit allocation", setting))
for bins in (2, 3, 5):
    rows.append(summarize(f"sensitivity/bins_{bins}", "Risk bins", str(bins)))
for directory, setting in [
    ("sensitivity/risk_equal", "Equal entropy-margin (main)"),
    ("sensitivity/risk_entropy", "Entropy only"),
    ("sensitivity/risk_weighted", "Weighted entropy-margin (0.65/0.35)"),
    ("sensitivity/risk_instability", "Entropy-margin-instability"),
]:
    rows.append(summarize(directory, "Risk definition", setting))
for tag, setting in [("05", "0.05"), ("10", "0.10"), ("20", "0.20")]:
    rows.append(summarize(f"sensitivity/alpha_{tag}", "Evaluation alpha", setting))

out = pd.DataFrame(rows)
output_dir = RESULTS / "generated_summary"
output_dir.mkdir(parents=True, exist_ok=True)
out.to_csv(output_dir / "sensitivity_summary.csv", index=False)
print(out.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
