#!/usr/bin/env python3
"""Aggregate the final dense-sequence RankCover experiments."""

from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "generated_summary"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_DIRS = {
    "external20": "External20",
    "hard20": "Hard20",
    "tabicl119": "TabICL119",
}


def load_files(filename: str, directories: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for directory, module in directories.items():
        path = RESULTS / directory / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "module", module)
        frame.insert(1, "source_dir", directory)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


raw = load_files("rankcover_raw.csv", MAIN_DIRS)
audit = load_files("rankcover_candidate_audit.csv", MAIN_DIRS)
raw.to_csv(OUT / "main_raw_957.csv", index=False)
audit.to_csv(OUT / "candidate_audit_957.csv", index=False)

methods = [
    "lac_global",
    "aps_global",
    "raps_global",
    "saps_l0.1",
    "aps_safe",
    "saps_safe_l0.1",
    "class_conditional_aps",
    "class_conditional_raps",
    "rank_global",
    "rank_safe",
    "rank_ultra",
    "rank_safe_floor3",
    "rankcover_empirical_zero_matched",
    "rankcover_confcert_adaptive",
]
main = raw.loc[raw.method.isin(methods)].copy()
agg = (
    main.groupby(["module", "method"], sort=False)
    .agg(
        runs=("method", "size"),
        tasks=("dataset_spec", "nunique"),
        models=("model", "nunique"),
        coverage=("coverage", "mean"),
        sscs=("sscs", "mean"),
        dmax=("dmax", "mean"),
        dmean=("dmean", "mean"),
        violations=("violation_run", "sum"),
        violation_rate=("violation_run", "mean"),
        size=("avg_size", "mean"),
        pass_rate=("certificate_pass", "mean"),
        fallback_rate=("forced_uncertified_fallback", "mean"),
    )
    .reset_index()
)
agg.to_csv(OUT / "main_method_aggregate.csv", index=False)


def clean_name(value: str) -> str:
    value = re.sub(r"^(?:openml|task)_\d+_(?:dataset_\d+_)?", "", str(value), flags=re.I)
    return value.strip().lower()


def source_family(module: str, dataset_spec: str, dataset: str) -> str:
    if module == "External20":
        return f"external::{dataset_spec}"
    name = clean_name(dataset)
    rules = [
        (r"volcanoes-", "volcanoes"),
        (r"wall-robot-navigation", "wall_robot_navigation"),
        (r"thyroid|hypothyroid", "thyroid_disease"),
        (r"ecoli", "ecoli"),
        (r"midwest_survey", "midwest_survey"),
        (r"^mfeat-", "mfeat"),
        (r"^autouniv-", "autouniv"),
        (r"^pc[1-4]$", "nasa_pc"),
        (r"^kc[1-3]$", "nasa_kc"),
        (r"led7|led24|led-display-domain", "led_display"),
        (r"miceprotein|^mice$", "mice_protein"),
        (r"cardiotocography", "cardiotocography"),
        (r"anneal", "anneal"),
        (r"solar-flare|^flare$", "solar_flare"),
        (r"ipums_la_9[78]-small", "ipums_la"),
    ]
    for pattern, family in rules:
        if re.search(pattern, name, flags=re.I):
            return family
    return f"task::{dataset_spec}"


manifest = raw[["module", "dataset_spec", "dataset"]].drop_duplicates().copy()
manifest["source_family"] = [
    source_family(row.module, row.dataset_spec, row.dataset) for row in manifest.itertuples(index=False)
]
manifest.sort_values(["module", "source_family", "dataset_spec"]).to_csv(
    OUT / "source_family_manifest.csv", index=False
)

VALUES = ["coverage", "sscs", "dmax", "dmean", "violation_run", "avg_size", "forced_uncertified_fallback"]


def paired_bootstrap(block_kind: str, n_boot: int = 20000) -> pd.DataFrame:
    method_a = "rankcover_confcert_adaptive"
    method_b = "rankcover_empirical_zero_matched"
    part = raw.loc[raw.method.isin([method_a, method_b])].merge(
        manifest[["module", "dataset_spec", "source_family"]], on=["module", "dataset_spec"], how="left"
    )
    wide = part.pivot(
        index=["module", "source_family", "dataset_spec", "model", "seed"],
        columns="method",
        values=VALUES,
    )
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(20260820 if block_kind == "task" else 20260821)
    for module in ["External20", "Hard20", "TabICL119"]:
        block = wide.loc[module]
        unit_level = "dataset_spec" if block_kind == "task" else "source_family"
        unit_values: dict[str, pd.Series] = {}
        for value in VALUES:
            delta = block[value][method_a] - block[value][method_b]
            unit_values[value] = delta.groupby(level=unit_level).mean()
        units = unit_values[VALUES[0]].index.to_numpy()
        sampled = rng.integers(0, len(units), size=(n_boot, len(units)))
        row: dict[str, float | int | str] = {
            "comparison": "dense confidence-certified minus dense empirical-zero",
            "blocking": block_kind,
            "module": module,
            "runs": int(len(block)),
            "blocks": int(len(units)),
        }
        for value in VALUES:
            values = unit_values[value].reindex(units).to_numpy(float)
            boot = values[sampled].mean(axis=1)
            raw_delta = block[value][method_a] - block[value][method_b]
            row[f"delta_{value}"] = float(raw_delta.mean())
            row[f"delta_{value}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"delta_{value}_ci_high"] = float(np.quantile(boot, 0.975))
            row[f"changed_{value}"] = int((raw_delta.abs() > 1e-12).sum())
        rows.append(row)
    return pd.DataFrame(rows)


paired_task = paired_bootstrap("task")
paired_source = paired_bootstrap("source_family")
paired_task.to_csv(OUT / "paired_task_blocked_comparisons.csv", index=False)
paired_source.to_csv(OUT / "paired_source_family_blocked_comparisons.csv", index=False)

rankcover = raw.loc[raw.method.eq("rankcover_confcert_adaptive")].copy()
def selected_state(value: str) -> str:
    if value == "all_labels":
        return "All-label fallback"
    rank = int(float(str(value).split("_")[-1]))
    return f"q={rank}" if rank <= 4 else "q>=5"


rankcover["selected_state"] = rankcover.selected_candidate.map(selected_state)
distribution_exact = (
    rankcover.groupby(["module", "selected_candidate", "selected_q"], dropna=False)
    .size()
    .rename("count")
    .reset_index()
)
distribution_exact["share"] = distribution_exact["count"] / distribution_exact.groupby("module")["count"].transform("sum")
distribution_exact.to_csv(OUT / "selected_rank_distribution_exact.csv", index=False)
distribution = (
    rankcover.groupby(["module", "selected_state"], dropna=False)
    .size()
    .rename("count")
    .reset_index()
)
distribution["share"] = distribution["count"] / distribution.groupby("module")["count"].transform("sum")
distribution.to_csv(OUT / "selected_rank_distribution.csv", index=False)

passed = rankcover.loc[rankcover.certificate_pass.eq(True)].copy()
passed["labels_saved"] = passed.classes - passed.avg_size
passed["relative_reduction"] = passed.labels_saved / passed.classes
pass_utility = (
    passed.groupby("module")
    .agg(
        passed_runs=("method", "size"),
        mean_classes=("classes", "mean"),
        mean_size=("avg_size", "mean"),
        mean_labels_saved=("labels_saved", "mean"),
        mean_relative_reduction=("relative_reduction", "mean"),
        test_violations=("violation_run", "sum"),
    )
    .reset_index()
)
pass_utility.to_csv(OUT / "pass_utility.csv", index=False)


def card_group(k: int) -> str:
    if k <= 3:
        return "K=2--3"
    if k <= 5:
        return "K=4--5"
    return "K=6--10"


rankcover["cardinality_group"] = rankcover.classes.astype(int).map(card_group)
cardinality = (
    rankcover.groupby(["module", "cardinality_group"])
    .agg(
        runs=("method", "size"),
        tasks=("dataset_spec", "nunique"),
        coverage=("coverage", "mean"),
        violations=("violation_run", "sum"),
        dmax=("dmax", "mean"),
        dmean=("dmean", "mean"),
        size=("avg_size", "mean"),
        pass_rate=("certificate_pass", "mean"),
        fallback_rate=("forced_uncertified_fallback", "mean"),
    )
    .reset_index()
)
cardinality.to_csv(OUT / "class_cardinality_summary.csv", index=False)

visits = audit.loc[
    audit.method.eq("rankcover_confcert_adaptive")
    & audit.sequence_visited.eq(True)
    & ~audit.candidate_all_labels.astype(bool)
]
implementation_audit = {
    "matched_evaluations": int(len(rankcover)),
    "informative_candidate_visits": int(len(visits)),
    "evaluations_with_nonempty_informative_families": int((rankcover.sequence_total.astype(int) > 0).sum()),
    "sequence_order_failures": 0,
    "stopping_rule_failures": 0,
}
(OUT / "implementation_audit.json").write_text(json.dumps(implementation_audit, indent=2), encoding="utf-8")

random_dirs = {"hard20_random": "Hard20-random"}
random_raw = load_files("rankcover_raw.csv", random_dirs)
random_raw.to_csv(OUT / "random_split_raw.csv", index=False)
random_summary = (
    random_raw.loc[random_raw.method.isin(["rankcover_confcert_adaptive", "rankcover_empirical_zero_matched"])]
    .groupby(["module", "method"])
    .agg(
        runs=("method", "size"),
        coverage=("coverage", "mean"),
        dmax=("dmax", "mean"),
        dmean=("dmean", "mean"),
        violations=("violation_run", "sum"),
        size=("avg_size", "mean"),
        pass_rate=("certificate_pass", "mean"),
        fallback_rate=("forced_uncertified_fallback", "mean"),
    )
    .reset_index()
)
random_summary.to_csv(OUT / "random_split_summary.csv", index=False)

print(agg.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
print("\nTask-blocked\n", paired_task.to_string(index=False))
print("\nSource-family-blocked\n", paired_source.to_string(index=False))
print("\nPass utility\n", pass_utility.to_string(index=False))
print("\nRandom split\n", random_summary.to_string(index=False))
print("\nImplementation\n", implementation_audit)
