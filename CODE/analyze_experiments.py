"""
Analyse Auto EA-SORAN energy-saving experiments.

Energy efficiency follows the 3GPP form:
    data volume / energy consumption, in bit/J.

The simulation's analytical energy model and runtime-signal KPIs
remain explicitly labelled in the exported metadata.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp, wilcoxon

SCENARIO_ORDER = [
    "no_control",
    "static_threshold",
    "deterministic",
    "ai",
]

SCENARIO_LABELS = {
    "no_control": "No control",
    "static_threshold": "Static threshold",
    "deterministic": "Rule-based",
    "ai": "AI closed-loop",
}

SCENARIO_COLORS = {
    "no_control": "#9D9D9D",
    "static_threshold": "#72B7B2",
    "deterministic": "#4C78A8",
    "ai": "#F58518",
}

SCENARIO_MARKERS = {
    "no_control": "s",
    "static_threshold": "D",
    "deterministic": "o",
    "ai": "^",
}

NUMERIC_COLUMNS = [
    "replicate",
    "seed",
    "sim_time_s",
    "physical_sim_time_s",
    "logical_time_scale",
    "hour_of_day",
    "represented_interval_s",
    "represented_energy_j",
    "energy_consumption_w",
    "dl_mbps",
    "ul_mbps",
    "generated_dl_mbps",
    "avg_latency_ms",
    "packet_loss_pct",
    "avg_sinr_db",
    "available_prbs",
    "used_prbs",
    "prb_util_pct",
    "active_ues",
]

RUN_METRICS = [
    "represented_energy_kwh",
    "average_network_power_w",
    "average_network_throughput_mbps",
    "energy_efficiency_mbit_per_j",
    "aggregate_packet_loss_pct",
    "traffic_weighted_latency_ms",
    "weighted_prb_utilization_pct",
    "average_active_ues",
    "sleeping_cell_time_pct",
]

FORMULA_METADATA = {
    "represented_energy_kwh": (
        "sum(represented_energy_j) / 3.6e6"
    ),
    "average_network_power_w": (
        "total represented energy / represented run duration"
    ),
    "average_network_throughput_mbps": (
        "represented received bits / represented run duration / 1e6"
    ),
    "energy_efficiency_mbit_per_j": (
        "represented received data volume in Mbit / "
        "represented energy in joules"
    ),
    "aggregate_packet_loss_pct": (
        "100 * max(0, offered bits - received bits) / offered bits"
    ),
    "weighted_prb_utilization_pct": (
        "100 * sum(used PRBs * represented interval) / "
        "sum(available PRBs * represented interval)"
    ),
    "traffic_weighted_latency_ms": (
        "mean cell latency weighted by represented received bits"
    ),
    "sleeping_cell_time_pct": (
        "100 * sleeping represented cell-time / total represented cell-time"
    ),
    "standards_note": (
        "3GPP TS 28.552 defines 5G performance measurements; "
        "TS 28.554 defines 5G KPIs. Energy efficiency uses "
        "data volume divided by energy consumption in bit/J."
    ),
    "references": [
        (
            "https://portal.3gpp.org/desktopmodules/"
            "Specifications/SpecificationDetails.aspx?"
            "specificationId=3413"
        ),
        "https://www.3gpp.org/technologies/ee-article",
    ],
    "timestamp_alignment": (
        "Retain only logical simulation timestamps present "
        "in every experiment run across all scenarios and seeds "
        "before calculating KPIs."
    ),
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return (
        Path(__file__).resolve().parent / path
    ).resolve()


def load_experiments(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*_cells.csv"))

    if not files:
        raise FileNotFoundError(
            f"No *_cells.csv files found in {input_dir}"
        )

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    required = {
        "scenario",
        "run_id",
        "replicate",
        "seed",
        "sim_time_s",
        "bs_id",
        "represented_interval_s",
        "represented_energy_j",
        "energy_consumption_w",
        "dl_mbps",
        "generated_dl_mbps",
        "available_prbs",
        "used_prbs",
        "es_mode",
        "traffic_level",
        "kpi_source",
        "energy_source",
    }

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            f"Experiment data is missing columns: {missing}"
        )

    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            ).fillna(0.0)

    data["scenario"] = data["scenario"].astype(str)
    data["run_id"] = data["run_id"].astype(str)
    data["bs_id"] = data["bs_id"].astype(str)
    data["es_mode"] = data["es_mode"].fillna(
        "active"
    ).astype(str)
    data["traffic_level"] = data[
        "traffic_level"
    ].fillna("unknown").astype(str)

    data["radio_sleeping"] = data[
        "radio_sleeping"
    ].astype(str).str.lower().isin(
        {"1", "true", "yes"}
    )

    scenarios = set(data["scenario"].unique())
    known_scenarios = set(SCENARIO_ORDER)
    unknown_scenarios = sorted(scenarios - known_scenarios)

    if unknown_scenarios:
        raise ValueError(
            "Unknown scenario values found: "
            f"{unknown_scenarios}. Expected one or more of "
            f"{SCENARIO_ORDER}."
        )

    if len(scenarios) < 2:
        raise ValueError(
            "At least two scenarios are required for comparison; "
            f"found {sorted(scenarios)}"
        )

    return data

def align_common_timestamps(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align matched scenario runs to identical logical timestamps.

    Matching is performed by replicate and seed. Raw input files are
    never modified. A timestamp is retained only when it exists in
    every available scenario for that matched experiment.
    """
    aligned_frames = []
    report_records = []

    working = data.copy()
    working["_alignment_time"] = (
        working["sim_time_s"].round(9)
    )

    run_timestamp_sets = [
        set(run_group["_alignment_time"].unique())
        for _, run_group in working.groupby(
            [
                "scenario",
                "run_id",
                "replicate",
                "seed",
            ],
            sort=True,
        )
    ]

    global_common_timestamps = set.intersection(
        *run_timestamp_sets
    )
    if not global_common_timestamps:
        raise ValueError(
            "No logical timestamps are common to every "
            "experiment run."
        )

    match_columns = ["replicate", "seed"]

    for (replicate, seed), matched_group in working.groupby(
        match_columns,
        sort=True,
    ):
        scenario_timestamp_sets = {}

        for scenario, scenario_group in matched_group.groupby(
            "scenario",
            sort=True,
        ):
            run_ids = sorted(
                scenario_group["run_id"].unique()
            )

            if len(run_ids) != 1:
                raise ValueError(
                    "Timestamp alignment requires exactly one "
                    f"run per scenario for replicate={replicate}, "
                    f"seed={seed}. Scenario '{scenario}' has "
                    f"run IDs: {run_ids}"
                )

            scenario_timestamp_sets[scenario] = set(
                scenario_group["_alignment_time"].unique()
            )

        if len(scenario_timestamp_sets) < 2:
            raise ValueError(
                "Timestamp alignment requires at least two "
                f"scenarios for replicate={replicate}, "
                f"seed={seed}."
            )

        common_timestamps = sorted(
            global_common_timestamps
        )

        for scenario, scenario_group in matched_group.groupby(
            "scenario",
            sort=True,
        ):
            original_timestamps = sorted(
                scenario_group["_alignment_time"].unique()
            )

            aligned_group = scenario_group[
                scenario_group["_alignment_time"].isin(
                    common_timestamps
                )
            ].copy()

            aligned_timestamps = sorted(
                aligned_group["_alignment_time"].unique()
            )

            report_records.append(
                {
                    "scenario": scenario,
                    "run_id": aligned_group[
                        "run_id"
                    ].iloc[0],
                    "replicate": int(replicate),
                    "seed": int(seed),
                    "original_snapshot_count": len(
                        original_timestamps
                    ),
                    "aligned_snapshot_count": len(
                        aligned_timestamps
                    ),
                    "dropped_snapshot_count": (
                        len(original_timestamps)
                        - len(aligned_timestamps)
                    ),
                    "original_first_sim_time_s": (
                        original_timestamps[0]
                    ),
                    "original_last_sim_time_s": (
                        original_timestamps[-1]
                    ),
                    "aligned_first_sim_time_s": (
                        aligned_timestamps[0]
                    ),
                    "aligned_last_sim_time_s": (
                        aligned_timestamps[-1]
                    ),
                }
            )

            aligned_frames.append(aligned_group)

        logging.info(
            "Aligned replicate=%s seed=%s across %d scenarios: "
            "%d common timestamps from %.3f to %.3f",
            replicate,
            seed,
            len(scenario_timestamp_sets),
            len(common_timestamps),
            common_timestamps[0],
            common_timestamps[-1],
        )

    aligned_data = pd.concat(
        aligned_frames,
        ignore_index=True,
    ).drop(columns=["_alignment_time"])

    alignment_report = pd.DataFrame(report_records)

    return aligned_data, alignment_report

def available_scenarios(
    frame: pd.DataFrame,
) -> List[str]:
    """Return scenarios in paper-preferred order."""

    present = set(frame["scenario"].dropna().astype(str))
    ordered = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in present
    ]
    extras = sorted(present - set(ordered))

    return ordered + extras


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(
        scenario,
        scenario.replace("_", " ").title(),
    )


def weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return float(values.mean()) if len(values) else 0.0

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def summarize_run(group: pd.DataFrame) -> Dict[str, float]:
    represented_interval = group[
        "represented_interval_s"
    ].clip(lower=0.0)

    represented_energy = group[
        "represented_energy_j"
    ].clip(lower=0.0)

    received_bits = (
        group["dl_mbps"].clip(lower=0.0)
        + group.get(
            "ul_mbps",
            pd.Series(0.0, index=group.index),
        ).clip(lower=0.0)
    ) * 1_000_000.0 * represented_interval

    offered_bits = (
        group["generated_dl_mbps"].clip(lower=0.0)
        * 1_000_000.0
        * represented_interval
    )

    snapshot_intervals = (
        group.groupby("sim_time_s")[
            "represented_interval_s"
        ]
        .max()
        .clip(lower=0.0)
    )
    represented_duration = float(
        snapshot_intervals.sum()
    )

    total_energy_j = float(
        represented_energy.sum()
    )
    total_received_bits = float(received_bits.sum())
    total_offered_bits = float(offered_bits.sum())

    if represented_duration > 0.0:
        average_network_power = (
            total_energy_j / represented_duration
        )
        average_network_throughput = (
            total_received_bits
            / represented_duration
            / 1_000_000.0
        )
    else:
        average_network_power = 0.0
        average_network_throughput = 0.0

    if total_energy_j > 0.0:
        energy_efficiency = (
            total_received_bits
            / total_energy_j
            / 1_000_000.0
        )
    else:
        energy_efficiency = 0.0

    if total_offered_bits > 0.0:
        packet_loss = 100.0 * max(
            0.0,
            total_offered_bits - total_received_bits,
        ) / total_offered_bits
        packet_loss = min(100.0, packet_loss)
    else:
        packet_loss = 0.0

    prb_denominator = float(
        (
            group["available_prbs"].clip(lower=0.0)
            * represented_interval
        ).sum()
    )
    prb_numerator = float(
        (
            group["used_prbs"].clip(lower=0.0)
            * represented_interval
        ).sum()
    )

    weighted_prb = (
        100.0 * prb_numerator / prb_denominator
        if prb_denominator > 0.0
        else 0.0
    )

    cell_time = float(represented_interval.sum())
    sleeping_time = float(
        represented_interval[
            group["radio_sleeping"]
        ].sum()
    )
    sleeping_pct = (
        100.0 * sleeping_time / cell_time
        if cell_time > 0.0
        else 0.0
    )

    latency = weighted_mean(
        group["avg_latency_ms"],
        received_bits,
    )

    active_ue_snapshot = group.groupby(
        "sim_time_s"
    )["active_ues"].sum()

    return {
        "scenario": group["scenario"].iloc[0],
        "run_id": group["run_id"].iloc[0],
        "replicate": int(group["replicate"].iloc[0]),
        "seed": int(group["seed"].iloc[0]),
        "represented_duration_s": represented_duration,
        "represented_energy_kwh": (
            total_energy_j / 3_600_000.0
        ),
        "average_network_power_w": (
            average_network_power
        ),
        "average_network_throughput_mbps": (
            average_network_throughput
        ),
        "energy_efficiency_mbit_per_j": (
            energy_efficiency
        ),
        "aggregate_packet_loss_pct": packet_loss,
        "traffic_weighted_latency_ms": latency,
        "weighted_prb_utilization_pct": weighted_prb,
        "average_active_ues": float(
            active_ue_snapshot.mean()
        ),
        "sleeping_cell_time_pct": sleeping_pct,
        "kpi_source": "|".join(
            sorted(group["kpi_source"].dropna().unique())
        ),
        "energy_source": "|".join(
            sorted(
                group["energy_source"].dropna().unique()
            )
        ),
    }


def build_run_summaries(
    data: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "run_id",
        "replicate",
        "seed",
    ]

    records = [
        summarize_run(group)
        for _, group in data.groupby(
            group_columns,
            sort=True,
        )
    ]

    return pd.DataFrame(records)


def build_scenario_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []

    for scenario, group in runs.groupby("scenario"):
        record: Dict[str, float] = {
            "scenario": scenario,
            "run_count": len(group),
        }

        for metric in RUN_METRICS:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            ci95 = (
                t.ppf(
                    0.975,
                    df=len(values) - 1,
                )
                * std
                / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )

            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95"] = ci95

        records.append(record)

    summary = pd.DataFrame(records)

    if summary.empty:
        return summary

    summary["scenario_order"] = summary["scenario"].map(
        {
            scenario: index
            for index, scenario in enumerate(
                SCENARIO_ORDER
            )
        }
    ).fillna(len(SCENARIO_ORDER))

    summary = summary.sort_values(
        ["scenario_order", "scenario"]
    ).drop(columns=["scenario_order"])

    return summary.reset_index(drop=True)


def build_paired_comparison(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    paired = runs.pivot_table(
        index=["replicate", "seed"],
        columns="scenario",
        values=RUN_METRICS,
        aggfunc="mean",
    )

    records = []
    for index, row in paired.iterrows():
        for baseline in [
            "no_control",
            "static_threshold",
            "deterministic",
        ]:
            if (
                ("represented_energy_kwh", baseline)
                not in row.index
                or ("represented_energy_kwh", "ai")
                not in row.index
            ):
                continue

            baseline_energy = row[
                ("represented_energy_kwh", baseline)
            ]
            ai_energy = row[
                ("represented_energy_kwh", "ai")
            ]

            baseline_throughput = row[
                (
                    "average_network_throughput_mbps",
                    baseline,
                )
            ]
            ai_throughput = row[
                ("average_network_throughput_mbps", "ai")
            ]

            baseline_latency = row[
                (
                    "traffic_weighted_latency_ms",
                    baseline,
                )
            ]
            ai_latency = row[
                ("traffic_weighted_latency_ms", "ai")
            ]

            baseline_packet_loss = row[
                (
                    "aggregate_packet_loss_pct",
                    baseline,
                )
            ]
            ai_packet_loss = row[
                ("aggregate_packet_loss_pct", "ai")
            ]

            baseline_efficiency = row[
                (
                    "energy_efficiency_mbit_per_j",
                    baseline,
                )
            ]
            ai_efficiency = row[
                ("energy_efficiency_mbit_per_j", "ai")
            ]

            records.append(
                {
                    "replicate": index[0],
                    "seed": index[1],
                    "baseline_scenario": baseline,
                    "comparison": (
                        f"ai_vs_{baseline}"
                    ),
                    "energy_saving_pct": (
                        100.0
                        * (
                            baseline_energy - ai_energy
                        )
                        / baseline_energy
                        if baseline_energy > 0.0
                        else 0.0
                    ),
                    "throughput_change_pct": (
                        100.0
                        * (
                            ai_throughput
                            - baseline_throughput
                        )
                        / baseline_throughput
                        if baseline_throughput > 0.0
                        else 0.0
                    ),
                    "latency_change_pct": (
                        100.0
                        * (
                            ai_latency
                            - baseline_latency
                        )
                        / baseline_latency
                        if baseline_latency > 0.0
                        else 0.0
                    ),
                    "packet_loss_delta_pct_points": (
                        ai_packet_loss
                        - baseline_packet_loss
                    ),
                    "energy_efficiency_change_pct": (
                        100.0
                        * (
                            ai_efficiency
                            - baseline_efficiency
                        )
                        / baseline_efficiency
                        if baseline_efficiency > 0.0
                        else 0.0
                    ),
                }
            )

    return pd.DataFrame(records)


def build_paired_comparison_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise AI-vs-baseline paired deltas."""

    if paired.empty:
        return pd.DataFrame()

    metrics = [
        "energy_saving_pct",
        "throughput_change_pct",
        "latency_change_pct",
        "packet_loss_delta_pct_points",
        "energy_efficiency_change_pct",
    ]

    records = []
    for comparison, group in paired.groupby(
        "comparison",
        sort=True,
    ):
        record = {
            "comparison": comparison,
            "baseline_scenario": group[
                "baseline_scenario"
            ].iloc[0],
            "paired_run_count": len(group),
        }

        for metric in metrics:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            ci95 = (
                t.ppf(
                    0.975,
                    df=len(values) - 1,
                )
                * std
                / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )

            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95"] = ci95

        records.append(record)

    summary = pd.DataFrame(records)
    summary["baseline_order"] = summary[
        "baseline_scenario"
    ].map(
        {
            scenario: index
            for index, scenario in enumerate(
                SCENARIO_ORDER
            )
        }
    ).fillna(len(SCENARIO_ORDER))

    return (
        summary.sort_values(
            ["baseline_order", "baseline_scenario"]
        )
        .drop(columns=["baseline_order"])
        .reset_index(drop=True)
    )

def holm_adjust(
    p_values: pd.Series,
) -> pd.Series:
    """Apply Holm's step-down correction."""

    values = p_values.astype(float).to_numpy()
    result = np.full(len(values), np.nan)

    valid_indices = np.where(np.isfinite(values))[0]
    if len(valid_indices) == 0:
        return pd.Series(
            result,
            index=p_values.index,
        )

    ordered_indices = valid_indices[
        np.argsort(values[valid_indices])
    ]

    number_of_tests = len(ordered_indices)
    running_maximum = 0.0

    for rank, position in enumerate(ordered_indices):
        adjusted = min(
            1.0,
            (
                number_of_tests - rank
            ) * values[position],
        )

        running_maximum = max(
            running_maximum,
            adjusted,
        )
        result[position] = running_maximum

    return pd.Series(
        result,
        index=p_values.index,
    )


def build_statistical_tests(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform paired seed-level inference on AI-minus-baseline
    effects.

    The paired comparison table already contains one effect
    value per matched seed, so testing against zero preserves
    the paired experimental design.
    """

    metrics = [
        "energy_saving_pct",
        "throughput_change_pct",
        "latency_change_pct",
        "packet_loss_delta_pct_points",
        "energy_efficiency_change_pct",
    ]

    records = []

    for comparison, comparison_group in paired.groupby(
        "comparison",
        sort=True,
    ):
        baseline = comparison_group[
            "baseline_scenario"
        ].iloc[0]

        for metric in metrics:
            values = (
                comparison_group[metric]
                .astype(float)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy()
            )

            sample_count = len(values)

            if sample_count == 0:
                continue

            mean = float(np.mean(values))
            std = (
                float(np.std(values, ddof=1))
                if sample_count > 1
                else 0.0
            )

            ci95 = (
                float(
                    t.ppf(
                        0.975,
                        df=sample_count - 1,
                    )
                    * std
                    / np.sqrt(sample_count)
                )
                if sample_count > 1
                else 0.0
            )

            if sample_count > 1:
                t_result = ttest_1samp(
                    values,
                    popmean=0.0,
                    alternative="two-sided",
                )
                t_statistic = float(
                    t_result.statistic
                )
                t_pvalue = float(t_result.pvalue)
            else:
                t_statistic = np.nan
                t_pvalue = np.nan

            if np.allclose(values, 0.0):
                wilcoxon_statistic = 0.0
                wilcoxon_pvalue = 1.0
            elif sample_count > 1:
                try:
                    wilcoxon_result = wilcoxon(
                        values,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                    wilcoxon_statistic = float(
                        wilcoxon_result.statistic
                    )
                    wilcoxon_pvalue = float(
                        wilcoxon_result.pvalue
                    )
                except ValueError:
                    wilcoxon_statistic = np.nan
                    wilcoxon_pvalue = np.nan
            else:
                wilcoxon_statistic = np.nan
                wilcoxon_pvalue = np.nan

            cohen_dz = (
                mean / std
                if std > 0.0
                else 0.0
            )

            records.append(
                {
                    "comparison": comparison,
                    "baseline_scenario": baseline,
                    "metric": metric,
                    "paired_run_count": sample_count,
                    "mean_effect": mean,
                    "std_effect": std,
                    "ci95_lower": mean - ci95,
                    "ci95_upper": mean + ci95,
                    "t_statistic": t_statistic,
                    "t_pvalue": t_pvalue,
                    "wilcoxon_statistic": (
                        wilcoxon_statistic
                    ),
                    "wilcoxon_pvalue": (
                        wilcoxon_pvalue
                    ),
                    "cohen_dz": cohen_dz,
                    "effect_direction": (
                        "positive"
                        if mean > 0.0
                        else (
                            "negative"
                            if mean < 0.0
                            else "zero"
                        )
                    ),
                }
            )

    tests = pd.DataFrame(records)

    if tests.empty:
        return tests

    tests["t_pvalue_holm"] = np.nan
    tests["wilcoxon_pvalue_holm"] = np.nan

    # Correct the three controller comparisons separately
    # within each KPI family.
    for metric, indices in tests.groupby(
        "metric"
    ).groups.items():
        tests.loc[
            indices,
            "t_pvalue_holm",
        ] = holm_adjust(
            tests.loc[indices, "t_pvalue"]
        )

        tests.loc[
            indices,
            "wilcoxon_pvalue_holm",
        ] = holm_adjust(
            tests.loc[
                indices,
                "wilcoxon_pvalue",
            ]
        )

    tests["t_significant_holm_0_05"] = (
        tests["t_pvalue_holm"] < 0.05
    )
    tests["wilcoxon_significant_holm_0_05"] = (
        tests["wilcoxon_pvalue_holm"] < 0.05
    )

    return tests.sort_values(
        ["metric", "comparison"]
    ).reset_index(drop=True)

def plot_kpi_comparison(
    scenario_summary: pd.DataFrame,
    output_dir: Path,
):
    metrics = [
        (
            "represented_energy_kwh",
            "Represented energy (kWh)",
        ),
        (
            "average_network_power_w",
            "Average network power (W)",
        ),
        (
            "average_network_throughput_mbps",
            "Network throughput (Mbit/s)",
        ),
        (
            "energy_efficiency_mbit_per_j",
            "Energy efficiency (Mbit/J)",
        ),
        (
            "traffic_weighted_latency_ms",
            "Traffic-weighted latency (ms)",
        ),
        (
            "aggregate_packet_loss_pct",
            "Aggregate packet loss (%)",
        ),
    ]

    scenarios = available_scenarios(scenario_summary)
    colors = [
        SCENARIO_COLORS.get(scenario, "#333333")
        for scenario in scenarios
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    indexed = scenario_summary.set_index("scenario")

    for axis, (metric, label) in zip(
        axes.flat,
        metrics,
    ):
        means = [
            indexed.loc[
                scenario,
                f"{metric}_mean",
            ]
            for scenario in scenarios
        ]
        errors = [
            indexed.loc[
                scenario,
                f"{metric}_ci95",
            ]
            for scenario in scenarios
        ]

        labels = [
            scenario_label(scenario)
            for scenario in scenarios
        ]

        axis.bar(
            labels,
            means,
            yerr=errors,
            color=colors,
            capsize=5,
        )
        axis.set_ylabel(label)
        axis.tick_params(
            axis="x",
            labelrotation=20,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Controller KPI Comparison",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "kpi_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hourly_profiles(
    data: pd.DataFrame,
    output_dir: Path,
):
    data = data.copy()
    data["hour_bin"] = np.floor(
        data["hour_of_day"]
    ).astype(int) % 24

    snapshot = (
        data.groupby(
            [
                "scenario",
                "run_id",
                "replicate",
                "seed",
                "sim_time_s",
                "hour_bin",
            ],
            as_index=False,
        )
        .agg(
            network_power_w=(
                "energy_consumption_w",
                "sum",
            ),
            network_throughput_mbps=(
                "dl_mbps",
                "sum",
            ),
            mean_prb_utilization_pct=(
                "prb_util_pct",
                "mean",
            ),
            mean_latency_ms=(
                "avg_latency_ms",
                "mean",
            ),
        )
    )

    hourly = (
        snapshot.groupby(
            ["scenario", "hour_bin"],
            as_index=False,
        )
        .agg(
            network_power_w=(
                "network_power_w",
                "mean",
            ),
            network_throughput_mbps=(
                "network_throughput_mbps",
                "mean",
            ),
            mean_prb_utilization_pct=(
                "mean_prb_utilization_pct",
                "mean",
            ),
            mean_latency_ms=(
                "mean_latency_ms",
                "mean",
            ),
        )
    )

    panels = [
        ("network_power_w", "Network power (W)"),
        (
            "network_throughput_mbps",
            "DL throughput (Mbit/s)",
        ),
        (
            "mean_prb_utilization_pct",
            "PRB utilization (%)",
        ),
        ("mean_latency_ms", "Latency (ms)"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        sharex=True,
    )

    for axis, (column, label) in zip(
        axes.flat,
        panels,
    ):
        for scenario in available_scenarios(hourly):
            color = SCENARIO_COLORS.get(
                scenario,
                "#333333",
            )
            subset = hourly[
                hourly["scenario"] == scenario
            ]
            axis.plot(
                subset["hour_bin"],
                subset[column],
                marker="o",
                label=scenario_label(scenario),
                color=color,
            )

        axis.set_ylabel(label)
        axis.set_xticks(range(0, 24, 3))
        axis.grid(alpha=0.25)
        axis.legend()

    axes[1, 0].set_xlabel("Logical hour of day")
    axes[1, 1].set_xlabel("Logical hour of day")

    fig.suptitle(
        "Daily Traffic and Energy Profiles",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "hourly_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_es_modes(
    data: pd.DataFrame,
    output_dir: Path,
):
    traffic_order = [
        "very_low",
        "low",
        "medium",
        "peak",
        "unknown",
    ]

    mode_time = (
        data.groupby(
            ["scenario", "traffic_level", "es_mode"],
            as_index=False,
        )["represented_interval_s"]
        .sum()
    )

    totals = mode_time.groupby(
        ["scenario", "traffic_level"]
    )["represented_interval_s"].transform("sum")

    mode_time["share_pct"] = np.where(
        totals > 0.0,
        100.0
        * mode_time["represented_interval_s"]
        / totals,
        0.0,
    )

    scenarios = available_scenarios(mode_time)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.5 * len(scenarios), 7),
        sharey=True,
    )

    if len(scenarios) == 1:
        axes = [axes]

    for axis, scenario in zip(axes, scenarios):
        subset = mode_time[
            mode_time["scenario"] == scenario
        ]
        pivot = subset.pivot_table(
            index="traffic_level",
            columns="es_mode",
            values="share_pct",
            fill_value=0.0,
        )

        ordered_index = [
            level
            for level in traffic_order
            if level in pivot.index
        ]
        pivot = pivot.reindex(ordered_index)

        pivot.plot(
            kind="bar",
            stacked=True,
            ax=axis,
            colormap="tab20",
        )
        axis.set_title(scenario_label(scenario))
        axis.set_xlabel("Traffic level")
        axis.set_ylabel(
            "Represented cell-time share (%)"
        )
        axis.legend(
            title="ES mode",
            fontsize=8,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Energy-Saving Mode Selection by Traffic Level",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "es_modes_by_traffic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_paired_energy_savings(
    paired: pd.DataFrame,
    output_dir: Path,
):
    """Plot matched-seed AI energy effects."""

    comparison_order = [
        "ai_vs_no_control",
        "ai_vs_static_threshold",
        "ai_vs_deterministic",
    ]

    comparison_labels = {
        "ai_vs_no_control": "AI vs\nNo control",
        "ai_vs_static_threshold": (
            "AI vs\nStatic threshold"
        ),
        "ai_vs_deterministic": (
            "AI vs\nRule-based"
        ),
    }

    baseline_colors = {
        "ai_vs_no_control": (
            SCENARIO_COLORS["no_control"]
        ),
        "ai_vs_static_threshold": (
            SCENARIO_COLORS["static_threshold"]
        ),
        "ai_vs_deterministic": (
            SCENARIO_COLORS["deterministic"]
        ),
    }

    figure, axis = plt.subplots(
        figsize=(10, 6),
    )

    for position, comparison in enumerate(
        comparison_order
    ):
        group = paired[
            paired["comparison"] == comparison
        ].sort_values("seed")

        if group.empty:
            continue

        values = group[
            "energy_saving_pct"
        ].astype(float).to_numpy()

        seeds = group["seed"].astype(int).to_numpy()

        mean = float(np.mean(values))
        std = (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        )
        ci95 = (
            float(
                t.ppf(
                    0.975,
                    df=len(values) - 1,
                )
                * std
                / np.sqrt(len(values))
            )
            if len(values) > 1
            else 0.0
        )

        offsets = np.linspace(
            -0.10,
            0.10,
            len(values),
        )

        axis.scatter(
            position + offsets,
            values,
            s=70,
            color=baseline_colors[comparison],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
            label=(
                "Matched seeds"
                if position == 0
                else None
            ),
            zorder=3,
        )

        for offset, value, seed in zip(
            offsets,
            values,
            seeds,
        ):
            axis.annotate(
                str(seed),
                (position + offset, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        axis.errorbar(
            position,
            mean,
            yerr=ci95,
            fmt="D",
            markersize=8,
            color="black",
            capsize=7,
            linewidth=2,
            label=(
                "Mean ± 95% t-CI"
                if position == 0
                else None
            ),
            zorder=4,
        )

    axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_xticks(
        range(len(comparison_order))
    )
    axis.set_xticklabels(
        [
            comparison_labels[item]
            for item in comparison_order
        ]
    )
    axis.set_ylabel(
        "AI energy saving relative to baseline (%)"
    )
    axis.set_title(
        "Matched-Seed AI Energy Effects"
    )
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_dir
        / "paired_energy_savings_by_seed.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

def plot_energy_tradeoff(
    runs: pd.DataFrame,
    output_dir: Path,
):
    fig, axis = plt.subplots(
        figsize=(10, 7)
    )

    for scenario in available_scenarios(runs):
        color = SCENARIO_COLORS.get(
            scenario,
            "#333333",
        )
        marker = SCENARIO_MARKERS.get(
            scenario,
            "o",
        )
        subset = runs[
            runs["scenario"] == scenario
        ]
        axis.scatter(
            subset["average_network_throughput_mbps"],
            subset["represented_energy_kwh"],
            label=scenario_label(scenario),
            color=color,
            marker=marker,
            s=80,
            alpha=0.8,
        )

    axis.set_xlabel(
        "Average network throughput (Mbit/s)"
    )
    axis.set_ylabel(
        "Represented energy (kWh)"
    )
    axis.set_title(
        "Energy–Throughput Trade-off Across Runs"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "energy_throughput_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Auto EA-SORAN experiment results"
        )
    )
    parser.add_argument(
        "--input-dir",
        default="../OUTPUTs/experiments/raw",
    )
    parser.add_argument(
        "--output-dir",
        default="../OUTPUTs/experiments/analysis",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | %(message)s"
        ),
    )

    data = load_experiments(input_dir)

    data, alignment_report = align_common_timestamps(
        data
    )

    alignment_report.to_csv(
        output_dir / "timestamp_alignment.csv",
        index=False,
    )

    runs = build_run_summaries(data)
    scenario_summary = build_scenario_summary(runs)
    paired = build_paired_comparison(runs)
    paired_summary = build_paired_comparison_summary(
        paired
    )
    statistical_tests = build_statistical_tests(
        paired
    )
    runs.to_csv(
        output_dir / "run_summary.csv",
        index=False,
    )
    scenario_summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )
    paired.to_csv(
        output_dir / "paired_comparison.csv",
        index=False,
    )
    paired_summary.to_csv(
        output_dir / "paired_comparison_summary.csv",
        index=False,
    )
    statistical_tests.to_csv(
        output_dir / "statistical_tests.csv",
        index=False,
    )
    with (
        output_dir / "formula_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            FORMULA_METADATA,
            metadata_file,
            indent=2,
        )

    plot_kpi_comparison(
        scenario_summary,
        output_dir,
    )
    plot_hourly_profiles(
        data,
        output_dir,
    )
    plot_es_modes(
        data,
        output_dir,
    )
    plot_energy_tradeoff(
        runs,
        output_dir,
    )
    plot_paired_energy_savings(
        paired,
        output_dir,
    )
    run_counts = runs.groupby("scenario").size()
    if run_counts.min() < 5:
        logging.warning(
            "Fewer than five runs per scenario were found. "
            "Use multiple matched seeds before drawing "
            "paper-level statistical conclusions."
        )

    missing_recommended = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario not in set(runs["scenario"])
    ]
    if missing_recommended:
        logging.warning(
            "Recommended journal-study scenarios missing: %s",
            ", ".join(missing_recommended),
        )

    logging.info(
        "Analysis written to %s",
        output_dir,
    )


if __name__ == "__main__":
    main()
