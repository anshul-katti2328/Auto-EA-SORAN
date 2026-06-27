"""
Analyse deterministic and AI energy-saving experiments.

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
    required_scenarios = {"deterministic", "ai"}

    if not required_scenarios.issubset(scenarios):
        raise ValueError(
            "Both deterministic and ai experiment data "
            f"are required; found {sorted(scenarios)}"
        )

    return data


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
                1.96 * std / np.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )

            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci95"] = ci95

        records.append(record)

    return pd.DataFrame(records)


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
        if (
            ("represented_energy_kwh", "deterministic")
            not in row.index
            or ("represented_energy_kwh", "ai")
            not in row.index
        ):
            continue

        deterministic_energy = row[
            ("represented_energy_kwh", "deterministic")
        ]
        ai_energy = row[
            ("represented_energy_kwh", "ai")
        ]

        deterministic_throughput = row[
            (
                "average_network_throughput_mbps",
                "deterministic",
            )
        ]
        ai_throughput = row[
            ("average_network_throughput_mbps", "ai")
        ]

        records.append(
            {
                "replicate": index[0],
                "seed": index[1],
                "energy_saving_pct": (
                    100.0
                    * (
                        deterministic_energy - ai_energy
                    )
                    / deterministic_energy
                    if deterministic_energy > 0.0
                    else 0.0
                ),
                "throughput_change_pct": (
                    100.0
                    * (
                        ai_throughput
                        - deterministic_throughput
                    )
                    / deterministic_throughput
                    if deterministic_throughput > 0.0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(records)


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

    scenarios = ["deterministic", "ai"]
    colors = ["#4C78A8", "#F58518"]

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

        axis.bar(
            scenarios,
            means,
            yerr=errors,
            color=colors,
            capsize=5,
        )
        axis.set_ylabel(label)
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    fig.suptitle(
        "Deterministic vs AI Energy-Saving Performance",
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
        for scenario, color in [
            ("deterministic", "#4C78A8"),
            ("ai", "#F58518"),
        ]:
            subset = hourly[
                hourly["scenario"] == scenario
            ]
            axis.plot(
                subset["hour_bin"],
                subset[column],
                marker="o",
                label=scenario,
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

    scenarios = ["deterministic", "ai"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        sharey=True,
    )

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
        axis.set_title(scenario.capitalize())
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


def plot_energy_tradeoff(
    runs: pd.DataFrame,
    output_dir: Path,
):
    fig, axis = plt.subplots(
        figsize=(10, 7)
    )

    for scenario, color, marker in [
        ("deterministic", "#4C78A8", "o"),
        ("ai", "#F58518", "^"),
    ]:
        subset = runs[
            runs["scenario"] == scenario
        ]
        axis.scatter(
            subset["average_network_throughput_mbps"],
            subset["represented_energy_kwh"],
            label=scenario,
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
            "Compare deterministic and AI experiment results"
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
    runs = build_run_summaries(data)
    scenario_summary = build_scenario_summary(runs)
    paired = build_paired_comparison(runs)

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

    if runs.groupby("scenario").size().min() < 5:
        logging.warning(
            "Fewer than five runs per scenario were found. "
            "Use multiple matched seeds before drawing "
            "paper-level statistical conclusions."
        )

    logging.info(
        "Analysis written to %s",
        output_dir,
    )


if __name__ == "__main__":
    main()