#!/usr/bin/env python3
"""Generate reproducible, stratified SHAP plots for the Keras ANN."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from inferenceENGINE import KerasEnergyPredictor


LOAD_BINS = [-np.inf, 0.10, 0.40, 0.70, np.inf]
LOAD_LABELS = ["low", "medium", "high", "very_high"]


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (base_dir / path).resolve()


def stratified_sample(
    data: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """Balance samples across load and ES-state strata."""

    work = data.reset_index(
        names="_source_row"
    ).copy()

    load = pd.to_numeric(
        work["load"],
        errors="coerce",
    ).fillna(0.0)

    work["_load_stratum"] = pd.cut(
        load,
        bins=LOAD_BINS,
        labels=LOAD_LABELS,
        include_lowest=True,
    ).astype(str)

    es_columns = [
        f"ESMode{index}"
        for index in range(1, 7)
        if f"ESMode{index}" in work.columns
    ]

    if es_columns:
        es_values = (
            work[es_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        work["_es_state"] = np.where(
            es_values.max(axis=1) >= 0.5,
            "energy_saving",
            "active",
        )
    else:
        work["_es_state"] = "unknown"

    work["_shap_stratum"] = (
        work["_load_stratum"]
        + "__"
        + work["_es_state"]
    )

    groups = [
        group
        for _, group in work.groupby(
            "_shap_stratum",
            observed=True,
            sort=True,
        )
    ]

    if not groups:
        raise ValueError("No SHAP sampling strata were created")

    target_count = min(
        max(1, int(sample_count)),
        len(work),
    )

    base_size = target_count // len(groups)
    remainder = target_count % len(groups)

    selected_parts = []

    for index, group in enumerate(groups):
        group_target = (
            base_size
            + (1 if index < remainder else 0)
        )

        if group_target <= 0:
            continue

        selected_parts.append(
            group.sample(
                n=min(group_target, len(group)),
                random_state=seed + index,
            )
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=False,
    )

    missing_count = target_count - len(selected)

    if missing_count > 0:
        remaining = work.drop(
            index=selected.index,
            errors="ignore",
        )

        if not remaining.empty:
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(
                        n=min(missing_count, len(remaining)),
                        random_state=seed + 10_000,
                    ),
                ],
                ignore_index=False,
            )

    return selected.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready SHAP plots"
    )
    parser.add_argument(
        "--data",
        default="../archive/merged_training_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="../OUTPUTs/shap/paper",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--background-samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    module_dir = Path(__file__).resolve().parent
    data_path = resolve_path(args.data, module_dir)
    output_dir = resolve_path(
        args.output_dir,
        module_dir,
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"SHAP dataset not found: {data_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(data_path)

    if data.empty:
        raise ValueError("SHAP dataset is empty")

    required_columns = {
        "load",
        "RUType",
        "Mode",
        "Frequency",
        "Bandwidth",
        "Antennas",
        "TXpower",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"SHAP dataset is missing columns: {sorted(missing)}"
        )

    cohort = stratified_sample(
        data,
        sample_count=args.samples,
        seed=args.seed,
    )

    original_columns = list(data.columns)

    background_count = min(
        max(1, args.background_samples),
        len(data),
    )

    background = data.sample(
        n=background_count,
        random_state=args.seed,
    )

    background_path = (
        output_dir / "shap_background_sample.csv"
    )
    cohort_path = (
        output_dir / "shap_analysis_cohort.csv"
    )

    background.to_csv(
        background_path,
        index=False,
    )
    cohort.to_csv(
        cohort_path,
        index=False,
    )

    predictor = KerasEnergyPredictor(
        enable_shap=True
    )

    predictor.initialize_shap(
        background_samples=len(background),
        background_data_path=str(background_path),
    )

    cohort_input = cohort[original_columns].copy()

    representative_index = int(
        (
            pd.to_numeric(
                cohort_input["load"],
                errors="coerce",
            ).fillna(0.0)
            - pd.to_numeric(
                cohort_input["load"],
                errors="coerce",
            ).median()
        ).abs().idxmin()
    )

    result = predictor.explain_prediction(
        cohort_input,
        sample_idx=representative_index,
        plot=True,
        save_path=str(
            output_dir / "ann_energy_shap.png"
        ),
        n_samples=len(cohort_input),
    )

    if not result:
        raise RuntimeError(
            "SHAP calculation returned no result"
        )

    importance = result["feature_importance"]
    importance.to_csv(
        output_dir / "shap_feature_importance.csv",
        index=False,
    )

    shap_values = pd.DataFrame(
        result["shap_values"],
        columns=result["feature_names"],
    )
    shap_values.insert(
        0,
        "source_row",
        cohort["_source_row"].to_numpy(),
    )
    shap_values.insert(
        1,
        "stratum",
        cohort["_shap_stratum"].to_numpy(),
    )
    shap_values.to_csv(
        output_dir / "shap_values.csv",
        index=False,
    )

    metadata = {
        "method": "Kernel SHAP",
        "model": str(predictor.model_path),
        "dataset": str(data_path),
        "random_seed": args.seed,
        "background_sampling": "representative_random",
        "analysis_sampling": (
            "balanced_load_and_energy_saving_strata"
        ),
        "background_samples": len(background),
        "analysis_samples": len(cohort),
        "load_bins": LOAD_BINS[1:-1],
        "stratum_counts": (
            cohort["_shap_stratum"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "waterfall_sample_index": representative_index,
        "base_value_w": float(result["base_value"]),
    }

    with open(
        output_dir / "shap_metadata.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metadata,
            output_file,
            indent=2,
        )

    print(f"SHAP outputs saved to: {output_dir}")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()