from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "figures"


def save(name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    frame = pd.read_csv(ROOT / "data/processed/model_input.csv")
    plt.style.use("grayscale")

    plt.figure(figsize=(8, 4.8))
    stages = ["Raw listings", "Prepared rows", "Train", "Calibration", "Test"]
    values = [
        metrics["raw_rows"],
        metrics["model_rows"],
        metrics["train_rows"],
        metrics["calibration_rows"],
        metrics["test_rows"],
    ]
    bars = plt.bar(stages, values, color="0.72", edgecolor="black")
    plt.bar_label(bars, fmt="%d")
    plt.title("Dataset preparation and split sizes")
    plt.ylabel("Rows")
    save("01_dataset_flow.png")

    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(
        ["MetroRent model", "City-median baseline"],
        [metrics["mae_inr"], metrics["city_median_baseline_mae_inr"]],
        color=["0.45", "0.8"],
        edgecolor="black",
    )
    plt.bar_label(bars, fmt="INR %.0f")
    plt.title("Held-out mean absolute error")
    plt.ylabel("MAE in INR (lower is better)")
    save("02_model_baseline.png")

    city_metrics = metrics["by_city"]
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(
        list(city_metrics),
        [item["mae_inr"] for item in city_metrics.values()],
        color="0.65",
        edgecolor="black",
    )
    plt.bar_label(bars, fmt="%.0f")
    plt.title("Test error by city")
    plt.ylabel("MAE in INR")
    save("03_city_error.png")

    plt.figure(figsize=(7, 4.8))
    bars = plt.bar(
        ["Observed coverage", "Target coverage"],
        [metrics["interval_coverage"] * 100, 90],
        color=["0.5", "0.82"],
        edgecolor="black",
    )
    plt.bar_label(bars, fmt="%.1f%%")
    plt.title(f"Prediction interval coverage (radius INR {metrics['interval_radius_inr']:,})")
    plt.ylabel("Coverage percentage")
    plt.ylim(0, 100)
    save("04_interval_coverage.png")

    city_medians = frame.groupby("city")["monthly_rent_inr"].median().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(city_medians.index, city_medians.values, color="0.72", edgecolor="black")
    plt.bar_label(bars, fmt="%.0f")
    plt.title("Median asking rent in the prepared data")
    plt.ylabel("Monthly rent in INR")
    save("05_city_rent_distribution.png")
    print("Wrote five report figures")


if __name__ == "__main__":
    main()
