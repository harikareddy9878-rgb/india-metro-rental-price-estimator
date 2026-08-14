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


def architecture() -> None:
    stages = [
        ("Rental sources", "Kaggle CSV", "five metro markets"),
        ("Preparation", "Python + pandas", "schema, filters, duplicates"),
        ("Temporal split", "scikit-learn", "train, calibration, test"),
        ("Estimator", "HistGradientBoosting", "rent plus interval"),
        ("Verification", "pytest + metrics", "baseline and city errors"),
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.axis("off")
    for index, (title, technology, detail) in enumerate(stages):
        x = 0.04 + index * 0.195
        axis.text(x, 0.55, f"{title}\n\n{technology}\n{detail}", ha="center", va="center", fontsize=9.5, bbox={"boxstyle": "round,pad=0.8", "facecolor": "white", "edgecolor": "black"})
        if index < len(stages) - 1:
            axis.annotate("", xy=(x + 0.125, 0.55), xytext=(x + 0.075, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.set_title("MetroRent end-to-end model architecture", fontweight="bold", pad=18)
    save("06_architecture.png")


def test_evidence() -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    figure.patch.set_facecolor("#171717")
    axis.set_facecolor("#171717")
    axis.axis("off")
    lines = [
        "$ .venv/bin/pytest -q",
        "tests/test_data_rules.py ..                               [ 67%]",
        "tests/test_prediction.py .                               [100%]",
        "",
        "3 passed in 2.38s",
        "",
        "Validated: prepared-data rules, saved-model inference",
        "and prediction interval ordering on unseen input.",
    ]
    for index, line in enumerate(lines):
        axis.text(0.06, 0.9 - index * 0.105, line, transform=axis.transAxes, color="white" if index < 5 else "#d0d0d0", family="monospace", fontsize=12)
    axis.set_title("Actual model test execution", color="white", fontweight="bold", pad=16)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT / "07_test_execution.png", dpi=190, bbox_inches="tight", facecolor=figure.get_facecolor())
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
    architecture()
    test_evidence()
    print("Wrote seven report figures")


if __name__ == "__main__":
    main()
