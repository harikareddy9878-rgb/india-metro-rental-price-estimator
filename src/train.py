"""Train and evaluate the MetroRent price estimator."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/model_input.csv"
CATEGORICAL = ["source", "city", "locality", "property_type", "furnishing", "seller_type"]
NUMERIC = ["beds", "bathrooms", "balconies", "area_sqft"]
FEATURES = CATEGORICAL + NUMERIC


def read_data(path: Path = DATA) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["monthly_rent_inr"] = pd.to_numeric(frame["monthly_rent_inr"], errors="coerce")
    return frame.dropna(subset=["monthly_rent_inr", "city", "locality"]).reset_index(drop=True)


def build_model(random_state: int = 42) -> Pipeline:
    categorical = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    numeric = Pipeline([("fill", SimpleImputer(strategy="median"))])
    prepare = ColumnTransformer(
        [("categorical", categorical, CATEGORICAL), ("numeric", numeric, NUMERIC)]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=240,
        max_leaf_nodes=31,
        l2_regularization=1.5,
        random_state=random_state,
    )
    return Pipeline([("prepare", prepare), ("model", model)])


def split_data(frame: pd.DataFrame):
    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=42,
        stratify=frame["city"],
    )
    calibration, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=42,
        stratify=remainder["city"],
    )
    return train, calibration, test


def train_and_evaluate() -> dict:
    frame = read_data()
    train, calibration, test = split_data(frame)
    model = build_model()
    model.fit(train[FEATURES], np.log1p(train["monthly_rent_inr"]))

    calibration_prediction = np.expm1(model.predict(calibration[FEATURES]))
    interval_radius = float(
        np.quantile(np.abs(calibration["monthly_rent_inr"] - calibration_prediction), 0.90)
    )
    prediction = np.expm1(model.predict(test[FEATURES]))
    actual = test["monthly_rent_inr"].to_numpy()
    lower = np.maximum(0, prediction - interval_radius)
    upper = prediction + interval_radius
    city_median = train.groupby("city")["monthly_rent_inr"].median()
    baseline = test["city"].map(city_median).fillna(train["monthly_rent_inr"].median())

    by_city = {}
    for city, group in test.assign(prediction=prediction).groupby("city"):
        by_city[city] = {
            "rows": int(len(group)),
            "mae_inr": round(mean_absolute_error(group["monthly_rent_inr"], group["prediction"])),
        }
    metrics = {
        "raw_rows": int(json.loads((ROOT / "data/raw/source_manifest.json").read_text())["raw_rows"]),
        "model_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "cities": int(frame["city"].nunique()),
        "localities": int(frame["locality"].nunique()),
        "mae_inr": round(mean_absolute_error(actual, prediction)),
        "rmse_inr": round(mean_squared_error(actual, prediction) ** 0.5),
        "r2": round(r2_score(actual, prediction), 3),
        "city_median_baseline_mae_inr": round(mean_absolute_error(actual, baseline)),
        "interval_radius_inr": round(interval_radius),
        "interval_coverage": round(float(np.mean((actual >= lower) & (actual <= upper))), 3),
        "by_city": by_city,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "data/processed").mkdir(exist_ok=True)
    (ROOT / "evidence").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "interval_radius_inr": interval_radius, "features": FEATURES},
        ROOT / "models/rent_estimator.joblib",
    )
    (ROOT / "data/processed/evaluation.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    residual = actual - prediction
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#f6f4f0")
    axes[0, 0].hexbin(actual, prediction, gridsize=32, mincnt=1, cmap="Blues")
    limit = min(200_000, max(float(actual.max()), float(prediction.max())))
    axes[0, 0].plot([0, limit], [0, limit], color="#c6543c", linestyle="--")
    axes[0, 0].set(xlim=(0, limit), ylim=(0, limit), title="Actual and predicted rent", xlabel="Actual monthly rent (INR)", ylabel="Predicted monthly rent (INR)")
    axes[0, 1].hist(residual, bins=45, color="#2a7b82", edgecolor="white")
    axes[0, 1].axvline(0, color="#1f2d3d", linestyle="--")
    axes[0, 1].set(title="Residual distribution", xlabel="Actual minus predicted (INR)", ylabel="Listings")
    ordered_city = sorted(by_city.items(), key=lambda item: item[1]["mae_inr"])
    axes[1, 0].barh([item[0] for item in ordered_city], [item[1]["mae_inr"] for item in ordered_city], color="#d6843d")
    axes[1, 0].set(title="Test MAE by city", xlabel="MAE (INR)")
    source_counts = frame["source"].value_counts()
    axes[1, 1].bar(source_counts.index, source_counts.values, color=["#385a7c", "#8a5f99"])
    axes[1, 1].set(title="Prepared listings by source", ylabel="Listings")
    axes[1, 1].tick_params(axis="x", labelrotation=12)
    fig.suptitle(
        f"Indian Rental Price Estimator | {len(frame):,} listings | MAE INR {metrics['mae_inr']:,} | R2 {metrics['r2']}",
        fontsize=16,
        weight="bold",
        color="#223149",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "evidence/model_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
