"""Train and evaluate the Bengaluru rent estimator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bengaluru_rent_listings.csv"
CATEGORICAL = ["locality", "furnishing"]
NUMERIC = ["bhk", "sqft", "bathrooms", "property_age_years", "metro_distance_km", "parking", "gated_community"]
FEATURES = CATEGORICAL + NUMERIC


def read_rows(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in NUMERIC + ["monthly_rent_inr"]:
            row[field] = float(row[field])
    return rows


def build_model(random_state: int = 42) -> Pipeline:
    processor = DictVectorizer(sparse=False)
    regressor = RandomForestRegressor(n_estimators=180, min_samples_leaf=3, random_state=random_state, n_jobs=-1)
    return Pipeline([("prepare", processor), ("model", regressor)])


def prediction_interval(model: Pipeline, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared = model.named_steps["prepare"].transform(rows)
    forest = model.named_steps["model"]
    tree_predictions = np.vstack([tree.predict(prepared) for tree in forest.estimators_])
    return np.percentile(tree_predictions, 10, axis=0), np.median(tree_predictions, axis=0), np.percentile(tree_predictions, 90, axis=0)


def train_and_evaluate() -> dict:
    rows = read_rows()
    x = [{field: row[field] for field in FEATURES} for row in rows]
    y = [row["monthly_rent_inr"] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42)
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    lower, median, upper = prediction_interval(model, x_test)
    metrics = {
        "rows": len(rows),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "mae_inr": round(mean_absolute_error(y_test, predictions), 0),
        "rmse_inr": round(mean_squared_error(y_test, predictions) ** 0.5, 0),
        "r2": round(r2_score(y_test, predictions), 3),
        "p10_p90_coverage": round(float(np.mean((np.array(y_test) >= lower) & (np.array(y_test) <= upper))), 3),
        "median_interval_width_inr": round(float(np.median(upper - lower)), 0),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "data/processed").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/rent_estimator.joblib")
    (ROOT / "data/processed/evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    residuals = np.array(y_test) - predictions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#f7f4ef")
    axes[0].scatter(y_test, predictions, alpha=0.62, color="#aa4a30", edgecolors="none")
    limit = max(max(y_test), max(predictions))
    axes[0].plot([0, limit], [0, limit], color="#25344a", linestyle="--")
    axes[0].set(title="Actual and predicted rent", xlabel="Actual monthly rent (₹)", ylabel="Predicted monthly rent (₹)")
    axes[1].hist(residuals, bins=24, color="#2f7483", edgecolor="white")
    axes[1].axvline(0, color="#25344a", linestyle="--")
    axes[1].set(title="Residual distribution", xlabel="Actual minus predicted (₹)", ylabel="Listings")
    fig.suptitle(f"Bengaluru Rent Estimator  |  MAE ₹{metrics['mae_inr']:,.0f}  |  R² {metrics['r2']}", fontsize=15, weight="bold", color="#25344a")
    fig.tight_layout()
    fig.savefig(ROOT / "evidence/model_evaluation.png", dpi=180)
    plt.close(fig)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
