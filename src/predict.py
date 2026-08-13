"""Load the trained model and return a rent estimate with an uncertainty range."""

from __future__ import annotations

from pathlib import Path

import joblib

from src.train import prediction_interval

ROOT = Path(__file__).resolve().parents[1]


def estimate(features: dict) -> dict:
    model = joblib.load(ROOT / "models/rent_estimator.joblib")
    lower, median, upper = prediction_interval(model, [features])
    return {"estimate_inr": round(float(median[0])), "lower_inr": round(float(lower[0])), "upper_inr": round(float(upper[0]))}


if __name__ == "__main__":
    example = {"locality": "Whitefield", "furnishing": "Semi Furnished", "bhk": 2, "sqft": 1050, "bathrooms": 2, "property_age_years": 5, "metro_distance_km": 2.3, "parking": 1, "gated_community": 1}
    print(estimate(example))

