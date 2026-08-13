"""Load the trained model and return a rent estimate with an uncertainty range."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def estimate(features: dict) -> dict:
    bundle = joblib.load(ROOT / "models/rent_estimator.joblib")
    row = {field: features.get(field) for field in FEATURES}
    prediction = float(np.expm1(bundle["model"].predict(pd.DataFrame([row]))[0]))
    radius = float(bundle["interval_radius_inr"])
    return {
        "estimate_inr": round(prediction),
        "lower_inr": round(max(0, prediction - radius)),
        "upper_inr": round(prediction + radius),
    }


if __name__ == "__main__":
    example = {
        "source": "India House Rent Prediction",
        "city": "Bangalore",
        "locality": "Whitefield",
        "property_type": "2 BHK Flat",
        "furnishing": "Semi-Furnished",
        "seller_type": "Unknown",
        "beds": 2,
        "bathrooms": 2,
        "balconies": 1,
        "area_sqft": 1050,
    }
    print(estimate(example))
