import pandas as pd

from scripts.acquire_data import deduplicate, valid
from src.train import FEATURES, build_model


def test_invalid_extreme_listing_is_rejected():
    row = {"monthly_rent_inr": 8_000_000, "area_sqft": 900, "beds": 2}
    assert not valid(row)


def test_duplicate_signature_is_removed():
    row = {
        "city": "Bangalore",
        "locality": "Whitefield",
        "property_type": "2 BHK Flat",
        "beds": 2,
        "bathrooms": 2,
        "area_sqft": 1000,
        "monthly_rent_inr": 32000,
    }
    clean, removed = deduplicate([row, row.copy()])
    assert len(clean) == 1
    assert removed == 1


def test_model_accepts_unseen_locality():
    records = []
    for index in range(40):
        records.append(
            {
                "source": "Test",
                "city": "Bangalore" if index % 2 else "New Delhi",
                "locality": "Whitefield" if index % 3 else "Dwarka",
                "property_type": "2 BHK Flat",
                "furnishing": "Semi-Furnished",
                "seller_type": "Owner",
                "beds": 2,
                "bathrooms": 2,
                "balconies": 1,
                "area_sqft": 800 + index * 10,
            }
        )
    frame = pd.DataFrame(records)
    model = build_model()
    model.fit(frame[FEATURES], [10.0 + index / 100 for index in range(40)])
    candidate = frame.iloc[[0]].copy()
    candidate.loc[:, "locality"] = "Unseen Layout"
    assert model.predict(candidate[FEATURES])[0] > 0
