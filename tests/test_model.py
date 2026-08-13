from src.generate_data import generate_rows
from src.train import build_model


def test_generation_is_repeatable():
    assert generate_rows(3, seed=7) == generate_rows(3, seed=7)


def test_generated_values_are_valid():
    rows = generate_rows(30)
    assert all(row["monthly_rent_inr"] >= 7000 for row in rows)
    assert all(row["sqft"] >= 350 for row in rows)


def test_model_accepts_unseen_locality():
    model = build_model()
    x = [{"locality": "Whitefield", "furnishing": "Unfurnished", "bhk": 1, "sqft": 500, "bathrooms": 1, "property_age_years": 2, "metro_distance_km": 1.5, "parking": 0, "gated_community": 0}, {"locality": "Jayanagar", "furnishing": "Fully Furnished", "bhk": 3, "sqft": 1400, "bathrooms": 3, "property_age_years": 8, "metro_distance_km": 0.8, "parking": 1, "gated_community": 1}, {"locality": "Hebbal", "furnishing": "Semi Furnished", "bhk": 2, "sqft": 900, "bathrooms": 2, "property_age_years": 5, "metro_distance_km": 2.0, "parking": 1, "gated_community": 1}, {"locality": "Kengeri", "furnishing": "Unfurnished", "bhk": 2, "sqft": 800, "bathrooms": 1, "property_age_years": 10, "metro_distance_km": 4.0, "parking": 0, "gated_community": 0}]
    model.fit(x, [12000, 52000, 28000, 15000])
    prediction = model.predict([{**x[0], "locality": "Unseen Layout"}])
    assert prediction[0] > 0

