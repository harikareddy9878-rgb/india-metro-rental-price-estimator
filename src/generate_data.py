"""Generate a transparent synthetic Bengaluru rent listing sample."""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/bengaluru_rent_listings.csv"
LOCALITIES = {
    "Whitefield": 1.18,
    "HSR Layout": 1.34,
    "Electronic City": 0.88,
    "Indiranagar": 1.62,
    "Yelahanka": 0.84,
    "Marathahalli": 1.12,
    "Jayanagar": 1.38,
    "Hebbal": 1.08,
    "Banashankari": 1.02,
    "Sarjapur Road": 1.16,
    "Koramangala": 1.55,
    "Kengeri": 0.76,
}
FURNISHING = {"Unfurnished": 0, "Semi Furnished": 4200, "Fully Furnished": 9200}


def generate_rows(count: int = 960, seed: int = 1714) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    localities = list(LOCALITIES)
    for listing_id in range(1, count + 1):
        locality = rng.choice(localities)
        bhk = rng.choices([1, 2, 3, 4], weights=[18, 46, 29, 7])[0]
        sqft = max(350, int(rng.gauss(360 + bhk * 390, 115)))
        furnishing = rng.choices(list(FURNISHING), weights=[27, 54, 19])[0]
        age = rng.randint(0, 20)
        metro_km = round(max(0.2, rng.gauss(3.2, 2.1)), 1)
        bathrooms = max(1, bhk + rng.choice([-1, 0, 0, 0, 1]))
        parking = rng.choices([0, 1], weights=[25, 75])[0]
        gated = rng.choices([0, 1], weights=[32, 68])[0]
        base = 5200 + sqft * 16.5 + bhk * 1900 + bathrooms * 1150
        adjustments = FURNISHING[furnishing] + parking * 1800 + gated * 2400 - metro_km * 780 - age * 130
        rent = int(max(7000, (base + adjustments) * LOCALITIES[locality] + rng.gauss(0, 3900)))
        rows.append({
            "listing_id": f"BLR-{listing_id:04d}",
            "locality": locality,
            "bhk": bhk,
            "sqft": sqft,
            "bathrooms": bathrooms,
            "furnishing": furnishing,
            "property_age_years": age,
            "metro_distance_km": metro_km,
            "parking": parking,
            "gated_community": gated,
            "monthly_rent_inr": rent,
        })
    return rows


def write_data() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return OUTPUT


if __name__ == "__main__":
    print(write_data())

