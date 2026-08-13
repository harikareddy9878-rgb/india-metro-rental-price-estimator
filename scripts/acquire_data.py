"""Download and prepare two licensed Indian rental listing datasets."""

from __future__ import annotations

import csv
import io
import json
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/india_rental_listings.csv"
MODEL_PATH = ROOT / "data/processed/model_input.csv"
MANIFEST_PATH = ROOT / "data/raw/source_manifest.json"

SOURCES = [
    {
        "name": "India House Rent Prediction",
        "url": "https://www.kaggle.com/api/v1/datasets/download/pranavshinde36/india-house-rent-prediction",
        "page": "https://www.kaggle.com/datasets/pranavshinde36/india-house-rent-prediction",
        "license": "MIT",
        "file": "data.csv",
    },
    {
        "name": "New Delhi Rental Properties",
        "url": "https://www.kaggle.com/api/v1/datasets/download/divyanshug40/data-for-houses-available-for-rent",
        "page": "https://www.kaggle.com/datasets/divyanshug40/data-for-houses-available-for-rent",
        "license": "MIT",
        "file": "Makaan_data_700pages.csv",
    },
]

FIELDS = [
    "listing_id",
    "source",
    "city",
    "locality",
    "property_type",
    "beds",
    "bathrooms",
    "balconies",
    "area_sqft",
    "furnishing",
    "seller_type",
    "monthly_rent_inr",
]


def download_zip(url: str) -> zipfile.ZipFile:
    request = urllib.request.Request(url, headers={"User-Agent": "HarikaRentalProject/2.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, timeout=120, context=context) as response:  # noqa: S310
        payload = response.read()
    return zipfile.ZipFile(io.BytesIO(payload))


def text(value: object, fallback: str = "Unknown") -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
    return cleaned or fallback


def number(value: object) -> float | None:
    cleaned = re.sub(r"[^0-9.]", "", str(value or ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def read_csv_from_zip(archive: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    with archive.open(filename) as stream:
        wrapper = io.TextIOWrapper(stream, encoding="utf-8-sig", errors="replace")
        return list(csv.DictReader(wrapper))


def prepare_general(rows: list[dict[str, str]]) -> list[dict]:
    prepared = []
    for index, row in enumerate(rows, start=1):
        prepared.append(
            {
                "listing_id": f"IR-{index:06d}",
                "source": "India House Rent Prediction",
                "city": text(row.get("city")),
                "locality": text(row.get("locality")),
                "property_type": text(row.get("house_type"), "Rental home"),
                "beds": number(row.get("beds")),
                "bathrooms": number(row.get("bathrooms")),
                "balconies": number(row.get("balconies")),
                "area_sqft": number(row.get("area")),
                "furnishing": text(row.get("furnishing")),
                "seller_type": "Unknown",
                "monthly_rent_inr": number(row.get("rent")),
            }
        )
    return prepared


def prepare_delhi(rows: list[dict[str, str]]) -> list[dict]:
    prepared = []
    for index, row in enumerate(rows, start=1):
        prepared.append(
            {
                "listing_id": f"DL-{index:06d}",
                "source": "New Delhi Rental Properties",
                "city": "New Delhi",
                "locality": text(row.get("Location")),
                "property_type": text(row.get("Property_type"), "Rental home"),
                "beds": number(row.get("Size")),
                "bathrooms": number(row.get("Bathroom")),
                "balconies": None,
                "area_sqft": number(row.get("Area_sqft")),
                "furnishing": text(row.get("Status")),
                "seller_type": text(row.get("Seller_type")),
                "monthly_rent_inr": number(row.get("Rent_price")),
            }
        )
    return prepared


def valid(row: dict) -> bool:
    return bool(
        row["monthly_rent_inr"]
        and 1_000 <= row["monthly_rent_inr"] <= 300_000
        and row["area_sqft"]
        and 150 <= row["area_sqft"] <= 8_000
        and row["beds"]
        and 0.5 <= row["beds"] <= 10
    )


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    seen: set[tuple] = set()
    clean: list[dict] = []
    for row in rows:
        signature = (
            row["city"].casefold(),
            row["locality"].casefold(),
            row["property_type"].casefold(),
            row["beds"],
            row["bathrooms"],
            row["area_sqft"],
            row["monthly_rent_inr"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        clean.append(row)
    return clean, len(rows) - len(clean)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def acquire() -> dict:
    first = download_zip(SOURCES[0]["url"])
    second = download_zip(SOURCES[1]["url"])
    raw = prepare_general(read_csv_from_zip(first, SOURCES[0]["file"]))
    raw.extend(prepare_delhi(read_csv_from_zip(second, SOURCES[1]["file"])))
    write_rows(RAW_PATH, raw)
    valid_rows = [row for row in raw if valid(row)]
    clean, duplicates = deduplicate(valid_rows)
    write_rows(MODEL_PATH, clean)
    summary = {
        "raw_rows": len(raw),
        "valid_rows_before_deduplication": len(valid_rows),
        "model_rows": len(clean),
        "invalid_rows_removed": len(raw) - len(valid_rows),
        "duplicate_rows_removed": duplicates,
        "sources": SOURCES,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(acquire(), indent=2))
