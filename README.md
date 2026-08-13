# Bengaluru Rent Estimator

This machine learning project estimates the monthly asking rent of a Bengaluru home from its locality, size, furnishing, property age, and distance to the nearest metro station. It also reports an uncertainty range so the prediction is not presented as an exact market value.

![Model evaluation](evidence/model_evaluation.png)

## Problem

Rental listings are difficult to compare because locality, floor area, furnishing, and access to transport affect price at the same time. A simple city-wide average hides these differences.

## Root cause addressed

The main cause of inconsistent comparisons is that similar homes are not grouped by the same features. The project creates a repeatable feature pipeline and evaluates predictions on a held-out test set.

## Purpose

The estimator is an educational model for comparing listings. It is not a valuation service and should not be used as the only basis for a rental agreement.

## Model results

Run `python src/train.py` to reproduce the current MAE, RMSE, R², and interval coverage. The evaluation image and JSON report are generated from the same test predictions.

## Data statement

The committed dataset is a deterministic synthetic sample modelled on common Bengaluru rental listing fields. It contains no scraped personal information and makes no claim to represent the current market. The generation assumptions are documented in `docs/data_card.md`.

## Repository guide

| Folder | Contents |
| --- | --- |
| `data` | Generated listing sample and train/test outputs |
| `src` | Data generation, training, evaluation, and prediction code |
| `models` | Serialized model artifact after training |
| `evidence` | Evaluation charts |
| `reports` | Detailed PDF project report |
| `tests` | Feature and prediction checks |

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
python scripts/build_report.py
pytest
ruff check src scripts tests
```

## Author

Harika

