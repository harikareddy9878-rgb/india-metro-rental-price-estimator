from __future__ import annotations

import json
from pathlib import Path

from report_template import build_research_report

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/MetroRent_Price_Estimator_Report.pdf"
FIGURES = ROOT / "reports/figures"


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    sections = [
        {
            "title": "Project overview and problem statement",
            "paragraphs": [
                "MetroRent estimates monthly asking rent for listings across Bangalore, Mumbai, Nagpur, New Delhi, and Pune. I built the project because city averages cannot represent the interaction between locality, property size, bedrooms, bathrooms, furnishing, and property type.",
                "The project tests whether a reproducible model improves on a transparent city-median baseline and whether a separate calibration split can produce a useful uncertainty interval. It is an educational listing-comparison tool, not a formal valuation or negotiation instruction.",
            ],
        },
        {
            "title": "Data sources and preparation",
            "paragraphs": [
                "Two MIT-licensed Kaggle sources contribute 21,691 raw rental listings. I harmonised city and locality names, converted numerical fields, removed impossible configurations, capped implausible values, and removed duplicate listing signatures.",
                "The final table contains 12,201 model rows across five cities and 2,102 localities. The split uses 8,540 training rows, 1,830 calibration rows, and 1,831 final test rows.",
            ],
            "table": [
                ["Stage", "Rows"],
                ["Raw listings", "21,691"],
                ["Prepared model table", "12,201"],
                ["Training", "8,540"],
                ["Calibration", "1,830"],
                ["Test", "1,831"],
            ],
        },
        {
            "title": "Model and evaluation design",
            "paragraphs": [
                "The model combines numerical and categorical listing attributes in a scikit-learn pipeline and trains a gradient-boosting regressor. A city-median predictor is the required baseline. Model selection and fitting are separated from final test evaluation.",
                "A conformal-style interval radius is estimated from absolute calibration errors and applied unchanged to the test set. I report MAE, RMSE, R-squared, interval coverage, and city-level error.",
            ],
        },
        {
            "title": "Experiment 1: data flow",
            "figure": FIGURES / "01_dataset_flow.png",
            "caption": "Figure 1. Raw, prepared, training, calibration, and test row counts.",
            "explanation": [
                [
                    "What I tested",
                    "How much source data remains after quality rules and how the final rows are allocated.",
                ],
                [
                    "What the graph shows",
                    "12,201 of 21,691 raw listings remain, with independent calibration and test partitions.",
                ],
                [
                    "Conclusion",
                    "The preparation removes substantial source noise while keeping enough rows for multi-city evaluation.",
                ],
            ],
        },
        {
            "title": "Experiment 2: baseline comparison",
            "figure": FIGURES / "02_model_baseline.png",
            "caption": "Figure 2. Held-out MAE for MetroRent and the city-median baseline.",
            "explanation": [
                [
                    "What I tested",
                    "Whether the learned model adds value beyond a simple city-level typical rent.",
                ],
                [
                    "What the graph shows",
                    f"Model MAE is INR {metrics['mae_inr']:,} compared with INR {metrics['city_median_baseline_mae_inr']:,} for the baseline.",
                ],
                [
                    "Conclusion",
                    "Property and locality features materially improve average test error, although the remaining INR error is still substantial.",
                ],
            ],
        },
        {
            "title": "Experiment 3: city error",
            "figure": FIGURES / "03_city_error.png",
            "caption": "Figure 3. Test MAE by city.",
            "explanation": [
                ["What I tested", "Whether one aggregate error represents all five cities."],
                [
                    "What the graph shows",
                    "Mumbai and Bangalore are the hardest, while New Delhi has the lowest MAE in this test split.",
                ],
                [
                    "Conclusion",
                    "Performance is geographically uneven and should always be reported with city-level sample sizes.",
                ],
            ],
        },
        {
            "title": "Experiment 4: interval calibration",
            "figure": FIGURES / "04_interval_coverage.png",
            "caption": "Figure 4. Observed test coverage against the 90 percent target.",
            "explanation": [
                [
                    "What I tested",
                    "Whether the calibration-derived interval covers approximately nine of ten unseen asking rents.",
                ],
                [
                    "What the graph shows",
                    f"The fixed plus-or-minus INR {metrics['interval_radius_inr']:,} interval covers {metrics['interval_coverage']:.1%} of test listings.",
                ],
                [
                    "Conclusion",
                    "The interval communicates uncertainty more honestly than a single point estimate, but it remains wide and non-local.",
                ],
            ],
        },
        {
            "title": "Experiment 5: city rent distribution",
            "figure": FIGURES / "05_city_rent_distribution.png",
            "caption": "Figure 5. Median asking rent by city in the prepared dataset.",
            "explanation": [
                [
                    "What I tested",
                    "How strongly the central rent level differs across cities before modelling other attributes.",
                ],
                [
                    "What the graph shows",
                    "Mumbai has the highest median asking rent, while Nagpur is substantially lower.",
                ],
                [
                    "Conclusion",
                    "City is an essential feature and a sensible baseline grouping, but locality and property characteristics are still needed.",
                ],
            ],
        },
        {
            "title": "Results and interpretation",
            "paragraphs": [
                f"On {metrics['test_rows']:,} held-out listings, MetroRent achieves MAE INR {metrics['mae_inr']:,}, RMSE INR {metrics['rmse_inr']:,}, and R-squared {metrics['r2']:.3f}. The city-median baseline MAE is INR {metrics['city_median_baseline_mae_inr']:,}.",
                "The improvement demonstrates useful predictive structure, but asking-rent data is noisy and heavy-tailed. A large error in premium Mumbai or Bangalore listings has a much greater rupee effect than the same relative error in Nagpur.",
            ],
        },
        {
            "title": "Limitations and reproducibility",
            "paragraphs": [
                "The sources are listing snapshots, not signed lease transactions. Duplicate advertisements may remain, property age and exact coordinates are incomplete, and market timing differs between sources. The model must not be used as a legal valuation or affordability recommendation.",
                "The repository contains source provenance, acquisition, cleaning, training, a prediction example, metrics, tests, five figures, and this report. A future version should use time-aware collection, locality grouping, transaction data, quantile models, and interval coverage by city and price band.",
            ],
        },
        {
            "title": "Conclusion",
            "paragraphs": [
                "I found that a structured machine-learning pipeline reduces average error substantially against a city-median baseline while calibrated intervals expose the remaining uncertainty. MetroRent demonstrates multi-source data preparation, reproducible modelling, baseline comparison, held-out evaluation, calibration, and honest geographic error analysis."
            ],
        },
    ]
    return build_research_report(
        OUTPUT,
        "MetroRent Price Estimator",
        "Harika",
        [
            "This report presents a multi-city monthly asking-rent estimator built from 21,691 public rental listings. I harmonised two source schemas, applied property and rent validity rules, removed duplicate listing signatures, and retained 12,201 rows for model development.",
            "The final model achieved INR 11,652 mean absolute error on 1,831 held-out listings compared with INR 22,122 for a city-median baseline. A separate calibration partition produced an interval with 89.7 percent test coverage. Five experiments explain dataset retention, baseline value, city-level error, interval calibration, and city rent differences.",
        ],
        "rental price modelling; gradient boosting; calibration; baseline comparison; Indian housing",
        sections,
    )


if __name__ == "__main__":
    print(build_report())
