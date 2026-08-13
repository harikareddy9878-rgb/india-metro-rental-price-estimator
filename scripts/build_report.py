"""Create a ten page report for the Bengaluru rent estimator."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/Bengaluru_Rent_Estimator_Report.pdf"
INK = colors.HexColor("#25344a")
RUST = colors.HexColor("#aa4a30")
PALE = colors.HexColor("#f4eee7")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6a6f77"))
    canvas.drawString(2 * cm, 1.15 * cm, "Bengaluru Rent Estimator")
    canvas.drawRightString(19 * cm, 1.15 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    with (ROOT / "data/raw/bengaluru_rent_listings.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    locality_counts = Counter(row["locality"] for row in rows)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=29, leading=34, textColor=INK, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=20, leading=25, textColor=INK, spaceAfter=14))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=12, leading=16, textColor=RUST, spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#363a40"), spaceAfter=9))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Bengaluru Rent Estimator", author="Harika", subject="Machine learning project report")
    story = []
    table_style = TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9cec2")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)])

    story.extend([Spacer(1, 3.2 * cm), Paragraph("Bengaluru Rent Estimator", styles["Title2"]), Paragraph("Regression, uncertainty, and responsible interpretation", ParagraphStyle(name="SubTitle", parent=styles["Bodyx"], fontSize=14, leading=20, textColor=RUST, alignment=TA_CENTER)), Spacer(1, 1.3 * cm), Table([["Project type", "Applied machine learning"], ["Geography", "Bengaluru, India"], ["Dataset", "Deterministic synthetic listing sample"], ["Prepared by", "Harika"]], colWidths=[4 * cm, 9 * cm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE), ("TEXTCOLOR", (0, 0), (0, -1), INK), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d9cec2")), ("PADDING", (0, 0), (-1, -1), 9)])), Spacer(1, 1.1 * cm), Paragraph("This report covers the problem, data assumptions, feature pipeline, model evaluation, error analysis, uncertainty, limitations, and reproducibility.", styles["Bodyx"]), PageBreak()])

    pages = [
        ("1. Executive summary", [
            "The project estimates monthly asking rent from seven property attributes and two categorical attributes. A random forest model is evaluated on a fixed held-out test set, and predictions are accompanied by an empirical tree-based interval.",
            "The current test result is an MAE of INR {:,.0f}, RMSE of INR {:,.0f}, and R² of {:.3f}. The sample is synthetic, so these numbers demonstrate the workflow rather than real market accuracy.".format(metrics["mae_inr"], metrics["rmse_inr"], metrics["r2"]),
            "The central design decision is to communicate a range instead of a single confident price. The interval still needs calibration because the current 10th to 90th percentile coverage is {:.1%}.".format(metrics["p10_p90_coverage"]),
        ]),
        ("2. Problem and intended use", [
            "Rental comparisons become misleading when locality, floor area, furnishing, transport access, and building age are considered separately. This project combines them in one repeatable model.",
            "The output can help a learner compare fictional listings, inspect model behaviour, and practise error analysis. It is not a verified market valuation, legal recommendation, or substitute for visiting a property.",
            "Success means a reproducible split, clear metrics, tolerance of unseen categorical values, a saved model, a testable prediction function, and honest limitations.",
        ]),
        ("3. Dataset and assumptions", [
            "The dataset contains 960 fictional listings across twelve Bengaluru localities. It is generated with a fixed seed and contains no scraped names, phone numbers, addresses, or user activity.",
            "Rent is influenced by floor area, bedroom and bathroom count, furnishing, parking, gated community status, locality, property age, and metro distance. Random variation prevents the target from being a perfectly deterministic formula.",
            "Synthetic data creates a cleaner relationship than a real marketplace. Evaluation results therefore describe this generator and must not be presented as evidence of current Bengaluru rents.",
        ]),
        ("4. Feature and model pipeline", [
            "Categorical fields are converted with a dictionary vectorizer that ignores unseen combinations safely. Numeric fields pass through unchanged. A random forest regressor learns nonlinear relationships and interactions.",
            "The split holds out 22 percent of rows with a fixed random state. No test target is used during fitting. The fitted preprocessing and model objects are saved together, reducing training and prediction mismatch.",
            "The feature set is intentionally compact: locality, furnishing, BHK, square feet, bathrooms, age, metro distance, parking, and gated community status.",
        ]),
    ]
    for title, paragraphs in pages:
        story.append(Paragraph(title, styles["H1x"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["Bodyx"]))
        if title.startswith("3."):
            story.append(Paragraph("Locality representation", styles["H2x"]))
            story.append(Table([["Locality", "Rows"]] + [[name, str(count)] for name, count in locality_counts.most_common(8)], colWidths=[10 * cm, 4 * cm], style=table_style))
        if title.startswith("4."):
            story.append(Table([["Step", "Implementation"], ["Encoding", "DictVectorizer"], ["Estimator", "Random forest regressor"], ["Validation", "Fixed 78 / 22 train test split"], ["Persistence", "Single joblib pipeline"], ["Inference", "Median plus 10th and 90th percentiles"]], colWidths=[4 * cm, 10 * cm], style=table_style))
        story.append(PageBreak())

    story.extend([Paragraph("5. Evaluation results", styles["H1x"]), Image(str(ROOT / "evidence/model_evaluation.png"), width=17 * cm, height=6.55 * cm), Spacer(1, 0.4 * cm), Table([["Metric", "Result"], ["Test rows", str(metrics["test_rows"])], ["MAE", f"INR {metrics['mae_inr']:,.0f}"], ["RMSE", f"INR {metrics['rmse_inr']:,.0f}"], ["R²", str(metrics["r2"])], ["Interval coverage", f"{metrics['p10_p90_coverage']:.1%}"]], colWidths=[8 * cm, 5 * cm], style=table_style), Paragraph("The actual versus predicted plot follows the diagonal but shows wider errors for higher rents. Residuals are centred near zero, while the tails confirm that a point estimate should not be treated as exact.", styles["Bodyx"]), PageBreak()])

    remaining = [
        ("6. Error analysis", ["MAE describes the typical absolute difference in rupees and is easier to communicate than squared error. RMSE is higher because it gives more weight to larger misses.", "The highest-rent homes are less frequent and combine larger floor areas with premium locality multipliers. Their sparse representation increases variance. A future real-data model should report errors by price band and locality, not only one city-wide score.", "A residual review should also check systematic underestimation of premium furnished homes and overestimation of older homes far from transport."]),
        ("7. Prediction uncertainty", ["Each tree produces a prediction. The project uses the 10th percentile, median, and 90th percentile of those tree predictions as a practical uncertainty summary.", "The median interval width is INR {:,.0f}. Current coverage is {:.1%}, below the nominal 80 percent span, so the range is descriptive rather than calibrated.".format(metrics["median_interval_width_inr"], metrics["p10_p90_coverage"]), "Conformal calibration on a separate validation split would be a stronger next step. Intervals should also widen for unfamiliar localities or feature combinations outside the training sample."]),
        ("8. Limitations and responsible use", ["The sample is synthetic and reflects its written assumptions. It excludes maintenance charges, deposits, exact street location, building quality, availability date, floor number, negotiation, and market change.", "A high R² on generated data is not proof of real-world performance. Real listings may include duplicates, errors, broker advertisements, and hidden selection bias.", "The estimator should be described as a learning project. Any real rental decision needs current comparable listings, property inspection, contract review, and local judgement."]),
        ("9. Reproducibility and next steps", ["The generator, training code, saved metrics, evaluation image, tests, and report script are versioned together. The README gives the exact commands and continuous integration repeats generation, training, tests, and static checks.", "Next steps are to replace the sample with a licensed timestamped dataset, add grouped cross-validation, calibrate intervals, compare a linear baseline, inspect feature influence, and monitor error across locality and rent bands.", "A small input application can be added only after validation. Its output should always show the model date, range, assumptions, and a clear educational-use statement."]),
    ]
    for title, paragraphs in remaining:
        story.append(Paragraph(title, styles["H1x"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["Bodyx"]))
        if title.startswith("8."):
            story.append(Table([["Risk", "Control"], ["Synthetic data mistaken for market data", "Prominent data statement"], ["Single price appears certain", "Return an interval"], ["Unknown category breaks inference", "Ignore unseen categories safely"], ["Training and prediction mismatch", "Save one complete pipeline"]], colWidths=[7 * cm, 7 * cm], style=table_style))
        story.append(PageBreak() if not title.startswith("9.") else Spacer(1, 0))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
