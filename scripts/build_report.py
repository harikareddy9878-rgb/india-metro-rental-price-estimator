"""Create a ten page report for the Indian metro rental estimator."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/Indian_Metro_Rental_Price_Estimator_Report.pdf"
INK = colors.HexColor("#223149")
RUST = colors.HexColor("#bd563d")
PALE = colors.HexColor("#f3eee8")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#68717d"))
    canvas.drawString(2 * cm, 1.1 * cm, "Indian Metro Rental Price Estimator")
    canvas.drawRightString(19 * cm, 1.1 * cm, f"Page {document.page}")
    canvas.restoreState()


def build_report() -> Path:
    metrics = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    manifest = json.loads((ROOT / "data/raw/source_manifest.json").read_text())
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=28, leading=34, textColor=INK, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=19, leading=24, textColor=INK, spaceAfter=13))
    styles.add(ParagraphStyle(name="Sub", parent=styles["Heading2"], fontSize=12, leading=16, textColor=RUST, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyR", parent=styles["BodyText"], fontSize=10, leading=15, textColor=colors.HexColor("#353b44"), spaceAfter=9))
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Indian Metro Rental Price Estimator", author="Harika")
    story = []
    table_style = TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7cec4")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)])

    story.extend([Spacer(1, 3.1 * cm), Paragraph("Indian Metro Rental<br/>Price Estimator", styles["CoverTitle"]), Paragraph("Real listing preparation, baseline comparison, and calibrated uncertainty", ParagraphStyle(name="CoverSub", parent=styles["BodyR"], fontSize=14, leading=20, textColor=RUST, alignment=TA_CENTER)), Spacer(1, 1.2 * cm), Table([["Project type", "Applied machine learning"], ["Coverage", "Bangalore, New Delhi, Mumbai, Pune, and Nagpur"], ["Raw listings", f"{metrics['raw_rows']:,}"], ["Prepared by", "Harika"]], colWidths=[4 * cm, 9 * cm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), PALE), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7cec4")), ("PADDING", (0, 0), (-1, -1), 9)])), PageBreak()])

    pages = [
        ("1. Executive summary", [f"Two licensed Indian rental datasets contribute {metrics['raw_rows']:,} raw listings. After validity rules and exact signature deduplication, {metrics['model_rows']:,} rows remain for modelling.", f"The held-out MAE is INR {metrics['mae_inr']:,}, compared with INR {metrics['city_median_baseline_mae_inr']:,} for a city-median baseline. The model R2 is {metrics['r2']:.3f}.", f"A separate calibration split produces a plus or minus INR {metrics['interval_radius_inr']:,} interval with {metrics['interval_coverage']:.1%} coverage on the final test set."]),
        ("2. Problem and intended use", ["Rental listings mix location, floor area, furnishing, property configuration, and seller information. A city-wide average cannot represent those interactions.", "The project tests whether a reproducible model improves on a transparent baseline and whether its uncertainty can be communicated without presenting an asking rent as an exact value.", "The output supports learning and listing comparison. It is not a rent valuation, negotiation instruction, legal recommendation, or guarantee of availability."]),
        ("3. Sources and provenance", ["India House Rent Prediction contributes listings from Bangalore, New Delhi, Pune, Mumbai, and Nagpur. New Delhi Rental Properties contributes a larger scraped Delhi snapshot.", "Both Kaggle pages state an MIT licence. The acquisition script records page URLs, file names, row counts, and filtering results in the source manifest.", "The source snapshots were collected at different times and do not represent a probability sample of Indian renters or properties."]),
        ("4. Data preparation", [f"The two schemas are mapped to a common set of twelve fields. {manifest['invalid_rows_removed']:,} rows are removed for missing or implausible target, area, or bedroom values. {manifest['duplicate_rows_removed']:,} exact signatures are removed.", "Raw combined data and prepared model data are stored separately. The cleaning rules are testable functions and the final table can be regenerated from the source endpoints.", "The prepared table contains city, locality, property type, furnishing, seller type, bedrooms, bathrooms, balconies, area, and monthly asking rent."]),
    ]
    for title, paragraphs in pages:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if title.startswith("3."):
            story.append(Table([["Source", "Rows before shared cleaning", "Licence"], ["India House Rent Prediction", "7,691", "MIT"], ["New Delhi Rental Properties", "14,000", "MIT"]], colWidths=[7.5 * cm, 4 * cm, 2.5 * cm], style=table_style))
        if title.startswith("4."):
            story.append(Table([["Quality rule", "Reason"], ["Rent INR 1,000 to 300,000", "Remove impossible and extreme entries for this scope"], ["Area 150 to 8,000 sq ft", "Remove invalid units and obvious entry errors"], ["Bedrooms 0.5 to 10", "Retain RK-style values and remove invalid values"], ["Exact listing signature", "Reduce repeated listings without fuzzy deletion"]], colWidths=[5 * cm, 9 * cm], style=table_style))
        story.append(PageBreak())

    story.extend([Paragraph("5. Evaluation dashboard", styles["Section"]), Image(str(ROOT / "evidence/model_evaluation.png"), width=17 * cm, height=10.9 * cm), Paragraph("The evaluation page combines prediction fit, residual shape, city-level MAE, and prepared source mix so that one headline metric does not hide important variation.", styles["BodyR"]), PageBreak()])

    final_pages = [
        ("6. Model and baseline", ["Categorical fields use an ordinal encoder that preserves an explicit unknown category. Numeric gaps use median imputation. Histogram gradient boosting learns nonlinear effects without requiring a very large one-hot matrix.", "The comparison baseline predicts the median training rent for each city. This is simple, reproducible, and difficult to beat by chance when city is a dominant factor.", f"The model improves MAE by {100 * (1 - metrics['mae_inr'] / metrics['city_median_baseline_mae_inr']):.1f}% over that baseline on the untouched test set."]),
        ("7. Validation and uncertainty", [f"The split uses {metrics['train_rows']:,} training rows, {metrics['calibration_rows']:,} calibration rows, and {metrics['test_rows']:,} final test rows, stratified by city.", "Absolute calibration residuals define a 90th-percentile interval radius. This split-conformal style construction does not assume normally distributed errors.", f"Final interval coverage is {metrics['interval_coverage']:.1%}. The interval is intentionally wide because listing noise and omitted variables remain substantial."]),
        ("8. Error analysis", ["Bangalore and Mumbai have higher MAE than New Delhi, Pune, and Nagpur in this prepared sample. Those city results are published in the evaluation JSON instead of being averaged away.", "Higher-price listings contribute disproportionately to RMSE. Residual review also shows that omitted building condition, maintenance, exact position, and timing remain important.", "A next version should test time-aware validation when trustworthy listing dates are available and should publish error by rent band and locality support."]),
        ("9. Limitations, reproducibility, and next steps", ["Asking rent is not contracted rent. Scraped listings may be stale, promotional, repeated across sites, or entered incorrectly. The data does not cover every Indian city or locality evenly.", "The repository contains acquisition, transformation, modelling, evaluation, prediction, tests, visual evidence, and this report. The README lists one command sequence for a complete rebuild.", "Next steps are to add a linear baseline, obtain reliable collection dates, calibrate city-specific intervals, review locality support, and create a small interface that always shows the model date and uncertainty."]),
    ]
    for title, paragraphs in final_pages:
        story.append(Paragraph(title, styles["Section"]))
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, styles["BodyR"]))
        if title.startswith("6."):
            story.append(Table([["Metric", "Result"], ["MAE", f"INR {metrics['mae_inr']:,}"], ["Baseline MAE", f"INR {metrics['city_median_baseline_mae_inr']:,}"], ["RMSE", f"INR {metrics['rmse_inr']:,}"], ["R2", str(metrics["r2"])]], colWidths=[8 * cm, 5 * cm], style=table_style))
        if not title.startswith("9."):
            story.append(PageBreak())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return OUTPUT


if __name__ == "__main__":
    print(build_report())
