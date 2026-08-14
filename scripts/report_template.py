from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def build_research_report(
    output: Path,
    title: str,
    author: str,
    abstract: list[str],
    keywords: str,
    sections: list[dict],
) -> Path:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ResearchTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=20,
            leading=25,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ResearchHeading",
            parent=styles["Heading1"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.black,
            spaceBefore=4,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ResearchBody",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=10.5,
            leading=15,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ResearchSmall",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            textColor=colors.black,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FigureCaption",
            parent=styles["BodyText"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            textColor=colors.black,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=8,
        )
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(0.35)
        canvas.line(22 * mm, 15 * mm, 188 * mm, 15 * mm)
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(colors.black)
        canvas.drawString(22 * mm, 10 * mm, title)
        canvas.drawRightString(188 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=22 * mm,
        title=title,
        author=author,
    )
    story = [
        Spacer(1, 18 * mm),
        Paragraph(title, styles["ResearchTitle"]),
        Paragraph(
            "Project Report",
            ParagraphStyle(
                name="ReportType", parent=styles["ResearchBody"], alignment=TA_CENTER, fontSize=12
            ),
        ),
        Spacer(1, 10 * mm),
        Paragraph(
            f"Prepared by: {author}",
            ParagraphStyle(name="Author", parent=styles["ResearchBody"], alignment=TA_CENTER),
        ),
        Spacer(1, 14 * mm),
        Paragraph("Abstract", styles["ResearchHeading"]),
    ]
    story.extend(Paragraph(text, styles["ResearchBody"]) for text in abstract)
    story.extend(
        [
            Spacer(1, 4 * mm),
            Paragraph(f"<b>Keywords:</b> {keywords}", styles["ResearchBody"]),
            Spacer(1, 8 * mm),
            Paragraph("Report structure", styles["ResearchHeading"]),
            Paragraph(
                "The report presents the problem, source data, preparation method, evaluation design, five evidence-based experiments, limitations, reproducibility details, and conclusion.",
                styles["ResearchBody"],
            ),
            PageBreak(),
        ]
    )

    table_style = TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.45, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )

    for index, section in enumerate(sections, start=1):
        story.append(Paragraph(f"{index}. {section['title']}", styles["ResearchHeading"]))
        for paragraph in section.get("paragraphs", []):
            story.append(Paragraph(paragraph, styles["ResearchBody"]))
        if section.get("table"):
            rows = [
                [Paragraph(str(cell), styles["ResearchSmall"]) for cell in row] for row in section["table"]
            ]
            column_count = len(rows[0])
            table = Table(rows, colWidths=[164 * mm / column_count] * column_count, repeatRows=1)
            table.setStyle(table_style)
            story.extend([Spacer(1, 3 * mm), table])
        if section.get("figure"):
            image = Image(str(section["figure"]))
            image._restrictSize(164 * mm, 92 * mm)
            image.hAlign = "CENTER"
            story.extend(
                [
                    Spacer(1, 3 * mm),
                    image,
                    Paragraph(section["caption"], styles["FigureCaption"]),
                ]
            )
            for label, text in section.get("explanation", []):
                story.append(Paragraph(f"<b>{label}:</b> {text}", styles["ResearchBody"]))
        if index < len(sections):
            story.append(PageBreak())

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output
