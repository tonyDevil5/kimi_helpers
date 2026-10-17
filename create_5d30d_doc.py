"""Create a Word document summarizing the 5d vs 30d MBS coupon signal analysis."""

from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUTPUT_PATH = Path(r"D:\python\kimi_helpers\MBS_5d_vs_30d_Signal_Analysis.docx")


def set_cell_shading(cell, fill: str):
    """Set background color of a table cell (fill as hex, e.g. 'D9E1F2')."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    return p


def add_paragraph(doc, text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(11)
    return p


def add_table(doc, headers, rows, header_fill="D9E1F2"):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_shading(hdr_cells[i], header_fill)
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)

    for row in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
            for paragraph in row_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    return table


def main():
    doc = Document()

    # Title
    title = doc.add_paragraph()
    run = title.add_run("MBS Coupon Signal Analysis: 5-Day vs 30-Day Performance vs 7Y UST")
    run.bold = True
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # Introduction
    add_heading(doc, "1. What the metric means", level=1)
    add_paragraph(doc,
        "In the Citi MBS vs 7Y UST section of the MBS RV report, performance is measured as the "
        "excess return of an MBS coupon over the 7-year Treasury benchmark, expressed in ticks (×32).")
    add_paragraph(doc,
        "• Positive value = MBS outperformed 7Y UST.\n"
        "• Negative value = MBS underperformed 7Y UST.")
    add_paragraph(doc,
        "The 5-day (5d) and 30-day (30d) cumulative figures act as a short-term momentum oscillator "
        "relative to a one-month trend. Comparing them helps identify when a coupon’s recent price "
        "action is diverging from its broader trend — an early sign of a potential flip.")

    # Scenarios table
    add_heading(doc, "2. Six hypothetical signal scenarios", level=1)
    headers = ["Scenario", "5d perf", "30d perf", "Relationship", "Interpretation"]
    rows = [
        ["A. Downtrend slowing", "−2", "−25", "5d > 30d", "Coupon was weak over the month, but the last week was much less bad. Possible bottoming / short-covering."],
        ["B. Short-term reversal", "+5", "−20", "5d > 30d", "Clear bounce after a weak month. If it continues, 30d turns positive — early bullish flip."],
        ["C. Momentum accelerating", "+12", "+8", "5d > 30d", "Already in an uptrend and recent performance is even stronger. Bullish continuation."],
        ["D. Momentum fading", "+3", "+15", "5d < 30d", "Still positive, but the rally is cooling. Watch for 5d to turn negative — early warning."],
        ["E. Early rollover", "−6", "+10", "5d < 30d", "Strong month, but the last week gave back gains. Distribution / bearish flip forming."],
        ["F. Extreme divergence", "+18", "−22", "gap = +40", "Massive snap-back vs the month. Could be overdone short-term; mean-reversion risk if fundamentals haven’t changed."],
    ]
    add_table(doc, headers, rows, header_fill="B4C7E7")

    # Progressions
    add_heading(doc, "3. How a bullish flip evolves", level=1)
    add_paragraph(doc, "Scenario A → B → C shows the typical bullish progression:")
    add_paragraph(doc,
        "A. 5d = −2, 30d = −25   → downtrend intact but slowing\n"
        "B. 5d = +5, 30d = −20   → short-term turns positive → bullish flip triggers\n"
        "C. 5d = +12, 30d = +8   → both positive, 5d leads → trend confirmed")
    add_paragraph(doc,
        "A trader might flag the flip at B and confirm it at C. Waiting for C reduces false signals "
        "but gives up some of the move.")

    add_heading(doc, "4. How a bearish flip evolves", level=1)
    add_paragraph(doc, "Scenario D → E shows the bearish mirror:")
    add_paragraph(doc,
        "D. 5d = +3, 30d = +15   → still bullish, but momentum fading\n"
        "E. 5d = −6, 30d = +10   → 5d crosses below 30d → bearish flip")

    # Curve trade example
    add_heading(doc, "5. Curve trade example", level=1)
    add_paragraph(doc,
        "Suppose two coupons show opposite signals on the same day:")
    headers2 = ["Coupon", "5d perf", "30d perf", "Signal"]
    rows2 = [
        ["FN 4.0", "+4", "−18", "Strong short-term reversal"],
        ["FN 6.5", "−2", "+14", "Momentum fading / early rollover"],
    ]
    add_table(doc, headers2, rows2, header_fill="C6E0B4")
    add_paragraph(doc,
        "This divergence suggests a relative-value rotation: money is moving out of premium coupons "
        "and into discount coupons. A possible trade expression is to buy FN 4.0 and sell FN 6.5 "
        "if you believe the rotation continues. The 5d/30d pair flags the turn; you confirm it by "
        "watching whether the 30d trend follows the 5d lead over the next few sessions.")

    # Practical rule
    add_heading(doc, "6. Simple rule to code", level=1)
    add_paragraph(doc,
        "perf['gap'] = perf['last_5d'] - perf['last_30d']\n"
        "perf['signal'] = np.where(perf['gap'] > 0, 'bullish', 'bearish')\n"
        "perf['early_bull'] = (perf['last_5d'] > 0) & (perf['last_30d'] < 0)")
    add_paragraph(doc,
        "Use the gap to rank coupons by short-term momentum improvement, and use the early-bull "
        "filter to catch reversals before the 30-day trend has turned.")

    # Footer note
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Generated from MBS RV report analysis — 5d vs 30d performance vs 7Y UST.")
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)

    doc.save(OUTPUT_PATH)
    print(f"Saved Word document to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
