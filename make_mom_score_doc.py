"""Generate DM_GOVY mom_score methodology Word document."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

OUT = Path(__file__).parent / "DM_GOVY_MomScore_Methodology.docx"

doc = Document()

# Base style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def h(text, level=1):
    doc.add_heading(text, level=level)

def p(text, bold=False, italic=False, size=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return par

def bullet(text):
    doc.add_paragraph(text, style="List Bullet")

def code(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x1F, 0x3B, 0x66)

# ---------------------------------------------------------------- title
title = doc.add_heading("DM GOVY Report — mom_score: Momentum Signal Methodology", level=0)

h("1. What is mom_score?", 1)
p('mom_score is a multi-horizon trend alignment score shown on the DM Government '
  'Bond summary pages of the daily DM_GOVY report. It answers one question: '
  'is the recent move in a bond\u2019s yield a confirmed trend or noise?')
p('For each of the 36 DM government bonds (US, France, Germany, Spain, Japan, UK '
  'across 2Y/5Y/7Y/10Y/20Y/30Y), the score measures whether yields have been moving '
  'in the same direction across three horizons: 1 week, 1 month, and 3 months.', )

h("2. Method", 1)
p('For each bond, take the yield changes over three horizons and count how many of '
  'them are falling vs rising:')
code('mom_score = -( sign(chg_1w) + sign(chg_1m) + sign(chg_3m) )')
p('where sign() returns +1 if the yield went up, -1 if it went down, and 0 if '
  'unchanged. The leading minus flips the sign so the score reads from a '
  'bondholder\u2019s perspective: falling yields (a bond rally) score positive.')
p('Each horizon contributes \u00b11, so the score ranges from -3 to +3:', bold=True)

tbl = doc.add_table(rows=1, cols=2)
tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
hdr[0].text = "Score"
hdr[1].text = "Meaning"
rows = [
    ("+3", "Yields fell over 1w, 1m AND 3m \u2192 strong, consistent bullish momentum (bond rally)"),
    ("+1", "2 of 3 horizons falling \u2192 mildly bullish"),
    ("0",  "Mixed / flat \u2192 no clear trend"),
    ("-1", "2 of 3 horizons rising \u2192 mildly bearish"),
    ("-3", "Yields rose over all three horizons \u2192 strong, consistent bearish momentum (bond selloff)"),
]
for s, m in rows:
    c = tbl.add_row().cells
    c[0].text = s
    c[1].text = m

doc.add_paragraph()

h("3. Companion column: mom_dir", 1)
p('mom_dir translates the score into a plain-language label:')
bullet('Bear / Bull \u2014 score is \u00b13: all horizons agree (clean, confirmed trend)')
bullet('Bearish / Bullish \u2014 score is \u00b11: partial agreement (trend with some divergence)')
bullet('Mixed \u2014 score 0, or insufficient history')

h("4. Worked examples (2026-09-07 data)", 1)

p('Example 1 \u2014 us_10y (score -3, "Bear")', bold=True)
t1 = doc.add_table(rows=1, cols=3)
t1.style = "Light Grid Accent 1"
h1 = t1.rows[0].cells
h1[0].text, h1[1].text, h1[2].text = "Horizon", "Change", "sign"
for hz, chg, sg in [("1w", "+3.2 bps", "+1"), ("1m", "+13.7 bps", "+1"), ("3m", "+23.0 bps", "+1")]:
    c = t1.add_row().cells
    c[0].text, c[1].text, c[2].text = hz, chg, sg
doc.add_paragraph()
code('mom_score = -(+1 + 1 + 1) = -3  \u2192 Bear')
p('Treasuries sold off consistently at every horizon \u2014 a confirmed trend, not a one-day wobble.')

doc.add_paragraph()
p('Example 2 \u2014 jpy_30y (score -1, "Bearish")', bold=True)
t2 = doc.add_table(rows=1, cols=3)
t2.style = "Light Grid Accent 1"
h2 = t2.rows[0].cells
h2[0].text, h2[1].text, h2[2].text = "Horizon", "Change", "sign"
for hz, chg, sg in [("1w", "-10.0 bps", "-1"), ("1m", "+10.9 bps", "+1"), ("3m", "+17.3 bps", "+1")]:
    c = t2.add_row().cells
    c[0].text, c[1].text, c[2].text = hz, chg, sg
doc.add_paragraph()
code('mom_score = -(-1 + 1 + 1) = -1  \u2192 Bearish')
p('The long end is still in a 3-month selloff, but the 1-week move has turned down, '
  'so bearish momentum is fading at the margin.')

h("5. Why it is useful", 1)
bullet('A single-day z-score tells you if a move is unusual; mom_score tells you if a '
       'move is confirmed \u2014 a one-week spike scores at most \u00b11, never \u00b13.')
bullet('Because it uses sign agreement rather than magnitude, it is immune to volatility '
       'differences between markets (e.g. JGB vs Treasury) \u2014 comparable across all 36 bonds.')
bullet('The heatmap on the report pages makes regime shifts visible at a glance: a fully '
       'red grid (-3 Bear) shows a broad global selloff; pockets of -1 show where the long '
       'end is starting to stabilize.')

h("6. Interpretation guide for the daily report", 1)
p('Suggested reading of the mom_score / mom_dir columns on the summary pages:', italic=True)
bullet('|mom_score| = 3 across a whole country row \u2192 broad, confirmed trend in that curve')
bullet('Front end -3 but long end -1 \u2192 bear-flattening losing steam at the long end')
bullet('Scores flipping from -3 to +3 in a few days \u2192 sharp regime change; check the '
       'chg_1d / z-score columns for the catalyst')

doc.add_paragraph()
par = doc.add_paragraph()
run = par.add_run("Source: dm_govy_report.py (build_summary_table) \u2014 computed from Bloomberg "
                  "generic government yield tickers (GT/GTDEM/GTFRF/GTESP/GTJPY/GUKG).")
run.italic = True
run.font.size = Pt(9)

doc.save(OUT)
print(f"Saved: {OUT}")
