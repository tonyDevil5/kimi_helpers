from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import pandas as pd
import numpy as np
from datetime import datetime

# Load model output
df = pd.read_csv('cta_output.csv')
latest = df.iloc[-1]

# Create document
doc = Document()

# Helper to add a table from dict/list
def add_table_from_data(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(10)
    
    for row in rows:
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
            for paragraph in row_cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    return table

# Title
title = doc.add_heading('CTA Model Summary Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = doc.add_paragraph('TY1 Treasury Futures Analysis')
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle2 = doc.add_paragraph(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
subtitle2.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# Executive Summary
p = doc.add_heading('Executive Summary', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.add_paragraph(
    'This report summarizes the output of a CTA trend-following model applied to '
    'TY1 (10-Year U.S. Treasury) futures. The model uses time-series momentum '
    '(past 60-day return normalized by realised volatility) to classify market '
    'positioning into four discrete regimes: max long, moderately long, moderately short, and max short.'
)

doc.add_paragraph(
    f'As of {latest["date"]}, TY1 is trading at {latest["price"]:.4f}. '
    f'The model currently flags the position as {latest["regime"].replace("_", " ")} '
    f'(position weight: {latest["position"]}). This reflects negative medium-term momentum '
    'following the recent selloff in Treasury futures.'
)

# Current Positioning
p = doc.add_heading('Current Positioning', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.add_paragraph('The table below shows the latest market data and model signals:')

pos_rows = [
    ['Ticker', 'TY1 COMB Comdty'],
    ['Date', latest['date']],
    ['Last Price', f"{latest['price']:.4f}"],
    ['EMA (20-day)', f"{latest['ema_fast']:.4f}"],
    ['EMA (60-day)', f"{latest['ema_medium']:.4f}"],
    ['EMA (120-day)', f"{latest['ema_slow']:.4f}"],
    ['60-day Momentum', f"{latest['momentum']*100:.2f}%"],
    ['Realised Vol (ann.)', f"{latest['vol']*100:.2f}%"],
    ['Z-Score', f"{latest['z']:.2f}"],
    ['Current Regime', latest['regime'].replace('_', ' ')],
    ['Position Weight', str(int(latest['position']))],
]
add_table_from_data(doc, ['Metric', 'Value'], pos_rows)
doc.add_paragraph()

# Key Levels
p = doc.add_heading('Key Levels to Watch', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.add_paragraph(
    'The model calculates the exact price levels at which the z-score crosses '
    'regime boundaries. These are the "trigger prices" for CTA flow.'
)

# Compute levels inline
p_lookback = latest['price'] / (1 + latest['momentum']) if latest['momentum'] != -1 else np.nan
vol = latest['vol']

if pd.notna(p_lookback) and pd.notna(vol) and vol > 0:
    reduce_longs = p_lookback * (1 + 1.0 * vol)
    flip_long = p_lookback * (1 + 0.25 * vol)
    flip_short = p_lookback * (1 - 0.25 * vol)
    reduce_shorts = p_lookback * (1 - 1.0 * vol)
    
    level_rows = [
        ['Reduce longs (max → mod)', f'{reduce_longs:.4f}', f'{reduce_longs - latest["price"]:.4f}'],
        ['Flip long (neutral → mod long)', f'{flip_long:.4f}', f'{flip_long - latest["price"]:.4f}'],
        ['Flip short (neutral → mod short)', f'{flip_short:.4f}', f'{flip_short - latest["price"]:.4f}'],
        ['Reduce shorts (max → mod)', f'{reduce_shorts:.4f}', f'{reduce_shorts - latest["price"]:.4f}'],
    ]
    add_table_from_data(doc, ['Signal', 'Price Level', 'Distance from Spot'], level_rows)

doc.add_paragraph()

# Model Methodology
p = doc.add_heading('Model Methodology', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.add_paragraph('Engine: Time-Series Momentum (TSMOM)')
doc.add_paragraph(
    '1. Momentum: Compute the N-day return (default 60 days).\n'
    '2. Volatility: Compute the rolling realised volatility over the same window, annualised.\n'
    '3. Z-Score: Normalise momentum by volatility.\n'
    '   z = momentum / vol\n'
    '4. Regime Mapping: Discretise the continuous z-score into four regimes.\n'
    '   • z >= +1.0     -> max_long (+2)\n'
    '   • +0.25 <= z < +1.0 -> mod_long (+1)\n'
    '   • -0.25 < z < +0.25 -> neutral (0)\n'
    '   • -1.0 < z <= -0.25 -> mod_short (-1)\n'
    '   • z <= -1.0     -> max_short (-2)'
)

# Backtest Results
p = doc.add_heading('Backtest Results (In-Sample)', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

bt = pd.read_csv('cta_output.csv')
bt['daily_ret'] = bt['price'].pct_change()
bt['pos_lag'] = bt['position'].shift(1).fillna(0)
bt['pnl'] = bt['pos_lag'] * bt['daily_ret']
bt['tc'] = bt['pos_lag'].diff().abs().fillna(0) * 0.04 / bt['price']
bt['pnl_tc'] = bt['pnl'] - bt['tc']
bt['cumulative'] = bt['pnl_tc'].cumsum()

total_ret = bt['pnl_tc'].sum()
gross_ret = bt['pnl'].sum()
ann_vol = bt['pnl_tc'].std() * np.sqrt(252)
sharpe = (bt['pnl_tc'].mean() / bt['pnl_tc'].std()) * np.sqrt(252) if bt['pnl_tc'].std() > 0 else 0
max_dd = (bt['cumulative'] - bt['cumulative'].cummax()).min()

bt_rows = [
    ['Total Return (net)', f'{total_ret*100:.2f}%'],
    ['Total Return (gross)', f'{gross_ret*100:.2f}%'],
    ['Annualised Volatility', f'{ann_vol*100:.2f}%'],
    ['Sharpe Ratio', f'{sharpe:.2f}'],
    ['Max Drawdown', f'{max_dd*100:.2f}%'],
    ['Days in Sample', str(len(bt))],
]
add_table_from_data(doc, ['Metric', 'Value'], bt_rows)
doc.add_paragraph()

# Regime Distribution
p = doc.add_heading('Regime Distribution', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

regime_counts = bt['regime'].value_counts().to_dict()
regime_rows = []
for reg, cnt in regime_counts.items():
    pct = cnt / len(bt) * 100
    regime_rows.append([reg.replace('_', ' '), str(cnt), f'{pct:.1f}%'])
add_table_from_data(doc, ['Regime', 'Days', 'Share'], regime_rows)
doc.add_paragraph()

# Interpretation
p = doc.add_heading('Interpretation', 1)
p.alignment = WD_ALIGN_PARAGRAPH.LEFT

doc.add_paragraph(
    '• The model has been in neutral roughly half the time, reflecting the choppy, range-bound '
    'price action in Treasuries over the past two years.'
)
doc.add_paragraph(
    '• Current mod_short positioning is consistent with the post-February selloff. '
    'TY would need to rally to ~113.58 (z = +0.25) to flip the model to mod_long.'
)
doc.add_paragraph(
    '• The "reduce_longs" level at 117.47 is far above spot, indicating that even a significant '
    'rally would not immediately push CTAs into max_long territory given current volatility.'
)
doc.add_paragraph(
    '• The "reduce_shorts" level at 107.09 marks the boundary between max_short and mod_short. '
    'A drop toward 107 would trigger deeper CTA selling.'
)

# Footer
doc.add_paragraph()
footer = doc.add_paragraph('Report generated by cta_model.py')
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.runs[0].font.size = Pt(9)
footer.runs[0].font.color.rgb = RGBColor(128, 128, 128)

# Save
doc_path = Path(__file__).parent / 'CTA_Model_Summary.docx'
doc.save(doc_path)
print(f'Saved Word document to: {doc_path}')
