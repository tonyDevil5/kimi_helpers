"""Interactive ATM normal volatility and payer-minus-receiver skew monitor."""
from __future__ import annotations

import html
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from swaption_skew_report import OPTION_TENORS, OUTPUT_DIR, UNDERLYING_COL

HORIZONS = [1, 5, 20, 30]


def build_change_summary(df, rr_df, tenors=OPTION_TENORS):
    """Absolute level changes at fixed source-date offsets, not percentage changes."""
    if df.empty:
        raise ValueError('No observations for the swaption change summary.')
    rows = []
    for tenor in tenors:
        row = {'Tenor': tenor, 'As of': str(df.index[-1].date())}
        for label, series in [('ATM', df[f'{tenor}_atm']), ('Skew', rr_df[f'{tenor}_rr'].reindex(df.index))]:
            row[f'{label} level'] = series.iloc[-1]
            for n in HORIZONS:
                row[f'{label} Δ{n}d'] = series.iloc[-1]-series.iloc[-1-n] if len(series)>n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(frame):
    parts = ['<div class="table-wrap"><table id="changes"><thead><tr><th rowspan="2">Tenor</th>',
        '<th colspan="5">ATM normal volatility</th><th colspan="5">Skew: payer +50bp − receiver −50bp</th></tr><tr>']
    parts += ['<th>'+label+'</th>' for _ in range(2) for label in ['Latest', 'Δ1d', 'Δ5d', 'Δ20d', 'Δ30d']]
    parts.append('</tr></thead><tbody>')
    for _, row in frame.iterrows():
        parts.append('<tr><td>'+html.escape(row.Tenor)+'</td>')
        for label in ['ATM', 'Skew']:
            for key in [f'{label} level']+[f'{label} Δ{n}d' for n in HORIZONS]:
                value = row[key]
                change = 'Δ' in key
                style = ' class="positive"' if change and value>0 else ' class="negative"' if change and value<0 else ''
                text = 'Unavailable' if pd.isna(value) else f'{value:+.2f}' if change else f'{value:.2f}'
                parts.append(f'<td{style}>{text}</td>')
        parts.append('</tr>')
    return ''.join(parts)+'</tbody></table></div>'


def build_interactive_report(df, rr_df, source_path, report_path=None):
    summary = build_change_summary(df, rr_df)
    report_path = Path(report_path) if report_path else OUTPUT_DIR / f'Swaption_Skew_Report_Interactive_{df.index[-1]:%Y%m%d}.html'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plots = []

    def add(fig, title):
        fig.update_layout(template='plotly_white', title=title, height=540, hovermode='x unified',
            legend=dict(orientation='h', y=1.12), margin=dict(t=95))
        fig.update_xaxes(rangeslider_visible=True)
        plots.append(fig.to_html(full_html=False, include_plotlyjs=False, config=dict(responsive=True, displaylogo=False)))

    fig = go.Figure()
    for tenor in OPTION_TENORS:
        series = df[f'{tenor}_atm']
        fig.add_trace(go.Scatter(x=df.index, y=series, name=f'{tenor} ATM ({series.iloc[-1]:.2f})', connectgaps=False))
    fig.update_yaxes(title='Annualized normal volatility (bp)')
    add(fig, 'ATM normal volatility — all three option expiries')

    fig = make_subplots(specs=[[dict(secondary_y=True)]])
    for tenor in OPTION_TENORS:
        rr = rr_df[f'{tenor}_rr']
        fig.add_trace(go.Scatter(x=rr.index, y=rr, name=f'{tenor} skew ({rr.iloc[-1]:.2f})', connectgaps=False), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.index, y=df[UNDERLYING_COL], name='CT10 proxy', line=dict(color='black', dash='dash')), secondary_y=True)
    fig.update_yaxes(title='Skew (normal-vol bp)', secondary_y=False)
    fig.update_yaxes(title='CT10 proxy yield (%)', secondary_y=True)
    add(fig, 'Swaption risk reversals vs CT10 proxy')

    for tenor in OPTION_TENORS:
        fig = make_subplots(specs=[[dict(secondary_y=True)]])
        fig.add_trace(go.Scatter(x=df.index, y=rr_df[f'{tenor}_rr'], name=f'{tenor} skew'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df[UNDERLYING_COL], name='CT10 proxy', line=dict(color='#f97316')), secondary_y=True)
        fig.update_yaxes(title='Skew (normal-vol bp)', secondary_y=False)
        fig.update_yaxes(title='Yield (%)', secondary_y=True)
        add(fig, f'{tenor} skew vs CT10 proxy')

    fig = make_subplots(rows=1, cols=3, subplot_titles=OPTION_TENORS)
    stats = []
    for i, tenor in enumerate(OPTION_TENORS, 1):
        pair = pd.concat([df[UNDERLYING_COL].rename('yield'), rr_df[f'{tenor}_rr'].rename('skew')], axis=1).dropna()
        fig.add_trace(go.Scatter(x=pair['yield'], y=pair['skew'], customdata=pair.index.strftime('%Y-%m-%d'),
            mode='markers', marker=dict(size=4, opacity=.45), name=tenor,
            hovertemplate='%{customdata}<br>Yield %{x:.3f}%<br>Skew %{y:.2f}<extra></extra>'), row=1, col=i)
        if len(pair)>1 and pair['yield'].nunique()>1:
            coeff = np.polyfit(pair['yield'], pair['skew'], 1)
            x = np.array([pair['yield'].min(), pair['yield'].max()])
            fig.add_trace(go.Scatter(x=x, y=np.polyval(coeff, x), mode='lines', showlegend=False), row=1, col=i)
        fig.update_xaxes(title='CT10 proxy yield (%)', row=1, col=i)
        fig.update_yaxes(title='Skew (normal-vol bp)', row=1, col=i)
        stats.append({'Tenor': tenor, 'Paired observations': len(pair), 'Mean skew': pair['skew'].mean(),
            'Skew std': pair['skew'].std(), 'Correlation with CT10 yield level': pair['yield'].corr(pair['skew'])})
    fig.update_layout(template='plotly_white', height=480, title='Skew vs CT10 yield — full-sample paired observations', showlegend=False)
    scatter = fig.to_html(full_html=False, include_plotlyjs=False, config=dict(responsive=True, displaylogo=False))
    refs = ' · '.join(f'{n}d reference: {df.index[-1-n]:%Y-%m-%d}' if len(df)>n else f'{n}d reference: unavailable' for n in HORIZONS)
    stats_html = pd.DataFrame(stats).round(3).to_html(index=False, border=0)
    atm_html = '<section>'+plots[0]+'</section>'
    skew_html = ''.join('<section>'+plot+'</section>' for plot in plots[1:])
    summary_json = summary.to_json(orient='records').replace('</', '<\\/')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UST Swaption ATM Volatility &amp; Skew</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f7;color:#183247;font:14px/1.45 Arial,sans-serif}}main{{max-width:1500px;margin:auto;padding:20px}}section{{background:white;padding:20px;border:1px solid #d6e0e8;border-radius:9px;margin-bottom:16px}}h1{{margin:0 0 8px;font-size:25px}}h2{{font-size:18px}}.note{{color:#586c7d;font-size:12px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:13px}}th{{background:#173e56;color:white;text-align:center;padding:9px}}td{{padding:10px;text-align:right;border-bottom:1px solid #dbe3ea}}td:first-child{{text-align:left;font-weight:bold}}tbody tr:nth-child(even){{background:#f4f7fa}}.positive{{color:#b45309;background:#fff5e7}}.negative{{color:#1d4ed8;background:#eef5ff}}button{{padding:7px 12px;border:1px solid #9eb4c3;background:white;border-radius:5px;cursor:pointer}}a{{color:#125b85}}
</style><script>{get_plotlyjs()}</script></head><body><main>
<section id="swaption-summary"><h1>UST swaption ATM volatility &amp; skew</h1>
<p>Data as of <strong>{df.index[-1]:%Y-%m-%d}</strong> · {len(df):,} observations · {df.index[0]:%Y-%m-%d} onward</p>
<p>All values are in annualized normal-volatility bp, as stored in the workbook's ABPVOL fields; no ×100 conversion or relative percentage change is applied.
Skew = +50bp payer vol − −50bp receiver vol. Positive changes mean higher ATM vol or more payer-rich skew.</p>
<p><a href="#swaption-atm">ATM history</a> · <a href="#swaption-skew">Skew histories</a> · <a href="#swaption-scatter">CT10 correlation</a></p>
<h2>ATM and skew changes — one table</h2>{summary_table(summary)}
<p class="note">ΔNd = latest value − value N available source-date observations earlier. Δ1d is latest minus the previous observation. These are observation intervals, not calendar days. Quotes are not filled; missing latest or reference values show Unavailable.</p>
<p class="note">{html.escape(refs)}</p><button id="download-changes" type="button">Download table CSV</button>
<p class="note">Source: {html.escape(str(source_path))} · Data1; Raw1 holds DB(FDER, SOFRSWAPTION, …, ABPVOL) fields. Cached workbook data is read without editing or recalculating the source. Re-running does not refresh the workbook's upstream feed.</p></section>
<div id="swaption-atm">{atm_html}</div><div id="swaption-skew">{skew_html}</div>
<section id="swaption-scatter"><h2>Historical skew statistics and CT10 correlation</h2><div class="table-wrap">{stats_html}</div>{scatter}</section>
<script type="application/json" id="changes-data">{summary_json}</script><script>
document.getElementById('download-changes').addEventListener('click',()=>{{
const rows=JSON.parse(document.getElementById('changes-data').textContent),keys=Object.keys(rows[0]);
const quote=v=>'"'+String(v===null?'Unavailable':v).replace(/"/g,'""')+'"';
const csv=[keys.map(quote).join(','),...rows.map(row=>keys.map(k=>quote(row[k])).join(','))].join('\\r\\n');
const url=URL.createObjectURL(new Blob(['\\ufeff'+csv],{{type:'text/csv;charset=utf-8'}}));
const a=document.createElement('a');a.href=url;a.download='swaption_atm_skew_changes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}});
</script></main></body></html>'''
    report_path.write_text(page, encoding='utf-8')
    summary.to_csv(report_path.parent/'swaption_atm_skew_changes.csv', index=False)
    print(f'Interactive swaption report: {report_path}')
    print(summary.to_string(index=False))
    return report_path
