"""DM-specific analytics for the technical monitor, reusing its yield download.

Common yield/RSI/Bollinger/volatility summaries stay in the technical overview.
This module adds the distinct performance, carry, spread, curve and quantile views.
"""
from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dm_govy_report as dg

CURRENCIES = dict(ust='usd', frf='eur', btp='eur', bund='eur', jgb='jpy', esp='eur', uk='gbp')
SPREADS = {'UST − Bund 10Y': 'bund', 'UST − OAT 10Y': 'frf', 'UST − Bono 10Y': 'esp'}


def build_payload(data, funding, names, tenors):
    """Use each yield's available observations, not a global filled calendar."""
    rows, quantiles, curves = [], {}, {}
    for country, ccy in CURRENCIES.items():
        quantile_rows, curve_rows = [], []
        for tenor in tenors:
            series = data.get(country, {}).get(tenor)
            if series is None or series.empty:
                continue
            series = series.dropna().sort_index()
            latest = float(series.iloc[-1])
            as_of = str(series.index[-1].date())
            frame = series.to_frame('yield')
            changes = dg.compute_yield_changes(frame)
            row = {'Country': names[country], 'Tenor': tenor, 'Yield as of': as_of}
            for label in ['1m', '3m', '6m', '1y']:
                row[f'Chg {label} (bp)'] = changes[(label, 'yield')].iloc[-1]
            votes = [changes[(h, 'yield')].iloc[-1] for h in ['1w', '1m', '3m']]
            score = -sum(np.sign(v) for v in votes) if all(pd.notna(v) for v in votes) else np.nan
            row['Momentum score'] = score
            row['Momentum'] = ('Unavailable' if pd.isna(score) else 'Mixed' if score == 0
                else ('Bull' if score == 3 else 'Bullish') if score > 0 else ('Bear' if score == -3 else 'Bearish'))
            row['Currency'] = ccy.upper()
            if ccy in funding:
                rate = funding[ccy].dropna().sort_index()
                # Union preserves funding updates on dates without a yield observation;
                # ffill is causal and never backfills from a later funding publication.
                aligned = rate.reindex(rate.index.union(series.index)).ffill().reindex(series.index)
                current_rate = aligned.iloc[-1]
                row['Funding (%)'] = current_rate
                known = rate.loc[:series.index[-1]]
                row['Funding as of'] = str(known.index[-1].date()) if len(known) else 'Unavailable'
                row['Yield − funding (bp/yr)'] = (latest - current_rate) * 100
                daily_std = series.diff().mul(100).rolling(dg.CARRY_WINDOW).std().iloc[-1]
                row['Carry / daily vol'] = row['Yield − funding (bp/yr)'] / daily_std if daily_std > 0 else np.nan
            rows.append(row)
            recent = series.tail(dg.QUANTILE_WINDOW)
            q = recent.quantile([0, .25, .5, .75, 1])
            quantile_rows.append({'Tenor': tenor, 'As of': as_of, 'Obs': len(recent), 'Current (%)': latest,
                'Min (%)': q[0], 'Q25 (%)': q[.25], 'Median (%)': q[.5], 'Q75 (%)': q[.75],
                'Max (%)': q[1], 'Range position (%)': (latest-q[0])/(q[1]-q[0])*100 if q[1]>q[0] else np.nan})
            curve_rows.append((int(tenor[:-1]), latest, as_of))
        if quantile_rows:
            quantiles[country] = pd.DataFrame(quantile_rows)
            curves[country] = curve_rows

    spread_rows, histories = [], {}
    for label, country in SPREADS.items():
        us, other = data.get('ust', {}).get('10Y'), data.get(country, {}).get('10Y')
        if us is None or other is None:
            continue
        pair = pd.concat([us.rename('us'), other.rename('other')], axis=1).dropna()
        spread = pair.us - pair.other  # percent; existing DM helpers convert changes to bp
        if spread.empty:
            continue
        changes = dg.compute_yield_changes(spread.to_frame('spread'))
        _, upper, lower = dg.compute_bbands(spread)
        tail = spread.tail(250)
        last = spread.iloc[-1]
        row = {'Spread': label, 'As of': str(spread.index[-1].date()), 'Level (bp)': last*100}
        for horizon in dg.HORIZONS:
            row[f'Chg {horizon} (bp)'] = changes[(horizon, 'spread')].iloc[-1]
        row.update({'RSI(14)': dg.compute_rsi(spread).iloc[-1],
            '%B': (last-lower.iloc[-1])/(upper.iloc[-1]-lower.iloc[-1]) if upper.iloc[-1]>lower.iloc[-1] else np.nan,
            'BB low (bp)': lower.iloc[-1]*100, 'BB high (bp)': upper.iloc[-1]*100,
            'Vol20 (bp/yr)': dg.compute_realized_volatility(spread).iloc[-1], 'Quantile obs': len(tail),
            'Min (bp)': tail.min()*100, 'Q25 (bp)': tail.quantile(.25)*100,
            'Median (bp)': tail.median()*100, 'Q75 (bp)': tail.quantile(.75)*100,
            'Max (bp)': tail.max()*100, 'Percentile (%)': tail.le(last).mean()*100})
        spread_rows.append(row)
        histories[label] = spread.tail(252)*100
    return dict(performance=pd.DataFrame(rows), quantiles=quantiles, curves=curves,
        spreads=pd.DataFrame(spread_rows), histories=histories, names=names)


def table_html(frame):
    display = frame.copy()
    for col in display:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')
    return '<div class="table-wrap">'+display.to_html(index=False, escape=True,
        classes='data-table sortable-table', border=0)+'</div>'


def section_html(builder, payload):
    esc = html.escape
    names, perf = payload['names'], payload['performance']
    sections = ['<h2>DM government bonds — comparison and relative value</h2>',
        '<p>Shared yield levels, daily changes, RSI, Bollinger bands, realized volatility and technical charts '
        'are in ALL and the country views. This section adds the DM-specific analytics without repeating that summary.</p>',
        '<p><a href="#dm-performance">Performance &amp; carry</a> · <a href="#dm-spreads">10Y spreads</a> · '
        '<a href="#dm-curves">Yield curves</a> · <a href="#dm-quantiles">60-observation quantiles</a> · '
        '<a href="#dm-heatmap">1-month heatmap</a></p>']
    def block(key, title, body):
        return f'<div class="tenor-block" data-always="1" id="{key}" style="scroll-margin-top:160px"><h3>{title}</h3>{body}</div>'
    sections.append(block('dm-performance', 'Performance, momentum and funding/carry',
        '<p>Changes use 21/63/126/252 available observations for 1m/3m/6m/1y; the technical view retains its 20/60-observation '
        'changes. Momentum = minus the sum of yield-change signs over 5/21/63 observations; +3 is falling yields in all three. '
        'Funding: SOFR (USD), ESTR (EUR), TONAR (JPY), SONIA (GBP), using the latest rate available by each yield date. '
        'Yield minus funding is an annualized yield-spread proxy. Carry / daily vol divides it by the 50-observation daily yield-change standard deviation; '
        'it is the legacy DM ratio, not a return Sharpe or realized bond carry.</p>'+table_html(perf)))
    if payload['histories']:
        fig = make_subplots(rows=len(payload['histories']), cols=1, shared_xaxes=True,
            subplot_titles=list(payload['histories']))
        for row, (label, series) in enumerate(payload['histories'].items(), 1):
            fig.add_trace(go.Scatter(x=series.index, y=series, name=label), row=row, col=1)
            fig.update_yaxes(title_text='bp', row=row, col=1)
        fig.update_layout(template='plotly_white', height=780, showlegend=False, hovermode='x unified',
            title='10Y relative yield spreads — last 252 common observations')
        sections.append(block('dm-spreads', 'UST 10Y vs Bund / OAT / Bono',
            '<p>Levels, band boundaries and quantiles are all in bp. Series use paired observed dates without filling missing yields. '
            'Quantiles/percentile use the last 250 common observations; the chart uses 252. The legacy DM 1d change is the '
            'previous observation minus the one before it; other horizons end at the latest observation. This differs from technical-view 1d changes.</p>'
            +table_html(payload['spreads'])+builder.fig_block(fig)))
    fig = go.Figure()
    for country, rows in payload['curves'].items():
        fig.add_trace(go.Scatter(x=[r[0] for r in rows], y=[r[1] for r in rows], customdata=[r[2] for r in rows],
            name=names[country], mode='lines+markers', hovertemplate='%{x}Y: %{y:.3f}%<br>As of %{customdata}<extra>%{fullData.name}</extra>'))
    fig.update_layout(template='plotly_white', height=530, title='Latest DM yield curves',
        xaxis=dict(title='Tenor (years)', tickvals=[2, 5, 7, 10, 20, 30]), yaxis_title='Yield (%)')
    sections.append(block('dm-curves', 'Latest yield curves', '<p>Latest available value per series; hover to see its observation date.</p>'+builder.fig_block(fig)))
    options, panels = [], []
    for i, (country, frame) in enumerate(payload['quantiles'].items()):
        options.append(f'<option value="{country}">{esc(names[country])}</option>')
        panels.append(f'<div data-dm-quantile="{country}"'+(' hidden' if i else '')+'>'+table_html(frame)+'</div>')
    sections.append(block('dm-quantiles', '60-observation yield quantiles',
        '<p>Each series uses its last 60 available observations. Range position is (current − min) / (max − min), '
        'not an empirical percentile.</p><label>Country <select id="dm-quantile-country">'+''.join(options)+'</select></label>'+''.join(panels)))
    matrix = perf.pivot(index='Country', columns='Tenor', values='Chg 1m (bp)').reindex(
        columns=[f'{v}Y' for v in [2, 5, 7, 10, 20, 30]])
    fig = go.Figure(go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index,
        zmid=0, colorscale='RdBu_r', colorbar=dict(title='bp'), hovertemplate='%{y} %{x}: %{z:.2f} bp<extra></extra>'))
    fig.update_layout(template='plotly_white', height=470, title='1-month yield change — 21 available observations')
    sections.append(block('dm-heatmap', '1-month yield-change heatmap', builder.fig_block(fig)))
    return ''.join(sections)
