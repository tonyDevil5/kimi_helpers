"""PDF-inspired, causal 10Y fixed-DV01 CTA comparison.

Run with mercury_env Python. Reads existing Bloomberg caches without modifying
the original BackTestingTool; writes a self-contained HTML and audit CSVs here.
"""
from pathlib import Path
import argparse
import re
import sys
import numpy as np
import pandas as pd

LEGACY = Path('D:/kimi_code/BackTestingTool/strategy_example')
sys.path.insert(0, str(LEGACY))
from ust10y_yield_momentum import build_signals, kama, SCENARIOS, run_scenario
from ust10y_cta_comparison import build_new_signals, SCENARIOS as CTA_SCENARIOS

OUT = Path(__file__).resolve().parent / 'walkforward_output'
TENORS = {'2Y': 'gt02_yield.csv', '5Y': 'gt05_yield.csv',
          '10Y': 'gt10_yield.csv', '30Y': 'gt30_yield.csv'}


def breakout(ma, window):
    """Strict breaks of prior window; opposite break exits, never flips directly."""
    high = ma.shift().rolling(window).max().to_numpy()
    low = ma.shift().rolling(window).min().to_numpy()
    values = ma.to_numpy()
    out = np.zeros(len(ma))
    state = 0
    for t, value in enumerate(values):
        up, down = value > high[t], value < low[t]
        if state == 0:
            state = -1 if up else 1 if down else 0
        elif (state == 1 and up) or (state == -1 and down):
            state = 0
        out[t] = state
    return out


def candidates(data, family):
    arrays, names = [], []
    for tenor in TENORS:
        y = data[tenor]
        if family == 'EWMA':
            for span in range(5, 61, 5):
                ma = y.ewm(span=span, adjust=False).mean()
                for window in range(5, 61, 5):
                    arrays.append(breakout(ma, window))
                    names.append(f'{tenor}|EWMA({span},{window})')
        else:
            for n in [5, 10, 15, 20, 30]:
                for fast in [2, 3, 5]:
                    for slow in [10, 15, 20, 30]:
                        ma = kama(y, n, fast, slow)['kama']
                        arrays.append(breakout(ma, 3))
                        names.append(f'{tenor}|KAMA({n},{fast},{slow},3)')
    return np.array(arrays).T, names


def walkforward(signals, names, yields, initial=504, step=20):
    """Train through t-1, choose at close t; exposure earns only t+1 move."""
    moves = -yields.diff().fillna(0).to_numpy() * 100
    candidate_pnl = np.vstack([np.zeros(signals.shape[1]), signals[:-1]]) * moves[:, None]
    sums = candidate_pnl.cumsum(axis=0)
    squares = (candidate_pnl ** 2).cumsum(axis=0)
    target = np.zeros(len(yields))
    selected, audit = [], []
    for t in range(initial, len(yields)):
        if (t-initial) % step == 0:
            mean = sums[t-1] / t
            variance = np.maximum((squares[t-1] - t * mean**2) / (t-1), 0)
            score = np.divide(mean * np.sqrt(252), np.sqrt(variance),
                              out=np.full(len(names), -np.inf), where=variance > 1e-12)
            pool = np.argsort(-score, kind='stable')[:20].tolist()
            selected = [i for i in selected if i in pool]
            while len(selected) < 2:
                remaining = [i for i in pool if i not in selected]
                drivers = {names[i].split('|')[0] for i in selected}
                diverse = [i for i in remaining if names[i].split('|')[0] not in drivers]
                selected.append((diverse or remaining)[0])
            audit.append({'decision_date': str(yields.index[t].date()),
                          'training_end': str(yields.index[t-1].date()),
                          'training_observations': t,
                          'selected_1': names[selected[0]], 'selected_2': names[selected[1]],
                          'train_sharpe_1': score[selected[0]], 'train_sharpe_2': score[selected[1]]})
        target[t] = signals[t, selected].mean()
    return pd.Series(target, index=yields.index), pd.DataFrame(audit)


def daily_pnl(target, yields):
    held = target.shift().fillna(0)
    pnl = -held * yields.diff().fillna(0) * 100
    return pd.DataFrame({'yield': yields, 'target': target, 'held_position': held,
                         'pnl_bps': pnl, 'pnl_usd': pnl * 1000})


def metrics(frame):
    pnl = frame.pnl_bps
    cumulative = pnl.cumsum()
    dd = cumulative - cumulative.cummax().clip(lower=0)
    vol = pnl.std(ddof=1)
    annual = pnl.mean()*252
    maxdd = -dd.min()
    return {'PnL bp': pnl.sum(), 'PnL USD': pnl.sum()*1000,
            'Annual bp': annual, 'Sharpe': pnl.mean()/vol*np.sqrt(252) if vol > 0 else 0,
            'Max DD bp': maxdd, 'Calmar': annual/maxdd if maxdd > 0 else 0,
            'Exposure %': frame.held_position.abs().mean()*100}


def load_data(update=False):
    columns = {}
    for tenor, filename in TENORS.items():
        if update:
            from xbbg import blp
            ticker = {'2Y':'GT02 Govt','5Y':'GT05 Govt','10Y':'GT10 Govt','30Y':'GT30 Govt'}[tenor]
            raw = blp.bdh(ticker, 'PX_LAST', '20160104', pd.Timestamp.today().strftime('%Y%m%d'))
            s = raw.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
        else:
            s = pd.read_csv(LEGACY/'ust10y_momentum_output'/filename,
                            index_col='date', parse_dates=True)['yield']
        columns[tenor] = pd.to_numeric(s, errors='coerce')
    # Intersection only: no backfill or stale yield substitutions.
    return pd.DataFrame(columns).sort_index().loc['2016-01-04':].dropna()


def build_report(data, results, summary, audits):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    panels = []
    for label, start in [('Walk-forward evaluation', data.index[504]),
                         ('2026 close-up', pd.Timestamp('2026-01-01'))]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=('Cumulative PnL (bp; × $1,000 for USD)', 'Drawdown (bp)'))
        for name, frame in results.items():
            sub = frame.loc[start:]
            cum = sub.pnl_bps.cumsum()
            dd = cum-cum.cummax().clip(lower=0)
            fig.add_trace(go.Scatter(x=sub.index,y=cum,name=name,legendgroup=name),row=1,col=1)
            fig.add_trace(go.Scatter(x=sub.index,y=dd,name=name,legendgroup=name,showlegend=False),row=2,col=1)
        fig.update_layout(height=650, title=label, hovermode='x unified',
                          legend=dict(orientation='h'), template='plotly_white')
        panels.append(fig.to_html(full_html=False,include_plotlyjs=False))
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=('10Y yield (%)','Exposure at close (long duration +1)'))
    fig.add_trace(go.Scatter(x=data.index,y=data['10Y'],name='10Y yield'),row=1,col=1)
    for name in ['EWMA walk-forward','KAMA walk-forward','EWMA/KAMA blend']:
        fig.add_trace(go.Scatter(x=data.index,y=results[name].target,name=name,line_shape='hv'),row=2,col=1)
    fig.update_layout(height=550,template='plotly_white',hovermode='x unified')
    fig.update_xaxes(range=['2026-01-01',str(data.index[-1].date())])
    fig.update_xaxes(rangeslider=dict(visible=True),row=2,col=1)
    panels.append(fig.to_html(full_html=False,include_plotlyjs=False))
    tables = ''
    for period in summary.Period.unique():
        tables += '<h2>'+period+'</h2>'+summary[summary.Period==period].drop(columns='Period').round(2).to_html(index=False)
    audit_html = ''.join('<h3>'+family+' selection history</h3>'+audit.round(3).to_html(index=False)
                         for family,audit in audits.items())
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>10Y CTA walk-forward comparison</title>
    <style>body{font:15px Arial;color:#213047;background:#f4f6fa;margin:30px}main{background:white;padding:25px;max-width:1500px;margin:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}h1,h2{color:#133960}.note{background:#edf3fa;padding:15px}details{overflow:auto}</style>
    <script>'''+get_plotlyjs()+'''</script></head><body><main><h1>10Y Treasury CTA: PDF-inspired walk-forward extension</h1>'''
    page += f'<p>Bloomberg GT yields: {data.index[0].date()}–{data.index[-1].date()}; {len(data):,} common observations. Fixed maximum DV01 $1,000/bp; transaction costs zero.</p>'
    page += f'<p>Common evaluation: {data.index[504].date()}–{data.index[-1].date()}. Cached Bloomberg observation calendar retained; 20/504 refer to observations, not a separately validated US Treasury trading calendar. This differs from the PDF’s 2018-01-09 evaluation start.</p>'
    page += '''<div class="note">Click legend names to toggle strategies; double-click to isolate. Drag to zoom. 2026 PnL restarts at zero, while positions carry over from 2025. Positive exposure is long duration; rising yields lose money.</div>'''
    page += tables+''.join(panels)
    page += '''<h2>Method and safeguards</h2><p>The original fixed-rule strategies remain unchanged. New EWMA (span N=5…60, breakout M=5…60, step 5) and KAMA (ER 5/10/15/20/30; fast 2/3/5; slow 10/15/20/30; breakout 3) use strict new highs/lows of the prior smoothed-yield window. A low enters long, a high enters short; an opposite break exits to flat before any later re-entry.</p>
    <p>10Y is always the traded yield. Signals may come from 2Y/5Y/10Y/30Y. After 504 observations, each family is reselected every 20 observations using expanding training ending the previous day. Rank by training daily annualized Sharpe; retain previous selections if still in Top20 and prefer a different signal tenor for replacements. Two selected signals are averaged. The blend averages the two families without further fitted weights. Decisions at close t earn only the subsequent observed yield move: PnL = −position(t−1) × Δyield(%) × 100.</p>
    <p>All comparisons use the same evaluation dates. New methods use fractional exposure; their nominal DV01 cap is the same, but average risk is not identical (see exposure). No new stop/TP overlay is imposed on the PDF strategies; the existing risk-controlled momentum scenario is retained separately. Daily close data cannot guarantee intraday stop fills. Carry, financing, roll, convexity and slippage are excluded.</p>
    <p>This is a reference-inspired implementation, not an exact reproduction: strict break tie handling, recursive EWMA/KAMA initialization, calendar intersection and deterministic selection details are explicit implementation choices. The PDF and 2026 history have already been viewed during design: historical walk-forward results are not a pristine untouched holdout. Training Sharpe is not a promise of future Sharpe. Annual drawdown restarts from a zero PnL anchor.</p><details><summary>Full selection audit (training cutoffs and parameters)</summary>'''
    page += audit_html+'</details></main></body></html>'
    path=OUT/'ust10y_walkforward_report.html'
    path.write_text(page,encoding='utf-8')
    return path


def merge_into_comparison(report_path, walkforward_path):
    """Embed the 10Y-only extension in the original multi-tenor report."""
    report_path, walkforward_path = Path(report_path), Path(walkforward_path)
    original = report_path.read_text(encoding='utf-8')
    source = walkforward_path.read_text(encoding='utf-8')
    body = source.split('<body><main>', 1)[1].rsplit('</main></body>', 1)[0]
    # Original report already embeds the same Plotly library. The extension
    # embeds only its chart initialization scripts, not another library copy.
    body = body.replace('<h1>10Y Treasury CTA: PDF-inspired walk-forward extension</h1>',
                        '<h2>10Y EWMA/KAMA walk-forward comparison</h2>')
    section = ('<!-- WALKFORWARD START --><section id="walkforward-10y">'
               '<p class="subtle">10Y traded yield only. The risk-rule toggle above does not apply here. '
               'This section is hidden when another trade tenor is selected.</p>'+body+
               '</section><script>document.querySelectorAll(".tt-btn").forEach(function(btn){'
               'btn.addEventListener("click",function(){const section=document.getElementById("walkforward-10y");'
               'section.style.display=btn.dataset.tenor==="10Y"?"":"none";'
               'if(btn.dataset.tenor==="10Y")requestAnimationFrame(function(){'
               'section.querySelectorAll(".plotly-graph-div").forEach(function(chart){Plotly.Plots.resize(chart);});});'
               '});});</script><!-- WALKFORWARD END -->')
    original = re.sub(r'<!-- WALKFORWARD START -->.*?<!-- WALKFORWARD END -->',
                      '', original, flags=re.S)
    if '</main></body></html>' not in original:
        raise ValueError('Unexpected comparison HTML structure')
    original = original.replace('</main></body></html>', section+'</main></body></html>')
    report_path.write_text(original, encoding='utf-8')
    return report_path


def main(update=False):
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_data(update)
    if len(data)<505: raise ValueError('Need more than 504 common observations')
    data.to_csv(OUT/'aligned_bloomberg_yields.csv',index_label='date')
    results, audits = {}, {}
    for family in ['EWMA','KAMA']:
        signals,names=candidates(data,family)
        target,audit=walkforward(signals,names,data['10Y'])
        results[family+' walk-forward']=daily_pnl(target,data['10Y'])
        audits[family]=audit
        audit.to_csv(OUT/(family.lower()+'_selection_audit.csv'),index=False)
    blend=(results['EWMA walk-forward'].target+results['KAMA walk-forward'].target)/2
    results['EWMA/KAMA blend']=daily_pnl(blend,data['10Y'])
    legacy=build_new_signals(build_signals(data[['10Y']].rename(columns={'10Y':'yield'})))
    risk_specs=[s for s in SCENARIOS if s.name=='Multi + stop 20bp + take profit 10bp + max hold 30d']
    for spec in CTA_SCENARIOS+ risk_specs:
        _,daily,_=run_scenario(legacy,spec)
        results['Existing: '+spec.name]=pd.DataFrame({'yield':data['10Y'],'target':daily.position,
            'held_position':daily.position.shift().fillna(0),'pnl_bps':daily.net_pnl_bps,'pnl_usd':daily.net_pnl_usd})
    rows=[]
    for name,frame in results.items():
        frame.to_csv(OUT/(name.replace(':','').replace('/','_').replace(' ','_')+'_daily.csv'),index_label='date')
        for period,start in [('Common evaluation',data.index[504]),('2026 YTD',pd.Timestamp('2026-01-01'))]:
            rows.append({'Strategy':name,'Period':period,**metrics(frame.loc[start:])})
        for year,sub in frame.loc[data.index[504]:].groupby(frame.loc[data.index[504]:].index.year):
            rows.append({'Strategy':name,'Period':str(year),**metrics(sub)})
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'strategy_comparison.csv',index=False)
    report=build_report(data,results,summary,audits)
    print(summary[summary.Period.isin(['Common evaluation','2026 YTD'])].round(2).to_string(index=False))
    print(report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update',action='store_true',help='Refresh Bloomberg directly into this extension output')
    main(parser.parse_args().update)
