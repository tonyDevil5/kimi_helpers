"""Reproducible, observation-age-matched FN shock analogue analysis.

Uses saved Bloomberg PX_LAST snapshots; no future outcomes enter distances.
Run with D:/Anaconda/envs/mercury_env/python.exe fn_basis_analogues.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).parent / 'mbs_rv_output' / 'fn_shocks'
GROUPS = {
    'basis': ['baseline_bps', 'initial_score', 'daily_vol_bps'],
    'path': [],
    'rates': ['yield_10y', 'yield_change_5_bps', 'yield_change_20_bps'],
    'swaption_vol': ['normal_vol', 'vol_change_5', 'vol_change_20'],
    'policy': ['policy_rate', 'policy_change_1m_bps', 'policy_change_3m_bps', 'policy_change_6m_bps'],
}
WEIGHTS = {
    'balanced': dict(basis=.2, path=.2, rates=.2, swaption_vol=.2, policy=.2),
    'basis_only': dict(basis=.5, path=.5, rates=0, swaption_vol=0, policy=0),
    'macro_heavy': dict(basis=.1, path=.1, rates=.25, swaption_vol=.25, policy=.3),
    'no_policy': dict(basis=.25, path=.25, rates=.25, swaption_vol=.25, policy=0),
}


def main():
    scores = pd.read_csv(OUT / 'FN_Basis_Shock_Scores.csv', index_col=0, parse_dates=True).sort_index()
    events = pd.read_csv(OUT / 'FN_Basis_Shock_Episodes.csv')
    raw = pd.read_csv(OUT / 'FN_Analogue_Bloomberg_Raw.csv', header=[0, 1], index_col=0, parse_dates=True).sort_index()
    raw.columns = raw.columns.get_level_values(0)
    assert raw.index.is_unique and scores.index.is_unique
    raw.to_csv(OUT / 'FN_Analogue_Bloomberg_Sorted.csv', index_label='date')
    current = events.iloc[-1]
    current_i = scores.index.get_loc(pd.Timestamp(current.trigger_date))
    age = len(scores) - 1 - current_i
    groups = {k: list(v) for k, v in GROUPS.items()}
    groups['path'] = [f'path_{n}' for n in range(1, age + 1)]
    assert age >= 1
    audit = []

    def value_at(ticker, date, event):
        available = raw.loc[:date, ticker].dropna()
        if available.empty:
            raise ValueError(f'Missing history: {ticker} {date}')
        used = available.index[-1]
        assert used <= date and (date - used).days <= 7
        audit.append(dict(event=event, ticker=ticker, requested_date=date.date(), source_date=used.date(), value=float(available.iloc[-1])))
        return float(available.iloc[-1])

    rows = []
    for r in events.itertuples():
        first = scores.index.get_loc(pd.Timestamp(r.trigger_date))
        stop = first + age
        assert stop < len(scores)
        snapshot = scores.index[stop]
        event = f'E{r.episode}'
        basis_path = scores.basis_bps.iloc[first:stop+1].to_numpy()
        scale = r.pre_shock_daily_vol_bps * np.sqrt(5)
        policy = value_at('FEDL01 Index', snapshot, event)
        row = dict(event=event, first_signal=r.trigger_date, snapshot_date=str(snapshot.date()),
                   baseline_bps=r.baseline_bps, initial_score=r.trigger_score,
                   initial_widening_bps=r.trigger_change_bps, daily_vol_bps=r.pre_shock_daily_vol_bps,
                   basis_at_snapshot=basis_path[-1], post_signal_change_bps=basis_path[-1]-basis_path[0],
                   yield_10y=value_at('GT10 Govt', snapshot, event),
                   normal_vol=value_at('USSNA110 FMCA Curncy', snapshot, event), policy_rate=policy)
        for n in (5, 20):
            earlier = scores.index[stop-n]
            row[f'yield_change_{n}_bps'] = 100 * (row['yield_10y'] - value_at('GT10 Govt', earlier, event))
            row[f'vol_change_{n}'] = row['normal_vol'] - value_at('USSNA110 FMCA Curncy', earlier, event)
        for months in (1, 3, 6):
            row[f'policy_change_{months}m_bps'] = 100 * (policy - value_at('FEDL01 Index', snapshot - pd.DateOffset(months=months), event))
        for n in range(1, age+1):
            row[f'path_{n}'] = (basis_path[n]-basis_path[0]) / scale
        for n in (5, 10, 20):
            row[f'future_{n}obs_change_bps'] = scores.basis_bps.iloc[stop+n] - basis_path[-1] if stop+n < len(scores) else np.nan
        row['peak_to_half_obs'] = r.half_recovery_observations
        row['peak_to_full_obs'] = r.full_recovery_observations
        row['first_signal_to_full_obs'] = r.trigger_to_full_recovery_observations
        rows.append(row)
    features = pd.DataFrame(rows).set_index('event')
    target = features.index[-1]
    hist = features.iloc[:-1].copy()
    assert len(hist) == 18
    for name, cols in groups.items():
        sd = hist[cols].std(ddof=1).replace(0, 1)
        z = (hist[cols] - features.loc[target, cols]) / sd
        assert np.isfinite(z.to_numpy(dtype=float)).all()
        hist[name + '_distance2'] = (z.astype(float)**2).mean(axis=1)
        hist[name + '_absolute'] = z.astype(float).abs().mean(axis=1)
    for name, weights in WEIGHTS.items():
        hist[name + '_distance'] = np.sqrt(sum(hist[g+'_distance2'] * w for g, w in weights.items()))
        hist[name + '_rank'] = hist[name + '_distance'].rank(method='min').astype(int)
        hist[name + '_absolute_distance'] = sum(hist[g+'_absolute'] * w for g, w in weights.items())
        hist[name + '_absolute_rank'] = hist[name + '_absolute_distance'].rank(method='min').astype(int)
    hist = hist.sort_values('balanced_distance')
    features.to_csv(OUT / 'FN_Analogue_Features.csv')
    hist.to_csv(OUT / 'FN_Analogue_Rankings.csv')
    pd.DataFrame(audit).to_csv(OUT / 'FN_Analogue_Data_Audit.csv', index=False)
    (OUT / 'FN_Analogue_Method.json').write_text(json.dumps(dict(age=int(age), target=target, groups=groups, weights=WEIGHTS,
        scaling='Sample standard deviation across 18 historical snapshots, excluding current event',
        distance='RMS: square root of weighted mean squared standardized differences. Absolute: weighted mean absolute standardized differences. Equal feature weights within each group.',
        policy='Effective rate, not target range; daily observation dates, not a vintage-correct publication-time backtest',
        missing='Backward as-of only, maximum 7 calendar days; no backfill'), indent=2))
    render(scores, events, features, hist, groups, age, target)
    print('TARGET\n', features.loc[target].to_string())
    print('\nRANKINGS\n', hist[['balanced_distance','balanced_rank','balanced_absolute_distance','balanced_absolute_rank','basis_only_rank','basis_only_absolute_rank','macro_heavy_rank','macro_heavy_absolute_rank','no_policy_rank','no_policy_absolute_rank','future_5obs_change_bps','future_10obs_change_bps','future_20obs_change_bps']].round(3).to_string())


def render(scores, events, features, hist, groups, age, target):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    picks = list(hist.index[:3])
    absolute_picks = list(hist.sort_values('balanced_absolute_distance').index[:3])
    colors = dict(zip([target]+picks, ['#111827', '#2563eb', '#b45309', '#0f766e']))
    pdf = OUT / 'FN_Basis_E19_Analogues.pdf'
    def footer(fig, page):
        fig.text(.045,.025,'Source: saved Bloomberg PX_LAST; FN basis x 100. As of '+str(scores.index[-1].date())+f' | {page}',fontsize=8,color='#475569')
    def table(ax, data, headers, font=9):
        ax.axis('off')
        t = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(font)
        t.scale(1,1.8)
        for (i,j), c in t.get_celld().items():
            c.set_edgecolor('#e2e8f0')
            if i==0:
                c.set_facecolor('#243447'); c.set_text_props(color='white',weight='bold')
            elif i%2:
                c.set_facecolor('#f1f5f9')
    with PdfPages(pdf) as output:
        fig = plt.figure(figsize=(14,9))
        fig.suptitle(f'{target}: closest historical FN basis shocks',fontsize=20,weight='bold',y=.95)
        fig.text(.05,.88,f'RMS leaders: {", ".join(picks)} | Absolute leaders: {", ".join(absolute_picks)} | First signal + {age} observations',fontsize=11.5)
        fig.text(.05,.835,'Five equally weighted groups: basis conditions, early path, Treasury yields, swaption vol, policy rate.',fontsize=11)
        ax = fig.add_axes([.08,.43,.86,.34])
        top = hist.head(8)
        ax.barh(top.index[::-1], top.balanced_distance.iloc[::-1],color='#2563eb')
        ax.set_xlabel('Standardized distance (lower = closer; not a probability)')
        ax.grid(axis='x',alpha=.2)
        data = []
        for event in picks:
            r=hist.loc[event]
            data.append([event,r.first_signal,f'{r.balanced_distance:.2f}',int(r.balanced_rank),f'{r.balanced_absolute_distance:.2f}',int(r.balanced_absolute_rank)])
        table(fig.add_axes([.05,.17,.90,.19]),data,['Event','First signal','RMS distance','RMS rank','Absolute distance','Absolute rank'])
        fig.text(.05,.105,'E18 leads all RMS and absolute weighting schemes, but its policy backdrop differs: E19 has a recent +25 bp effective-rate move.',fontsize=10)
        fig.text(.05,.075,'Ranks are exploratory: 18 historical events; weights are chosen, not fitted. Historical outcomes do not enter matching.',fontsize=10)
        footer(fig,1); output.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(14,9))
        fig.suptitle('Market conditions at the same event age',fontsize=19,weight='bold',y=.95)
        fields=[('Snapshot date','snapshot_date'),('FN baseline (bps)','baseline_bps'),('First-signal widening (bps)','initial_widening_bps'),
            ('First-signal score','initial_score'),('Pre-shock daily vol (bps)','daily_vol_bps'),('FN basis at snapshot (bps)','basis_at_snapshot'),
            ('Change since first signal (bps)','post_signal_change_bps'),('10y yield (%)','yield_10y'),('10y change, 5 obs (bps)','yield_change_5_bps'),
            ('10y change, 20 obs (bps)','yield_change_20_bps'),('1y10y normal annualized vol (bps)','normal_vol'),('Vol change, 5 obs','vol_change_5'),
            ('Vol change, 20 obs','vol_change_20'),('Effective policy rate (%)','policy_rate'),('Policy change, 1 month (bps)','policy_change_1m_bps'),
            ('Policy change, 3 months (bps)','policy_change_3m_bps'),('Policy change, 6 months (bps)','policy_change_6m_bps')]
        data=[[label]+[f'{features.loc[e,key]:.2f}' if key!='snapshot_date' else features.loc[e,key] for e in [target]+picks] for label,key in fields]
        ax=fig.add_axes([.04,.17,.92,.68]); ax.axis('off')
        t=ax.table(cellText=data,colLabels=['Measure',target]+picks,loc='center',cellLoc='center',colWidths=[.36,.16,.16,.16,.16])
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1,1.8)
        for (i,j),cell in t.get_celld().items():
            cell.set_edgecolor('#e2e8f0')
            if i==0: cell.set_facecolor('#243447'); cell.set_text_props(color='white',weight='bold')
            elif i%2: cell.set_facecolor('#f1f5f9')
        fig.text(.05,.11,'FEDL01: effective-rate proxy. E19 uses Sep 17 value of 3.88%; Sep 18 is missing. No future value is filled backward.',fontsize=9)
        fig.text(.05,.075,'Ticker definitions: GT10 Govt; USSNA110 FMCA Curncy (normal annualized bps); FEDL01 Index.',fontsize=9)
        footer(fig,2); output.savefig(fig); plt.close(fig)

        fig, axes=plt.subplots(1,2,figsize=(14,7))
        fig.suptitle('Observed early path and subsequent historical paths',fontsize=18,weight='bold')
        for event in [target]+picks:
            r=events.loc[events.episode==int(event[1:])].iloc[0]
            i=scores.index.get_loc(pd.Timestamp(r.trigger_date))
            early=scores.basis_bps.iloc[i:i+age+1].to_numpy()
            axes[0].plot(np.arange(len(early)),early-early[0],marker='o',label=event,color=colors[event],lw=2)
            start=i+age
            later=scores.basis_bps.iloc[start:min(start+21,len(scores))].to_numpy()
            axes[1].plot(np.arange(len(later)),later-later[0],marker='o' if event==target else None,label=event,color=colors[event],lw=2)
        for ax in axes:
            ax.axhline(0,color='#64748b',lw=.7); ax.grid(alpha=.2); ax.legend(frameon=False)
        axes[0].set(title='Only this early period is available for E19',xlabel='Observations after first signal',ylabel='Basis change from first signal (bps)')
        axes[1].set(title='Historical outcomes: excluded from ranking',xlabel='Observations after matched snapshot',ylabel='Further basis change (bps; negative = tightening)')
        fig.text(.05,.075,'E19 has no observed forward path yet. Historical trajectories can cross later shocks and are not independent forecasts.',fontsize=10)
        fig.tight_layout(rect=[.02,.13,.98,.92]); footer(fig,3); output.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(14,9))
        fig.suptitle('All candidates and method limits',fontsize=19,weight='bold',y=.95)
        data=[]
        for e,r in hist.iterrows():
            data.append([e,r.first_signal,f'{r.balanced_distance:.2f}',int(r.balanced_rank),f'{r.balanced_absolute_distance:.2f}',int(r.balanced_absolute_rank),
                         f'{r.future_5obs_change_bps:+.2f}',f'{r.future_10obs_change_bps:+.2f}',f'{r.future_20obs_change_bps:+.2f}'])
        table(fig.add_axes([.025,.31,.95,.53]),data,['Event','First signal','RMS dist','RMS rank','Abs dist','Abs rank','Next 5 obs','Next 10 obs','Next 20 obs'],7.3)
        notes=[
            'Distances: standardize each feature by its standard deviation across the 18 historical snapshots; exclude E19 from scaling.',
            'RMS distance squares standardized differences; absolute distance averages their magnitudes. Forward columns are basis changes in bps.',
            'Sensitivity: basis-only = basis/path 50% each; macro-heavy = 10/10/25/25/30%; no-policy = remaining four groups 25% each.',
            'Only first signal and the next three observations define each snapshot. Final signal dates, final peaks and recovery times do not select matches.',
            'Event catalogue was identified retrospectively. FEDL01 uses observation dates; publication timestamps/vintages are unavailable.',
            'These are descriptive comparisons, not a real-time backtest. Dataset observations include some holidays; counts are not exchange sessions.',
            'Backward as-of joins use values no more than seven calendar days old. Policy changes compare calendar-month lookbacks.',
        ]
        for n,note in enumerate(notes): fig.text(.05,.23-n*.026,note,fontsize=8.5)
        footer(fig,4); output.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(14,9))
        fig.suptitle('Squared versus absolute-distance sensitivity',fontsize=19,weight='bold',y=.95)
        ax = fig.add_axes([.07,.48,.43,.36])
        ax.scatter(hist.balanced_distance,hist.balanced_absolute_distance,color='#2563eb',s=35)
        for event,r in hist.iterrows():
            if r.balanced_rank <= 6 or r.balanced_absolute_rank <= 6:
                ax.annotate(event,(r.balanced_distance,r.balanced_absolute_distance),xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.set_xlabel('RMS standardized distance')
        ax.set_ylabel('Mean absolute standardized distance')
        ax.grid(alpha=.2)
        abs_top = hist.sort_values('balanced_absolute_distance').head(8)
        data=[]
        for event,r in abs_top.iterrows():
            data.append([event,f'{r.balanced_absolute_distance:.3f}',int(r.balanced_absolute_rank),f'{r.balanced_distance:.3f}',int(r.balanced_rank)])
        table(fig.add_axes([.54,.48,.40,.36]),data,['Event','Abs dist','Abs rank','RMS dist','RMS rank'],8.2)
        compare = list(dict.fromkeys(list(hist.head(4).index)+list(abs_top.head(4).index)))
        data=[]
        for event in compare:
            r=hist.loc[event]
            data.append([event,f'{r.basis_absolute:.3f}',f'{r.path_absolute:.3f}',f'{r.rates_absolute:.3f}',
                         f'{r.swaption_vol_absolute:.3f}',f'{r.policy_absolute:.3f}',f'{r.balanced_absolute_distance:.3f}'])
        table(fig.add_axes([.06,.18,.88,.22]),data,['Event','Basis','Early path','Rates','Swaption vol','Policy','Overall abs'],8.2)
        fig.text(.06,.125,'Absolute method: take |z| for every feature, average within each group, then average the five group scores.',fontsize=9)
        fig.text(.06,.095,'RMS method penalizes a single large mismatch more strongly. Absolute distance is more robust to one unusual feature.',fontsize=9)
        fig.text(.06,.065,'Raw distance levels are on different scales; compare event ranks and stability across methods.',fontsize=9)
        footer(fig,5); output.savefig(fig); plt.close(fig)
    print('PDF:',pdf)


if __name__ == '__main__':
    main()
