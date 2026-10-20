"""Plot workbook FN 6 HEDGE_10Y and price, September 2026; source unchanged."""
from pathlib import Path
import json
import pandas as pd

SOURCE = Path('C:/Users/tonyy/Nutstore/1/python/notebook/data/mbs/mbs_rv.xlsx')
OUT = Path(__file__).resolve().parent / 'adhoc_output'


def main():
    raw = pd.read_excel(SOURCE, sheet_name='Raw', header=5,
                        usecols='D,AG,DL')
    assert {'FN 6 Price', 'FN 6 HR'}.issubset(raw.columns)
    raw = raw.rename(columns={raw.columns[0]: 'date'})
    raw['date'] = pd.to_datetime(raw.date, errors='coerce')
    data = raw.loc[raw.date.between('2026-09-01', '2026-09-30')].copy()
    data = data.sort_values('date')
    assert not data.date.duplicated().any()
    for col in ['FN 6 Price', 'FN 6 HR']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    assert len(data) == 21 and data.notna().all().all()
    OUT.mkdir(exist_ok=True)
    data.assign(date=data.date.dt.strftime('%Y-%m-%d')).to_json(
        OUT/'fn6_hr_price_sep2026_data.json', orient='records', indent=2)
    # Fixed, clearly labeled axes keep the original series units separate.
    left, right, top, bottom = 115, 1065, 160, 565
    x = lambda date: left+(date-pd.Timestamp('2026-09-01')).days/29*(right-left)
    yh = lambda v: bottom-(v-.45)/.45*(bottom-top)
    yp = lambda v: bottom-(v-97.5)/4*(bottom-top)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="710" viewBox="0 0 1200 710">',
           '<rect width="1200" height="710" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#203449">',
           '<text x="65" y="55" font-size="28" font-weight="bold">FN 6 hedge ratio and price</text>',
           '<text x="65" y="88" font-size="17" fill="#607487">September 2026 · observed workbook dates · no interpolation of missing observations</text>',
           '<text x="115" y="130" font-size="17" fill="#1769aa">Hedge ratio (HEDGE_10Y, left)</text>',
           '<text x="730" y="130" font-size="17" fill="#bc5419">Price ($ per $100 par, right)</text>']
    for v in [.45,.50,.55,.60,.65,.70,.75,.80,.85,.90]:
        y=yh(v)
        svg += [f'<line x1="{left}" x2="{right}" y1="{y}" y2="{y}" stroke="#e7edf2"/>',
                f'<text x="100" y="{y+5}" text-anchor="end" font-size="14" fill="#1769aa">{v:.2f}</text>']
    for v in [97.5,98,98.5,99,99.5,100,100.5,101,101.5]:
        svg += [f'<text x="1080" y="{yp(v)+5}" font-size="14" fill="#bc5419">{v:.1f}</text>']
    for day in [1,4,8,11,15,18,22,25,30]:
        xx=x(pd.Timestamp(2026,9,day))
        svg += [f'<text x="{xx}" y="595" text-anchor="middle" font-size="14">Sep {day:02d}</text>']
    for col,func,color in [('FN 6 HR',yh,'#1769aa'),('FN 6 Price',yp,'#bc5419')]:
        points=' '.join(f'{x(d):.2f},{func(v):.2f}' for d,v in zip(data.date,data[col]))
        svg += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>']
        for d,v in zip(data.date,data[col]):
            svg += [f'<circle cx="{x(d):.2f}" cy="{func(v):.2f}" r="3.5" fill="{color}"/>']
        for row,dx,anchor in [(data.iloc[0],10,'start'),(data.iloc[-1],-8,'end')]:
            svg += [f'<text x="{x(row.date)+dx:.2f}" y="{func(row[col])-14:.2f}" text-anchor="{anchor}" font-size="15" font-weight="bold" fill="{color}">{row[col]:.3f}</text>']
    hr_delta=data['FN 6 HR'].iloc[-1]-data['FN 6 HR'].iloc[0]
    price_delta=data['FN 6 Price'].iloc[-1]-data['FN 6 Price'].iloc[0]
    svg += [f'<text x="115" y="638" font-size="18">Sep 1–30 change: hedge ratio {hr_delta:+.3f} · price {price_delta:+.3f} points</text>',
            '<text x="115" y="671" font-size="14" fill="#607487">Source: mbs_rv.xlsx · Raw sheet · FN 6 HR (DL) and FN 6 Price (AG); dates in D.</text>',
            '<text x="115" y="693" font-size="13" fill="#607487">Lines connect available observations. Separate axis scales; hedge ratio shown in workbook units.</text>',
            '</g></svg>']
    (OUT/'fn6_hr_price_sep2026.svg').write_text('\n'.join(svg),encoding='utf-8')
    print(json.dumps({'observations':len(data),'first':str(data.date.iloc[0].date()),
                      'last':str(data.date.iloc[-1].date()),'HR_change':hr_delta,
                      'price_change':price_delta}))


if __name__=='__main__': main()
