"""PDF: non-overlapping two-month 7Y Treasury selloffs strictly above 60 bp."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'adhoc_output'/'ust7y_2m_change'
TMP=ROOT/'tmp'/'pdfs'/'ust7y_over60'
OUTPUT=ROOT/'output'/'pdf'/'UST7Y_TwoMonth_Rises_Over60bp.pdf'

SOURCES=[
    ('Federal Reserve, Monetary Policy Report, 1 March 2011, Part 2.',
     'https://www.federalreserve.gov/monetarypolicy/mpr_20110301_part2.htm'),
    ('Federal Reserve Bank of St. Louis, "No taper tantrum this time?", 30 August 2021.',
     'https://fred.stlouisfed.org/blog/2021/08/no-taper-tantrum-this-time/'),
    ('Federal Reserve, Monetary Policy Report, 14 February 2017, Financial Developments.',
     'https://www.federalreserve.gov/monetarypolicy/files/20170214_mprfullreport.pdf'),
    ('Federal Reserve, Monetary Policy Report, 17 June 2022, Part 1: Financial Developments.',
     'https://www.federalreserve.gov/monetarypolicy/2022-06-mpr-part1.htm'),
    ('Federal Reserve, Monetary Policy Report, 3 March 2023, Part 1: Financial Developments.',
     'https://www.federalreserve.gov/monetarypolicy/2023-03-mpr-part1.htm'),
]
BACKDROPS={
    '2010-11':('Stronger economic news, a growth-supportive tax-cut agreement and reassessment of the scale of Fed Treasury purchases contributed to rising yields during QE2.',1),
    '2013-05':('Taper tantrum: signals that the Fed could reduce asset purchases triggered a broad Treasury selloff and repricing of longer-term yields.',2),
    '2016-11':('Post-election reflation: investors anticipated more expansionary fiscal policy. The expected policy-rate path steepened, with some increase in term premiums.',3),
    '2022-03':('Inflation and tightening repricing: inflation pressure and expectations of faster Fed tightening pushed Treasury yields higher, largely through rising real yields.',4),
    '2022-09':('Further hawkish repricing: expectations of a more restrictive policy path and higher real yields accompanied the second-half 2022 Treasury selloff.',5),
}


def main():
    TMP.mkdir(parents=True,exist_ok=True);OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    periods=pd.read_csv(DATA/'calendar_nonoverlap_jan_mar_periods.csv',parse_dates=['start_date','end_date'])
    selected=periods.loc[periods.change_bp>60].copy()
    assert len(selected)==5
    next_rows=[]
    for idx,row in selected.iterrows():
        nxt=periods.iloc[idx+1]
        assert nxt.start_date==row.end_date
        assert np.isclose(row.change_bp,(row.end_yield_pct-row.start_yield_pct)*100)
        next_rows.append((nxt.start_date,nxt.end_date,nxt.change_bp))
    selected['next_start_date']=[r[0] for r in next_rows]
    selected['next_end_date']=[r[1] for r in next_rows]
    selected['next_change_bp']=[r[2] for r in next_rows]
    selected.to_csv(DATA/'over_60bp_cases_and_followthrough.csv',index=False)
    moves=periods.change_bp
    q5,q95=moves.quantile([.05,.95])
    bins=np.arange(np.floor(moves.min()/10)*10,np.ceil(moves.max()/10)*10+10,10)
    chart=TMP/'distribution.png'
    if '--pdf-only' not in sys.argv:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12.5,4.2),layout='constrained')
        lower=moves[moves<=60];upper=moves[moves>60]
        ax.hist([lower,upper],bins=bins,weights=[np.full(len(lower),100/len(moves)),np.full(len(upper),100/len(moves))],
            stacked=True,color=['#26688e','#d37c28'],edgecolor='white',linewidth=.6,
            label=['All other completed periods','Rise >60 bp: 5 periods'])
        ax.axvline(60,color='#ab302c',linewidth=1.7,linestyle='--')
        ax.text(63,ax.get_ylim()[1]*.74,'Filter: >60 bp',fontsize=10,color='#9b322b')
        ax.set(xlabel='7Y Treasury yield change over two calendar months (bp)',ylabel='Share of all 105 periods (%)')
        ax.legend(loc='upper left',frameon=False,fontsize=10)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        fig.savefig(chart,dpi=220);plt.close(fig)
    if '--chart-only' in sys.argv:
        print('CHART',chart)
        return

    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold',r'C:\Windows\Fonts\arialbd.ttf'))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial',boldItalic='Arial-Bold')
    W,H=landscape(A4);margin=36;usable=W-2*margin
    navy=colors.HexColor('#173e56');gray=colors.HexColor('#566b7b')
    body=ParagraphStyle('Body',fontName='Arial',fontSize=9.2,leading=12,textColor=navy)
    small=ParagraphStyle('Small',parent=body,fontSize=8,leading=10.5,textColor=gray)
    cell=ParagraphStyle('Cell',parent=body,fontSize=9,leading=12)
    centered=ParagraphStyle('Centered',parent=cell,alignment=1)
    head=ParagraphStyle('Head',parent=cell,fontName='Arial-Bold',textColor=colors.white,fontSize=9,leading=11,alignment=1)
    c=canvas.Canvas(str(OUTPUT),pagesize=(W,H))
    c.setTitle('UST 7Y: Non-overlapping Two-Month Yield Rises Above 60 bp')
    c.setAuthor('Treasury research')
    def para(text,x,top,width,style=body):
        p=Paragraph(text,style);_,height=p.wrap(width,H);p.drawOn(c,x,top-height);return top-height
    def heading(title,subtitle):
        c.setFillColor(navy);c.setFont('Arial-Bold',21);c.drawString(margin,H-44,title)
        para(subtitle,margin,H-58,usable,small)
    def footer(page):
        c.setStrokeColor(colors.HexColor('#d7e1e8'));c.line(margin,31,W-margin,31)
        c.setFont('Arial',7.5);c.setFillColor(gray)
        c.drawString(margin,19,'Bloomberg GT07 Govt | PX_LAST | Prepared '+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%d %b %Y'))
        c.drawRightString(W-margin,19,f'{page} / 2')

    heading('UST 7Y | Two-month selloffs above 60 bp',
        'Non-overlapping month-start schedule: Jan-Mar, Mar-May, May-Jul, Jul-Sep, Sep-Nov, Nov-Jan.')
    para('105 completed periods | First observed quote in each boundary month | Changes in basis points',margin,H-82,usable)
    c.drawImage(str(chart),margin,207,width=usable,height=usable*4.2/12.5,mask='auto')
    stats=[['COMPLETED PERIODS','RISES >60 BP','MEAN / STANDARD DEVIATION','5TH / 95TH PERCENTILE'],
        ['105',f'5 of 105 ({len(selected)/len(periods)*100:.1f}%)',f'{moves.mean():+.1f} / {moves.std():.1f} bp',f'{q5:+.1f} / {q95:+.1f} bp']]
    strip=Table(stats,colWidths=[usable*.19,usable*.21,usable*.32,usable*.28])
    strip.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Arial'),('FONTNAME',(0,1),(-1,1),'Arial-Bold'),
        ('FONTSIZE',(0,0),(-1,0),8),('FONTSIZE',(0,1),(-1,1),12),('TEXTCOLOR',(0,0),(-1,-1),navy),
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eef4f8')),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    _,height=strip.wrap(usable,H);strip.drawOn(c,margin,146)
    note='<b>Method.</b> Yield change = (ending yield % - initial yield %) x 100. Endpoints use the first available Bloomberg weekday quote of the boundary month; adjacent periods share only that boundary quote. No filling or interpolation. The strict filter is rise &gt;60 bp, not absolute change.'
    y=para(note,margin,126,usable,small)
    y=para('<b>Sample.</b> Completed intervals run from 2 Mar-1 May 2009 through 1 Jul-1 Sep 2026. Bloomberg history begins 26 Feb 2009, so its partial February is excluded. Quotes were downloaded through 26 Oct 2026; the Sep-Nov 2026 interval is not yet complete and is excluded.',margin,y-8,usable,small)
    para('<b>Interpretation.</b> The orange tail contains five episodes. In the next two-month period, yields rose further in 3 cases and fell in 2; the mean subsequent move was +2.7 bp. These are descriptive results from a small sample, not a mean-reversion forecast.',margin,y-8,usable,small)
    footer(1);c.showPage()

    heading('The five qualifying episodes',
        'Starting and ending yields are Bloomberg generic 7Y yields. Positive subsequent changes mean the selloff continued.')
    headers=['Episode<br/>start / end','Initial 7Y<br/>(%)','End 7Y<br/>(%)','Yield rise<br/>(bp)',
        'Subsequent 2m<br/>start / end','Next 2m<br/>(bp)','Market backdrop (historical context)']
    table_rows=[[Paragraph(h,head) for h in headers]]
    for _,row in selected.iterrows():
        context,ref=BACKDROPS[row.start_month]
        values=[f'{row.start_date:%d %b %Y}<br/>{row.end_date:%d %b %Y}',
            f'{row.start_yield_pct:.3f}',f'{row.end_yield_pct:.3f}',f'<b>{row.change_bp:+.1f}</b>',
            f'{row.next_start_date:%d %b %Y}<br/>{row.next_end_date:%d %b %Y}',
            f'<b>{row.next_change_bp:+.1f}</b>',html.escape(context)+f' [{ref}]']
        table_rows.append([Paragraph(value,cell if i in [0,4,6] else centered) for i,value in enumerate(values)])
    widths=[91,53,53,57,91,61,usable-406]
    table=Table(table_rows,colWidths=widths,repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f8')]),
        ('LINEBELOW',(0,0),(-1,0),.7,navy),('LINEBELOW',(0,1),(-1,-1),.4,colors.HexColor('#d7e1e8'))]))
    _,height=table.wrap(usable,H)
    top=H-95;bottom=top-height
    assert bottom>160, f'Table too tall: bottom {bottom}'
    table.drawOn(c,margin,bottom)
    y=para('<b>Follow-through:</b> 3 continued selloffs versus 2 partial reversals; mean next-period change +2.7 bp. '
        'Subsequent moves are measured between the two dates shown, rather than from the original episode start. '
        'The backdrop summaries contextualize the broader Treasury market; they are not a causal decomposition of each 7Y move.',margin,bottom-12,usable,small)
    y-=12
    para('<b>Sources for market backdrop (click to open)</b>',margin,y,usable,small);y-=15
    for i,(title,url) in enumerate(SOURCES,1):
        y=para(f'[{i}] <link href="{url}" color="#155b85">{html.escape(title)}</link>',margin,y,usable,small)-4
    assert y>36, f'Sources overlap footer: {y}'
    footer(2);c.save()
    print('PDF',OUTPUT);print('PAGES 2; CASES',len(selected));print('NEXT MEAN',selected.next_change_bp.mean())


if __name__=='__main__':main()
