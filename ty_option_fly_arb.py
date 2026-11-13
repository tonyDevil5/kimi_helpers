"""TY (10Y Treasury) futures option butterfly (convexity) arbitrage monitor.

Checks the next-month serial TY option for butterfly and vertical-spread
arbitrage across strikes ATM-5.0 .. ATM+5.0 (0.5 increments), calls and puts
separately, on mid prices ((bid+ask)/2, falling back to PX_LAST when a quote
side is missing) and on bid/ask (tradeability). Renders an
interactive HTML report to ty_option_fly_arb_output/.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from xbbg import blp

REPORT_TITLE = "TY Option Butterfly Arbitrage Monitor"
OUTPUT_DIR = Path("ty_option_fly_arb_output")

STRIKE_STEP = 0.5
STRIKE_HALF_WIDTH = 5.0
# Minimum DTE for the "next month" serial option; if the month+1 serial is
# closer to expiry than this, roll to the month after.
MIN_DTE = 10
FLY_TOL = 1 / 64  # one half-32nd; ignore noise-level negatives

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
# Serial option month -> underlying quarterly future month code.
UNDL_CODES = {1: "H", 2: "H", 3: "H", 4: "M", 5: "M", 6: "M",
              7: "U", 8: "U", 9: "U", 10: "Z", 11: "Z", 12: "Z"}

OPT_FIELDS = ["PX_LAST", "BID", "ASK", "OPT_UNDL_TICKER", "LAST_TRADEABLE_DT"]


def bdp_long(tickers, fields) -> pd.DataFrame:
    """bdp returning a pandas long frame with ticker/field/value columns."""
    df = blp.bdp(tickers=tickers, flds=fields)
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    return df


def bdp_value(ticker: str, field: str):
    df = bdp_long([ticker], [field])
    row = df[(df["ticker"] == ticker) & (df["field"] == field)]
    return row["value"].iloc[0] if not row.empty else None

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{background:#1a2a4a;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px}
header .meta{margin:6px 0 0;font-size:13px;opacity:.85}
main{padding:20px 28px;max-width:1200px;margin:0 auto}
.panel{background:#fff;border:1px solid #e0e3e8;border-radius:8px;margin-bottom:20px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.panel h2{margin:0;padding:12px 16px;font-size:15px;border-bottom:1px solid #eceff3}
.panel .body{padding:14px 16px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:5px 9px;text-align:right;border-bottom:1px solid #eceff3;white-space:nowrap}
th{background:#f2f4f8;font-weight:600}
td.strike,th.strike{text-align:center;font-weight:600}
tr.atm td{background:#eef4ff}
td.viol{background:#ffd6d6;color:#a00;font-weight:700}
td.ok{color:#1a7a33}
.summary-arb{color:#a00;font-weight:700}
.summary-ok{color:#1a7a33;font-weight:700}
ul{margin:6px 0;padding-left:20px}
li{margin:3px 0;font-size:13px}
.tabbar{display:flex;gap:6px;margin-bottom:16px}
.tabbar button{padding:8px 18px;font-size:13.5px;font-weight:600;border:1px solid #cdd3dd;
  border-radius:6px;background:#fff;color:#334;cursor:pointer}
.tabbar button.active{background:#1a2a4a;color:#fff;border-color:#1a2a4a}
.tabbar button:hover:not(.active){background:#eef1f6}
.expiry-tab.hidden{display:none}
.expiry-meta{font-size:13px;color:#556;margin:-8px 0 14px}
"""

_JS = """
function showExpiry(idx) {
  document.querySelectorAll('.expiry-tab').forEach(function(el, i) {
    el.classList.toggle('hidden', i !== idx);
  });
  document.querySelectorAll('.tabbar button').forEach(function(el, i) {
    el.classList.toggle('active', i === idx);
  });
  var tab = document.querySelectorAll('.expiry-tab')[idx];
  tab.querySelectorAll('.plotly-graph-div').forEach(function(gd) {
    if (window.Plotly) Plotly.Plots.resize(gd);
  });
}
"""


def fmt_strike(k: float) -> str:
    s = f"{k:.1f}"
    return s[:-2] if s.endswith(".0") else s


def fmt_ticks(v: float, signed: bool = False) -> str:
    """Points -> 64th ticks; one decimal only for fractional ticks."""
    t = v * 64
    s = f"{t:+.1f}" if signed else f"{t:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def pick_option_roots(today: dt.date, n: int = 2) -> list[str]:
    """Return the next n serial option roots (e.g. ['TYF7', 'TYG7'])."""
    roots = []
    month, year = today.month, today.year
    for _ in range(6):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        root = f"TY{MONTH_CODES[month]}{year % 10}"
        probe = f"{root}C 100 Comdty"
        expiry = pd.Timestamp(bdp_value(probe, "LAST_TRADEABLE_DT")).date()
        if (expiry - today).days >= MIN_DTE:
            roots.append(root)
            if len(roots) == n:
                return roots
    raise RuntimeError("Not enough suitable serial options found")


def fetch_data(root: str, strikes: list[float]) -> tuple[pd.DataFrame, str, dt.date]:
    tickers = [f"{root}{cp} {fmt_strike(k)} Comdty" for k in strikes for cp in ("C", "P")]
    df = bdp_long(tickers, OPT_FIELDS)
    df = df.assign(
        strike=[float(t.split()[1]) for t in df["ticker"]],
        cp=[t.split()[0][-1] for t in df["ticker"]],
    )
    rows = []
    for (k, cp), grp in df.groupby(["strike", "cp"]):
        g = grp.set_index("field")["value"]
        rows.append({"strike": k, "cp": cp, **g.to_dict()})
    out = pd.DataFrame(rows)
    undl = out["OPT_UNDL_TICKER"].dropna().iloc[0]
    expiry = pd.Timestamp(out["LAST_TRADEABLE_DT"].dropna().iloc[0]).date()
    return out, undl, expiry


def compute_checks(df: pd.DataFrame, atm: float) -> dict[str, pd.DataFrame]:
    """Per-side frames around ATM: fly edge (mid and tradeable) and violation flags.

    Calls cover strikes >= ATM, puts strikes <= ATM (ATM appears in both).
    Fly edge = 2x price(body) - price(wings); positive = arbitrage (short body,
    long wings collects premium with non-negative payoff). Wings use the full
    pulled chain; monotonicity is checked within each range only.
    """
    wide = df.pivot(index="strike", columns="cp")
    wide.columns = [f"{cp}_{f}".lower() for f, cp in wide.columns]
    wide = wide.sort_index()
    for f in ("px_last", "bid", "ask"):
        for cp in ("c", "p"):
            wide[f"{cp}_{f}"] = pd.to_numeric(wide[f"{cp}_{f}"], errors="coerce")
    for cp in ("c", "p"):
        bid, ask = wide[f"{cp}_bid"], wide[f"{cp}_ask"]
        # Mid = (bid+ask)/2, falling back to PX_LAST when a quote side is missing.
        wide[f"{cp}_mid_fallback"] = bid.isna() | ask.isna()
        wide[f"{cp}_mid"] = ((bid + ask) / 2).where(~wide[f"{cp}_mid_fallback"],
                                                    wide[f"{cp}_px_last"])
        mid = wide[f"{cp}_mid"]
        # Edge from shorting 2x body and buying the wings: positive = arbitrage.
        wide[f"{cp}_fly"] = 2 * mid - mid.shift(1) - mid.shift(-1)
        # Executable version: sell body at bid, buy wings at ask.
        wide[f"{cp}_fly_cost"] = 2 * bid - ask.shift(1) - ask.shift(-1)
    sides = {}
    for cp, mask, sign in (("c", wide.index >= atm, 1), ("p", wide.index <= atm, -1)):
        cols = [f"{cp}_{f}" for f in
                ("px_last", "bid", "ask", "mid", "mid_fallback", "fly", "fly_cost")]
        s = wide.loc[mask, cols].copy()
        s.columns = [c[len(cp) + 1:] for c in cols]
        s["mono_viol"] = s["mid"].diff() * sign > FLY_TOL
        s["fly_viol"] = s["fly"] > FLY_TOL
        s["tradeable_arb"] = s["fly_cost"] > 0
        sides[cp] = s.reset_index()
    return sides


def fly_bg(v_ticks: float, vmin: float, vmax: float = 1.0) -> str:
    """Heat color for a fly edge in ticks: green at vmin (deep convexity),
    white mid-range, red as it approaches the +1 tick arb threshold."""
    t = min(max((v_ticks - vmin) / (vmax - vmin), 0.0), 1.0)
    if t < 0.5:  # green -> white
        u = t / 0.5
        r = int(120 + (255 - 120) * u)
        g = int(200 + (255 - 200) * u)
        b = int(140 + (255 - 140) * u)
    else:  # white -> red
        u = (t - 0.5) / 0.5
        r = 255
        g = int(255 - (255 - 150) * u)
        b = int(255 - (255 - 150) * u)
    return f"rgb({r},{g},{b})"


def table_html(s: pd.DataFrame, cp_label: str) -> str:
    head = (f"<tr><th class='strike'>Strike</th><th>{cp_label} Mid (ticks)</th><th>Bid</th><th>Ask</th>"
            f"<th>Fly Edge (mid)</th><th>Fly Edge (tradeable)</th></tr>")

    def cell(val, viol=False, signed=False, mark=False):
        if pd.isna(val):
            return "<td>—</td>"
        cls = " class='viol'" if viol else ""
        star = "*" if mark else ""
        return f"<td{cls}>{fmt_ticks(val, signed)}{star}</td>"

    def fly_cell(val, viol, vmin):
        if pd.isna(val):
            return "<td>—</td>"
        if viol:
            return f"<td class='viol'>{fmt_ticks(val, signed=True)}</td>"
        return (f"<td style='background:{fly_bg(val * 64, vmin)}'>"
                f"{fmt_ticks(val, signed=True)}</td>")

    vmin = min(s["fly"].min(), s["fly_cost"].min()) * 64

    rows = []
    for _, r in s.iterrows():
        rows.append(
            f"<tr><td class='strike'>{fmt_strike(r['strike'])}</td>"
            + cell(r["mid"], bool(r["mono_viol"]), mark=bool(r["mid_fallback"]))
            + cell(r["bid"]) + cell(r["ask"])
            + fly_cell(r["fly"], bool(r["fly_viol"]), vmin)
            + fly_cell(r["fly_cost"], bool(r["tradeable_arb"]), vmin)
            + "</tr>"
        )
    return (f"<table>{head}{''.join(rows)}</table>"
            "<p style='font-size:11px;color:#777;margin:8px 0 0'>"
            "Prices and edges in 64th-ticks (points × 64). "
            "* Mid fell back to PX_LAST (bid or ask missing) — possibly a stale print. "
            "Fly edge = 2×mid(body) − mid(wings); positive = arbitrage (tolerance 1 tick). "
            "Tradeable edge = 2×bid(body) − ask(wings); positive = executable arbitrage.</p>")


def price_fig(sides: dict[str, pd.DataFrame], atm: float) -> go.Figure:
    fig = go.Figure()
    for cp, name, color in (("c", "Calls", "#1f77b4"), ("p", "Puts", "#d62728")):
        s = sides[cp]
        fig.add_trace(go.Scatter(x=s["strike"], y=s["mid"] * 64, mode="lines+markers",
                                 name=name, marker_color=color))
    fig.add_vline(x=atm, line_dash="dash", line_color="gray", annotation_text="ATM")
    fig.update_layout(title="Option mid prices by strike (calls K ≥ ATM, puts K ≤ ATM)", xaxis_title="Strike",
                      yaxis_title="Mid price (64th-ticks)", height=420, template="plotly_white")
    return fig


def fly_fig(sides: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for cp, name, color in (("c", "Calls fly edge (mid)", "#1f77b4"),
                            ("p", "Puts fly edge (mid)", "#d62728")):
        s = sides[cp]
        fig.add_trace(go.Bar(x=s["strike"], y=s["fly"] * 64, name=name, opacity=0.75,
                             marker_color=color))
    fig.add_hline(y=FLY_TOL * 64, line_dash="dash", line_color="red",
                  annotation_text="arb threshold (+1 tick)")
    fig.update_layout(
        title="Butterfly edge on mid prices: 2×mid(K) − mid(K−0.5) − mid(K+0.5); positive = arbitrage",
        xaxis_title="Body strike K", yaxis_title="Fly edge (64th-ticks)",
        barmode="group", height=420, template="plotly_white")
    return fig


def summary_html(sides: dict[str, pd.DataFrame]) -> str:
    items = []
    for cp, name in (("c", "Calls"), ("p", "Puts")):
        s = sides[cp]
        for _, r in s[s["fly_viol"]].iterrows():
            items.append(f"<li class='summary-arb'>{name}: positive fly edge at K={fmt_strike(r['strike'])} "
                         f"(edge = {fmt_ticks(r['fly'], signed=True)} ticks — body overpriced vs wings)</li>")
        for _, r in s[s["tradeable_arb"]].iterrows():
            items.append(f"<li class='summary-arb'>{name}: TRADEABLE arb at K={fmt_strike(r['strike'])} "
                         f"(edge 2×bid(body) − ask(wings) = {fmt_ticks(r['fly_cost'], signed=True)} ticks)</li>")
        for _, r in s[s["mono_viol"]].iterrows():
            items.append(f"<li class='summary-arb'>{name}: vertical monotonicity violation at "
                         f"K={fmt_strike(r['strike'])}</li>")
    if not items:
        return "<p class='summary-ok'>No butterfly or vertical-spread arbitrage detected.</p>"
    return f"<ul>{''.join(items)}</ul>"


def analyze_expiry(root: str, today: dt.date) -> dict:
    """Run the full chain pull and arb checks for one option root."""
    probe = f"{root}C 100 Comdty"
    meta = bdp_long([probe], OPT_FIELDS).set_index("field")["value"]
    undl_root = meta["OPT_UNDL_TICKER"]
    expiry = pd.Timestamp(meta["LAST_TRADEABLE_DT"]).date()
    dte = (expiry - today).days

    undl_px = float(bdp_value(f"{undl_root} Comdty", "PX_LAST"))
    atm = round(undl_px / STRIKE_STEP) * STRIKE_STEP

    strikes = [round(atm + i * STRIKE_STEP, 2)
               for i in range(-int(STRIKE_HALF_WIDTH / STRIKE_STEP), int(STRIKE_HALF_WIDTH / STRIKE_STEP) + 1)]
    df, undl_chain, expiry_chain = fetch_data(root, strikes)
    sides = compute_checks(df, atm)
    print(f"{root}: underlying {undl_root} (chain confirms {undl_chain}) @ {undl_px:.3f} "
          f"-> ATM {fmt_strike(atm)}; expiry {expiry} (DTE {dte})")
    return {"root": root, "undl_root": undl_root, "undl_px": undl_px, "atm": atm,
            "expiry": expiry, "dte": dte, "sides": sides}


def expiry_tab_html(ex: dict, idx: int, hidden: bool) -> str:
    sides, atm, root = ex["sides"], ex["atm"], ex["root"]
    figs = "".join(
        f'<div class="panel"><h2>{html.escape(f.layout.title.text)}</h2><div class="body">'
        + f.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig{idx}-{i}")
        + "</div></div>"
        for i, f in enumerate([price_fig(sides, atm), fly_fig(sides)])
    )
    cls = "expiry-tab hidden" if hidden else "expiry-tab"
    return f"""<div class="{cls}" id="tab-{idx}">
<p class="expiry-meta">Underlying {ex['undl_root']} @ {ex['undl_px']:.3f} ·
ATM {fmt_strike(atm)} · expiry {ex['expiry']} (DTE {ex['dte']})</p>
<div class="panel"><h2>Arbitrage summary — {root}</h2><div class="body">{summary_html(sides)}</div></div>
<div class="panel"><h2>Calls (K &ge; {fmt_strike(atm)}) — {root} — red = violation</h2><div class="body">{table_html(sides['c'], 'Call')}</div></div>
<div class="panel"><h2>Puts (K &le; {fmt_strike(atm)}) — {root} — red = violation</h2><div class="body">{table_html(sides['p'], 'Put')}</div></div>
{figs}
</div>"""


def build_report() -> Path:
    today = dt.date.today()
    roots = pick_option_roots(today, n=2)
    print(f"Option roots: {roots}")
    expiries = [analyze_expiry(root, today) for root in roots]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"TY_Option_Fly_Arb_{today:%Y%m%d}.html"

    tab_buttons = "".join(
        f'<button class="{"active" if i == 0 else ""}" onclick="showExpiry({i})">'
        f'{html.escape(ex["root"])} (exp {ex["expiry"]})</button>'
        for i, ex in enumerate(expiries)
    )
    tabs = "".join(expiry_tab_html(ex, i, hidden=i != 0) for i, ex in enumerate(expiries))

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<script>{get_plotlyjs()}</script>
<style>{_CSS}</style>
</head>
<body>
<header><h1>{html.escape(REPORT_TITLE)}</h1>
<p class="meta">Expiries {' · '.join(ex['root'] for ex in expiries)} · generated {today} ·
calls K &ge; ATM, puts K &le; ATM ·
fly edge = 2×mid(body) − mid(wings); positive = arbitrage</p></header>
<main>
<div class="tabbar">{tab_buttons}</div>
{tabs}
</main>
<script>{_JS}</script>
</body></html>"""
    out.write_text(document, encoding="utf-8")
    print(f"Report saved to: {out}")
    return out


if __name__ == "__main__":
    build_report()
