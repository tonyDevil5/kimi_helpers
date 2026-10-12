"""
Ad-hoc analysis: two CMO bonds vs CT10, CC basis, 2s10s and 1y10y vol.

Bonds:
    FHS 402 F7 Mtge        (7 cap, Freddie)   -> FN CC basis
    GNR 2023-113 FD Mtge   (6.5 cap, Ginnie)  -> G2 CC basis

Pulls daily prices from Bloomberg since deal inception (Aug 2023), together
with CT10 (USGG10YR), 2y (USGG2YR -> 2s10s slope) and the FN/G2 current-
coupon OAS (same tickers as adhoc_g2_6_vs_ct10.py).  1y10y ATM normal vol
comes from mbs_rv.xlsx (same source as mbs_rv_report.py).  For each bond an
OLS regression

    price = a + b1*CT10 + b2*CC_basis + b3*2s10s + b4*1y10y_vol

is fitted over the full history; the current residual (actual - fitted)
measures how rich/cheap the bond is versus its own history.

Output:
    ./adhoc_output/adhoc_cmo_valuation.html   (self-contained, plotly)

Run:
    python adhoc_cmo_valuation.py
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(__file__).parent / "adhoc_output"
CACHE = OUTPUT_DIR / "cmo_valuation_data.csv"
MBS_RV_XLSX = r"C:\Users\tonyy\Nutstore\1\python\notebook\data\mbs\mbs_rv.xlsx"

START = "2023-08-01"  # both deals priced Aug 2023

BONDS = {
    "FHS 402 F7 Mtge": {"label": "FHS 402 F7 (7 cap)", "cc": "FN CC"},
    "GNR 2023-113 FD Mtge": {"label": "GNR 2023-113 FD (6.5 cap)", "cc": "G2 CC"},
}
CC_TICKERS = {"FN CC": ".FNCC105 G Index", "G2 CC": ".G2CC105 G Index"}
CT10_TICKER = "USGG10YR Index"
CT2_TICKER = "USGG2YR Index"

# display names / units for predictors
PRED_META = {
    "CT10": ("CT10", "%"),
    "cc": ("CC basis", "bp"),
    "2s10s": ("2s10s", "bp"),
    "vol": ("1y10y vol", "bp"),
}

# Scenario shocks in predictor units (CT10 in %, others in bp).
# Add more dicts to show additional scenarios in the report.
SCENARIOS = [
    {"name": "Bear: CT10 +20bp, CC +10bp",
     "shocks": {"CT10": 0.20, "cc": 10.0, "2s10s": 0.0, "vol": 0.0}},
    {"name": "Bull: CT10 -20bp, CC -10bp",
     "shocks": {"CT10": -0.20, "cc": -10.0, "2s10s": 0.0, "vol": 0.0}},
]

C_PX = "#1f4e79"
C_FIT = "#7a9cc6"
C_CT10 = "#c00000"
C_CC = "#e08e0b"
C_2S10S = "#2e8b57"
C_VOL = "#7b3fb5"


def load_bbg() -> pd.DataFrame | None:
    tickers = list(BONDS) + [CT10_TICKER, CT2_TICKER] + list(CC_TICKERS.values())
    try:
        from xbbg import blp
    except ImportError:
        print("xbbg not available")
        return None
    try:
        print(f"Downloading {len(tickers)} series from Bloomberg since {START}...")
        df = blp.bdh(tickers=tickers, flds=["px_last"], start_date=START)
        if df is None or df.empty:
            raise RuntimeError("empty bdh response")
        df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        for k, v in CC_TICKERS.items():
            df[k] = df[v] * 100.0  # OAS % -> bps
        df = df.rename(columns={CT10_TICKER: "CT10", CT2_TICKER: "CT2"})
        df["2s10s"] = (df["CT10"] - df["CT2"]) * 100.0  # % -> bp
        keep = list(BONDS) + ["CT10", "2s10s"] + list(CC_TICKERS)
        df = df[keep]
        print(f"Downloaded {len(df)} rows, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as exc:
        print(f"Bloomberg download failed ({exc})")
        return None


def load_vol() -> pd.Series | None:
    """1y10y ATM normal vol (bp) from the mbs_rv workbook."""
    try:
        df = pd.read_excel(MBS_RV_XLSX, sheet_name="Data1")
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        s = df.dropna(subset=["date"]).set_index("date")["1y10y_atm"].sort_index()
        print(f"1y10y vol loaded from workbook: {len(s)} obs, "
              f"{s.index[0].date()} to {s.index[-1].date()}")
        return s.rename("vol")
    except Exception as exc:
        print(f"1y10y vol load failed ({exc}); vol predictor will be skipped")
        return None


def load_data() -> pd.DataFrame:
    df = load_bbg()
    if df is None:
        if not CACHE.exists():
            raise RuntimeError("No Bloomberg data and no cache available")
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"Loaded cache: {len(df)} rows, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    vol = load_vol()
    if vol is not None:
        df["vol"] = vol.reindex(df.index).ffill()
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


def _solve(a: list, b: list) -> list:
    """Gaussian elimination with partial pivoting (pure python).

    np.linalg is unusable in this env (broken LAPACK DLL), and the system
    is tiny (<=5x5)."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            raise ValueError("singular design matrix")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def regress(y: pd.Series, X: pd.DataFrame) -> dict:
    """OLS y = a + sum b_i * X_i via normal equations (no numpy.linalg)."""
    cols = list(X.columns)
    vecs = [np.ones(len(y))] + [X[c].values.astype(float) for c in cols]
    k = len(vecs)
    xtx = [[float((vecs[i] * vecs[j]).sum()) for j in range(k)]
           for i in range(k)]
    xty = [float((vecs[i] * y.values).sum()) for i in range(k)]
    beta = _solve(xtx, xty)
    fit = np.zeros(len(y))
    for i in range(k):
        fit = fit + beta[i] * vecs[i]
    resid = y.values - fit
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y.values - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return {
        "a": beta[0],
        "b": dict(zip(cols, beta[1:])),
        "fit": pd.Series(fit, index=y.index),
        "resid": pd.Series(resid, index=y.index),
        "r2": r2,
    }


def analyze(df: pd.DataFrame, ticker: str, cc_col: str) -> dict:
    preds = ["CT10", "cc", "2s10s"] + (["vol"] if "vol" in df.columns else [])
    sub = df[[ticker] + [cc_col if p == "cc" else p for p in preds]].copy()
    sub = sub.rename(columns={ticker: "px", cc_col: "cc"}).dropna()
    reg = regress(sub["px"], sub[preds])
    resid = reg["resid"]
    cur = {
        "date": sub.index[-1],
        "px": sub["px"].iloc[-1],
        "ct10": sub["CT10"].iloc[-1],
        "fair": reg["fit"].iloc[-1],
        "resid": resid.iloc[-1],
        "resid_z": (resid.iloc[-1] - resid.mean()) / resid.std(),
        "resid_pct": float((resid < resid.iloc[-1]).mean()),
        "resid_std": resid.std(),
    }
    return {"data": sub, "reg": reg, "cur": cur, "preds": preds}


def _fmt_px(v: float) -> str:
    """Price as points+32nds string."""
    pts = int(v)
    ticks = round((v - pts) * 32)
    if ticks == 32:
        pts, ticks = pts + 1, 0
    return f"{pts}+{ticks:02d}/32"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_levels(r: dict, label: str, cc_name: str) -> go.Figure:
    d = r["data"]
    reg = r["reg"]
    has_vol = "vol" in d.columns
    row_heights = [0.5, 0.25, 0.25] if has_vol else [0.62, 0.38]
    specs = [[{}]] + [[{"secondary_y": True}]] * (len(row_heights) - 1)
    titles = ["Price vs model fit", f"CT10 (%) and {cc_name} (bp)"]
    if has_vol:
        titles.append("2s10s (bp) and 1y10y vol (bp)")
    fig = make_subplots(
        rows=len(row_heights), cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=row_heights, specs=specs,
        subplot_titles=tuple(titles),
    )
    fig.add_trace(go.Scatter(x=d.index, y=d["px"], name="Price",
                             line=dict(color=C_PX, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=reg["fit"], name="Model fit",
                             line=dict(color=C_FIT, width=1.4, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["CT10"], name="CT10 (%)",
                             line=dict(color=C_CT10, width=1.3)),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=d.index, y=d["cc"], name=f"{cc_name} (bp)",
                             line=dict(color=C_CC, width=1.3)),
                  row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CT10 (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"{cc_name} (bp)", row=2, col=1,
                     secondary_y=True)
    if has_vol:
        fig.add_trace(go.Scatter(x=d.index, y=d["2s10s"], name="2s10s (bp)",
                                 line=dict(color=C_2S10S, width=1.3)),
                      row=3, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=d.index, y=d["vol"], name="1y10y vol (bp)",
                                 line=dict(color=C_VOL, width=1.3)),
                      row=3, col=1, secondary_y=True)
        fig.update_yaxes(title_text="2s10s (bp)", row=3, col=1,
                         secondary_y=False)
        fig.update_yaxes(title_text="vol (bp)", row=3, col=1, secondary_y=True)
    fig.update_layout(height=780 if has_vol else 620, title=label,
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=60, r=60, t=90, b=40))
    return fig


def _fig_residual(r: dict, label: str) -> go.Figure:
    d = r["data"]
    reg = r["reg"]
    c = r["cur"]
    resid_ticks = reg["resid"] * 32.0
    std_ticks = c["resid_std"] * 32.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=resid_ticks, name="Residual",
                             line=dict(color=C_PX, width=1.3)))
    for k, col in ((1, "#bbbbbb"), (2, "#dddddd")):
        for sgn in (1, -1):
            fig.add_hline(y=sgn * k * std_ticks, line=dict(color=col, width=1,
                                                           dash="dot"))
    fig.add_hline(y=0, line=dict(color="#888888", width=1))
    fig.add_trace(go.Scatter(x=[c["date"]], y=[c["resid"] * 32],
                             mode="markers", name="Current",
                             marker=dict(color=C_CT10, size=10)))
    fig.update_layout(
        height=420,
        title=(f"{label} — residual (actual − model), current "
               f"{c['resid']*32:+.1f} ticks "
               f"(z {c['resid_z']:+.2f}, {c['resid_pct']*100:.0f} pct)"),
        yaxis_title="ticks (32nds)", hovermode="x unified",
        showlegend=False, margin=dict(l=60, r=40, t=70, b=40),
    )
    return fig


def _fig_scatter(r: dict, label: str, cc_name: str) -> go.Figure:
    d = r["data"]
    reg = r["reg"]
    c = r["cur"]
    xs = np.linspace(d["CT10"].min(), d["CT10"].max(), 50)
    others = sum(b * d[p].median() for p, b in reg["b"].items() if p != "CT10")
    ys = reg["a"] + reg["b"]["CT10"] * xs + others
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["CT10"], y=d["px"], mode="markers",
                             name="daily obs",
                             marker=dict(color=C_PX, size=4, opacity=0.35)))
    fig.add_trace(go.Scatter(x=xs, y=ys, name="fit @ median other predictors",
                             line=dict(color=C_CT10, width=1.6)))
    fig.add_trace(go.Scatter(x=[c["ct10"]], y=[c["px"]], mode="markers",
                             name="current",
                             marker=dict(color=C_CT10, size=11,
                                         symbol="diamond")))
    fig.update_layout(height=480,
                      title=(f"{label} — price vs CT10 "
                             f"(R² {reg['r2']:.3f})"),
                      xaxis_title="CT10 (%)", yaxis_title="Price",
                      legend=dict(orientation="h", y=1.10),
                      margin=dict(l=60, r=40, t=70, b=40))
    return fig


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_CSS = """
html, body { background: #ffffff; }
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 24px auto;
       max-width: 1200px; color: #222; padding: 0 12px; }
h1 { font-size: 22px; margin-bottom: 4px; color: #1a1a1a; }
h2 { font-size: 17px; margin: 28px 0 8px; border-bottom: 2px solid #1f4e79;
     padding-bottom: 4px; color: #1a1a1a; }
.subtitle { color: #666; font-size: 13px; margin-bottom: 16px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: center; }
th { background: #1f4e79; color: #fff; font-weight: 600; }
td { background: #fff; color: #222; }
td.rowlabel { background: #eef2f8; font-weight: 600; text-align: left; }
.cheap { color: #1f7a33; font-weight: 700; }
.rich { color: #c00000; font-weight: 700; }
.note { font-size: 12px; color: #666; margin-top: 6px; }
.eq { font-family: Consolas, monospace; font-size: 13px; background: #f6f8fb;
      color: #222; border: 1px solid #dde3ec; padding: 10px 14px;
      margin: 8px 0; white-space: pre-wrap; }
.chart { margin: 18px 0; }
"""


def _summary_table(results: dict) -> str:
    head = ("<tr><th>Bond</th><th>Date</th><th>Price</th><th>Model FV</th>"
            "<th>Residual</th><th>Resid z</th><th>Resid pct</th>"
            "<th>R²</th><th>Read</th></tr>")
    rows = []
    for tk, r in results.items():
        c = r["cur"]
        cheap = c["resid"] < 0
        read = "CHEAP" if cheap else "RICH"
        rows.append(
            f"<tr><td class='rowlabel'>{html.escape(BONDS[tk]['label'])}</td>"
            f"<td>{c['date'].date()}</td>"
            f"<td>{c['px']:.3f} ({_fmt_px(c['px'])})</td>"
            f"<td>{c['fair']:.3f} ({_fmt_px(c['fair'])})</td>"
            f"<td>{c['resid']*32:+.1f} ticks</td>"
            f"<td>{c['resid_z']:+.2f}</td>"
            f"<td>{c['resid_pct']*100:.0f}%</td>"
            f"<td>{r['reg']['r2']:.3f}</td>"
            f"<td class='{'cheap' if cheap else 'rich'}'>{read}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _scenario_rows(results: dict, scen: dict) -> str:
    preds = [p for p in ("CT10", "cc", "2s10s", "vol")]
    shocks = scen["shocks"]
    head = ["<tr><th>Bond</th>"]
    for p in preds:
        name, unit = PRED_META[p]
        s = shocks.get(p, 0.0)
        head.append(f"<th>{html.escape(name)} {s:+.4g}{unit}</th>")
    head += ["<th>Total Δpx</th>", "<th>New price</th></tr>"]
    rows = []
    for tk, r in results.items():
        reg = r["reg"]
        c = r["cur"]
        cells = []
        total = 0.0
        for p in preds:
            contrib_pts = reg["b"].get(p, 0.0) * shocks.get(p, 0.0)
            total += contrib_pts
            cells.append(f"<td>{contrib_pts*32:+.1f}</td>")
        new_px = c["px"] + total
        rows.append(
            f"<tr><td class='rowlabel'>{html.escape(BONDS[tk]['label'])}</td>"
            + "".join(cells)
            + f"<td><b>{total*32:+.1f} ticks</b></td>"
            f"<td>{new_px:.3f} ({_fmt_px(new_px)})</td></tr>"
        )
    return f"<table>{''.join(head)}{''.join(rows)}</table>"


def _scenario_section(results: dict) -> str:
    parts = ["<h2>Scenario analysis (model-implied price impact)</h2>"]
    for scen in SCENARIOS:
        parts.append(f"<h3 style='font-size:14px;margin:14px 0 4px;'>"
                     f"{html.escape(scen['name'])}</h3>")
        parts.append(_scenario_rows(results, scen))
    parts.append("<div class='note'>Factor contributions in ticks "
                 "(32nds) = regression coefficient × shock; other factors "
                 "held unchanged. Linear full-sample model — first-order "
                 "estimate only.</div>")
    return "".join(parts)


def _regression_block(results: dict) -> str:
    blocks = []
    for tk, r in results.items():
        reg = r["reg"]
        d = r["data"]
        terms = []
        for p, b in reg["b"].items():
            name, unit = PRED_META[p]
            if p == "cc":
                name = BONDS[tk]["cc"]
            terms.append(f"{b:+.5f} × {name}({unit})")
        blocks.append(
            f"<div class='eq'><b>{html.escape(BONDS[tk]['label'])}</b>"
            f"  ({html.escape(tk)})\n"
            f"px = {reg['a']:.3f}  {'  '.join(terms)}\n"
            f"n = {len(d)}, R² = {reg['r2']:.3f}, "
            f"sample {d.index[0].date()} → {d.index[-1].date()}</div>"
        )
    return "".join(blocks)


def build_html(results: dict, out_path: Path) -> None:
    last_date = max(r["cur"]["date"] for r in results.values())
    pred_desc = " + ".join(m[0] for m in PRED_META.values())
    parts = [
        "<h1>CMO Valuation vs CT10, CC Basis, 2s10s and 1y10y Vol</h1>",
        f"<div class='subtitle'>Data through {last_date.date()} · "
        "Bloomberg PX_LAST, USGG10YR/USGG2YR, .FNCC105/.G2CC105 G Index, "
        "1y10y ATM vol from mbs_rv.xlsx · "
        f"full-sample OLS: price ~ {pred_desc}</div>",
        "<h2>Current valuation</h2>",
        _summary_table(results),
        "<div class='note'>Negative residual = price below model "
        "(cheap); positive = rich. Residual pct = percentile of the current "
        "residual within its own history since inception.</div>",
        _scenario_section(results),
        "<h2>Regression (full sample, daily obs)</h2>",
        _regression_block(results),
    ]
    for tk, r in results.items():
        label = BONDS[tk]["label"]
        cc_name = BONDS[tk]["cc"]
        parts.append(f"<h2>{html.escape(label)}</h2>")
        for fig in (_fig_levels(r, label, cc_name),
                    _fig_residual(r, label),
                    _fig_scatter(r, label, cc_name)):
            div = pio.to_html(fig, full_html=False, include_plotlyjs=False,
                              config={"displaylogo": False})
            parts.append(f"<div class='chart'>{div}</div>")

    document = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>CMO Valuation vs CT10 / CC Basis / 2s10s / Vol</title>"
        f"<style>{_CSS}</style>"
        f"<script>{get_plotlyjs()}</script>"
        "</head><body>" + "".join(parts) + "</body></html>"
    )
    out_path.write_text(document, encoding="utf-8")
    print(f"\nHTML written: {out_path}")


def main() -> None:
    df = load_data()
    results = {}
    for tk, meta in BONDS.items():
        r = analyze(df, tk, meta["cc"])
        results[tk] = r
        c = r["cur"]
        reg = r["reg"]
        terms = "  ".join(f"{p} {b:+.5f}" for p, b in reg["b"].items())
        print(f"\n=== {meta['label']} ({tk}) ===")
        print(f"sample: {r['data'].index[0].date()} to "
              f"{r['data'].index[-1].date()} ({len(r['data'])} obs)")
        print(f"px = {reg['a']:.3f} | {terms}   R2={reg['r2']:.3f}")
        print(f"current ({c['date'].date()}): px {c['px']:.3f} "
              f"({_fmt_px(c['px'])}), model {c['fair']:.3f}, "
              f"resid {c['resid']*32:+.1f} ticks "
              f"(z {c['resid_z']:+.2f}, {c['resid_pct']*100:.0f} pct) "
              f"-> {'RICH' if c['resid'] > 0 else 'CHEAP'} vs model")
    OUTPUT_DIR.mkdir(exist_ok=True)
    build_html(results, OUTPUT_DIR / "adhoc_cmo_valuation.html")


if __name__ == "__main__":
    main()
