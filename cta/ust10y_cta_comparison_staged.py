"""Compare Treasury CTA signals across tenors, with an interactive 2026 close-up.

Uses cached Bloomberg yield histories for GT02/GT05/GT10/GT30 Govt and the same
rates-mode backtester, fixed $1,000 DV01, and zero transaction costs as the
existing momentum and trend-filter reports.  A tenor toggle on the report page
switches all content between 2Y, 5Y, 10Y and 30Y; a separate toggle switches
the summary tables between risk-rule versions.

Run from the repository root:
    python strategy_example/ust10y_cta_comparison.py            # use cached yields
    python strategy_example/ust10y_cta_comparison.py --update   # refresh all tenor caches
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR.parent), str(SCRIPT_DIR)]

from ust10y_yield_momentum import (  # noqa: E402
    DV01_USD, FIELD, OUTPUT_DIR, START_DATE, Scenario, _normalize_bdh,
    _performance_metrics, build_signals, run_scenario,
)


REPORT_PATH = OUTPUT_DIR / "ust10y_cta_strategy_comparison.html"
SUMMARY_PATH = OUTPUT_DIR / "cta_strategy_comparison.csv"
DAILY_PATH = OUTPUT_DIR / "cta_strategy_daily_2026.csv"

TENORS = {
    "2Y": ("GT02 Govt", "gt02_yield.csv"),
    "5Y": ("GT05 Govt", "gt05_yield.csv"),
    "10Y": ("GT10 Govt", "gt10_yield.csv"),
    "30Y": ("GT30 Govt", "gt30_yield.csv"),
}
TENOR_ORDER = ["2Y", "5Y", "10Y", "30Y"]
DEFAULT_TENOR = "10Y"

CHANNEL_ENTRY = 55
CHANNEL_EXIT = 20
SLOPE_WINDOW = 60
SLOPE_T_THRESHOLD = 2.0
VOL_WINDOWS = (20, 60, 120)
VOL_ESTIMATION = 60
VOL_SCORE_THRESHOLD = 0.75
VOL_LEVEL_WINDOW = 252


def load_tenor_yields(tenor: str, update: bool = False) -> pd.DataFrame:
    """Load one tenor's yield history, refreshing its cache when ``update``."""
    ticker, cache_name = TENORS[tenor]
    cache = OUTPUT_DIR / cache_name
    if cache.exists() and not update:
        data = pd.read_csv(cache, index_col="date", parse_dates=True)
        return data.sort_index().dropna(subset=["yield"])

    from xbbg import blp

    print(f"Downloading {ticker} from Bloomberg ({START_DATE} to today)...")
    raw = blp.bdh(
        tickers=ticker,
        flds=FIELD,
        start_date=pd.Timestamp(START_DATE).strftime("%Y%m%d"),
        end_date=datetime.now().strftime("%Y%m%d"),
    )
    data = _normalize_bdh(raw).to_frame()
    data.index.name = "date"
    data.to_csv(cache)
    print(f"Saved {len(data):,} observations to {cache}")
    return data


def _channel_signal(yield_percent: pd.Series) -> pd.Series:
    """55-day yield breakout, held until the opposite 20-day exit channel."""
    high_entry = yield_percent.shift(1).rolling(CHANNEL_ENTRY).max()
    low_entry = yield_percent.shift(1).rolling(CHANNEL_ENTRY).min()
    high_exit = yield_percent.shift(1).rolling(CHANNEL_EXIT).max()
    low_exit = yield_percent.shift(1).rolling(CHANNEL_EXIT).min()
    signals = np.zeros(len(yield_percent), dtype=np.int8)
    state = 0
    for i, value in enumerate(yield_percent.to_numpy(dtype=float)):
        if pd.isna(high_entry.iat[i]) or pd.isna(low_entry.iat[i]):
            continue
        # Short duration on rising yields; long duration on falling yields.
        if value >= high_entry.iat[i]:
            state = -1
        elif value <= low_entry.iat[i]:
            state = 1
        elif state == -1 and value <= low_exit.iat[i]:
            state = 0
        elif state == 1 and value >= high_exit.iat[i]:
            state = 0
        signals[i] = state
    return pd.Series(signals, index=yield_percent.index, name="signal_channel")


def _slope_t_score(window_values: np.ndarray) -> float:
    """Signed OLS slope divided by its residual standard error."""
    n = len(window_values)
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    y = window_values * 100.0  # yield percent -> basis points
    y_centered = y - y.mean()
    xx = float(np.dot(x_centered, x_centered))
    slope = float(np.dot(x_centered, y_centered) / xx)
    residual = y_centered - slope * x_centered
    residual_var = float(np.dot(residual, residual) / (n - 2))
    if residual_var <= 0:
        return float(np.sign(slope) * 1e9) if slope != 0 else 0.0
    return slope / np.sqrt(residual_var / xx)


def build_new_signals(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    yields = out["yield"]
    out["signal_channel"] = _channel_signal(yields)

    out["slope_t_60d"] = yields.rolling(SLOPE_WINDOW).apply(
        _slope_t_score, raw=True
    )
    out["signal_slope"] = np.select(
        [out["slope_t_60d"] >= SLOPE_T_THRESHOLD,
         out["slope_t_60d"] <= -SLOPE_T_THRESHOLD],
        [-1, 1], default=0,
    ).astype(int)

    daily_yield_move_bps = yields.diff() * 100.0
    daily_vol = daily_yield_move_bps.rolling(VOL_ESTIMATION).std(ddof=1)
    out["daily_yield_vol_bps"] = daily_vol
    votes = []
    for horizon in VOL_WINDOWS:
        score = out[f"trend_{horizon}d_bps"] / (daily_vol * np.sqrt(horizon))
        score = score.replace([np.inf, -np.inf], np.nan)
        out[f"vol_score_{horizon}d"] = score
        vote = np.select(
            [score >= VOL_SCORE_THRESHOLD, score <= -VOL_SCORE_THRESHOLD],
            [-1, 1], default=0,
        ).astype(int)
        out[f"vol_vote_{horizon}d"] = vote
        votes.append(vote)
    vote_sum = sum(votes)
    out["signal_vol_momentum"] = np.select(
        [vote_sum >= 2, vote_sum <= -2], [1, -1], default=0
    ).astype(int)

    # High-vol-only variant: same votes, but flat unless the current 60d
    # vol sits above its own trailing 1-year median.
    vol_median = daily_vol.rolling(VOL_LEVEL_WINDOW).median()
    high_vol = daily_vol > vol_median
    out["vol_level_median"] = vol_median
    out["signal_vol_momentum_hv"] = out["signal_vol_momentum"].where(
        high_vol, 0
    ).astype(int)
    return out


SCENARIOS = [
    Scenario("20d momentum", "signal_20d"),
    Scenario("60d momentum", "signal_60d"),
    Scenario("120d momentum", "signal_120d"),
    Scenario("Multi-horizon momentum", "signal_multi"),
    Scenario("EWMA 10d / 2bp", "signal_ewma_h10_thr2"),
    Scenario("EWMA 20d / 5bp", "signal_ewma_h20_thr5"),
    Scenario("EWMA 40d / 8bp", "signal_ewma_h40_thr8"),
    Scenario("KAMA / 2bp", "signal_kama_plain"),
    Scenario("KAMA + efficiency filter", "signal_kama_er"),
    Scenario("55d channel breakout", "signal_channel"),
    Scenario("60d trend strength", "signal_slope"),
    Scenario("Vol-adjusted momentum", "signal_vol_momentum"),
    Scenario("Vol-adj momentum (high vol only)", "signal_vol_momentum_hv"),
]

FAMILY = {
    "20d momentum": "Momentum", "60d momentum": "Momentum",
    "120d momentum": "Momentum", "Multi-horizon momentum": "Momentum",
    "EWMA 10d / 2bp": "EWMA", "EWMA 20d / 5bp": "EWMA",
    "EWMA 40d / 8bp": "EWMA", "KAMA / 2bp": "KAMA",
    "KAMA + efficiency filter": "KAMA",
    "55d channel breakout": "Channel breakout",
    "60d trend strength": "Trend strength",
    "Vol-adjusted momentum": "Volatility adjusted",
    "Vol-adj momentum (high vol only)": "Volatility adjusted",
}

# Risk-rule versions of every strategy.  The non-Base versions require a fresh
# signal after a risk exit and freeze entries for 5 days after a stop-loss;
# otherwise the engine re-enters at the same close and the rules do nothing.
RISK_VERSIONS = {
    "Base": {},
    "10/20/30d": dict(stop_loss=10, take_profit=20, max_holding_days=30),
    "20/30/60d": dict(stop_loss=20, take_profit=30, max_holding_days=60),
}
RISK_REENTRY = dict(fresh_signal=True, freeze_days=5)

_SIGNAL_BADGE = {
    1: ("LONG", "sig-long"),
    -1: ("SHORT", "sig-short"),
    0: ("FLAT", "sig-flat"),
}


def _current_signals_table(data: pd.DataFrame) -> str:
    """One row per strategy: latest signal value and its trailing run length."""
    latest = data.index.max().date()
    rows = []
    for spec in SCENARIOS:
        signal = data[spec.signal_col]
        current = int(signal.iloc[-1])
        run_days = 0
        for value in signal.to_numpy()[::-1]:
            if int(value) != current:
                break
            run_days += 1
        label, css = _SIGNAL_BADGE[current]
        rows.append(
            f"<tr><td>{html.escape(spec.name)}</td>"
            f"<td>{html.escape(FAMILY[spec.name])}</td>"
            f'<td class="{css}">{label}</td><td>{run_days}</td></tr>'
        )
    return (
        "<table><thead><tr><th>Strategy</th><th>Family</th>"
        f"<th>Signal as of {latest}</th><th>Consecutive days in state</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _display_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].copy()
    for column in ("net_pnl_usd", "max_drawdown_usd"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"${value:,.0f}")
    for column in ("net_pnl_bps", "daily_sharpe", "time_in_market_pct"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:,.2f}")
    return shown.to_html(index=False, classes="result-table", escape=True)


def _daily_detail_figure(data: pd.DataFrame, daily: pd.DataFrame, signal_col: str,
                         label: str, tenor: str, year: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    start = pd.Timestamp(f"{year}-01-01")
    display = data.loc[data.index >= start]
    subset = daily.loc[daily.index >= start]
    cumulative_usd = subset["net_pnl_usd"].cumsum()
    drawdown_usd = (cumulative_usd - cumulative_usd.cummax().clip(lower=0))
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=[0.29, 0.20, 0.30, 0.21],
        subplot_titles=(f"{tenor} yield", "Signal and position",
                        "YTD cumulative P&L", "YTD drawdown"),
    )
    fig.add_trace(go.Scatter(x=display.index, y=display["yield"], name=f"{tenor} yield",
                             line=dict(color="#172554")), row=1, col=1)
    fig.add_trace(go.Scatter(x=display.index, y=display[signal_col], name="Signal",
                             line=dict(color="#2563eb", shape="hv")), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset.index, y=subset["position"], name="Held position",
                             line=dict(color="#dc2626", shape="hv")), row=2, col=1)
    fig.add_trace(go.Scatter(x=subset.index, y=cumulative_usd, name="Cumulative P&L",
                             line=dict(color="#047857", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=subset.index, y=drawdown_usd, name="Drawdown",
                             fill="tozeroy", line=dict(color="#7c3aed"),
                             fillcolor="rgba(124,58,237,.14)"), row=4, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=1)
    fig.update_yaxes(title_text="−1 / 0 / +1", range=[-1.25, 1.25], row=2, col=1)
    fig.update_yaxes(title_text="USD", tickformat="$,.0f", row=3, col=1)
    fig.update_yaxes(title_text="USD", tickformat="$,.0f", row=4, col=1)
    fig.update_layout(
        template="plotly_white", height=910, margin=dict(t=75, b=50, l=70, r=30),
        title=f"{label} — {tenor} — {year} behavior", hovermode="x unified",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def _overview_figure(results: dict, tenor: str, year: int):
    """YTD cumulative P&L of every strategy of one tenor on a shared zero base."""
    import plotly.graph_objects as go

    featured = {"60d momentum", "55d channel breakout",
                "60d trend strength", "Vol-adjusted momentum"}
    overview = go.Figure()
    for spec in SCENARIOS:
        daily = results[(tenor, spec.name)][0]
        slice_ = daily.loc[daily.index >= pd.Timestamp(f"{year}-01-01")]
        overview.add_trace(go.Scatter(
            x=slice_.index, y=slice_["net_pnl_usd"].cumsum(), mode="lines",
            name=spec.name,
            line=dict(width=2 if spec.name in featured else 1.2),
            visible=True if spec.name in {"20d momentum"} | featured else "legendonly",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>%{fullData.name}</extra>",
        ))
    overview.update_layout(
        template="plotly_white", height=560, hovermode="x unified",
        title=f"{tenor} — {year} cumulative P&L by strategy ($1,000 DV01)",
        xaxis_title="Date", yaxis_title="USD", yaxis_tickformat="$,.0f",
        legend=dict(orientation="h", y=-0.22), margin=dict(b=160),
    )
    return overview


def build_report(data_by_tenor: dict[str, pd.DataFrame], summary: pd.DataFrame,
                 results: dict, year: int) -> Path:
    import plotly.io as pio

    latest = max(data.index.max() for data in data_by_tenor.values()).date()
    table_columns = ["scenario", "family", "net_pnl_bps", "net_pnl_usd",
                     "max_drawdown_usd", "daily_sharpe", "trades", "time_in_market_pct"]

    def _period_tables(period: str, sort_col: str) -> str:
        divs = []
        for tenor in TENOR_ORDER:
            for version in RISK_VERSIONS:
                subset = summary[
                    (summary.period == period)
                    & (summary.tenor == tenor)
                    & (summary.version == version)
                ].sort_values(sort_col, ascending=False)
                hidden = (
                    "" if (tenor == DEFAULT_TENOR and version == "Base")
                    else ' style="display:none"'
                )
                divs.append(
                    f'<div class="version-table" data-tenor="{tenor}" '
                    f'data-version="{html.escape(version)}"{hidden}>'
                    f'<div class="table-wrap">{_display_table(subset, table_columns)}</div></div>'
                )
        return "".join(divs)

    ytd_tables = _period_tables(f"{year} YTD", "net_pnl_usd")
    full_tables = _period_tables("Full sample", "daily_sharpe")

    signals_divs = ""
    for tenor in TENOR_ORDER:
        hidden = "" if tenor == DEFAULT_TENOR else ' style="display:none"'
        signals_divs += (
            f'<div class="tenor-signals" data-tenor="{tenor}"{hidden}>'
            f'<div class="table-wrap">{_current_signals_table(data_by_tenor[tenor])}</div></div>'
        )

    overview_payloads = {}
    for tenor in TENOR_ORDER:
        overview_payloads[tenor] = pio.to_json(
            _overview_figure(results, tenor, year)
        ).replace("</", "<\\/")

    figure_payload = {}
    for tenor in TENOR_ORDER:
        data = data_by_tenor[tenor]
        for index, spec in enumerate(SCENARIOS):
            daily, _ = results[(tenor, spec.name)]
            figure_payload[f"detail-{tenor}-{index}"] = pio.to_json(
                _daily_detail_figure(data, daily, spec.signal_col, spec.name, tenor, year)
            ).replace("</", "<\\/")

    details = "".join(
        f'<option value="{i}">{html.escape(spec.name)}</option>'
        for i, spec in enumerate(SCENARIOS)
    )
    payload_tags = "".join(
        f'<script type="application/json" class="figure-json" data-key="{key}">{raw}</script>'
        for key, raw in figure_payload.items()
    )
    overview_tags = "".join(
        f'<script type="application/json" class="overview-json" data-tenor="{tenor}">{raw}</script>'
        for tenor, raw in overview_payloads.items()
    )
    script = f"""
const payloads = {{}};
document.querySelectorAll('.figure-json').forEach(node => {{ payloads[node.dataset.key] = node.textContent; }});
const overviewPayloads = {{}};
document.querySelectorAll('.overview-json').forEach(node => {{ overviewPayloads[node.dataset.tenor] = node.textContent; }});
const picker = document.getElementById('strategy-picker');
const chart = document.getElementById('strategy-detail');
let currentTenor = '{DEFAULT_TENOR}';
let currentVersion = 'Base';
function drawSelected() {{
  const fig = JSON.parse(payloads['detail-' + currentTenor + '-' + picker.value]);
  Plotly.react(chart, fig.data, fig.layout, {{responsive: true, displaylogo: false}});
}}
function drawOverview() {{
  const fig = JSON.parse(overviewPayloads[currentTenor]);
  Plotly.react('overview-chart', fig.data, fig.layout, {{responsive: true, displaylogo: false}});
}}
function refreshTables() {{
  const walkforward = document.getElementById('walkforward-10y');
  if (walkforward) walkforward.style.display = currentTenor === '10Y' ? '' : 'none';
  document.querySelectorAll('.version-table').forEach(div => {{
    div.style.display = (div.dataset.tenor === currentTenor && div.dataset.version === currentVersion) ? '' : 'none';
  }});
  document.querySelectorAll('.tenor-signals').forEach(div => {{
    div.style.display = div.dataset.tenor === currentTenor ? '' : 'none';
  }});
}}
picker.addEventListener('change', drawSelected);
document.querySelectorAll('.tt-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    currentTenor = btn.dataset.tenor;
    document.querySelectorAll('.tt-btn').forEach(b => b.classList.toggle('active', b === btn));
    refreshTables();
    drawOverview();
    drawSelected();
  }});
}});
document.querySelectorAll('.vt-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    currentVersion = btn.dataset.version;
    document.querySelectorAll('.vt-btn').forEach(b => b.classList.toggle('active', b === btn));
    refreshTables();
  }});
}});
drawSelected();
"""
    script = script.replace("</", "<\\/")
    default_overview = _overview_figure(results, DEFAULT_TENOR, year)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    tenor_spans = " · ".join(
        f"{tenor} {data_by_tenor[tenor].index.min().date()}–{data_by_tenor[tenor].index.max().date()}"
        for tenor in TENOR_ORDER
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Treasury CTA Comparison</title>
<style>
:root{{--bg:#eef3f7;--ink:#17212c;--muted:#536574;--navy:#12344d;--line:#dbe4ec}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Arial,sans-serif}}
main{{max-width:1500px;margin:auto;padding:24px}}section{{background:white;border:1px solid var(--line);border-radius:10px;margin:0 0 18px;padding:20px;box-shadow:0 2px 8px #17212c0b}}
h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 14px}}p{{margin:8px 0}}.subtle{{color:var(--muted)}}.table-wrap{{overflow:auto}}
table{{border-collapse:collapse;min-width:950px;width:100%;font-size:12px}}th{{background:var(--navy);color:white;padding:8px;text-align:right;position:sticky;top:0}}
th:first-child,th:nth-child(2),td:first-child,td:nth-child(2){{text-align:left}}td{{border-bottom:1px solid var(--line);padding:7px;text-align:right;white-space:nowrap}}
tbody tr:nth-child(even){{background:#f5f8fa}}select{{padding:8px 12px;font-size:14px;border:1px solid #aebfcb;border-radius:6px;min-width:330px}}
.method{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}}.method div{{background:#f5f8fb;border:1px solid var(--line);border-radius:8px;padding:12px}}
.vt-btn,.tt-btn{{padding:7px 16px;margin-right:6px;font-size:13px;border:1px solid #aebfcb;border-radius:6px;background:#f5f8fb;color:var(--ink);cursor:pointer}}
.vt-btn.active,.tt-btn.active{{background:var(--navy);color:white;border-color:var(--navy)}}
.sig-long{{color:#15803d;font-weight:bold}}.sig-short{{color:#b91c1c;font-weight:bold}}.sig-flat{{color:#6b7280;font-weight:bold}}
@media(max-width:700px){{main{{padding:10px}}section{{padding:12px}}}}
</style></head><body><main>
<section><h1>Treasury CTA strategy comparison</h1><p class="subtle">Bloomberg GT02/GT05/GT10/GT30 Govt · {year} through {latest} · generated {generated}</p>
<p class="subtle">Sample span per tenor: {tenor_spans}</p>
<p>All strategies use the same daily yield series per tenor, next-period position accounting, fixed $1,000 DV01, and zero transaction costs. Long duration earns when yields fall; short duration earns when yields rise. Dollar P&amp;L is the bp-equivalent result multiplied by $1,000. Signal parameters are identical across tenors (thresholds are in bps).</p></section>
<section><h2>Tenor</h2><p><button class="tt-btn" data-tenor="2Y">2Y</button><button class="tt-btn" data-tenor="5Y">5Y</button><button class="tt-btn active" data-tenor="10Y">10Y</button><button class="tt-btn" data-tenor="30Y">30Y</button></p>
<p class="subtle">Switches the original signals, comparison tables, and charts between tenors. The added walk-forward section trades 10Y only and appears when 10Y is selected. Works independently from the risk-rule toggle below.</p></section>
<section><h2>Current signals</h2><p class="subtle">Latest signal value per strategy and the length of the current run of consecutive days in that state, for the selected tenor. LONG = long duration (earns when yields fall); SHORT = short duration.</p>{signals_divs}</section>
<section><h2>Risk rules</h2><p><button class="vt-btn active" data-version="Base">Base</button><button class="vt-btn" data-version="10/20/30d">10/20/30d</button><button class="vt-btn" data-version="20/30/60d">20/30/60d</button></p>
<p class="subtle">Switches both comparison tables between the raw signals (Base) and risk-managed versions (stop-loss / take-profit in bps / max holding days; risk exits require a fresh signal and freeze re-entry for 5 days after a stop). Charts always display the Base versions.</p></section>
<section><h2>{year} comparison</h2>{ytd_tables}</section>
<section><h2>{year} cumulative P&amp;L</h2><p class="subtle">Click legend names to show or hide curves. Every line is reset to zero at the previous year-end; the first point includes that day's P&amp;L. The chart follows the tenor toggle.</p>{pio.to_html(default_overview, include_plotlyjs='inline', full_html=False, div_id='overview-chart', config={'responsive':True,'displaylogo':False})}{overview_tags}</section>
<section><h2>Zoom into one strategy</h2><p><label for="strategy-picker">Strategy: </label><select id="strategy-picker">{details}</select> <span class="subtle">(chart follows the tenor toggle)</span></p><div id="strategy-detail"></div>{payload_tags}<script>{script}</script></section>
<section><h2>Full sample comparison</h2>{full_tables}</section>
<section><h2>Signal definitions (tenor-agnostic)</h2>
<p>All signals map the yield (y, in percent) of the selected tenor to a position in {{-1, 0, +1}} every day: <strong>+1 = long duration</strong> (earns when yields fall), <strong>−1 = short duration</strong> (earns when yields rise), 0 = flat. The backtester holds the position until the signal changes. The same parameters apply to every tenor; thresholds are in basis points.</p>
<div class="method">
<div><strong>20d momentum</strong><p>Raw price momentum over 20 business days with a 5bp dead-band.<br><code>d = (y_t − y_(t−20)) × 100</code> (bps)<br><code>d ≥ +5 → −1 (short); d ≤ −5 → +1 (long); else 0</code></p></div>
<div><strong>60d momentum</strong><p>Same rule on a 60-day change with a 10bp dead-band.<br><code>d = (y_t − y_(t−60)) × 100</code><br><code>d ≥ +10 → −1; d ≤ −10 → +1; else 0</code></p></div>
<div><strong>120d momentum</strong><p>Same rule on a 120-day change with a 15bp dead-band.<br><code>d = (y_t − y_(t−120)) × 100</code><br><code>d ≥ +15 → −1; d ≤ −15 → +1; else 0</code></p></div>
<div><strong>Multi-horizon momentum</strong><p>Majority vote of the three momentum signals above.<br><code>votes = signal_20d + signal_60d + signal_120d</code><br><code>votes ≥ +2 → +1; votes ≤ −2 → −1; else 0</code></p></div>
<div><strong>EWMA 10d / 2bp</strong><p>Distance of the yield from its exponentially weighted moving average, halflife 10 days.<br><code>ewma_t = λ·ewma_(t−1) + (1−λ)·y_t, λ from halflife 10d</code><br><code>d = (y_t − ewma_t) × 100; d ≥ +2 → −1; d ≤ −2 → +1; else 0</code></p></div>
<div><strong>EWMA 20d / 5bp</strong><p>Same EWMA-distance rule with halflife 20 days and a 5bp band.<br><code>d = (y_t − ewma_20d_t) × 100; d ≥ +5 → −1; d ≤ −5 → +1; else 0</code></p></div>
<div><strong>EWMA 40d / 8bp</strong><p>Same EWMA-distance rule with halflife 40 days and an 8bp band.<br><code>d = (y_t − ewma_40d_t) × 100; d ≥ +8 → −1; d ≤ −8 → +1; else 0</code></p></div>
<div><strong>KAMA / 2bp</strong><p>Distance from Kaufman's adaptive moving average (ER window 10, fast 2 / slow 30).<br><code>ER_t = |y_t − y_(t−10)| / Σ|Δy| over 10 days</code><br><code>SC_t = (ER_t·(2/3 − 2/31) + 2/31)²</code><br><code>KAMA_t = KAMA_(t−1) + SC_t·(y_t − KAMA_(t−1))</code><br><code>d = (y_t − KAMA_t) × 100; d ≥ +2 → −1; d ≤ −2 → +1; else 0</code></p></div>
<div><strong>KAMA + efficiency filter</strong><p>Same KAMA / 2bp signal, gated by the efficiency ratio — forced flat when the market is not sufficiently directional.<br><code>if ER_t ≤ 0.25 → 0 (flat), else the KAMA / 2bp signal</code></p></div>
<div><strong>55d channel breakout</strong><p>Donchian-style breakout with separate entry and exit channels; a state machine holds the position between triggers.<br><code>enter short (−1) when y_t ≥ max(y) of the prior 55 days; enter long (+1) when y_t ≤ min(y) of the prior 55 days</code><br><code>exit to 0 when y crosses the opposite 20-day channel (short exits at the 20-day low, long exits at the 20-day high)</code></p></div>
<div><strong>60d trend strength</strong><p>Statistical significance of the 60-day linear trend. Regress yield (bps) on time over a 60-day window; the score is the OLS slope divided by its residual standard error (a t-statistic).<br><code>t = slope / SE(slope) over 60d</code><br><code>t ≥ +2 → −1 (rising yields, short); t ≤ −2 → +1; else 0</code></p></div>
<div><strong>Vol-adjusted momentum</strong><p>Momentum measured in vol units instead of raw bps, voted across 20/60/120 days.<br><code>score_h = (y_t − y_(t−h)) × 100 / (σ_60d × √h)</code>, where <code>σ_60d</code> = 60-day std of daily yield changes (bps)<br><code>vote_h = −1 if score_h ≥ +0.75, +1 if ≤ −0.75, else 0</code><br><code>Σ votes ≥ +2 → +1; ≤ −2 → −1; else 0</code></p></div>
<div><strong>Vol-adj momentum (high vol only)</strong><p>Same vol-adjusted momentum votes, but gated by the vol level: trade only in elevated-volatility regimes, sit flat in quiet markets.<br><code>if σ_60d ≤ median(σ_60d, trailing 252d) → 0 (flat)</code><br><code>else the vol-adjusted momentum signal</code></p></div>
</div><p class="subtle">These are fixed-DV01 yield-move simulations. They do not include carry, financing, futures roll, convexity, or execution costs — at equal DV01, a 1bp yield move has the same dollar impact across tenors; realized yield volatility and signal activity determine differences in P&amp;L. Historical {year} results have been viewed during strategy design and should not be treated as a fresh out-of-sample test.</p></section>
</main></body></html>"""
    REPORT_PATH.write_text(page, encoding="utf-8")
    return REPORT_PATH


def main(update: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_by_tenor = {}
    for tenor in TENOR_ORDER:
        yields = load_tenor_yields(tenor, update=update)
        data = build_new_signals(build_signals(yields))
        data_by_tenor[tenor] = data.dropna(subset=["trend_120d_bps"]).copy()
    year = max(data.index.max().year for data in data_by_tenor.values())

    results = {}
    rows = []
    ytd_daily = {}
    exit_reasons = {version: {} for version in RISK_VERSIONS}
    for tenor in TENOR_ORDER:
        data = data_by_tenor[tenor]
        for version, risk in RISK_VERSIONS.items():
            counts = exit_reasons[version]
            for spec in SCENARIOS:
                print(f"Running [{tenor} / {version}] {spec.name}")
                if version == "Base":
                    run_spec = spec
                else:
                    run_spec = Scenario(spec.name, spec.signal_col, **risk, **RISK_REENTRY)
                _, daily, trades = run_scenario(data, run_spec)
                if not trades.empty:
                    for reason, count in trades["exit_reason"].value_counts().items():
                        counts[reason] = counts.get(reason, 0) + int(count)
                if version == "Base":
                    results[(tenor, spec.name)] = (daily, trades)
                    ytd_slice = daily.loc[daily.index >= pd.Timestamp(f"{year}-01-01")]
                    ytd_daily[f"{tenor} {spec.name} position"] = ytd_slice["position"]
                    ytd_daily[f"{tenor} {spec.name} pnl_usd"] = ytd_slice["net_pnl_usd"]
                for period, subset in (
                    ("Full sample", daily),
                    (f"{year} YTD", daily.loc[daily.index >= pd.Timestamp(f"{year}-01-01")]),
                ):
                    row = _performance_metrics(subset, trades, spec.name, period)
                    row["family"] = FAMILY[spec.name]
                    row["tenor"] = tenor
                    row["version"] = version
                    rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(ytd_daily).to_csv(DAILY_PATH, index_label="date")
    report = build_report(data_by_tenor, summary, results, year)
    # Keep the PDF-inspired extension in this report on every regeneration.
    from ust10y_walkforward import main as run_walkforward, merge_into_comparison
    walkforward_report = run_walkforward(update=False)
    merge_into_comparison(report, walkforward_report)

    ytd_summary = summary[(summary.period == f"{year} YTD") & (summary.version == "Base")]
    for tenor in TENOR_ORDER:
        print(f"\n{year} YTD - {tenor} - Base")
        print(ytd_summary.loc[ytd_summary.tenor == tenor,
                              ["scenario", "net_pnl_usd", "daily_sharpe", "max_drawdown_usd"]]
              .sort_values("net_pnl_usd", ascending=False).to_string(index=False))
    print("\nExit reasons per version (all tenors and strategies combined)")
    for version, counts in exit_reasons.items():
        print(f"  {version}: {counts}")
    print(f"\nSummary CSV: {SUMMARY_PATH}")
    print(f"Daily CSV: {DAILY_PATH}")
    print(f"Report: {report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Refresh all tenor yield caches from Bloomberg")
    args = parser.parse_args()
    main(update=args.update)
