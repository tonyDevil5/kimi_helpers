r"""
DM Government Bond Performance Comparison - interactive HTML report
==================================================================
HTML counterpart of dm_govy_report.py. Reuses the analysis layer of
``dm_govy_report`` (data loading, summary tables, yield changes) and the
shared ``html_report_utils`` helpers, and renders the report as a single
self-contained HTML file with Plotly charts.

Run:
    python dm_govy_html_report.py

Output:
    ./dm_govy_output/DM_GOVY_Report_<YYYYMMDD>.html
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dm_govy_report import (
    COUNTRIES,
    OUTPUT_DIR,
    TENORS,
    _idx_date,
    build_quantile_summary_table,
    build_spread_summary_table,
    build_summary_table,
    compute_yield_changes,
    load_data_from_bloomberg,
    load_funding_rates,
)
from html_report_utils import df_to_html_table, figs_to_html, page_template

HORIZONS = ["1d", "1w", "1m", "3m", "6m", "1y"]

CURVE_COLORS = {
    "us": "#1f77b4",
    "frf": "#d62728",
    "dem": "#2ca02c",
    "esp": "#8c564b",
    "jpy": "#ff7f0e",
    "uk": "#9467bd",
}


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------
def fig_spread_history_1y(df: pd.DataFrame) -> go.Figure:
    """3 stacked subplots: last 252 rows of GT10 vs Bund/OAT/Bono spreads (bps)."""
    recent = df.tail(252)
    spreads = {
        "GT10 - Bund10": ((recent["us_10y"] - recent["dem_10y"]) * 100, "green"),
        "GT10 - OAT10Y": ((recent["us_10y"] - recent["frf_10y"]) * 100, "red"),
        "GT10 - Bono10Y": ((recent["us_10y"] - recent["esp_10y"]) * 100, "orange"),
    }

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=[f"{name} 1-Year History" for name in spreads],
    )
    for i, (name, (spread_bps, color)) in enumerate(spreads.items(), start=1):
        fig.add_trace(
            go.Scatter(
                x=spread_bps.index, y=spread_bps,
                mode="lines", name=name,
                line=dict(color=color, width=1.2),
            ),
            row=i, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=i, col=1)
        fig.update_yaxes(title_text="Spread (bps)", row=i, col=1)

    fig.update_layout(template="plotly_white", height=800, showlegend=False,
                      title_text="GT10 Spread History (1y)")
    return fig


def fig_yield_curves(df: pd.DataFrame) -> go.Figure:
    """Latest yield curve per country."""
    tenors_num = [2, 5, 7, 10, 20, 30]
    fig = go.Figure()
    for country in COUNTRIES:
        yields = [df[f"{country}_{t}"].iloc[-1] if f"{country}_{t}" in df.columns else None
                  for t in TENORS]
        fig.add_trace(
            go.Scatter(
                x=tenors_num, y=yields,
                mode="lines+markers", name=country.upper(),
                line=dict(color=CURVE_COLORS.get(country, "gray")),
            )
        )
    fig.update_layout(
        template="plotly_white", height=500,
        title_text="DM Government Yield Curves (Latest)",
        xaxis_title="Tenor (years)", yaxis_title="Yield (%)",
    )
    return fig


def fig_chg_heatmap(changes: pd.DataFrame, horizon: str = "1m") -> go.Figure:
    """Heatmap of yield changes (bps): rows = bonds, cols = last 22 business days."""
    data = changes[horizon].T.iloc[:, -22:]
    fig = go.Figure(
        go.Heatmap(
            z=data.values,
            x=[d.strftime("%Y-%m-%d") for d in data.columns],
            y=list(data.index),
            colorscale="RdYlGn_r",
            reversescale=False,
            hovertemplate="bond=%{y}<br>date=%{x}<br>chg=%{z:.1f} bps<extra></extra>",
            colorbar=dict(title="bps"),
        )
    )
    fig.update_layout(
        template="plotly_white", height=700,
        title_text=f"Yield Change Heatmap ({horizon}) - last 22 business days (bps)",
    )
    return fig


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate_html_report(
    df: pd.DataFrame,
    funding_df: Optional[pd.DataFrame] = None,
    html_path: Optional[Path] = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if html_path is None:
        report_date = date.today().strftime("%Y%m%d")
        html_path = OUTPUT_DIR / f"DM_GOVY_Report_{report_date}.html"
        if html_path.exists():
            try:
                with open(html_path, "ab"):
                    pass
            except PermissionError:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_path = OUTPUT_DIR / f"DM_GOVY_Report_{report_date}_{ts}.html"

    latest = _idx_date(df.index[-1])
    start = _idx_date(df.index[0])
    if funding_df is not None:
        rfr_text = "  |  ".join(
            f"{ccy.upper()} RFR: {funding_df[ccy].iloc[-1]:.3f}%" for ccy in funding_df.columns
        )
        subtitle = f"Sample: {start} to {latest} ({len(df)} observations)  |  {rfr_text}"
    else:
        subtitle = f"Sample: {start} to {latest} ({len(df)} observations)"

    sections = []

    # 1. Summary table
    summary = build_summary_table(df, funding_df=funding_df)
    gradient_cols = [f"chg_{h}" for h in HORIZONS] + ["rsi", "bb_pos", "rv20", "mom_score"]
    if funding_df is not None:
        gradient_cols += ["carry_bps", "carry_ratio"]
    sections.append(
        "<section><h2>DM Government Bonds summary</h2>"
        '<p class="subtle">Changes in bps  |  momentum score: + = yields falling (bullish), - = yields rising (bearish)</p>'
        + df_to_html_table(summary, gradient_cols=gradient_cols, badge_cols=("mom_dir",))
        + "</section>"
    )

    # 2. Spread summary table
    spread_summary = build_spread_summary_table(df)
    if not spread_summary.empty:
        sections.append(
            "<section><h2>GT10 spread analysis</h2>"
            '<p class="subtle">GT10 vs 10Y Bund / OAT / Bono  |  changes in bps</p>'
            + df_to_html_table(
                spread_summary,
                gradient_cols=[f"chg_{h}" for h in HORIZONS] + ["pct_1y"],
            )
            + "</section>"
        )

    # 3. Spread history chart
    sections.append(
        "<section><h2>GT10 spread history (1y)</h2>"
        + figs_to_html([fig_spread_history_1y(df)])
        + "</section>"
    )

    # 4. Yield curves
    sections.append(
        "<section><h2>Latest yield curves</h2>"
        + figs_to_html([fig_yield_curves(df)])
        + "</section>"
    )

    # 5. 60-day quantiles by country
    quantile_summaries = build_quantile_summary_table(df)
    q_parts = []
    for country, q_df in quantile_summaries.items():
        country_name = COUNTRIES.get(country, country)
        q_parts.append(f"<h3>{country_name}</h3>")
        q_parts.append(df_to_html_table(q_df.reindex(TENORS), gradient_cols=("range_pct",)))
    sections.append(
        "<section><h2>60-day yield quantiles</h2>"
        f'<p class="subtle">Window: last 60 observations  |  Latest: {latest}</p>'
        + "".join(q_parts)
        + "</section>"
    )

    # 6. 1-month change heatmap
    changes = compute_yield_changes(df)
    sections.append(
        "<section><h2>1-month yield-change heatmap</h2>"
        + figs_to_html([fig_chg_heatmap(changes, horizon="1m")])
        + "</section>"
    )

    page = page_template(
        "DM Government Bond Performance Comparison",
        subtitle,
        "".join(sections),
    )
    html_path.write_text(page, encoding="utf-8")
    return html_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data_from_bloomberg()
    funding_df = load_funding_rates()
    html_path = generate_html_report(df, funding_df=funding_df)

    print("\n" + "=" * 70)
    print("DM Government Bond Performance Comparison (HTML)")
    print("=" * 70)
    print(f"Latest date : {_idx_date(df.index[-1])}")
    print(f"Observations: {len(df)}")
    print(f"HTML report saved to: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
