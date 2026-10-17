"""Interactive HTML version of the clearing-basis report (euribor_basis_report.py).

Covers LCH/Eurex EURIBOR 6M, LCH/CME SOFR and LCH/JSCC TONAR basis (Bloomberg),
plus the Sheet5 EURIBOR 3S6S and SOFR/FedFunds adjacent-tenor spread tables.

Run:
    python euribor_basis_html_report.py
Output:
    ./euribor_basis_output/Basis_Report_YYYYMMDD.html
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import plotly.graph_objects as go

from euribor_basis_report import (
    BASIS_CONFIG,
    OUTPUT_DIR,
    TENORS,
    build_basis_summary,
    build_spread_summary,
    compute_adjacent_tenor_spreads,
    load_basis_data,
    load_sheet5_basis_data,
)
from html_report_utils import df_to_html_table, figs_to_html, page_template

TENOR_COLORS = {
    "2y": "#1f77b4", "3y": "#ff7f0e", "4y": "#2ca02c", "5y": "#d62728",
    "7y": "#9467bd", "10y": "#8c564b", "20y": "#e377c2", "30y": "#7f7f7f",
}


def _basis_history_fig(df, title: str) -> go.Figure:
    fig = go.Figure()
    for tenor in TENORS:
        if tenor not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[tenor], name=tenor, mode="lines",
            line=dict(width=1.6, color=TENOR_COLORS.get(tenor)),
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
    fig.update_layout(
        template="plotly_white", height=520, title=title,
        xaxis_title="Date", yaxis_title="Basis (bp)", hovermode="x unified",
        legend=dict(orientation="h", y=-0.18), margin=dict(b=100),
    )
    return fig


def generate_html_report() -> Path:
    sections = []
    latest_dates = []

    for name, config in BASIS_CONFIG.items():
        try:
            df = load_basis_data(config["tickers"], name)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue
        summary = build_basis_summary(df)
        if summary.empty:
            continue
        latest_dates.append(df.index[-1].date())
        table = df_to_html_table(
            summary, gradient_cols=["z_score", "pct_1y"], fmt="{:.2f}"
        )
        chart = figs_to_html([_basis_history_fig(df, f"{name} Basis (1-Year History)")])
        sections.append(
            f"<section><h2>{name} — summary (1y)</h2>"
            f'<p class="subtle">{config["description"]}</p>{table}</section>'
            f"<section><h2>{name} — history</h2>{chart}</section>"
        )

    try:
        sheet5 = load_sheet5_basis_data()
    except Exception as exc:
        print(f"Skipping Sheet5 spreads: {exc}")
        sheet5 = {}
    for name, df in sheet5.items():
        spreads = compute_adjacent_tenor_spreads(df)
        if spreads.empty:
            continue
        summary = build_spread_summary(spreads, windows=(90, 250))
        if summary.empty:
            continue
        latest_dates.append(df.index[-1].date())
        table = df_to_html_table(
            summary, gradient_cols=["z_90d", "z_250d"], fmt="{:.2f}"
        )
        sections.append(
            f"<section><h2>{name} — adjacent tenor spreads</h2>"
            f'<p class="subtle">Spread = higher tenor minus lower tenor (bp); '
            f"z-scores over 90d and 250d windows | latest {df.index[-1].date()}</p>"
            f"{table}</section>"
        )

    latest = max(latest_dates) if latest_dates else "n/a"
    subtitle = (
        f"Latest data {latest} · generated "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · source: Bloomberg + "
        "mbs_rv_citi.xlsx Sheet5"
    )
    page = page_template("Clearing Basis Report (LCH / Eurex / CME / JSCC)",
                         subtitle, "".join(sections))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUTPUT_DIR / f"Basis_Report_{date.today():%Y%m%d}.html"
    try:
        html_path.write_text(page, encoding="utf-8")
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = OUTPUT_DIR / f"Basis_Report_{date.today():%Y%m%d}_{stamp}.html"
        html_path.write_text(page, encoding="utf-8")
    print(f"HTML report saved to: {html_path}")
    return html_path


if __name__ == "__main__":
    generate_html_report()
