r"""
UST 10Y Fair Value OLS Regression Report (interactive HTML)
============================================================
HTML counterpart of ust_fair_value_report.py. Reuses its analysis layer
(loaders, regressions, table builders) and renders an interactive,
self-contained HTML report with Plotly charts and PDF-style gradient tables.

Run:
    python ust_fair_value_html_report.py

Output:
    ./ust_fair_value_output/UST_FairValue_Report_<YYYYMMDD>.html
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from html_report_utils import df_to_html_table, figs_to_html, page_template
from ust_fair_value_report import (
    CURVE_PAIRS,
    FLY_TRIPLES,
    OUTPUT_DIR,
    RESPONSE_IDX,
    ROLLING_WINDOW,
    build_curve_summary_table,
    build_dat_final,
    build_fly_summary_table,
    build_merged_data,
    build_model_summary_dict,
    build_regression_params_table,
    build_regression_stats_table,
    load_bloomberg_data,
    load_fed_holdings_data,
    run_curve_ols,
    run_fly_ols,
    run_full_sample_regression,
    run_recent_rolling_fair_value,
    run_rolling_fair_value,
    _idx_date,
)

TEMPLATE = "plotly_white"
COLOR_ACTUAL = "red"
COLOR_FITTED = "blue"
COLOR_RESID = "purple"
OPEN_THRESHOLD = 1.25


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def fig_fitted_vs_actual(dat_final, fitted, summary) -> go.Figure:
    """Full-sample actual vs fitted 10Y UST yield."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dat_final.index, y=dat_final.iloc[:, RESPONSE_IDX],
        name="Actual 10Y Yield", line=dict(color=COLOR_ACTUAL), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=dat_final.index, y=fitted,
        name="Model Fitted", line=dict(color=COLOR_FITTED), opacity=0.8,
    ))
    fig.update_layout(
        template=TEMPLATE,
        height=480,
        title=(
            "Full-Sample Actual vs Model Fitted 10Y UST Yield"
            f" — mkt={summary['latest_actual']:.3f}, mdl={summary['latest_fitted']:.3f}, "
            f"resid={summary['latest_residual']:.3f}, z={summary['latest_resid_z']:.2f}"
        ),
        xaxis_title="Date",
        yaxis_title="Yield (%)",
    )
    return fig


def fig_rolling_mkt_mdl(rolling_df, recent: bool = False) -> go.Figure:
    """Rolling market vs model value chart."""
    latest = rolling_df.iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["mkt_val"],
        name="Market 10Y Yield", line=dict(color=COLOR_ACTUAL), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["mdl_val"],
        name="Model Fair Value", line=dict(color=COLOR_FITTED), opacity=0.8,
    ))
    title = (
        f"Latest: mkt={latest['mkt_val']:.3f}, mdl={latest['mdl_val']:.3f}, z={latest['resid_z']:.2f}"
        f" — {ROLLING_WINDOW}-day rolling OLS"
    )
    if recent:
        title = "Recent Rolling Market vs Model 10Y UST Yield — " + title
    else:
        title = "Rolling Market vs Model 10Y UST Yield — " + title
    fig.update_layout(
        template=TEMPLATE, height=480, title=title,
        xaxis_title="Date", yaxis_title="Yield (%)",
    )
    return fig


def fig_rolling_resid_z(rolling_df) -> go.Figure:
    """Rolling residual z-score with zero line and +/-1.25 thresholds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["resid_z"],
        name="Residual Z", line=dict(color=COLOR_RESID), opacity=0.85,
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_hline(
        y=OPEN_THRESHOLD, line=dict(color="red", width=1, dash="dash"),
        annotation_text="open threshold", annotation_position="top left",
    )
    fig.add_hline(y=-OPEN_THRESHOLD, line=dict(color="red", width=1, dash="dash"))
    fig.update_layout(
        template=TEMPLATE, height=450,
        title=f"Rolling Fair-Value Residual Z-Score ({ROLLING_WINDOW}-day rolling OLS)",
        xaxis_title="Date", yaxis_title="Residual / In-Sample Std",
    )
    return fig


def fig_recent_resid_bps(recent_df) -> go.Figure:
    """Recent rolling residual in bps."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_df.index, y=recent_df["resid_bps"],
        name="Residual (bps)", line=dict(color=COLOR_RESID), opacity=0.85,
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(
        template=TEMPLATE, height=450,
        title=f"Recent Rolling Fair-Value Residual ({ROLLING_WINDOW}-day rolling window)",
        xaxis_title="Date", yaxis_title="Residual (bps)",
    )
    return fig


def fig_curve_grid(merged_df, curve_summary) -> go.Figure:
    """2x3 subplot grid of actual vs fitted curve spreads (bps)."""
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[2] for p in CURVE_PAIRS])
    for i, (crv1, crv2, name) in enumerate(CURVE_PAIRS):
        row, col = divmod(i, 3)
        Y, model = run_curve_ols(merged_df, crv1, crv2)
        fig.add_trace(go.Scatter(
            x=Y.index, y=Y, name="Actual", legendgroup="actual",
            showlegend=(i == 0), line=dict(color=COLOR_ACTUAL), opacity=0.8,
        ), row=row + 1, col=col + 1)
        fig.add_trace(go.Scatter(
            x=Y.index, y=model.fittedvalues, name="Fitted", legendgroup="fitted",
            showlegend=(i == 0), line=dict(color=COLOR_FITTED), opacity=0.8,
        ), row=row + 1, col=col + 1)
        s = curve_summary.loc[name]
        fig.layout.annotations[i].text = (
            f"{name}  (diff={s['diff']:.1f}bp, ratio={s['ratio']:.2f})"
        )
    fig.update_layout(
        template=TEMPLATE, height=700,
        title="UST Curve Spreads: Actual vs Model Fitted (bps)",
    )
    return fig


def fig_fly_grid(merged_df, fly_summary) -> go.Figure:
    """1x3 subplot grid of actual vs fitted butterfly spreads (bps)."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t[3] for t in FLY_TRIPLES])
    for i, (short, belly, long, name) in enumerate(FLY_TRIPLES):
        Y, model = run_fly_ols(merged_df, short, belly, long)
        fig.add_trace(go.Scatter(
            x=Y.index, y=Y, name="Actual", legendgroup="actual",
            showlegend=(i == 0), line=dict(color=COLOR_ACTUAL), opacity=0.8,
        ), row=1, col=i + 1)
        fig.add_trace(go.Scatter(
            x=Y.index, y=model.fittedvalues, name="Fitted", legendgroup="fitted",
            showlegend=(i == 0), line=dict(color=COLOR_FITTED), opacity=0.8,
        ), row=1, col=i + 1)
        s = fly_summary.loc[name]
        fig.layout.annotations[i].text = (
            f"{name}  (diff={s['diff']:.1f}bp, ratio={s['ratio']:.2f})"
        )
    fig.update_layout(
        template=TEMPLATE, height=500,
        title="UST Butterfly Spreads: Actual vs Model Fitted (bps)",
    )
    return fig


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_html_report(dat_final, model, rolling_df, recent_df, merged_df):
    """Build all sections and write the self-contained HTML report."""
    summary = build_model_summary_dict(model, {})
    params_table = (
        build_regression_params_table(model)
        .set_index("parameter")
        .rename_axis("feature")
    )
    stats_table = build_regression_stats_table(model)
    curve_summary = build_curve_summary_table(merged_df)
    fly_summary = build_fly_summary_table(merged_df)

    subtitle = (
        f"Sample: {_idx_date(dat_final.index[0])} to {_idx_date(dat_final.index[-1])}"
        f"  |  Obs: {summary['obs']}"
        f"  |  R²: {summary['r_squared']:.3f}"
        f"  |  Adj R²: {summary['adj_r_squared']:.3f}"
        f"  |  Resid Std: {summary['resid_std']:.3f}"
        f"  |  Latest Residual Z: {summary['latest_resid_z']:.2f}"
        f"  |  Generated: {datetime.now():%Y-%m-%d %H:%M}"
    )

    sections = []

    # 1. Regression parameters & statistics (side by side)
    sections.append(
        "<section><h2>Regression parameters &amp; statistics</h2>"
        '<div class="side-by-side">'
        + df_to_html_table(params_table, gradient_cols=("coef", "p_value"))
        + df_to_html_table(stats_table, gradient_cols=("value",))
        + "</div></section>"
    )

    # 2. Fitted vs actual (full sample)
    sections.append(
        "<section><h2>Fitted vs actual (full sample)</h2>"
        + figs_to_html([fig_fitted_vs_actual(dat_final, model.fittedvalues, summary)])
        + "</section>"
    )

    # 3. Rolling fair value
    sections.append(
        "<section><h2>Rolling fair value</h2>"
        + figs_to_html([
            fig_rolling_mkt_mdl(rolling_df),
            fig_rolling_resid_z(rolling_df),
        ])
        + "</section>"
    )

    # 4. Recent rolling fair value
    sections.append(
        "<section><h2>Recent rolling fair value</h2>"
        + figs_to_html([
            fig_rolling_mkt_mdl(recent_df, recent=True),
            fig_recent_resid_bps(recent_df),
        ])
        + "</section>"
    )

    # 5. Curve spread fair value
    sections.append(
        "<section><h2>Curve spread fair value</h2>"
        '<p class="subtle">Predictors: SOFR 1y1y, 5y5y infl zero, FedHoldRatio, dummy'
        "  |  percentiles over trailing 252 observations (bps)</p>"
        + df_to_html_table(curve_summary, gradient_cols=("diff", "ratio"))
        + figs_to_html([fig_curve_grid(merged_df, curve_summary)])
        + "</section>"
    )

    # 6. Butterfly fair value
    sections.append(
        "<section><h2>Butterfly fair value</h2>"
        '<p class="subtle">Fly = 2 x belly - short - long (bps)'
        "  |  beta_10y = sensitivity of daily fly change (bps) to daily GT10 change (bps)</p>"
        + df_to_html_table(fly_summary, gradient_cols=("diff", "ratio", "beta_10y"))
        + figs_to_html([fig_fly_grid(merged_df, fly_summary)])
        + "</section>"
    )

    html = page_template(
        "UST 10Y Fair Value OLS Regression Report",
        subtitle,
        "".join(sections),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_date = date.today().strftime("%Y%m%d")
    html_path = OUTPUT_DIR / f"UST_FairValue_Report_{report_date}.html"
    if html_path.exists():
        try:
            with open(html_path, "ab"):
                pass
        except PermissionError:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = OUTPUT_DIR / f"UST_FairValue_Report_{report_date}_{ts}.html"

    try:
        html_path.write_text(html, encoding="utf-8")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = OUTPUT_DIR / f"UST_FairValue_Report_{report_date}_{ts}.html"
        html_path.write_text(html, encoding="utf-8")
    return html_path, summary, curve_summary, fly_summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_bloomberg_data()
    fed_holdings = load_fed_holdings_data()
    dat_final = build_dat_final(df, fed_holdings)

    print("Running full-sample OLS regression...")
    model, _ = run_full_sample_regression(dat_final)

    print("Running rolling fair-value model via USTDurationStrategy...")
    rolling_df = run_rolling_fair_value(dat_final)

    print("Running recent rolling fair-value window (2025-02-11 onwards)...")
    recent_df = run_recent_rolling_fair_value(dat_final)

    print("Building curve-spread / fly dataset...")
    merged_df = build_merged_data(dat_final)

    print("Generating HTML report...")
    html_path, summary, curve_summary, fly_summary = generate_html_report(
        dat_final, model, rolling_df, recent_df, merged_df
    )

    print("\n" + "=" * 70)
    print("UST 10Y Fair Value OLS Regression Report (HTML)")
    print("=" * 70)
    print(f"Latest date : {_idx_date(dat_final.index[-1])}")
    print(f"Observations: {len(dat_final)}")
    print(f"R-squared   : {summary['r_squared']:.4f}")
    print(f"Latest resid: {summary['latest_residual']:.3f} (z={summary['latest_resid_z']:.2f})")
    print("\nCurve spreads (latest):")
    print(curve_summary[["mkt_val", "mdl_val", "diff", "ratio"]].to_string())
    print("\nButterflies (latest):")
    print(fly_summary[["mkt_val", "mdl_val", "diff", "ratio"]].to_string())
    print(f"\nHTML report saved to: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
