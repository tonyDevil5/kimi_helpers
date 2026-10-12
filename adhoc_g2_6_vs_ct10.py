"""
Ad-hoc analysis: G2 6% price daily change vs CT10 (10y UST yield) daily change.

Loads mbs_rv.xlsx (same source as mbs_rv_report.py), computes daily diffs of
"G2 6 Price" and "CT10", and reports the largest daily moves (up and down),
with the corresponding move in the other series on those days. Also shows
FN/G2 coupon 5-6.5 behavior on the biggest 10y move days.

Output:
    ./adhoc_output/adhoc_g2_6_vs_ct10.pdf

Run:
    python adhoc_g2_6_vs_ct10.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

FNAME = r"C:\Users\tonyy\Nutstore\1\python\notebook\data\mbs\mbs_rv.xlsx"
OUTPUT_DIR = Path(__file__).parent / "adhoc_output"
PX_COL = "G2 6 Price"
YLD_COL = "CT10"  # 10y UST on-the-run yield (yield in %, so daily chg is in bp x 0.01)
COUPONS = [5, 5.5, 6, 6.5]
N_BIG_DAYS = 8
TOP_N_DOWN = 10
TOP_N_ROLL5 = 10


def load() -> pd.DataFrame:
    df = pd.read_excel(FNAME, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(subset=[PX_COL, YLD_COL])
    print(f"Loaded {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


CC_TICKERS = {  # current-coupon OAS, Bloomberg tickers
    "FN CC": ".FNCC105 G Index",
    "G2 CC": ".G2CC105 G Index",
}
CC_CACHE = OUTPUT_DIR / "cc_basis.csv"


def load_cc_basis(start_date, end_date) -> pd.DataFrame | None:
    """Download current-coupon OAS (bps) from Bloomberg; fall back to cache."""
    try:
        from xbbg import blp
    except ImportError:
        print("xbbg not available; CC basis columns will be empty")
        return None
    try:
        print(f"Downloading CC basis from Bloomberg "
              f"({start_date.date()} to {end_date.date()})...")
        df = blp.bdh(tickers=list(CC_TICKERS.values()), flds=["px_last"],
                     start_date=start_date.strftime("%Y-%m-%d"),
                     end_date=end_date.strftime("%Y-%m-%d"))
        if df is None or df.empty:
            raise RuntimeError("empty bdh response")
        df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        df = df * 100.0  # Bloomberg returns %, convert to bps
        df = df.rename(columns={v: k for k, v in CC_TICKERS.items()})
        df = df[list(CC_TICKERS)]
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(CC_CACHE)
        print(f"CC basis downloaded: {len(df)} rows")
        return df
    except Exception as exc:  # Bloomberg session down, etc.
        print(f"Bloomberg download failed ({exc}); trying cache")
        if CC_CACHE.exists():
            df = pd.read_csv(CC_CACHE, index_col=0, parse_dates=True)
            print(f"CC basis loaded from cache: {len(df)} rows")
            return df
        print("No CC basis cache; CC basis columns will be empty")
        return None


def compute_results(df: pd.DataFrame) -> dict:
    chg = pd.DataFrame(
        {
            "px_ticks": df[PX_COL].diff() * 32.0,  # price ticks (32nds) per day
            "yld_chg_bp": df[YLD_COL].diff() * 100.0,  # 10y yield change in bp per day
        }
    ).dropna()
    chg["yld_lvl"] = df[YLD_COL]  # absolute 10y yield level on the day

    # Current-coupon basis (bps), reindexed to the main calendar
    cc = load_cc_basis(df.index[0], df.index[-1])
    if cc is not None:
        cc = cc.reindex(df.index).ffill()
        for name in CC_TICKERS:
            chg[f"{name} lvl (bp)"] = cc[name]
            chg[f"{name} chg (bp)"] = cc[name].diff()

    corr = chg["px_ticks"].corr(chg["yld_chg_bp"])
    beta = chg["px_ticks"].cov(chg["yld_chg_bp"]) / chg["yld_chg_bp"].var()

    top_px = chg.reindex(chg["px_ticks"].abs().nlargest(5).index).sort_values(
        "px_ticks", key=abs, ascending=False
    )
    top_y = chg.reindex(chg["yld_chg_bp"].abs().nlargest(5).index).sort_values(
        "yld_chg_bp", key=abs, ascending=False
    )

    # Coupon behavior on the biggest 10y move days
    coupons = [f"{a} {c} Price" for a in ("FN", "G2") for c in COUPONS]
    coupon_chg_ticks = df[coupons].diff() * 32.0  # daily change in ticks

    big_days = chg["yld_chg_bp"].abs().nlargest(N_BIG_DAYS).index.sort_values()
    big_tbl = pd.DataFrame(
        {
            "10y chg (bp)": chg.loc[big_days, "yld_chg_bp"],
            "10y lvl (%)": chg.loc[big_days, "yld_lvl"],
            **{f"{a} {c}": coupon_chg_ticks.loc[big_days, f"{a} {c} Price"]
               for a in ("FN", "G2") for c in COUPONS},
            **{col: chg.loc[big_days, col]
               for col in chg.columns if col.startswith(("FN CC", "G2 CC"))},
        }
    )
    big_tbl.index = [d.date() for d in big_tbl.index]
    big_tbl.index.name = "date"

    avg_rows = {}
    for label, mask in (("10y UP", chg.loc[big_days, "yld_chg_bp"] > 0),
                        ("10y DOWN", chg.loc[big_days, "yld_chg_bp"] < 0)):
        days = big_days[mask]
        row = {f"{a} {c}": coupon_chg_ticks.loc[days, f"{a} {c} Price"].mean()
               for a in ("FN", "G2") for c in COUPONS}
        row["n"] = len(days)
        row["avg 10y (bp)"] = chg.loc[days, "yld_chg_bp"].mean()
        avg_rows[label] = row
    avg_tbl = pd.DataFrame(avg_rows).T
    avg_tbl = avg_tbl[["n", "avg 10y (bp)"] +
                      [f"{a} {c}" for a in ("FN", "G2") for c in COUPONS]]

    # Largest daily up / down move per coupon, with the 10y move that day
    names = [f"{a} {c}" for a in ("FN", "G2") for c in COUPONS]
    ext_rows = {}
    for c in coupons:
        s = coupon_chg_ticks[c].dropna()
        name = c.replace(" Price", "")
        for direction, d in (("UP", s.idxmax()), ("DOWN", s.idxmin())):
            ext_rows[(name, direction)] = {
                "date": d.date(),
                "px chg (ticks)": s.loc[d],
                "10y chg (bp)": chg.loc[d, "yld_chg_bp"],
                "10y lvl (%)": chg.loc[d, "yld_lvl"],
            }
    ext_tbl = pd.DataFrame(ext_rows).T
    ext_tbl.index.names = ["coupon", "direction"]

    # Top N largest daily DOWN moves per coupon, with the 10y / CC context
    # and the FOLLOWING day's price action
    yld_next = chg["yld_chg_bp"].shift(-1)
    top_dn_rows = []
    for c in coupons:
        s = coupon_chg_ticks[c].dropna()
        s_next = s.shift(-1)
        name = c.replace(" Price", "")
        for rank, (d, v) in enumerate(s.nsmallest(TOP_N_DOWN).items(), 1):
            row = {
                "coupon": name,
                "#": rank,
                "date": d.date(),
                "px chg (ticks)": v,
                "next px (ticks)": s_next.loc[d],
                "next 10y (bp)": yld_next.loc[d],
                "10y chg (bp)": chg.loc[d, "yld_chg_bp"],
                "10y lvl (%)": chg.loc[d, "yld_lvl"],
            }
            for col in chg.columns:
                if col.startswith(("FN CC", "G2 CC")):
                    row[col] = chg.loc[d, col]
            top_dn_rows.append(row)
    top_dn = pd.DataFrame(top_dn_rows)

    # Worst rolling 5-observation windows per coupon, with 10y / CC context
    yld_roll5 = chg["yld_chg_bp"].rolling(5).sum()
    roll5_rows = []
    for c in coupons:
        s = coupon_chg_ticks[c].dropna()
        roll5 = s.rolling(5).sum()
        name = c.replace(" Price", "")
        agency = name.split()[0]
        cc_lvl_col = f"{agency} CC lvl (bp)"
        cc_chg = chg[cc_lvl_col].diff(5) if cc_lvl_col in chg else None
        for rank, (d, v) in enumerate(roll5.nsmallest(TOP_N_ROLL5).items(), 1):
            roll5_rows.append({
                "coupon": name,
                "#": rank,
                "end date": d.date(),
                "5d px (ticks)": v,
                "5d 10y (bp)": yld_roll5.loc[d],
                "10y lvl (%)": chg.loc[d, "yld_lvl"],
                "CC lvl (bp)": chg.loc[d, cc_lvl_col] if cc_lvl_col in chg else np.nan,
                "CC chg 5d (bp)": cc_chg.loc[d] if cc_chg is not None else np.nan,
            })
    roll5_tbl = pd.DataFrame(roll5_rows)

    return {
        "chg": chg,
        "corr": corr,
        "beta": beta,
        "top_px": top_px,
        "top_y": top_y,
        "big_tbl": big_tbl,
        "avg_tbl": avg_tbl,
        "ext_tbl": ext_tbl,
        "top_dn": top_dn,
        "roll5_tbl": roll5_tbl,
    }


def print_results(df: pd.DataFrame, res: dict) -> None:
    chg = res["chg"]
    print("\n=== Daily change stats ===")
    print(chg[["px_ticks", "yld_chg_bp"]].describe().round(3))
    print(f"\nCorr(px daily chg, 10y yield daily chg): {res['corr']:.3f}")
    print(f"OLS beta: {res['beta']:.3f} ticks per 1bp move in 10y yield")

    print("\n=== Top 5 largest absolute G2 6 daily price moves (ticks) ===")
    for d, r in res["top_px"].iterrows():
        print(f"  {d.date()}  px {r['px_ticks']:+7.2f} ticks   "
              f"10y {r['yld_chg_bp']:+6.1f} bp  (lvl {r['yld_lvl']:.3f}%)")

    print("\n=== Top 5 largest absolute 10y yield daily moves ===")
    for d, r in res["top_y"].iterrows():
        print(f"  {d.date()}  10y {r['yld_chg_bp']:+6.1f} bp  "
              f"(lvl {r['yld_lvl']:.3f}%)   px {r['px_ticks']:+7.2f} ticks")

    max_abs = chg["px_ticks"].abs().max()
    print(f"\nMax |daily G2 6 price change| = {max_abs:.2f} ticks "
          f"({max_abs/32.0:.3f} pts, on {chg['px_ticks'].abs().idxmax().date()})")

    print(f"\n=== FN/G2 5-6.5 daily px change (ticks) on top {N_BIG_DAYS} "
          f"|10y| move days ===")
    print(res["big_tbl"].round(2).to_string())
    print("\n=== Average by direction ===")
    print(res["avg_tbl"].round(2).to_string())

    print("\n=== Largest daily move per coupon (with 10y context) ===")
    print(res["ext_tbl"].round(2).to_string())

    print(f"\n=== Top {TOP_N_DOWN} largest daily DOWN moves per coupon ===")
    print(res["top_dn"].round(2).to_string(index=False))

    print(f"\n=== Worst {TOP_N_ROLL5} rolling 5-day windows per coupon ===")
    print(res["roll5_tbl"].round(2).to_string(index=False))


def _fmt(v) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:+.2f}"
    return str(v)


def _add_table(fig, tbl: pd.DataFrame, title: str, ytop: float):
    """Render tbl as a table whose first row starts at figure fraction `ytop`
    (the title is drawn just above). The table grows downward from there."""
    ax = fig.add_axes([0.05, max(ytop - 0.05, 0.0), 0.9, 0.05])
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    cell_text = [[_fmt(v) for v in row] for row in
                 np.column_stack([tbl.index.astype(str), tbl.round(2).values])]
    col_labels = [tbl.index.name or ""] + list(tbl.columns)
    t = ax.table(cellText=cell_text, colLabels=col_labels, loc="upper left",
                 cellLoc="center", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.3)
    for j in range(len(col_labels)):
        t[0, j].set_facecolor("#d9e2f3")
        t[0, j].set_text_props(fontweight="bold")
    ncols = len(col_labels)
    for j in range(ncols):
        t.auto_set_column_width(j)
    return ax


def build_pdf(df: pd.DataFrame, res: dict, out_path: Path) -> None:
    chg = res["chg"]

    with PdfPages(out_path) as pdf:
        # ---- Page 1: summary + top moves tables ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("G2 6 Price vs CT10 (10y UST) — Daily Change Analysis",
                     fontsize=14, fontweight="bold", y=0.97)
        sub = (f"{chg.index[0].date()} to {chg.index[-1].date()}  |  "
               f"{len(chg)} daily obs  |  corr {res['corr']:.3f}  |  "
               f"beta {res['beta']:.3f} ticks/bp")
        fig.text(0.05, 0.935, sub, fontsize=10, color="#444444")

        top_px_tbl = res["top_px"][["px_ticks", "yld_chg_bp", "yld_lvl"]].copy()
        top_px_tbl.columns = ["G2 6 chg (ticks)", "10y chg (bp)", "10y lvl (%)"]
        top_px_tbl.index = [d.date() for d in top_px_tbl.index]
        top_px_tbl.index.name = "date"
        _add_table(fig, top_px_tbl.round(2),
                   "Top 5 largest |G2 6 daily price moves|", 0.88)

        top_y_tbl = res["top_y"][["yld_chg_bp", "yld_lvl", "px_ticks"]].copy()
        top_y_tbl.columns = ["10y chg (bp)", "10y lvl (%)", "G2 6 chg (ticks)"]
        top_y_tbl.index = [d.date() for d in top_y_tbl.index]
        top_y_tbl.index.name = "date"
        _add_table(fig, top_y_tbl.round(2),
                   "Top 5 largest |10y yield daily moves|", 0.62)

        max_abs = chg["px_ticks"].abs().max()
        stats_txt = (
            f"Max daily UP move   : +{chg['px_ticks'].max():.2f} ticks "
            f"({chg['px_ticks'].idxmax().date()}, 10y "
            f"{chg['yld_chg_bp'].loc[chg['px_ticks'].idxmax()]:+.1f} bp)\n"
            f"Max daily DOWN move : {chg['px_ticks'].min():+.2f} ticks "
            f"({chg['px_ticks'].idxmin().date()}, 10y "
            f"{chg['yld_chg_bp'].loc[chg['px_ticks'].idxmin()]:+.1f} bp)\n"
            f"Max |daily change|  : {max_abs:.2f} ticks "
            f"({max_abs/32.0:.3f} pts, {chg['px_ticks'].abs().idxmax().date()})\n"
            f"Max daily 10y move  : {chg['yld_chg_bp'].max():+.1f} bp "
            f"({chg['yld_chg_bp'].idxmax().date()}) / "
            f"{chg['yld_chg_bp'].min():+.1f} bp "
            f"({chg['yld_chg_bp'].idxmin().date()})"
        )
        fig.text(0.05, 0.16, stats_txt, fontsize=10, family="monospace",
                 va="bottom")
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: largest daily move per coupon (with 10y context) ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle("Largest Daily Move per Coupon — FN/G2 5-6.5 (ticks)",
                     fontsize=13, fontweight="bold", y=0.97)
        ext = res["ext_tbl"]
        for direction in ("UP", "DOWN"):
            tbl = ext.xs(direction, level="direction").copy()
            tbl.index.name = "coupon"
            _add_table(fig, tbl.reset_index().set_index("coupon"),
                       f"Max daily {direction} move per coupon",
                       0.88 if direction == "UP" else 0.55)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: top N down moves, one table per coupon (2 per page) ----
        top_dn = res["top_dn"]
        cc_cols = [c for c in top_dn.columns if c.startswith(("FN CC", "G2 CC"))]
        for agency in ("FN", "G2"):
            names = [f"{agency} {c}" for c in COUPONS]
            for c1, c2 in ((names[0], names[1]), (names[2], names[3])):
                fig = plt.figure(figsize=(8.5, 11))
                fig.suptitle(f"{c1} / {c2}: Top {TOP_N_DOWN} Largest Daily "
                             f"DOWN Moves (ticks)",
                             fontsize=13, fontweight="bold", y=0.97)
                for coupon, ytop in ((c1, 0.90), (c2, 0.46)):
                    agency = coupon.split()[0]  # "FN" or "G2"
                    cols = [c for c in cc_cols if c.startswith(agency)]
                    tbl = top_dn[top_dn["coupon"] == coupon].set_index("#")
                    tbl = tbl[["date", "px chg (ticks)", "next px (ticks)",
                               "next 10y (bp)", "10y chg (bp)",
                               "10y lvl (%)"] + cols]
                    _add_table(fig, tbl,
                               f"{coupon}: {TOP_N_DOWN} largest down days",
                               ytop)
                pdf.savefig(fig)
                plt.close(fig)

        # ---- Pages 4-5: worst rolling 5-day windows (2 coupons per page) ----
        roll5_tbl = res["roll5_tbl"]
        for agency in ("FN", "G2"):
            names = [f"{agency} {c}" for c in COUPONS]
            for c1, c2 in ((names[0], names[1]), (names[2], names[3])):
                fig = plt.figure(figsize=(8.5, 11))
                cc_name = f"{agency} CC"
                fig.suptitle(f"{c1} / {c2}: Worst Rolling 5-Day Windows (ticks)",
                             fontsize=13, fontweight="bold", y=0.97)
                for coupon, ytop in ((c1, 0.90), (c2, 0.42)):
                    tbl = roll5_tbl[roll5_tbl["coupon"] == coupon].set_index("#")
                    tbl = tbl.drop(columns=["coupon"])
                    tbl = tbl.rename(columns={"CC lvl (bp)": f"{cc_name} lvl (bp)",
                                              "CC chg 5d (bp)": f"{cc_name} chg 5d (bp)"})
                    _add_table(fig, tbl,
                               f"{coupon}: worst {TOP_N_ROLL5} 5d rolling windows",
                               ytop)
                pdf.savefig(fig)
                plt.close(fig)

        # ---- Page 6: coupon behavior on big 10y move days ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"FN/G2 5-6.5 Daily Price Change (ticks) on Top "
                     f"{N_BIG_DAYS} |10y| Move Days",
                     fontsize=13, fontweight="bold", y=0.97)
        big_tbl = res["big_tbl"]
        cc_cols = [c for c in big_tbl.columns if c.startswith(("FN CC", "G2 CC"))]
        _add_table(fig, big_tbl[[c for c in big_tbl.columns if c not in cc_cols]].round(2),
                   f"Daily px change (ticks) per coupon", 0.90)
        _add_table(fig, big_tbl[["10y chg (bp)"] + cc_cols].round(2),
                   "Current-coupon basis (bps) on those days", 0.56)
        _add_table(fig, res["avg_tbl"].round(2),
                   "Average px change by 10y direction", 0.30)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 8: time series + scatter ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11))
        fig.suptitle("G2 6 Price vs CT10 — Time Series and Daily Changes",
                     fontsize=13, fontweight="bold")

        ax1b = ax1.twinx()
        ax1.plot(df.index, df[PX_COL], color="#1f4e79", lw=1.2, label="G2 6 Price")
        ax1b.plot(df.index, df[YLD_COL], color="#c00000", lw=1.2, label="CT10")
        ax1.set_ylabel("G2 6 Price", color="#1f4e79")
        ax1b.set_ylabel("CT10 (%)", color="#c00000")
        ax1.set_title("Levels")
        ax1.tick_params(axis="x", rotation=30)
        ln1, lb1 = ax1.get_legend_handles_labels()
        ln2, lb2 = ax1b.get_legend_handles_labels()
        ax1.legend(ln1 + ln2, lb1 + lb2, loc="upper right", fontsize=8)

        ax2.scatter(chg["yld_chg_bp"], chg["px_ticks"], s=8, alpha=0.4,
                    color="#1f4e79")
        xs = np.linspace(chg["yld_chg_bp"].min(), chg["yld_chg_bp"].max(), 50)
        ax2.plot(xs, res["beta"] * xs, color="#c00000", lw=1.5,
                 label=f"OLS fit ({res['beta']:.2f} ticks/bp)")
        ax2.axhline(0, color="#999999", lw=0.5)
        ax2.axvline(0, color="#999999", lw=0.5)
        ax2.set_xlabel("CT10 daily change (bp)")
        ax2.set_ylabel("G2 6 daily change (ticks)")
        ax2.set_title(f"Daily changes (corr {res['corr']:.3f})")
        ax2.legend(fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPDF written: {out_path}")


def main() -> None:
    df = load()
    res = compute_results(df)
    print_results(df, res)
    OUTPUT_DIR.mkdir(exist_ok=True)
    build_pdf(df, res, OUTPUT_DIR / "adhoc_g2_6_vs_ct10.pdf")


if __name__ == "__main__":
    main()
