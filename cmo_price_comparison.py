"""Raw-price comparison, deliberately independent of valuation-model inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

START = "2026-08-01"
END = "2026-10-02"
BENCHMARKS = ["FN 5.5", "FN 6", "G2 5.5", "G2 6"]
BENCHMARK_TICKERS = {"FN 5.5": "FNCL 5.5 Mtge", "FN 6": "FNCL 6 Mtge",
                     "G2 5.5": "G2SF 5.5 Mtge", "G2 6": "G2SF 6 Mtge"}
BENCHMARK_CACHE = Path(__file__).parent / "adhoc_output" / "cmo_tba_bbg_history.csv"
TBA_WORKBOOK = Path("C:/Users/tonyy/Nutstore/1/python/notebook/data/mbs/mbs_rv.xlsx")


def refresh_benchmarks():
    from xbbg import blp
    df = blp.bdh(list(BENCHMARK_TICKERS.values()), "PX_LAST", start_date="2023-01-01")
    if df.empty:
        raise RuntimeError("Bloomberg returned no TBA benchmark prices")
    df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={ticker: label+" Price" for label, ticker in BENCHMARK_TICKERS.items()})
    required = [label+" Price" for label in BENCHMARKS]
    if any(c not in df or df[c].notna().sum()==0 for c in required):
        raise RuntimeError("Incomplete Bloomberg benchmark response")
    df[required].sort_index().to_csv(BENCHMARK_CACHE, index_label="date")
    print(f"Bloomberg TBA benchmark cache: {len(df)} dates")


def build_comparison(prices, bonds, workbook):
    tba = workbook.copy() if isinstance(workbook, pd.DataFrame) else pd.read_excel(workbook, index_col=0)
    tba.index = pd.to_datetime(tba.index)
    dates = pd.DatetimeIndex(pd.to_datetime(prices.index)).union(tba.index).sort_values().unique()

    def item(label, series, **metadata):
        series = pd.to_numeric(series, errors="coerce")
        series.index = pd.to_datetime(series.index)
        series = series.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
        series = series.reindex(dates).replace([np.inf, -np.inf], np.nan)
        start, end = series.get(START, np.nan), series.get(END, np.nan)
        ok = pd.notna(start) and pd.notna(end)
        missing = [date for date, value in [(START, start), (END, end)] if pd.isna(value)]
        return {"label": label, **metadata,
                "prices": [float(v) if pd.notna(v) else None for v in series],
                "start_price": float(start) if pd.notna(start) else None,
                "end_price": float(end) if pd.notna(end) else None,
                "change": float(end-start) if ok else None,
                "status": "Complete endpoints" if ok else "Missing price: " + ", ".join(missing)}

    benchmarks = [item(label, tba.get(label+" Price", pd.Series(dtype=float)),
                       source="mbs_rv.xlsx / first sheet / " + label+" Price")
                  for label in BENCHMARKS]
    rows = [item(f"{ticker.replace(' Mtge', '')} · {cap:g}% cap", prices.get(ticker, pd.Series(dtype=float)),
                 ticker=ticker, cap=cap, agency=agency,
                 source="Bloomberg PX_LAST / cmo_caps_data.csv")
            for ticker, cap, agency in bonds]
    for row in rows:
        row["excess"] = {b["label"]: row["change"]-b["change"]
                         if row["change"] is not None and b["change"] is not None else None
                         for b in benchmarks}
    return {"start": START, "end": END, "dates": dates.strftime("%Y-%m-%d").tolist(),
            "benchmarks": benchmarks, "bonds": rows}


def load_comparison(tickers=None):
    from cmo_forward_indicator import payload as indicator_payload
    from adhoc_cmo_caps import CACHE, BONDS, PORTFOLIO_TICKERS
    selected = [bond for bond in BONDS if tickers is None or bond[0] in tickers]
    benchmarks = pd.read_excel(TBA_WORKBOOK, index_col=0)
    payload = build_comparison(pd.read_csv(CACHE, index_col=0, parse_dates=True), selected, benchmarks)
    for row in payload["bonds"]:
        row["group"] = "portfolio" if row["ticker"] in PORTFOLIO_TICKERS else "analysis"
    payload["indicator"] = indicator_payload()
    return payload
