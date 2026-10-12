"""
Ad-hoc analysis: CMO watchlist vs CT10, CC basis, 2s10s and 1y10y vol,
grouped by coupon cap.

For each bond, full-history and prior-window OLS regressions use

    price = a + b1*CT10 + b2*CC_basis + b3*2s10s + b4*1y10y_vol

The prior-window model uses up to 120 observations before the evaluation date,
with at least 60 training observations. Paired held-out RMSE compares the
two-factor baseline with all four factors. CC basis: FN CC for
FHR/FNR (Fannie/Freddie), G2 CC for GNR (Ginnie).  1y10y ATM normal vol
comes directly from Bloomberg USSNA110 FMCA Curncy.

Output:
    ./adhoc_output/adhoc_cmo_caps.html   (self-contained, plotly)

Run:
    python adhoc_cmo_caps.py
    python adhoc_cmo_caps.py --cache  # reuse cached Bloomberg history
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq


OUTPUT_DIR = Path(__file__).parent / "adhoc_output"
CACHE = OUTPUT_DIR / "cmo_caps_data.csv"
REFERENCE_CACHE = OUTPUT_DIR / "cmo_caps_reference.csv"
REFERENCE_FIELDS = ["FLT_SPREAD", "MTG_COLLAT_TYP"]

START = "2023-01-01"  # retain early-2023 history wherever the ticker has observations
MIN_OBS = 20          # below this, regression is not meaningful
SHORT_N = 60          # below this, flag the fit as a short sample
ROLL_N = 120
VALIDATION_MIN = 60

# (ticker, cap, agency)  agency: "FN" -> FN CC basis, "G2" -> G2 CC basis
BONDS = [
    ("GNR 2026-3 FJ Mtge",   6.0, "G2"),
    ("FHR 5668 FB Mtge",     6.0, "FN"),
    ("GNR 2026-93 FD Mtge",  6.0, "G2"),
    ("GNR 2026-85 FE Mtge",  6.0, "G2"),
    ("GNR 2025-188 FA Mtge", 6.0, "G2"),
    ("GNR 2025-188 EF Mtge", 6.0, "G2"),
    ("GNR 2026-47 FB Mtge",  6.0, "G2"),
    ("FHR 5647 FC Mtge",     6.0, "FN"),
    ("FHR 5598 FD Mtge",     6.0, "FN"),
    ("FNR 2026-65 FA Mtge",  6.0, "FN"),
    ("GNR 2026-111 FJ Mtge", 6.5, "G2"),
    ("GNR 2025-124 GF Mtge", 6.5, "G2"),
    ("FNR 2025-103 DF Mtge", 6.5, "FN"),
    ("GNR 2026-140 FH Mtge", 6.5, "G2"),
    ("GNR 2026-157 FB Mtge", 6.5, "G2"),
    ("GNR 2026-2 FG Mtge",   7.0, "G2"),
    ("GNR 2025-198 FH Mtge", 7.0, "G2"),
    ("FNR 2026-70 FA Mtge",  7.0, "FN"),
    # 2023 GNR capped floaters (GNR 23-63/64/65/66)
    ("GNR 2023-63 AF Mtge",  6.5, "G2"),
    ("GNR 2023-63 BF Mtge",  6.5, "G2"),
    ("GNR 2023-63 FA Mtge",  6.5, "G2"),
    ("GNR 2023-63 FB Mtge",  6.5, "G2"),
    ("GNR 2023-63 FD Mtge",  7.0, "G2"),
    ("GNR 2023-63 FJ Mtge",  6.5, "G2"),
    ("GNR 2023-63 FW Mtge",  7.0, "G2"),
    ("GNR 2023-63 KF Mtge",  7.0, "G2"),
    ("GNR 2023-63 YF Mtge",  6.5, "G2"),
    ("GNR 2023-63 FQ Mtge",  6.5, "G2"),
    ("GNR 2023-63 NF Mtge",  6.5, "G2"),
    ("GNR 2023-63 TF Mtge",  6.5, "G2"),
    ("GNR 2023-63 FX Mtge",  6.5, "G2"),
    ("GNR 2023-63 FG Mtge",  6.5, "G2"),
    ("GNR 2023-63 JF Mtge",  6.5, "G2"),
    ("GNR 2023-63 FM Mtge",  7.0, "G2"),
    ("GNR 2023-64 FA Mtge",  6.5, "G2"),
    ("GNR 2023-64 FB Mtge",  6.5, "G2"),
    ("GNR 2023-64 FL Mtge",  6.5, "G2"),
    ("GNR 2023-64 FM Mtge",  7.0, "G2"),
    ("GNR 2023-64 FE Mtge",  6.5, "G2"),
    ("GNR 2023-64 FG Mtge",  6.5, "G2"),
    ("GNR 2023-65 AF Mtge",  7.0, "G2"),
    ("GNR 2023-65 BF Mtge",  7.0, "G2"),
    ("GNR 2023-65 FB Mtge",  7.0, "G2"),
    ("GNR 2023-65 FA Mtge",  7.0, "G2"),
    ("GNR 2023-65 FD Mtge",  7.0, "G2"),
    ("GNR 2023-65 FE Mtge",  7.0, "G2"),
    ("GNR 2023-65 FG Mtge",  7.0, "G2"),
    ("GNR 2023-65 GF Mtge",  7.0, "G2"),
    ("GNR 2023-65 FH Mtge",  7.0, "G2"),
    ("GNR 2023-65 FJ Mtge",  7.0, "G2"),
    ("GNR 2023-65 FK Mtge",  7.0, "G2"),
    ("GNR 2023-65 FL Mtge",  7.0, "G2"),
    ("GNR 2023-65 KF Mtge",  7.0, "G2"),
    ("GNR 2023-65 LF Mtge",  7.0, "G2"),
    ("GNR 2023-65 NF Mtge",  7.0, "G2"),
    ("GNR 2023-65 QF Mtge",  7.0, "G2"),
    ("GNR 2023-65 WF Mtge",  7.0, "G2"),
    ("GNR 2023-66 FB Mtge",  6.5, "G2"),
    ("GNR 2023-66 FC Mtge",  7.0, "G2"),
    ("GNR 2023-66 FE Mtge",  7.0, "G2"),
    ("GNR 2023-66 FG Mtge",  6.5, "G2"),
    ("GNR 2023-66 FL Mtge",  6.5, "G2"),
    ("GNR 2023-66 FN Mtge",  6.5, "G2"),
    ("GNR 2023-66 GF Mtge",  6.5, "G2"),
    ("GNR 2023-66 BF Mtge",  7.0, "G2"),
    ("GNR 2023-66 DF Mtge",  7.0, "G2"),
    ("GNR 2023-66 FH Mtge",  7.0, "G2"),
    ("GNR 2023-66 FA Mtge",  6.5, "G2"),
    ("GNR 2023-66 FD Mtge",  7.0, "G2"),
    ("GNR 2023-66 FJ Mtge",  7.0, "G2"),
    ("GNR 2023-66 FM Mtge",  7.0, "G2"),
    # 2022 GNR capped floaters (GNR 22-188 ~ 22-204)
    ("GNR 2022-188 FA Mtge", 6.5, "G2"),
    ("GNR 2022-188 F Mtge",  6.5, "G2"),
    ("GNR 2022-189 FG Mtge", 7.0, "G2"),
    ("GNR 2022-189 FA Mtge", 7.0, "G2"),
    ("GNR 2022-189 DF Mtge", 6.5, "G2"),
    ("GNR 2022-189 KF Mtge", 6.0, "G2"),
    ("GNR 2022-190 FJ Mtge", 6.5, "G2"),
    ("GNR 2022-190 FA Mtge", 6.5, "G2"),
    ("GNR 2022-191 FC Mtge", 6.5, "G2"),
    ("GNR 2022-193 FA Mtge", 6.0, "G2"),
    ("GNR 2022-195 FA Mtge", 7.0, "G2"),
    ("GNR 2022-195 FB Mtge", 7.0, "G2"),
    ("GNR 2022-197 FC Mtge", 6.0, "G2"),
    ("GNR 2022-197 CF Mtge", 6.5, "G2"),
    ("GNR 2022-197 DF Mtge", 7.5, "G2"),
    ("GNR 2022-197 BF Mtge", 6.5, "G2"),
    ("GNR 2022-197 FJ Mtge", 6.0, "G2"),
    ("GNR 2022-197 JF Mtge", 7.0, "G2"),
    ("GNR 2022-197 KF Mtge", 7.5, "G2"),
    ("GNR 2022-197 LF Mtge", 7.0, "G2"),
    ("GNR 2022-200 FA Mtge", 7.0, "G2"),
    ("GNR 2022-200 F Mtge",  7.0, "G2"),
    ("GNR 2022-201 FB Mtge", 6.5, "G2"),
    ("GNR 2022-204 FA Mtge", 6.0, "G2"),
]
CAP_ORDER = [6.0, 6.5, 7.0, 7.5]

# The first watchlist (18 bonds) is the portfolio; everything added later
# (GNR 2023 / 2022 floater deals) is the analysis universe.
PORTFOLIO_TICKERS = {
    "GNR 2026-3 FJ Mtge", "FHR 5668 FB Mtge", "GNR 2026-93 FD Mtge",
    "GNR 2026-85 FE Mtge", "GNR 2025-188 FA Mtge", "GNR 2025-188 EF Mtge",
    "GNR 2026-47 FB Mtge", "FHR 5647 FC Mtge", "FHR 5598 FD Mtge",
    "FNR 2026-65 FA Mtge", "GNR 2026-111 FJ Mtge", "GNR 2025-124 GF Mtge",
    "FNR 2025-103 DF Mtge", "GNR 2026-140 FH Mtge", "GNR 2026-157 FB Mtge",
    "GNR 2026-2 FG Mtge", "GNR 2025-198 FH Mtge", "FNR 2026-70 FA Mtge",
}
CC_TICKERS = {"FN CC": ".FNCC105 G Index", "G2 CC": ".G2CC105 G Index"}
CT10_TICKER = "USGG10YR Index"
CT2_TICKER = "USGG2YR Index"
VOL_TICKER = "USSNA110 FMCA Curncy"  # USD SOFR 1Y10Y ATM normal vol, annualized bp
ABNORMAL_VOL_DATES = {"2023-06-22", "2024-06-10"}


def clean_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Replace reviewed abnormal quotes with prior available vol, never future data."""
    df = df.sort_index().copy()
    if "vol_raw" not in df:
        df["vol_raw"] = df["vol"]
    raw = pd.to_numeric(df["vol_raw"], errors="coerce")
    corrected = raw.copy()
    flagged = df.index.strftime("%Y-%m-%d").isin(ABNORMAL_VOL_DATES) & raw.notna()
    corrected.loc[flagged] = np.nan
    prior = corrected.ffill()
    corrected.loc[flagged] = prior.loc[flagged]
    df["vol"] = corrected
    df["vol_replaced"] = flagged & raw.notna() & corrected.notna()
    log = pd.DataFrame({"date": df.index[flagged], "raw_vol": raw.loc[flagged].values,
                        "used_vol": corrected.loc[flagged].values,
                        "reason": "Reviewed isolated abnormal quote; use previous available observation"})
    OUTPUT_DIR.mkdir(exist_ok=True)
    log.to_csv(OUTPUT_DIR / "cmo_vol_replacements.csv", index=False)
    print(f"Vol cleaning: {int(df.vol_replaced.sum())} reviewed quotes replaced with previous values")
    return df

def _label(ticker: str) -> str:
    return ticker.replace(" Mtge", "")


def load_bbg() -> pd.DataFrame | None:
    tickers = ([t for t, _, _ in BONDS] + [CT10_TICKER, CT2_TICKER]
               + list(CC_TICKERS.values()) + [VOL_TICKER])
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
        df = df.sort_index()
        for k, v in CC_TICKERS.items():
            df[k] = df[v] * 100.0  # OAS % -> bps
        df = df.rename(columns={CT10_TICKER: "CT10", CT2_TICKER: "CT2", VOL_TICKER: "vol"})
        df["2s10s"] = (df["CT10"] - df["CT2"]) * 100.0  # % -> bp
        keep = [t for t, _, _ in BONDS] + ["CT10", "2s10s", "vol"] + list(CC_TICKERS)
        df = df[keep]
        if df["vol"].notna().sum() == 0:
            raise RuntimeError(f"No Bloomberg volatility history for {VOL_TICKER}")
        df["vol_source"] = VOL_TICKER
        print(f"Downloaded {len(df)} rows, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as exc:
        print(f"Bloomberg download failed ({exc})")
        return None


def load_vol() -> pd.Series | None:
    """Direct Bloomberg normal volatility in annualized basis points."""
    try:
        from xbbg import blp
        df = blp.bdh(VOL_TICKER, "PX_LAST", start_date=START)
        if df.empty:
            raise RuntimeError("empty Bloomberg volatility history")
        s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        print(f"1y10y vol loaded from Bloomberg: {len(s)} obs, "
              f"{s.index[0].date()} to {s.index[-1].date()}")
        return s.rename("vol")
    except Exception as exc:
        print(f"1y10y Bloomberg vol load failed ({exc})")
        return None


def _collateral_coupon(description) -> float | None:
    """Parse a single pass-through coupon, e.g. 'G2SF 5.5'; reject mixed pools."""
    if not isinstance(description, str):
        return None
    match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*\s+(\d+(?:\.\d+)?)\s*%?", description.strip())
    if match:
        coupon = float(match.group(1))
        return coupon if 0 < coupon <= 30 else None
    return None


def _reference_direct(tickers: list[str]) -> pd.DataFrame:
    """Bounded Desktop API fallback when xbbg returns an empty response."""
    import blpapi

    options = blpapi.SessionOptions()
    options.setServerHost("127.0.0.1")
    options.setServerPort(8194)
    options.setConnectTimeout(3000)
    options.setNumStartAttempts(1)
    session = blpapi.Session(options)
    values = {}
    try:
        if not session.start():
            raise RuntimeError("Bloomberg Desktop API connection unavailable")
        if not session.openService("//blp/refdata"):
            raise RuntimeError("Bloomberg reference-data service unavailable")
        request = session.getService("//blp/refdata").createRequest("ReferenceDataRequest")
        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        for field in REFERENCE_FIELDS:
            request.getElement("fields").appendValue(field)
        session.sendRequest(request)
        deadline = time.monotonic()+15
        complete = False
        while time.monotonic() < deadline:
            event = session.nextEvent(500)
            for message in event:
                if message.hasElement("responseError"):
                    raise RuntimeError(message.getElement("responseError").getElementAsString("message"))
                if not message.hasElement("securityData"):
                    continue
                securities = message.getElement("securityData")
                for i in range(securities.numValues()):
                    security = securities.getValueAsElement(i)
                    ticker = security.getElementAsString("security")
                    if security.hasElement("securityError"):
                        print(f"Reference fields unavailable: {ticker}")
                        continue
                    fields = security.getElement("fieldData")
                    row = {}
                    for field in REFERENCE_FIELDS:
                        if not fields.hasElement(field) or fields.getElement(field).isNull():
                            continue
                        if field == "MTG_COLLAT_TYP":
                            row[field.lower()] = fields.getElementAsString(field)
                        else:
                            value = fields.getElementAsFloat(field)
                            if np.isfinite(value):
                                row[field.lower()] = value
                    if row:
                        values[ticker] = row
            if event.eventType() == blpapi.Event.RESPONSE:
                complete = True
                break
        if not complete:
            raise RuntimeError("Bloomberg reference-data request timed out")
        return pd.DataFrame.from_dict(values, orient="index")
    finally:
        session.stop()


def load_reference(refresh: bool = True) -> dict:
    """Fetch spread and collateral reference data separately from model inputs."""
    tickers = [t for t, _, _ in BONDS]
    refs = {ticker: {"flt_spread": None, "mtg_collat_typ": None, "underlying_cpn": None,
                     "asof": "", "collateral_asof": "",
                     "status": "Bloomberg FLT_SPREAD unavailable",
                     "collateral_status": "Bloomberg collateral unavailable"} for ticker in tickers}
    has_collateral_cache = False
    if REFERENCE_CACHE.exists():
        snapshot = pd.read_csv(REFERENCE_CACHE).set_index("ticker")
        has_collateral_cache = "mtg_collat_typ" in snapshot.columns
        for ticker, row in snapshot.iterrows():
            if ticker not in refs:
                continue
            value = pd.to_numeric(row.get("flt_spread"), errors="coerce")
            if pd.notna(value) and np.isfinite(value):
                refs[ticker].update({"flt_spread": float(value),
                                    "asof": str(row["asof"]) if pd.notna(row.get("asof")) else "",
                                    "status": "Cached Bloomberg snapshot"})
            collateral = row.get("mtg_collat_typ")
            if isinstance(collateral, str) and collateral.strip():
                refs[ticker].update({"mtg_collat_typ": collateral,
                                    "underlying_cpn": _collateral_coupon(collateral),
                                    "collateral_asof": str(row["collateral_asof"]) if pd.notna(row.get("collateral_asof")) else "",
                                    "collateral_status": "Cached Bloomberg MTG_COLLAT_TYP"})
    if refresh or not has_collateral_cache:
        try:
            try:
                from xbbg import blp
                response = blp.bdp(tickers=tickers, flds=REFERENCE_FIELDS, timeout=5000)
            except Exception as exc:
                print(f"xbbg reference request failed ({exc}); trying Desktop API")
                response = None
            if response is None or response.empty:
                print("xbbg reference response empty; trying Bloomberg Desktop API directly")
                response = _reference_direct(tickers)
            if response is None or response.empty:
                raise RuntimeError("No Bloomberg reference-data response")
            response.columns = response.columns.map(lambda c: str(c).lower())
            stamp = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M %Z")
            for ticker in tickers:
                if "flt_spread" in response:
                    value = pd.to_numeric(response["flt_spread"].get(ticker), errors="coerce")
                    if pd.notna(value) and np.isfinite(value):
                        refs[ticker].update({"flt_spread": float(value), "asof": stamp,
                                            "status": "Bloomberg FLT_SPREAD"})
                if "mtg_collat_typ" in response:
                    collateral = response["mtg_collat_typ"].get(ticker)
                    if isinstance(collateral, str) and collateral.strip():
                        refs[ticker].update({"mtg_collat_typ": collateral,
                                            "underlying_cpn": _collateral_coupon(collateral),
                                            "collateral_asof": stamp,
                                            "collateral_status": "Bloomberg MTG_COLLAT_TYP"})
            valid = [{"ticker": t, **r} for t, r in refs.items()
                     if r["flt_spread"] is not None or r["mtg_collat_typ"] is not None]
            if valid:
                OUTPUT_DIR.mkdir(exist_ok=True)
                pd.DataFrame(valid).to_csv(REFERENCE_CACHE, index=False)
        except Exception as exc:
            print(f"Reference fetch failed ({exc}); displaying cached values or unavailable")
            for ref in refs.values():
                if ref["flt_spread"] is None:
                    ref["status"] = str(exc)
                if ref["mtg_collat_typ"] is None:
                    ref["collateral_status"] = str(exc)
    for field in ["flt_spread", "mtg_collat_typ", "underlying_cpn"]:
        print(f"{field}: {sum(r[field] is not None for r in refs.values())}/{len(tickers)} available")
    return refs


def load_data(cache_only: bool = False) -> pd.DataFrame:
    df = None if cache_only else load_bbg()
    if df is None:
        if not CACHE.exists():
            raise RuntimeError("No Bloomberg data and no cache available")
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        print(f"Loaded cache: {len(df)} rows, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
    if "vol_source" not in df or not df["vol_source"].eq(VOL_TICKER).all():
        vol = load_vol()
        if vol is None:
            raise RuntimeError("Cannot use legacy workbook vol; Bloomberg volatility is required")
        df["vol"] = vol.reindex(df.index)
        df["vol_raw"] = df["vol"]
        df["vol_source"] = VOL_TICKER
    required = ["CT10", "2s10s", "FN CC", "G2 CC", "vol"]
    missing = [c for c in required if c not in df]
    if missing:
        raise RuntimeError(f"Missing required predictors: {missing}")
    df = df.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    df = clean_vol(df)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


def regress(y: pd.Series, X: pd.DataFrame) -> dict:
    """Centered/scaled SVD least squares; avoids unstable normal equations."""
    cols = list(X.columns)
    means = X.mean()
    scales = X.std(ddof=0).replace(0, 1)
    design = np.column_stack([np.ones(len(y)), ((X-means)/scales).to_numpy()])
    beta, _, rank, singular = lstsq(design, y.to_numpy(dtype=float), lapack_driver="gelsd")
    if rank < len(cols)+1:
        raise ValueError("Predictors are collinear or constant in this sample")
    slopes = beta[1:] / scales.to_numpy()
    intercept = float(beta[0] - np.dot(slopes, means.to_numpy()))
    fit = design @ beta
    resid = y.values - fit
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y.values - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "a": intercept,
        "b": dict(zip(cols, slopes)),
        "fit": pd.Series(fit, index=y.index),
        "resid": pd.Series(resid, index=y.index),
        "r2": r2,
        "adj_r2": 1-(1-r2)*(len(y)-1)/(len(y)-len(cols)-1),
        "sigma": np.sqrt(ss_res/(len(y)-len(cols)-1)),
        "condition": float(singular[0]/singular[-1]),
        "n": len(y),
        "start": y.index[0], "end": y.index[-1],
    }


def analyze(df: pd.DataFrame, ticker: str, agency: str) -> dict:
    cc_col = f"{agency} CC"
    preds = ["CT10", "cc", "2s10s", "vol"]
    cols = [ticker, cc_col, "CT10", "2s10s"]
    if "vol" in df.columns:
        cols.append("vol")
    sub = df[cols].rename(columns={ticker: "px", cc_col: "cc"})
    sub = sub.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out = {"data": sub, "preds": preds, "cc_col": cc_col, "reg": None,
           "cur": None}
    if len(sub) < MIN_OBS:
        print(f"  {_label(ticker)}: only {len(sub)} obs - regression skipped")
        return out
    out.update({"models": {}, "validation": {}, "error": None})
    try:
        reg = regress(sub["px"], sub[preds])
    except ValueError as exc:
        out["error"] = str(exc)
        return out
    resid = reg["resid"]
    out["reg"] = reg
    out["cur"] = {
        "date": sub.index[-1],
        "px": sub["px"].iloc[-1],
        "fair": reg["fit"].iloc[-1],
        "resid": resid.iloc[-1],
        "resid_z": resid.iloc[-1] / reg["sigma"] if reg["sigma"] > 1e-12 else 0.0,
        "resid_pct": float((resid < resid.iloc[-1]).mean()),
        "resid_std": resid.std(),
        "n": len(sub),
        "start": sub.index[0],
    }
    out["models"]["full"] = reg
    # Each rolling fitted value uses coefficients from prior observations only.
    rolling_fit = pd.Series(np.nan, index=sub.index)
    rolling_sigma = rolling_fit.copy()
    baseline_fit = rolling_fit.copy()
    base_errors, enhanced_errors = [], []
    latest_rolling = None
    for i in range(VALIDATION_MIN, len(sub)):
        train = sub.iloc[max(0, i-ROLL_N):i]
        try:
            rolling = regress(train["px"], train[preds])
            baseline = regress(train["px"], train[["CT10", "cc"]])
        except ValueError:
            continue
        row = sub.iloc[i]
        predicted = rolling["a"] + sum(rolling["b"][p]*row[p] for p in preds)
        base = baseline["a"] + sum(baseline["b"][p]*row[p] for p in baseline["b"])
        rolling_fit.iloc[i] = predicted
        baseline_fit.iloc[i] = base
        rolling_sigma.iloc[i] = rolling["sigma"]
        enhanced_errors.append(row["px"]-predicted)
        base_errors.append(row["px"]-base)
        if i == len(sub)-1:
            latest_rolling = rolling
    if latest_rolling is not None:
        latest_rolling["fit"] = rolling_fit
        latest_rolling["resid"] = sub["px"]-rolling_fit
        latest_rolling["sigma_path"] = rolling_sigma
        out["models"]["rolling"] = latest_rolling
    if base_errors:
        base_rmse = float(np.sqrt(np.mean(np.square(base_errors)))*32)
        enhanced_rmse = float(np.sqrt(np.mean(np.square(enhanced_errors)))*32)
        out["validation"] = {"n": len(base_errors), "baseline_rmse": base_rmse,
                             "enhanced_rmse": enhanced_rmse,
                             "improvement": 100*(1-enhanced_rmse/base_rmse) if base_rmse else None}
    out["baseline_fit"] = baseline_fit
    return out


def _fmt_px(v: float) -> str:
    """Price as points+32nds string."""
    pts = int(v)
    ticks = round((v - pts) * 32)
    if ticks == 32:
        pts, ticks = pts + 1, 0
    return f"{pts}+{ticks:02d}/32"


def build_html(results: dict, out_path: Path) -> None:
    from cmo_caps_dashboard import build_dashboard
    portfolio = [b for b in BONDS if b[0] in PORTFOLIO_TICKERS]
    analysis = [b for b in BONDS if b[0] not in PORTFOLIO_TICKERS]
    build_dashboard(results, portfolio, out_path)
    build_dashboard(results, analysis,
                    OUTPUT_DIR / "adhoc_cmo_analysis.html",
                    summary_name="cmo_analysis_model_summary.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="store_true", help="Use cached Bloomberg history")
    parser.add_argument("--refresh-reference", action="store_true",
                        help="Refresh Bloomberg spread/collateral fields when using cached price history")
    args = parser.parse_args()
    df = load_data(cache_only=args.cache)
    reference = load_reference(refresh=not args.cache or args.refresh_reference)
    results = {}
    for number, (tk, cap, agency) in enumerate(BONDS, start=1):
        results[tk] = analyze(df, tk, agency)
        if number % 10 == 0:
            print(f"Model rebuild: {number}/{len(BONDS)} bonds complete", flush=True)
        results[tk]["price_history"] = pd.to_numeric(df[tk], errors="coerce").dropna()
        results[tk]["reference"] = reference[tk]
        results[tk]["group"] = ("portfolio" if tk in PORTFOLIO_TICKERS
                                else "analysis")

    for cap in CAP_ORDER:
        print(f"\n===== Cap {cap:g} =====")
        for tk, cp, agency in BONDS:
            if cp != cap:
                continue
            r = results[tk]
            if r["cur"] is None:
                print(f"{_label(tk):24s} n={len(r['data']):3d}  "
                      f"(insufficient history)")
                continue
            c = r["cur"]
            reg = r["reg"]
            flag = " SHORT-N" if c["n"] < SHORT_N else ""
            print(f"{_label(tk):24s} n={c['n']:3d} R2={reg['r2']:.2f} "
                  f"px {c['px']:7.3f} fv {c['fair']:7.3f} "
                  f"resid {c['resid']*32:+5.1f}t "
                  f"z {c['resid_z']:+5.2f} "
                  f"{c['resid_pct']*100:3.0f}pct "
                  f"{'CHEAP' if c['resid'] < 0 else 'RICH '}{flag}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    if not args.cache:
        from cmo_forward_indicator import refresh as refresh_indicator
        try:
            refresh_indicator()
        except Exception as exc:
            print(f"Standalone 1Y1Y indicator refresh failed ({exc}); keeping any cached history")
    build_html(results, OUTPUT_DIR / "adhoc_cmo_caps.html")


if __name__ == "__main__":
    main()
