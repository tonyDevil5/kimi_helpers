"""
CTA Trend-Following Model
=========================
Inspired by Citi's CTA positioning report (Feb 2026).

Two engines are available:
    1. 'tsmom'  – Time-Series Momentum (past N-day return normalised by vol).
       This is the default and tends to be more robust on rates.
    2. 'ma_cross' – Dual EMA crossover + percentage/ATR bands.

Both produce the four Citi-style regimes:
    - "max" long
    - moderately long
    - moderately short
    - "max" short

And they compute the price levels that would trigger:
    * flip long  / flip short
    * reduce longs / reduce shorts

Run from the command line:
    python cta_model.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
MODE = "tsmom"          # 'tsmom' or 'ma_cross'
LOOKBACK = 60           # days for momentum lookback
VOL_WINDOW = 60         # days for realised-vol estimation

# Z-score thresholds for TSMOM regime mapping
Z_MAX_LONG = 1.0
Z_MOD_LONG = 0.25
Z_MOD_SHORT = -0.25
Z_MAX_SHORT = -1.0

# MA-cross parameters (only used when MODE='ma_cross')
MA_FAST = 20
MA_MEDIUM = 60
MA_SLOW = 120
BAND_TYPE = "pct"       # 'pct' or 'atr'
MAX_BAND_PCT = 0.008    # 0.8%
MAX_BAND_ATR = 2.0

# Transaction cost (one-way, in price terms)
TC_ONE_WAY = 0.04

Regime = Literal["max_long", "mod_long", "neutral", "mod_short", "max_short"]


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_bloomberg_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load a Bloomberg-style XLSX export.

    Expected shape (one ticker):
        rows 0-6 : headers (cusip, field, dates, etc.)
        row 7    : ticker label
        row 8+   : [ ..., date, price ]

    Returns a DataFrame with columns ['date', 'price', 'ticker'].
    """
    df_raw = pd.read_excel(filepath, sheet_name=0, header=None)

    ticker = "Unknown"
    mask_ticker = df_raw.iloc[:, 2] == "cusip"
    if mask_ticker.any():
        ticker = str(df_raw.loc[mask_ticker, 3].iloc[0])

    dates = pd.to_datetime(df_raw.iloc[:, 2], errors="coerce", dayfirst=False, format="mixed")
    first_data_idx = dates.first_valid_index()
    if first_data_idx is None:
        raise ValueError("Could not find date column in data file.")

    prices = pd.to_numeric(df_raw.iloc[:, 3], errors="coerce")

    out = pd.DataFrame(
        {
            "date": dates.loc[first_data_idx:].values,
            "price": prices.loc[first_data_idx:].values,
            "ticker": ticker,
        }
    ).dropna(subset=["date", "price"])

    out = out.sort_values("date").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr_from_close(close: pd.Series, period: int = 14) -> pd.Series:
    tr = close.diff().abs()
    return tr.ewm(span=period, adjust=False).mean()


def regime_weight(regime: Regime) -> int:
    return {
        "max_long": 2,
        "mod_long": 1,
        "neutral": 0,
        "mod_short": -1,
        "max_short": -2,
    }[regime]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class CTALevels:
    flip_long: float | None = None
    flip_short: float | None = None
    reduce_longs: float | None = None
    reduce_shorts: float | None = None


class CTATrendModel:
    """
    CTA trend-following model.

    Parameters
    ----------
    mode : {'tsmom', 'ma_cross'}
        'tsmom'  -> momentum normalised by realised vol (default).
        'ma_cross' -> dual EMA crossover with bands.
    """

    def __init__(
        self,
        mode: str = MODE,
        lookback: int = LOOKBACK,
        vol_window: int = VOL_WINDOW,
        ma_fast: int = MA_FAST,
        ma_medium: int = MA_MEDIUM,
        ma_slow: int = MA_SLOW,
        band_type: str = BAND_TYPE,
        max_band_pct: float = MAX_BAND_PCT,
        max_band_atr: float = MAX_BAND_ATR,
    ):
        self.mode = mode
        self.lookback = lookback
        self.vol_window = vol_window
        self.ma_fast = ma_fast
        self.ma_medium = ma_medium
        self.ma_slow = ma_slow
        self.band_type = band_type
        self.max_band_pct = max_band_pct
        self.max_band_atr = max_band_atr
        self.data: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Core signal computation
    # ------------------------------------------------------------------

    def compute_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.mode == "tsmom":
            df = self._compute_tsmom(df)
        elif self.mode == "ma_cross":
            df = self._compute_ma_cross(df)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.data = df
        return df

    def _compute_tsmom(self, df: pd.DataFrame) -> pd.DataFrame:
        returns = df["price"].pct_change()
        df["vol"] = returns.rolling(self.vol_window).std() * np.sqrt(252)
        df["momentum"] = df["price"] / df["price"].shift(self.lookback) - 1
        df["z"] = df["momentum"] / df["vol"]
        df["z"] = df["z"].replace([np.inf, -np.inf], np.nan)

        def _regime(z: float) -> Regime:
            if pd.isna(z):
                return "neutral"
            if z >= Z_MAX_LONG:
                return "max_long"
            if z >= Z_MOD_LONG:
                return "mod_long"
            if z > Z_MOD_SHORT:
                return "neutral"
            if z > Z_MAX_SHORT:
                return "mod_short"
            return "max_short"

        df["regime"] = df["z"].apply(_regime)
        df["position"] = df["regime"].apply(regime_weight)

        # For completeness, add MAs so reports are consistent
        df["ema_fast"] = _ema(df["price"], self.ma_fast)
        df["ema_medium"] = _ema(df["price"], self.ma_medium)
        df["ema_slow"] = _ema(df["price"], self.ma_slow)
        df["atr"] = _atr_from_close(df["price"], period=14)

        return df

    def _compute_ma_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ema_fast"] = _ema(df["price"], self.ma_fast)
        df["ema_medium"] = _ema(df["price"], self.ma_medium)
        df["ema_slow"] = _ema(df["price"], self.ma_slow)
        df["atr"] = _atr_from_close(df["price"], period=14)

        trend = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        if self.band_type == "pct":
            dist = (df["price"] - df["ema_fast"]) / df["ema_fast"]
            max_b = self.max_band_pct
        else:
            dist = (df["price"] - df["ema_fast"]) / df["atr"]
            max_b = self.max_band_atr

        dist = dist.replace([np.inf, -np.inf], np.nan)

        def _regime(t: int, d: float) -> Regime:
            if pd.isna(d):
                return "neutral"
            if t > 0:
                if d >= max_b:
                    return "max_long"
                if d >= 0.0:
                    return "mod_long"
                if d >= -max_b:
                    return "neutral"
                return "mod_short"
            else:
                if d <= -max_b:
                    return "max_short"
                if d <= 0.0:
                    return "mod_short"
                if d <= max_b:
                    return "neutral"
                return "mod_long"

        df["dist"] = dist
        df["regime"] = pd.Series([_regime(t, d) for t, d in zip(trend, dist)], index=df.index)
        df["position"] = df["regime"].apply(regime_weight)

        # Fill vol/momentum so reports are consistent
        returns = df["price"].pct_change()
        df["vol"] = returns.rolling(self.vol_window).std() * np.sqrt(252)
        df["momentum"] = df["price"] / df["price"].shift(self.lookback) - 1
        df["z"] = df["momentum"] / df["vol"]

        return df

    # ------------------------------------------------------------------
    # Levels calculator
    # ------------------------------------------------------------------

    def compute_levels(self, row: pd.Series | None = None) -> CTALevels:
        if self.data is None or self.data.empty:
            raise RuntimeError("Run compute_signals() first.")

        if row is None:
            row = self.data.iloc[-1]

        levels = CTALevels()

        if self.mode == "tsmom":
            # z = (P / P_lookback - 1) / vol
            # => P = P_lookback * (1 + z * vol)
            p_lookback = row["price"] / (1 + row["momentum"]) if pd.notna(row["momentum"]) and row["momentum"] != -1 else np.nan
            vol = row["vol"]
            if pd.notna(p_lookback) and pd.notna(vol) and vol > 0:
                levels.reduce_longs = p_lookback * (1 + Z_MAX_LONG * vol)
                levels.flip_long = p_lookback * (1 + Z_MOD_LONG * vol)
                levels.flip_short = p_lookback * (1 + Z_MOD_SHORT * vol)
                levels.reduce_shorts = p_lookback * (1 + Z_MAX_SHORT * vol)
        else:
            ema_fast = row["ema_fast"]
            atr = row["atr"]
            if self.band_type == "pct":
                levels.reduce_longs = ema_fast * (1 + self.max_band_pct)
                levels.reduce_shorts = ema_fast * (1 - self.max_band_pct)
            else:
                levels.reduce_longs = ema_fast + self.max_band_atr * atr
                levels.reduce_shorts = ema_fast - self.max_band_atr * atr
            levels.flip_long = ema_fast
            levels.flip_short = ema_fast

        return levels

    # ------------------------------------------------------------------
    # Simple backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        df: pd.DataFrame | None = None,
        tc: float = TC_ONE_WAY,
    ) -> pd.DataFrame:
        if df is None:
            df = self.data
        if df is None:
            raise RuntimeError("No data loaded.")

        df = df.copy()
        df["daily_ret"] = df["price"].pct_change()
        df["pos_lag"] = df["position"].shift(1).fillna(0)
        df["pnl"] = df["pos_lag"] * df["daily_ret"]

        pos_change = df["pos_lag"].diff().abs().fillna(0)
        df["tc"] = pos_change * tc / df["price"]
        df["pnl_tc"] = df["pnl"] - df["tc"]

        df["cumulative"] = df["pnl_tc"].cumsum()
        df["cumulative_gross"] = df["pnl"].cumsum()

        return df

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        if self.data is None or self.data.empty:
            raise RuntimeError("No data.")

        latest = self.data.iloc[-1]
        levels = self.compute_levels(latest)
        bt = self.backtest()

        total_ret = bt["pnl_tc"].sum()
        gross_ret = bt["pnl"].sum()
        ann_vol = bt["pnl_tc"].std() * np.sqrt(252)
        sharpe = (bt["pnl_tc"].mean() / bt["pnl_tc"].std()) * np.sqrt(252) if bt["pnl_tc"].std() > 0 else 0

        cum = bt["cumulative"]
        running_max = cum.cummax()
        drawdown = cum - running_max
        max_dd = drawdown.min()

        regime_counts = bt["regime"].value_counts().to_dict()

        return {
            "ticker": latest.get("ticker", "N/A"),
            "date": latest["date"],
            "price": latest["price"],
            "ema_fast": latest["ema_fast"],
            "ema_medium": latest["ema_medium"],
            "ema_slow": latest["ema_slow"],
            "regime": latest["regime"],
            "position": latest["position"],
            "levels": {
                "flip_long": levels.flip_long,
                "flip_short": levels.flip_short,
                "reduce_longs": levels.reduce_longs,
                "reduce_shorts": levels.reduce_shorts,
            },
            "backtest": {
                "total_return": total_ret,
                "gross_return": gross_ret,
                "ann_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "days_in_sample": len(bt),
            },
            "regime_distribution": regime_counts,
        }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(summary: dict) -> None:
    print("=" * 60)
    print("CTA MODEL REPORT")
    print("=" * 60)
    print(f"Ticker : {summary['ticker']}")
    print(f"Date   : {summary['date']}")
    print(f"Price  : {summary['price']:.4f}")
    print("-" * 60)
    print("MOVING AVERAGES")
    print(f"  EMA({MA_FAST})  : {summary['ema_fast']:.4f}")
    print(f"  EMA({MA_MEDIUM})  : {summary['ema_medium']:.4f}")
    print(f"  EMA({MA_SLOW}) : {summary['ema_slow']:.4f}")
    print("-" * 60)
    print("CURRENT POSITIONING")
    print(f"  Regime   : {summary['regime']}")
    print(f"  Position : {summary['position']}  (+2=max long, -2=max short)")
    print("-" * 60)
    print("KEY LEVELS")
    for name, val in summary["levels"].items():
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            print(f"  {name:15s}: {val:.4f}")
    print("-" * 60)
    print("BACKTEST STATS (in-sample)")
    bt = summary["backtest"]
    print(f"  Total Return (net)  : {bt['total_return']:.2%}")
    print(f"  Total Return (gross): {bt['gross_return']:.2%}")
    print(f"  Ann. Volatility     : {bt['ann_volatility']:.2%}")
    print(f"  Sharpe Ratio        : {bt['sharpe']:.2f}")
    print(f"  Max Drawdown        : {bt['max_drawdown']:.2%}")
    print(f"  Days in sample      : {bt['days_in_sample']}")
    print("-" * 60)
    print("REGIME DISTRIBUTION")
    for reg, cnt in summary["regime_distribution"].items():
        pct = cnt / bt["days_in_sample"] * 100
        print(f"  {reg:15s}: {cnt:4d} days ({pct:5.1f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    data_path = Path(__file__).parent / "data.xlsx"

    print(f"Loading data from {data_path} ...")
    df = load_bloomberg_data(data_path)
    print(f"Loaded {len(df)} rows for {df['ticker'].iloc[0]}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()

    model = CTATrendModel(
        mode=MODE,
        lookback=LOOKBACK,
        vol_window=VOL_WINDOW,
        ma_fast=MA_FAST,
        ma_medium=MA_MEDIUM,
        ma_slow=MA_SLOW,
        band_type=BAND_TYPE,
        max_band_pct=MAX_BAND_PCT,
        max_band_atr=MAX_BAND_ATR,
    )
    model.compute_signals(df)
    report = model.summary()
    print_report(report)

    out_path = Path(__file__).parent / "cta_output.csv"
    model.data.to_csv(out_path, index=False)
    print(f"\nDetailed signals saved to {out_path}")


if __name__ == "__main__":
    main()
