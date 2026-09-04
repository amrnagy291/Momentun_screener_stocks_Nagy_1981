"""
unusual_activity_calc.py

Flags stocks showing "unusual activity" on their most recent trading day:
an outsized single-day price move combined with trading volume well above
its typical recent level. This is a day-over-day "movers" / relative-volume
screen -- distinct from the multi-week momentum and VCP patterns the other
tabs look for.

No dependency on Streamlit/yfinance, so it can be unit tested standalone,
matching the style of momentum_calc.py and vcp_calc.py.
"""

from __future__ import annotations

import pandas as pd


def compute_unusual_activity_factors(df: pd.DataFrame, volume_lookback: int = 50) -> dict:
    """
    df: OHLCV DataFrame for one stock, indexed by date ascending, with
        'Close' and 'Volume' columns.
    volume_lookback: number of PRIOR trading days (excluding the most
        recent one) used as the "typical" volume baseline, so today's
        volume is compared against a baseline that doesn't include itself.

    Returns:
      - last_price: latest close
      - price_change_pct: % change from the prior close to the latest close
        (None if there's only one day of history)
      - last_volume: latest day's volume
      - avg_volume: average volume over the `volume_lookback` days before
        the latest one (None if not enough history)
      - relative_volume: last_volume / avg_volume (None if avg_volume is
        missing or zero) -- 1.0 = typical, 2.0 = twice the usual volume
      - unusual_score: abs(price_change_pct) * relative_volume, a simple
        combined ranking score (not a 0-100 scale -- just for sorting
        candidates against each other). 0 if either input is missing.
    """
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    last_price = float(close.iloc[-1])
    price_change_pct = None
    if len(close) >= 2:
        prev_close = float(close.iloc[-2])
        if prev_close:
            price_change_pct = (last_price / prev_close - 1.0) * 100.0

    last_volume = None
    avg_volume = None
    relative_volume = None
    if volume is not None and len(volume) >= 1:
        last_volume = float(volume.iloc[-1])
        baseline = volume.iloc[-(volume_lookback + 1) : -1]
        if len(baseline.dropna()) >= max(10, volume_lookback // 2):
            avg_volume = float(baseline.mean())
            if avg_volume:
                relative_volume = last_volume / avg_volume

    unusual_score = 0.0
    if price_change_pct is not None and relative_volume is not None:
        unusual_score = round(abs(price_change_pct) * relative_volume, 2)

    return {
        "last_price": last_price,
        "price_change_pct": price_change_pct,
        "last_volume": last_volume,
        "avg_volume": avg_volume,
        "relative_volume": relative_volume,
        "unusual_score": unusual_score,
    }


def compute_universe_unusual_activity(price_data: dict[str, pd.DataFrame], bench_ticker: str,
                                        volume_lookback: int = 50) -> pd.DataFrame:
    """
    price_data: {ticker: OHLCV DataFrame}, as produced by
    data_fetch.download_price_history. Returns one row per non-benchmark
    ticker with compute_unusual_activity_factors() output plus a 'ticker'
    column. Unfiltered -- the app layer applies the volume/price thresholds
    and direction filter.
    """
    rows = []
    for ticker, df in price_data.items():
        if ticker == bench_ticker:
            continue
        try:
            factors = compute_unusual_activity_factors(df, volume_lookback=volume_lookback)
        except Exception:
            continue
        factors["ticker"] = ticker
        rows.append(factors)
    return pd.DataFrame(rows)
