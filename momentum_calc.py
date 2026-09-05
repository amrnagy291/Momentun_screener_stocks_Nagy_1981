"""
momentum_calc.py

Pure calculation functions for the momentum stock screener.
Deliberately has NO dependency on Streamlit or yfinance, so this
module can be unit-tested on its own with any OHLCV DataFrame.

Expected input for most functions: a pandas DataFrame indexed by date
(ascending) with at least a 'Close' column, and often 'Volume'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _price_n_days_ago(close: pd.Series, n: int) -> float | None:
    """Return the closing price ~n trading days ago, or None if not enough history."""
    if len(close) <= n:
        return None
    return float(close.iloc[-1 - n])


def pct_return(close: pd.Series, lookback_days: int) -> float | None:
    """Simple percentage return over the last `lookback_days` trading days."""
    end = float(close.iloc[-1])
    start = _price_n_days_ago(close, lookback_days)
    if start is None or start == 0:
        return None
    return (end / start) - 1.0


# ---------------------------------------------------------------------------
# Factor 1: Price momentum, 12 months minus most recent 1 month ("12-1")
# ---------------------------------------------------------------------------

def momentum_12_1(close: pd.Series, days_per_year: int = TRADING_DAYS_PER_YEAR,
                   days_per_month: int = TRADING_DAYS_PER_MONTH) -> float | None:
    """
    Classic academic momentum factor: total return from 12 months ago to
    1 month ago, deliberately excluding the most recent month to avoid
    short-term mean-reversion noise.

    days_per_year/days_per_month default to the equity trading-day calendar
    (252/21). Pass e.g. 365/30 for an asset that trades every calendar day
    (crypto) so "12 months" and "1 month" still mean actual calendar time.
    """
    price_12m_ago = _price_n_days_ago(close, days_per_year)
    price_1m_ago = _price_n_days_ago(close, days_per_month)
    if price_12m_ago is None or price_1m_ago is None or price_12m_ago == 0:
        return None
    return (price_1m_ago / price_12m_ago) - 1.0


# ---------------------------------------------------------------------------
# Factor 2: Relative strength vs a benchmark (e.g. S&P 500 / SPY)
# ---------------------------------------------------------------------------

def relative_strength(stock_close: pd.Series, bench_close: pd.Series, lookback_days: int) -> float | None:
    """
    Stock's return over `lookback_days` minus the benchmark's return over
    the same period. Positive = stock is outperforming the market
    (relative momentum), not just moving up in absolute terms.
    """
    stock_ret = pct_return(stock_close, lookback_days)
    bench_ret = pct_return(bench_close, lookback_days)
    if stock_ret is None or bench_ret is None:
        return None
    return stock_ret - bench_ret


# ---------------------------------------------------------------------------
# Factor 3: Trend + volume confirmation
# ---------------------------------------------------------------------------

def moving_average(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return float(close.rolling(window).mean().iloc[-1])


def trend_and_volume(df: pd.DataFrame) -> dict:
    """
    df must have 'Close' and 'Volume' columns.

    Returns a dict of boolean/float signals:
      - above_50ma:  price is above its 50-day moving average
      - above_200ma: price is above its 200-day moving average
      - golden_cross: 50-day MA is above the 200-day MA (established uptrend)
      - volume_ratio: 20-day average volume / 100-day average volume
                       (>1 means recent participation is picking up)
    """
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None

    last_price = float(close.iloc[-1])
    ma50 = moving_average(close, 50)
    ma200 = moving_average(close, 200)

    above_50ma = bool(ma50 is not None and last_price > ma50)
    above_200ma = bool(ma200 is not None and last_price > ma200)
    golden_cross = bool(ma50 is not None and ma200 is not None and ma50 > ma200)

    volume_ratio = None
    if volume is not None and len(volume.dropna()) >= 100:
        recent_20 = float(volume.rolling(20).mean().iloc[-1])
        base_100 = float(volume.rolling(100).mean().iloc[-1])
        if base_100 and base_100 > 0:
            volume_ratio = recent_20 / base_100

    return {
        "above_50ma": above_50ma,
        "above_200ma": above_200ma,
        "golden_cross": golden_cross,
        "volume_ratio": volume_ratio,
    }


# ---------------------------------------------------------------------------
# Factor 4: Short-term technical signals (RSI, MACD, 52-week high proximity)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    last_avg_gain = avg_gain.iloc[-1]
    last_avg_loss = avg_loss.iloc[-1]

    if pd.isna(last_avg_gain) or pd.isna(last_avg_loss):
        return None
    if last_avg_loss == 0:
        return 100.0
    rs = last_avg_gain / last_avg_loss
    return float(100 - (100 / (1 + rs)))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    Returns the MACD line, signal line, histogram, and whether a bullish
    crossover (MACD crossing above signal) happened within the last 3 bars.
    """
    if len(close) < slow + signal:
        return {"macd_line": None, "signal_line": None, "histogram": None, "bullish_crossover": False}

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    diff = macd_line - signal_line
    # bullish if currently positive and was negative within the last 3 bars
    recent = diff.iloc[-4:]
    bullish_crossover = bool(diff.iloc[-1] > 0 and (recent.iloc[:-1] < 0).any())

    return {
        "macd_line": float(macd_line.iloc[-1]),
        "signal_line": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
        "bullish_crossover": bullish_crossover,
    }


def pct_off_52w_high(close: pd.Series, days_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """
    How far (in %) the current price is below its trailing 52-week high.
    0 = at the high. Momentum stocks tend to trade close to their highs.

    days_per_year defaults to the equity trading-day calendar (252); pass
    365 for an asset that trades every calendar day (crypto) so this is a
    genuine trailing 52 weeks rather than ~8-9 months of data.
    """
    window = close.iloc[-days_per_year:]
    if window.empty:
        return None
    high = float(window.max())
    last = float(close.iloc[-1])
    if high == 0:
        return None
    return (high - last) / high * 100.0


def rsi_score(value: float | None) -> float:
    """
    Map an RSI reading to a 0-100 'momentum quality' score.
    Sweet spot for sustainable momentum is roughly 55-75:
    strong trend, not yet extreme/overbought.
    """
    if value is None or pd.isna(value):
        return 50.0
    if 55 <= value <= 75:
        return 100.0
    if value < 55:
        # linearly scale up from 0 at RSI=30 to 100 at RSI=55
        return float(np.clip((value - 30) / (55 - 30) * 100, 0, 100))
    # value > 75: taper off, fully discounted by RSI=95 (overbought / exhaustion risk)
    return float(np.clip(100 - (value - 75) / (95 - 75) * 100, 0, 100))


# ---------------------------------------------------------------------------
# Per-stock factor bundle
# ---------------------------------------------------------------------------

def compute_all_factors(df: pd.DataFrame, bench_close: pd.Series,
                         days_per_year: int = TRADING_DAYS_PER_YEAR,
                         days_per_month: int = TRADING_DAYS_PER_MONTH) -> dict:
    """
    df: OHLCV DataFrame for one stock, indexed by date ascending, with
        'Close' and 'Volume' columns.
    bench_close: Close price series for the benchmark (e.g. SPY), same
        date index universe.
    days_per_year/days_per_month: default to the equity trading-day calendar
        (252/21). Pass 365/30 for an asset that trades every calendar day
        (crypto) so "12-1 month" and "52-week high" stay calendar-accurate
        instead of covering a much shorter span than intended.

    Returns a flat dict of raw factor values (NaN-safe: missing values are None).
    Cross-sectional ranking/weighting into a single composite score happens
    at the app layer once all tickers have been computed (see rank_universe
    below), because relative strength/percentile ranks only make sense
    compared across the whole scanned universe.
    """
    close = df["Close"]

    trend = trend_and_volume(df)
    macd_vals = macd(close)

    return {
        "last_price": float(close.iloc[-1]),
        "momentum_12_1": momentum_12_1(close, days_per_year, days_per_month),
        "rs_3m": relative_strength(close, bench_close, 3 * days_per_month),
        "rs_6m": relative_strength(close, bench_close, 6 * days_per_month),
        "rs_12m": relative_strength(close, bench_close, 12 * days_per_month),
        "above_50ma": trend["above_50ma"],
        "above_200ma": trend["above_200ma"],
        "golden_cross": trend["golden_cross"],
        "volume_ratio": trend["volume_ratio"],
        "rsi": rsi(close),
        "macd_bullish_crossover": macd_vals["bullish_crossover"],
        "pct_off_52w_high": pct_off_52w_high(close, days_per_year),
    }


# ---------------------------------------------------------------------------
# Cross-sectional ranking + composite score across the whole scanned universe
# ---------------------------------------------------------------------------

def _percentile_rank(series: pd.Series) -> pd.Series:
    """0-100 percentile rank, NaNs excluded from ranking and left as NaN -> filled with 50 (neutral)."""
    ranked = series.rank(pct=True) * 100
    return ranked.fillna(50.0)


def rank_universe(factors_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    factors_df: one row per ticker, columns matching compute_all_factors() output,
                 plus a 'ticker' column.
    weights: dict with keys 'price_momentum', 'relative_strength',
             'trend_volume', 'short_term_technical' (0-100, should sum to ~100
             but will be re-normalized if not).

    Returns factors_df with added *_score columns (0-100 each) and a final
    'momentum_score' column (0-100), sorted descending by momentum_score.
    """
    out = factors_df.copy()

    # --- Category 1: price momentum (12-1) ---
    out["score_price_momentum"] = _percentile_rank(out["momentum_12_1"])

    # --- Category 2: relative strength (average of 3/6/12m, each percentile-ranked first) ---
    rs_components = [
        _percentile_rank(out["rs_3m"]),
        _percentile_rank(out["rs_6m"]),
        _percentile_rank(out["rs_12m"]),
    ]
    out["score_relative_strength"] = pd.concat(rs_components, axis=1).mean(axis=1)

    # --- Category 3: trend + volume confirmation ---
    vol_pctile = _percentile_rank(out["volume_ratio"])
    trend_components = pd.concat(
        [
            out["above_50ma"].astype(float) * 100,
            out["above_200ma"].astype(float) * 100,
            out["golden_cross"].astype(float) * 100,
            vol_pctile,
        ],
        axis=1,
    )
    out["score_trend_volume"] = trend_components.mean(axis=1)

    # --- Category 4: short-term technical signals ---
    rsi_scores = out["rsi"].apply(rsi_score)
    macd_scores = out["macd_bullish_crossover"].astype(float) * 100
    # closer to the 52-week high is better -> invert pct_off_high before ranking
    high_proximity_pctile = _percentile_rank(-out["pct_off_52w_high"])
    technical_components = pd.concat([rsi_scores, macd_scores, high_proximity_pctile], axis=1)
    out["score_short_term_technical"] = technical_components.mean(axis=1)

    # --- Composite ---
    w = {
        "price_momentum": weights.get("price_momentum", 0),
        "relative_strength": weights.get("relative_strength", 0),
        "trend_volume": weights.get("trend_volume", 0),
        "short_term_technical": weights.get("short_term_technical", 0),
    }
    total_w = sum(w.values()) or 1.0
    w = {k: v / total_w for k, v in w.items()}

    out["momentum_score"] = (
        out["score_price_momentum"] * w["price_momentum"]
        + out["score_relative_strength"] * w["relative_strength"]
        + out["score_trend_volume"] * w["trend_volume"]
        + out["score_short_term_technical"] * w["short_term_technical"]
    )

    out = out.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
