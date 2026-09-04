"""
vcp_calc.py

Heuristic detector for Minervini-style Volatility Contraction Patterns (VCP).

A VCP is a series of price pullbacks that get progressively shallower --
"contractions" -- while trading volume also dries up, typically forming
within an existing uptrend and resolving in a breakout above the pattern's
high (the "pivot") on a surge in volume.

There is no single, universally-agreed algorithmic definition of a VCP --
traders normally identify it by eye on a chart. This module implements a
documented, reasonable approximation:

  1. Find swing highs/lows (local price pivots) over a lookback window.
  2. Reduce those to an alternating high-low zigzag.
  3. Walk backward from the most recent contraction, keeping contractions
     whose pullback % keeps shrinking -- that chain is the candidate VCP.
  4. Score the chain on number/tightness of contractions, volume dry-up,
     and proximity to the pivot (the most recent swing high in the chain).
  5. Require the stock to already be in an uptrend (a simplified version
     of Minervini's "Trend Template") before it counts as a candidate.

Treat this as a screening aid, not a certified pattern match -- always
look at the annotated chart before trusting a result.

No dependency on Streamlit/yfinance, so it can be unit tested standalone,
matching the style of momentum_calc.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import momentum_calc as mc

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Swing point / zigzag detection
# ---------------------------------------------------------------------------

def find_swing_points(close: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """
    A point is a swing high if it's the single highest close within
    `window` trading days on both sides of it (and similarly for a swing
    low). Returns two boolean Series aligned to `close`'s index.

    Note: the most recent `window` days can never register as a swing
    point, since there aren't enough days afterward yet to confirm one.
    """
    values = close.to_numpy()
    n = len(values)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)

    for i in range(window, n - window):
        seg = values[i - window : i + window + 1]
        center = values[i]
        if center == seg.max() and np.sum(seg == center) == 1:
            is_high[i] = True
        if center == seg.min() and np.sum(seg == center) == 1:
            is_low[i] = True

    return pd.Series(is_high, index=close.index), pd.Series(is_low, index=close.index)


def _zigzag(close: pd.Series, is_high: pd.Series, is_low: pd.Series) -> list[dict]:
    """
    Collapse raw swing points into an alternating high/low zigzag: when two
    same-type points occur back-to-back (no opposite-type point between
    them), keep only the more extreme one. Returns a list of
    {"pos": int, "price": float, "type": "high"|"low"} in chronological
    order, where `pos` is a position within `close` (0-indexed).
    """
    points = []
    for pos in range(len(close)):
        if is_high.iloc[pos]:
            points.append({"pos": pos, "price": float(close.iloc[pos]), "type": "high"})
        elif is_low.iloc[pos]:
            points.append({"pos": pos, "price": float(close.iloc[pos]), "type": "low"})

    zigzag: list[dict] = []
    for p in points:
        if zigzag and zigzag[-1]["type"] == p["type"]:
            if p["type"] == "high" and p["price"] >= zigzag[-1]["price"]:
                zigzag[-1] = p
            elif p["type"] == "low" and p["price"] <= zigzag[-1]["price"]:
                zigzag[-1] = p
            # otherwise the existing point is more extreme -- keep it, skip p
        else:
            zigzag.append(p)
    return zigzag


# ---------------------------------------------------------------------------
# Contraction chain detection
# ---------------------------------------------------------------------------

def detect_contractions(df: pd.DataFrame, window: int = 5, lookback_days: int = 180,
                          max_contractions: int = 6) -> dict:
    """
    df: OHLCV DataFrame for one stock, indexed by date ascending, with
        'Close' and 'Volume' columns.

    Returns a dict describing the most recent chain of shrinking pullbacks:
      - num_contractions: length of the qualifying chain (0 if none found)
      - contraction_depths: list of % pullback for each contraction, oldest first
      - pivot_price: the high to beat (most recent swing high in the chain)
      - last_price: latest close
      - pct_below_pivot: how far (%) below the pivot price sits right now
                         (negative means it has already closed above the pivot)
      - volume_dryup_ratio: recent 10-day avg volume / trailing 50-day avg volume
                            (below 1 = volume contracting, a good VCP sign)
      - breakout_volume_ratio: latest day's volume / trailing 50-day avg volume
      - is_breakout: last_price above pivot_price AND breakout_volume_ratio >= 1.3
      - swing_highs / swing_lows: {pos, price} points in the chain, for charting
        (pos is a position in the ORIGINAL df passed in, not the sliced window)
    """
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else None
    recent = close.iloc[-lookback_days:] if len(close) > lookback_days else close
    offset = len(close) - len(recent)

    empty = {
        "num_contractions": 0,
        "contraction_depths": [],
        "pivot_price": None,
        "last_price": float(close.iloc[-1]),
        "pct_below_pivot": None,
        "volume_dryup_ratio": None,
        "breakout_volume_ratio": None,
        "is_breakout": False,
        "swing_highs": [],
        "swing_lows": [],
    }

    if len(recent) < 2 * window + 2:
        return empty

    is_high, is_low = find_swing_points(recent, window=window)
    zigzag = _zigzag(recent, is_high, is_low)

    # Build (high, low) contraction pairs in chronological order.
    contractions = []
    for a, b in zip(zigzag, zigzag[1:]):
        if a["type"] == "high" and b["type"] == "low" and a["price"] > 0:
            depth_pct = (a["price"] - b["price"]) / a["price"] * 100.0
            contractions.append({"high": a, "low": b, "depth_pct": depth_pct})

    if not contractions:
        return empty

    # Walk backward from the most recent contraction, extending the chain
    # while depths keep shrinking (earlier contractions must be deeper).
    chain = [contractions[-1]]
    for c in reversed(contractions[:-1]):
        if c["depth_pct"] > chain[0]["depth_pct"]:
            chain.insert(0, c)
            if len(chain) >= max_contractions:
                break
        else:
            break

    if len(chain) < 2:
        return empty

    last_price = float(close.iloc[-1])
    pivot_price = chain[-1]["high"]["price"]
    pct_below_pivot = (pivot_price - last_price) / pivot_price * 100.0 if pivot_price else None

    volume_dryup_ratio = None
    breakout_volume_ratio = None
    if volume is not None and len(volume.dropna()) >= 50:
        avg50 = float(volume.rolling(50).mean().iloc[-1])
        if avg50 and avg50 > 0:
            avg10 = float(volume.rolling(10).mean().iloc[-1])
            volume_dryup_ratio = avg10 / avg50
            breakout_volume_ratio = float(volume.iloc[-1]) / avg50

    is_breakout = bool(
        pivot_price is not None
        and last_price > pivot_price
        and breakout_volume_ratio is not None
        and breakout_volume_ratio >= 1.3
    )

    return {
        "num_contractions": len(chain),
        "contraction_depths": [round(c["depth_pct"], 1) for c in chain],
        "pivot_price": pivot_price,
        "last_price": last_price,
        "pct_below_pivot": pct_below_pivot,
        "volume_dryup_ratio": volume_dryup_ratio,
        "breakout_volume_ratio": breakout_volume_ratio,
        "is_breakout": is_breakout,
        "swing_highs": [{"pos": c["high"]["pos"] + offset, "price": c["high"]["price"]} for c in chain],
        "swing_lows": [{"pos": c["low"]["pos"] + offset, "price": c["low"]["price"]} for c in chain],
    }


# ---------------------------------------------------------------------------
# Trend template (simplified Minervini "Stage 2 uptrend" gate)
# ---------------------------------------------------------------------------

def pct_above_52w_low(close: pd.Series) -> float | None:
    """How far (in %) the current price sits above its trailing 52-week low."""
    window = close.iloc[-TRADING_DAYS_PER_YEAR:]
    if window.empty:
        return None
    low = float(window.min())
    last = float(close.iloc[-1])
    if low == 0:
        return None
    return (last - low) / low * 100.0


def passes_trend_template(df: pd.DataFrame) -> bool:
    """
    Simplified version of Minervini's "Trend Template": price above its
    50- and 200-day moving averages with the 50-day above the 200-day
    (an established uptrend / golden cross), trading well clear of its
    52-week low, and not too extended below its 52-week high.
    """
    close = df["Close"]
    trend = mc.trend_and_volume(df)
    off_high = mc.pct_off_52w_high(close)
    above_low = pct_above_52w_low(close)

    if not (trend["above_50ma"] and trend["above_200ma"] and trend["golden_cross"]):
        return False
    if off_high is None or off_high > 25:
        return False
    if above_low is None or above_low < 25:
        return False
    return True


# ---------------------------------------------------------------------------
# Scoring + per-ticker / per-universe wrappers
# ---------------------------------------------------------------------------

def vcp_score(info: dict) -> float:
    """
    0-100 heuristic score for ranking VCP candidates against each other.
    Rewards more contractions (up to 4), a tight/shallow final contraction,
    volume drying up, and sitting close to (or just breaking out of) the pivot.
    """
    if info["num_contractions"] < 2:
        return 0.0

    contraction_score = min(info["num_contractions"], 4) / 4 * 100

    final_depth = info["contraction_depths"][-1]
    tightness_score = float(np.clip(100 - (final_depth / 15) * 100, 0, 100))

    if info["volume_dryup_ratio"] is None:
        volume_score = 50.0
    else:
        volume_score = float(np.clip((1.0 - info["volume_dryup_ratio"]) / 0.5 * 100, 0, 100))

    if info["is_breakout"]:
        proximity_score = 100.0
    elif info["pct_below_pivot"] is None:
        proximity_score = 50.0
    else:
        proximity_score = float(np.clip(100 - (info["pct_below_pivot"] / 15) * 100, 0, 100))

    return round(
        contraction_score * 0.25
        + tightness_score * 0.25
        + volume_score * 0.25
        + proximity_score * 0.25,
        1,
    )


def compute_vcp_factors(df: pd.DataFrame, window: int = 5, lookback_days: int = 180) -> dict:
    """Per-stock VCP factor bundle for the app layer. NaN/None-safe."""
    trend_ok = passes_trend_template(df)
    info = detect_contractions(df, window=window, lookback_days=lookback_days)
    score = vcp_score(info) if trend_ok else 0.0

    return {
        "last_price": info["last_price"],
        "trend_ok": trend_ok,
        "num_contractions": info["num_contractions"],
        "contraction_depths": info["contraction_depths"],
        "pivot_price": info["pivot_price"],
        "pct_below_pivot": info["pct_below_pivot"],
        "volume_dryup_ratio": info["volume_dryup_ratio"],
        "breakout_volume_ratio": info["breakout_volume_ratio"],
        "is_breakout": info["is_breakout"],
        "is_vcp_candidate": bool(trend_ok and info["num_contractions"] >= 2),
        "vcp_score": score,
        "swing_highs": info["swing_highs"],
        "swing_lows": info["swing_lows"],
    }


def compute_universe_vcp(price_data: dict[str, pd.DataFrame], bench_ticker: str,
                           window: int = 5, lookback_days: int = 180) -> pd.DataFrame:
    """
    price_data: {ticker: OHLCV DataFrame}, as produced by data_fetch.download_price_history.
    Returns one row per non-benchmark ticker with compute_vcp_factors() output
    plus a 'ticker' column. Unfiltered/unranked -- the app layer filters by
    is_vcp_candidate / num_contractions / is_breakout and sorts by vcp_score.
    """
    rows = []
    for ticker, df in price_data.items():
        if ticker == bench_ticker:
            continue
        try:
            factors = compute_vcp_factors(df, window=window, lookback_days=lookback_days)
        except Exception:
            continue
        factors["ticker"] = ticker
        rows.append(factors)
    return pd.DataFrame(rows)


def rank_vcp_candidates(rows: list[dict]) -> pd.DataFrame:
    """
    rows: list of per-ticker dicts, each = compute_vcp_factors(...) output
          plus a 'ticker' key (e.g. compute_universe_vcp(...).to_dict("records")).
    Returns only the valid candidates (is_vcp_candidate), sorted by
    vcp_score descending, with a 'rank' column added.
    """
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["is_vcp_candidate"]].copy()
    if out.empty:
        return out
    out = out.sort_values("vcp_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
