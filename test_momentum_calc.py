"""
Quick sanity tests for momentum_calc.py using synthetic price data.
Run with: python3 test_momentum_calc.py
No network / yfinance / streamlit needed -- pure math check.
"""

import numpy as np
import pandas as pd

import momentum_calc as mc


def make_series(days=500, drift=0.0006, vol=0.015, start=100.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, days)
    prices = start * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    return pd.Series(prices, index=idx)


def make_df(days=500, drift=0.0006, vol=0.015, start=100.0, seed=0, volume_drift=1.0):
    close = make_series(days, drift, vol, start, seed)
    rng = np.random.default_rng(seed + 1)
    volume = rng.integers(1_000_000, 3_000_000, days).astype(float)
    # inject a recent volume pickup for testing volume_ratio
    volume[-20:] *= volume_drift
    df = pd.DataFrame({"Close": close.values, "Volume": volume}, index=close.index)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    return df


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    strong = make_df(days=500, drift=0.0015, vol=0.012, seed=1, volume_drift=1.8)  # strong uptrend, volume picking up
    weak = make_df(days=500, drift=-0.0004, vol=0.02, seed=2, volume_drift=0.6)    # downtrend, volume drying up
    bench = make_series(days=500, drift=0.0003, vol=0.01, seed=3)  # benchmark, modest drift

    # --- momentum_12_1 ---
    m_strong = mc.momentum_12_1(strong["Close"])
    m_weak = mc.momentum_12_1(weak["Close"])
    check("momentum_12_1 returns a float for strong uptrend", isinstance(m_strong, float))
    check("strong uptrend momentum > weak/downtrend momentum", m_strong > m_weak)
    print(f"    momentum_12_1 strong={m_strong:.3f} weak={m_weak:.3f}")

    # --- relative strength ---
    rs_strong = mc.relative_strength(strong["Close"], bench, 63)
    rs_weak = mc.relative_strength(weak["Close"], bench, 63)
    check("relative_strength: strong > weak vs benchmark", rs_strong > rs_weak)
    print(f"    rs_3m strong={rs_strong:.3f} weak={rs_weak:.3f}")

    # --- trend and volume ---
    t_strong = mc.trend_and_volume(strong)
    t_weak = mc.trend_and_volume(weak)
    check("strong uptrend is above its 50MA", t_strong["above_50ma"] is True)
    check("strong uptrend is above its 200MA", t_strong["above_200ma"] is True)
    check("strong uptrend has golden cross", t_strong["golden_cross"] is True)
    check("weak downtrend volume_ratio < strong uptrend volume_ratio",
          t_weak["volume_ratio"] < t_strong["volume_ratio"])
    print(f"    volume_ratio strong={t_strong['volume_ratio']:.2f} weak={t_weak['volume_ratio']:.2f}")

    # --- RSI ---
    rsi_strong = mc.rsi(strong["Close"])
    rsi_weak = mc.rsi(weak["Close"])
    check("RSI is between 0 and 100 (strong)", 0 <= rsi_strong <= 100)
    check("RSI is between 0 and 100 (weak)", 0 <= rsi_weak <= 100)
    check("strong uptrend RSI > weak downtrend RSI", rsi_strong > rsi_weak)
    print(f"    RSI strong={rsi_strong:.1f} weak={rsi_weak:.1f}")

    # --- MACD ---
    macd_strong = mc.macd(strong["Close"])
    check("MACD dict has expected keys", set(macd_strong.keys()) == {"macd_line", "signal_line", "histogram", "bullish_crossover"})
    check("MACD line is a float", isinstance(macd_strong["macd_line"], float))

    # --- 52-week high proximity ---
    off_high_strong = mc.pct_off_52w_high(strong["Close"])
    off_high_weak = mc.pct_off_52w_high(weak["Close"])
    check("pct_off_52w_high >= 0", off_high_strong >= 0 and off_high_weak >= 0)
    check("strong uptrend is closer to its 52w high than weak downtrend", off_high_strong < off_high_weak)
    print(f"    pct_off_52w_high strong={off_high_strong:.2f} weak={off_high_weak:.2f}")

    # --- rsi_score shape ---
    check("rsi_score sweet spot = 100", mc.rsi_score(65) == 100.0)
    check("rsi_score very overbought is discounted", mc.rsi_score(95) < mc.rsi_score(65))
    check("rsi_score very weak is discounted", mc.rsi_score(20) < mc.rsi_score(65))
    check("rsi_score handles None", mc.rsi_score(None) == 50.0)

    # --- calendar-aware day-counts (used for crypto's 365-day-a-year calendar) ---
    m12_1_stock_calendar = mc.momentum_12_1(strong["Close"])
    m12_1_crypto_calendar = mc.momentum_12_1(strong["Close"], days_per_year=365, days_per_month=30)
    check("momentum_12_1 with crypto day-counts differs from the stock default",
          m12_1_stock_calendar != m12_1_crypto_calendar)

    # A "hump" fixture where the true peak sits ~300 days before the last row:
    # the default 252-day window misses it, but a 365-day (crypto-calendar)
    # window should catch it, since it covers more calendar time per row when
    # the asset trades every day instead of just weekdays.
    hump_prices = np.concatenate([
        np.linspace(100, 250, 100, endpoint=False),  # rises to a peak of 250 at day 99
        np.linspace(250, 150, 300),                  # then declines for the rest of the series
    ])
    hump_close = pd.Series(hump_prices, index=pd.date_range("2023-01-01", periods=400, freq="B"))
    off_high_252 = mc.pct_off_52w_high(hump_close)
    off_high_365 = mc.pct_off_52w_high(hump_close, days_per_year=365)
    check("pct_off_52w_high with a 365-day window catches an older, higher peak the 252-day window misses",
          off_high_365 > off_high_252)
    print(f"    pct_off_52w_high (hump fixture) 252-day={off_high_252:.1f}% 365-day={off_high_365:.1f}%")

    factors_crypto_calendar = mc.compute_all_factors(strong, bench, days_per_year=365, days_per_month=30)
    check("compute_all_factors accepts crypto day-counts and still returns a momentum value",
          factors_crypto_calendar["momentum_12_1"] is not None)

    # --- full pipeline across a small universe ---
    universe = {
        "STRONG": strong,
        "WEAK": weak,
        "MID": make_df(days=500, drift=0.0002, vol=0.014, seed=4, volume_drift=1.0),
    }
    rows = []
    for ticker, df in universe.items():
        factors = mc.compute_all_factors(df, bench)
        factors["ticker"] = ticker
        rows.append(factors)
    factors_df = pd.DataFrame(rows)

    weights = {
        "price_momentum": 30,
        "relative_strength": 30,
        "trend_volume": 25,
        "short_term_technical": 15,
    }
    ranked = mc.rank_universe(factors_df, weights)
    check("rank_universe returns one row per ticker", len(ranked) == 3)
    check("momentum_score column exists and is numeric", pd.api.types.is_numeric_dtype(ranked["momentum_score"]))
    check("STRONG ranks above WEAK", ranked.loc[ranked.ticker == "STRONG", "rank"].iloc[0] <
                                      ranked.loc[ranked.ticker == "WEAK", "rank"].iloc[0])
    print("\nRanked universe:")
    print(ranked[["rank", "ticker", "momentum_score", "score_price_momentum",
                  "score_relative_strength", "score_trend_volume", "score_short_term_technical"]]
          .to_string(index=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
