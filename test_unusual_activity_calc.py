"""
Quick sanity tests for unusual_activity_calc.py using synthetic price data.
Run with: python3 test_unusual_activity_calc.py
No network / yfinance / streamlit needed -- pure math check.
"""

import numpy as np
import pandas as pd

import unusual_activity_calc as ua


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def make_df(days=90, seed=0, spike_pct=0.0, spike_volume_mult=1.0):
    """Ordinary noisy random-walk price/volume, with the LAST day optionally
    overridden to simulate a one-day price spike and/or volume spike."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, days)
    price = 100.0 * np.cumprod(1 + rets)
    volume = rng.integers(500_000, 1_500_000, days).astype(float)

    if spike_pct:
        price[-1] = price[-2] * (1 + spike_pct)
    if spike_volume_mult != 1.0:
        volume[-1] = volume[-51:-1].mean() * spike_volume_mult

    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    df = pd.DataFrame({"Close": price, "Volume": volume}, index=idx)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    return df


def main():
    normal = make_df(seed=1)
    gainer = make_df(seed=2, spike_pct=0.08, spike_volume_mult=3.0)   # +8% on 3x volume
    loser = make_df(seed=3, spike_pct=-0.09, spike_volume_mult=2.5)   # -9% on 2.5x volume

    f_normal = ua.compute_unusual_activity_factors(normal)
    f_gainer = ua.compute_unusual_activity_factors(gainer)
    f_loser = ua.compute_unusual_activity_factors(loser)

    check("normal day has a small price change (< 5%)", abs(f_normal["price_change_pct"]) < 5)
    check("normal day has roughly typical relative volume", 0.3 < f_normal["relative_volume"] < 3.0)
    check("normal day has a low unusual_score", f_normal["unusual_score"] < 10)

    check("gainer shows > 5% price change", f_gainer["price_change_pct"] > 5)
    check("gainer shows elevated relative volume (> 2x)", f_gainer["relative_volume"] > 2.0)

    check("loser shows < -5% price change", f_loser["price_change_pct"] < -5)
    check("loser shows elevated relative volume (> 2x)", f_loser["relative_volume"] > 2.0)

    check("gainer's unusual_score is much higher than normal's", f_gainer["unusual_score"] > f_normal["unusual_score"] * 3)
    check("loser's unusual_score is much higher than normal's", f_loser["unusual_score"] > f_normal["unusual_score"] * 3)

    print(f"    normal: change={f_normal['price_change_pct']:+.2f}% rel_vol={f_normal['relative_volume']:.2f}x score={f_normal['unusual_score']}")
    print(f"    gainer: change={f_gainer['price_change_pct']:+.2f}% rel_vol={f_gainer['relative_volume']:.2f}x score={f_gainer['unusual_score']}")
    print(f"    loser:  change={f_loser['price_change_pct']:+.2f}% rel_vol={f_loser['relative_volume']:.2f}x score={f_loser['unusual_score']}")

    # --- universe pipeline ---
    price_data = {"NORMAL": normal, "GAINER": gainer, "LOSER": loser, "SPY": make_df(seed=9)}
    universe = ua.compute_universe_unusual_activity(price_data, bench_ticker="SPY")
    check("universe has one row per non-benchmark ticker", len(universe) == 3)
    check("SPY excluded from universe", "SPY" not in universe["ticker"].tolist())
    check("universe rows contain the expected columns", {"price_change_pct", "relative_volume", "unusual_score"}.issubset(universe.columns))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
