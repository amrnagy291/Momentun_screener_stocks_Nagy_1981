"""
Quick sanity tests for vcp_calc.py using hand-constructed synthetic price
paths (not random walks -- a VCP is a specific shape, so the test builds
that shape deterministically and checks detection against it).
Run with: python3 test_vcp_calc.py
No network / yfinance / streamlit needed -- pure math check.
"""

import numpy as np
import pandas as pd

import vcp_calc as vc


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def _path(segments):
    """segments: list of (num_days, start_price, end_price), concatenated
    with linspace (endpoint excluded so segments connect smoothly)."""
    pieces = [np.linspace(start, end, days, endpoint=False) for days, start, end in segments]
    return np.concatenate(pieces)


def make_vcp_df(seed=1, breakout=False):
    """
    Builds: ~220 days of uptrend, then three progressively shallower
    pullback/recovery legs (~20% -> ~10% -> ~5%), tapering volume down
    each leg (volume 'drying up'). Optionally appends a breakout leg with
    a volume spike.
    """
    segments = [
        (220, 50.0, 180.0),   # Phase A: establishing uptrend
        (15, 180.0, 144.0),   # contraction 1 down (~20%)
        (15, 144.0, 182.0),   # bounce to a new marginal high
        (12, 182.0, 163.8),   # contraction 2 down (~10%)
        (12, 163.8, 184.0),   # bounce to a new marginal high
        (10, 184.0, 174.8),   # contraction 3 down (~5%)
        (10, 174.8, 182.0),   # tighten back up, just under the pivot
    ]
    if breakout:
        segments.append((5, 182.0, 196.0))  # breakout above the 184 pivot

    price = _path(segments)
    n = len(price)
    rng = np.random.default_rng(seed)
    price = price + rng.normal(0, 0.15, n)  # tiny noise so extrema are unique

    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    # Volume: high during the uptrend/first leg, tapering down each
    # subsequent leg (drying up), spiking on the breakout leg if present.
    vol_means = (
        [2_000_000] * 220
        + [1_800_000] * 30
        + [1_300_000] * 24
        + [800_000] * 20
    )
    if breakout:
        vol_means += [2_600_000] * 5
    vol_means = np.array(vol_means[:n], dtype=float)
    volume = vol_means * (1 + rng.normal(0, 0.08, n))
    volume = np.clip(volume, 50_000, None)

    df = pd.DataFrame({"Close": price, "Volume": volume}, index=idx)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] * 1.005
    df["Low"] = df["Close"] * 0.995
    return df


def make_no_pattern_df(seed=2):
    """A mild downtrend with noise -- should fail the trend template
    (no golden cross), so it should never register as a VCP candidate."""
    n = 300
    rng = np.random.default_rng(seed)
    rets = rng.normal(-0.0005, 0.015, n)
    price = 120.0 * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    df = pd.DataFrame({"Close": price, "Volume": volume}, index=idx)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    return df


def main():
    vcp_df = make_vcp_df(seed=1, breakout=False)
    breakout_df = make_vcp_df(seed=1, breakout=True)
    flat_df = make_no_pattern_df(seed=2)

    # --- trend template ---
    check("clean VCP setup passes the trend template", vc.passes_trend_template(vcp_df) is True)
    check("no-pattern downtrend fails the trend template", vc.passes_trend_template(flat_df) is False)

    # --- contraction detection on the clean (not-yet-broken-out) case ---
    info = vc.detect_contractions(vcp_df, window=5, lookback_days=180)
    check("finds at least 2 contractions", info["num_contractions"] >= 2)
    print(f"    contraction_depths={info['contraction_depths']}")
    depths = info["contraction_depths"]
    check("contraction depths are non-increasing", all(depths[i] >= depths[i + 1] for i in range(len(depths) - 1)))
    check("pivot price is near the pattern's highs (170-190)", info["pivot_price"] is not None and 170 <= info["pivot_price"] <= 190)
    check("not yet broken out", info["is_breakout"] is False)
    check("still sitting below the pivot (0-15%)", info["pct_below_pivot"] is not None and 0 <= info["pct_below_pivot"] <= 15)
    check("volume has dried up (ratio < 1)", info["volume_dryup_ratio"] is not None and info["volume_dryup_ratio"] < 1.0)
    print(f"    pivot={info['pivot_price']:.2f} pct_below_pivot={info['pct_below_pivot']:.2f} "
          f"volume_dryup_ratio={info['volume_dryup_ratio']:.2f}")

    # --- breakout case ---
    info_bo = vc.detect_contractions(breakout_df, window=5, lookback_days=180)
    check("breakout case is flagged as a breakout", info_bo["is_breakout"] is True)
    check("breakout case has a volume surge >= 1.3x", info_bo["breakout_volume_ratio"] >= 1.3)
    print(f"    breakout_volume_ratio={info_bo['breakout_volume_ratio']:.2f}")

    # --- scoring ---
    factors = vc.compute_vcp_factors(vcp_df)
    factors_bo = vc.compute_vcp_factors(breakout_df)
    factors_flat = vc.compute_vcp_factors(flat_df)
    check("clean VCP is a valid candidate", factors["is_vcp_candidate"] is True)
    check("breakout case is a valid candidate", factors_bo["is_vcp_candidate"] is True)
    check("no-pattern case is NOT a valid candidate", factors_flat["is_vcp_candidate"] is False)
    check("no-pattern case scores 0", factors_flat["vcp_score"] == 0.0)
    check("clean VCP scores meaningfully higher than no-pattern", factors["vcp_score"] > factors_flat["vcp_score"])
    print(f"    vcp_score clean={factors['vcp_score']} breakout={factors_bo['vcp_score']} flat={factors_flat['vcp_score']}")

    # --- full universe pipeline ---
    price_data = {
        "CLEAN": vcp_df,
        "BREAKOUT": breakout_df,
        "FLAT": flat_df,
        "SPY": make_vcp_df(seed=9, breakout=False),  # benchmark, should be excluded
    }
    universe = vc.compute_universe_vcp(price_data, bench_ticker="SPY")
    check("universe has one row per non-benchmark ticker", len(universe) == 3)
    ranked = vc.rank_vcp_candidates(universe.to_dict("records"))
    check("ranked output excludes FLAT", "FLAT" not in ranked["ticker"].tolist())
    check("ranked output includes CLEAN and BREAKOUT", set(["CLEAN", "BREAKOUT"]).issubset(set(ranked["ticker"])))
    print("\nRanked VCP universe:")
    print(ranked[["rank", "ticker", "vcp_score", "num_contractions", "is_breakout", "pct_below_pivot"]].to_string(index=False))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
