"""
Momentum Stock Screener -- Streamlit dashboard

Scans the S&P 500 (or the full US stock market, or a curated list of major
cryptocurrencies) and ranks them by a composite momentum score built from
four factor categories:
  1. Price momentum (12 months of return, excluding the most recent month)
  2. Relative strength vs a benchmark (3/6/12 month outperformance)
  3. Trend + volume confirmation (above 50/200-day MAs, golden cross, volume)
  4. Short-term technical signals (RSI, MACD crossover, 52-week high proximity)

...plus a VCP (Volatility Contraction Pattern) scanner and an Unusual
Activity (price/volume spike) scanner, each available for both stocks and
crypto.

Run it with:
    streamlit run app.py

This is an educational/research tool, NOT financial advice. Momentum
strategies carry real risk, including sharp reversals ("momentum crashes").
Always do your own research before trading on this or any screener's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_fetch
import momentum_calc as mc
import vcp_calc as vc
import unusual_activity_calc as ua

st.set_page_config(page_title="Momentum Stock Screener", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_tickers() -> list[str]:
    return data_fetch.get_sp500_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_us_tickers() -> list[str]:
    return data_fetch.get_all_us_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def load_crypto_tickers() -> list[str]:
    return data_fetch.get_crypto_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_factors(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bench_close = price_data[data_fetch.BENCHMARK_TICKER]["Close"]
    rows = []
    for ticker, df in price_data.items():
        if ticker == data_fetch.BENCHMARK_TICKER:
            continue
        try:
            factors = mc.compute_all_factors(df, bench_close)
        except Exception:
            continue
        factors["ticker"] = ticker
        rows.append(factors)
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_crypto_factors(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Crypto trades every calendar day, not just weekdays, so "12 months" /
    # "1 month" / "52 weeks" need a 365/30-day calendar instead of the
    # 252/21 trading-day calendar used for stocks -- otherwise these lookback
    # windows would only cover ~70% of the intended calendar time.
    bench_close = price_data[data_fetch.CRYPTO_BENCHMARK_TICKER]["Close"]
    rows = []
    for ticker, df in price_data.items():
        if ticker == data_fetch.CRYPTO_BENCHMARK_TICKER:
            continue
        try:
            factors = mc.compute_all_factors(df, bench_close, days_per_year=365, days_per_month=30)
        except Exception:
            continue
        factors["ticker"] = ticker
        rows.append(factors)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_vcp(price_data: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    return vc.compute_universe_vcp(price_data, data_fetch.BENCHMARK_TICKER, window=window)


@st.cache_data(show_spinner=False)
def compute_crypto_vcp(price_data: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    return vc.compute_universe_vcp(
        price_data, data_fetch.CRYPTO_BENCHMARK_TICKER, window=window, days_per_year=365
    )


@st.cache_data(show_spinner=False)
def compute_unusual(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ua.compute_universe_unusual_activity(price_data, data_fetch.BENCHMARK_TICKER)


@st.cache_data(show_spinner=False)
def compute_crypto_unusual(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ua.compute_universe_unusual_activity(price_data, data_fetch.CRYPTO_BENCHMARK_TICKER)


# ---------------------------------------------------------------------------
# Shared rendering helpers (used by both the stock tabs and the crypto tab,
# so the three screeners stay in sync instead of drifting apart as two
# copies of the same logic).
# ---------------------------------------------------------------------------

def render_momentum_results(*, factors_df, price_data, weights, total_downloaded,
                             benchmark_label, key_prefix, min_price, require_uptrend, top_n,
                             asset_noun="stocks", asset_noun_singular="stock"):
    ranked = mc.rank_universe(factors_df, weights)

    filtered = ranked[ranked["last_price"] >= min_price]
    if require_uptrend:
        filtered = filtered[filtered["above_200ma"]]
    filtered = filtered.head(top_n)

    st.caption(
        f"Scanned {len(factors_df)} {asset_noun} · {total_downloaded} "
        f"total tickers downloaded · showing top {len(filtered)} after filters"
    )

    if filtered.empty:
        st.warning("Nothing matches the current filters. Try lowering the minimum price or unchecking the uptrend filter.")
        return

    display_cols = {
        "rank": "Rank",
        "ticker": "Ticker",
        "last_price": "Price",
        "momentum_score": "Momentum Score",
        "score_price_momentum": "Price Mom.",
        "score_relative_strength": "Rel. Strength",
        "score_trend_volume": "Trend/Vol",
        "score_short_term_technical": "Technicals",
        "momentum_12_1": "12-1mo Return",
        "rs_3m": "RS 3mo",
        "pct_off_52w_high": "% Off 52w High",
        "rsi": "RSI",
    }
    table = filtered[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        table.style.format(
            {
                "Price": "${:.2f}",
                "Momentum Score": "{:.1f}",
                "Price Mom.": "{:.1f}",
                "Rel. Strength": "{:.1f}",
                "Trend/Vol": "{:.1f}",
                "Technicals": "{:.1f}",
                "12-1mo Return": "{:+.1%}",
                "RS 3mo": "{:+.1%}",
                "% Off 52w High": "{:.1f}%",
                "RSI": "{:.0f}",
            },
            na_rep="—",
        ).background_gradient(subset=["Momentum Score"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
        height=min(700, 45 * (len(table) + 1)),
    )

    st.subheader(f"Inspect a {asset_noun_singular}")
    selected = st.selectbox(
        "Pick a ticker from the ranked list above", filtered["ticker"].tolist(),
        key=f"{key_prefix}_momentum_select",
    )

    if selected:
        row = filtered[filtered["ticker"] == selected].iloc[0]
        df = price_data[selected]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Momentum Score", f"{row['momentum_score']:.1f} / 100")
        col2.metric("12-1mo Return", f"{row['momentum_12_1']:+.1%}" if pd.notna(row["momentum_12_1"]) else "—")
        col3.metric("RSI (14)", f"{row['rsi']:.0f}" if pd.notna(row["rsi"]) else "—")
        col4.metric("% Off 52w High", f"{row['pct_off_52w_high']:.1f}%" if pd.notna(row["pct_off_52w_high"]) else "—")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#2E7D32")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="50-day MA",
                                  line=dict(color="#1976D2", dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(200).mean(), name="200-day MA",
                                  line=dict(color="#E65100", dash="dot")))
        fig.update_layout(
            title=f"{selected} — price with 50/200-day moving averages",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How the Momentum Score is built"):
        st.markdown(
            f"""
Each {asset_noun_singular} gets a **percentile rank (0-100) within the scanned universe** for every
underlying signal, then those are blended into four category scores, which are combined using
your sidebar weights into the final **Momentum Score**:

- **Price momentum**: 12-month return excluding the most recent month (the standard academic
  momentum factor — skipping the last month avoids short-term reversal noise).
- **Relative strength**: average of how much it has out/underperformed {benchmark_label} over
  3, 6, and 12 months.
- **Trend + volume**: whether price is above its 50-day and 200-day moving averages, whether
  the 50-day MA is above the 200-day MA (a "golden cross" / established uptrend), and whether
  recent (20-day) volume is elevated versus the trailing 100-day average.
- **Short-term technicals**: RSI (favoring the 55–75 "strong but not overbought" zone), a
  recent bullish MACD crossover, and closeness to the 52-week high.

A score of 100 in a category means "best in the scanned universe on that dimension right now" —
scores are relative to what else was scanned, not absolute thresholds.
            """
        )


def render_vcp_results(*, price_data, bench_ticker, days_per_year, key_prefix,
                        asset_noun="stocks", asset_noun_singular="stock"):
    st.write(
        "Looks for assets tracing out a **Volatility Contraction Pattern (VCP)** — a series "
        "of pullbacks that get progressively shallower (and quieter, on volume) as price "
        "coils beneath a resistance level (the **pivot**), often resolving in a breakout. "
        "This is a heuristic approximation, not a certified pattern match — always look at "
        "the chart before trusting a result."
    )

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    min_contractions = ctrl1.slider(
        "Minimum contractions required", 2, 4, 2, key=f"{key_prefix}_vcp_min_contractions",
        help="A classic VCP usually has 2-4 (sometimes more) progressively smaller pullbacks.",
    )
    breakouts_only = ctrl2.checkbox(
        "Show breakout candidates only", value=False, key=f"{key_prefix}_vcp_breakouts_only"
    )
    swing_window = ctrl3.slider(
        "Swing sensitivity (days each side)", 3, 10, 5, key=f"{key_prefix}_vcp_swing_window",
        help="Smaller = picks up minor wiggles as contractions. Larger = only major swings count.",
    )

    with st.spinner("Scanning for VCP setups..."):
        if days_per_year == mc.TRADING_DAYS_PER_YEAR:
            vcp_all = compute_vcp(price_data, swing_window)
        else:
            vcp_all = compute_crypto_vcp(price_data, swing_window)

    if vcp_all.empty:
        st.warning(f"No {asset_noun} had enough history to scan for VCP setups.")
        return

    vcp_candidates = vcp_all[
        vcp_all["is_vcp_candidate"] & (vcp_all["num_contractions"] >= min_contractions)
    ].copy()
    if breakouts_only:
        vcp_candidates = vcp_candidates[vcp_candidates["is_breakout"]]
    vcp_candidates = vcp_candidates.sort_values("vcp_score", ascending=False).reset_index(drop=True)
    if not vcp_candidates.empty:
        vcp_candidates.insert(0, "rank", np.arange(1, len(vcp_candidates) + 1))

    st.caption(
        f"Scanned {len(vcp_all)} {asset_noun} · {int(vcp_all['is_vcp_candidate'].sum())} showing at least "
        f"2 shrinking contractions in an uptrend · {len(vcp_candidates)} match current filters"
    )

    if vcp_candidates.empty:
        st.warning(
            "No VCP candidates match the current filters. Try lowering the minimum "
            "contractions, unchecking 'breakouts only', or widening the universe size."
        )
    else:
        vcp_display_cols = {
            "rank": "Rank",
            "ticker": "Ticker",
            "last_price": "Price",
            "vcp_score": "VCP Score",
            "num_contractions": "# Contractions",
            "pivot_price": "Pivot",
            "pct_below_pivot": "% Below Pivot",
            "volume_dryup_ratio": "Vol Dry-up",
            "is_breakout": "Breakout",
        }
        vcp_table = vcp_candidates[list(vcp_display_cols.keys())].rename(columns=vcp_display_cols)
        vcp_table["Breakout"] = vcp_table["Breakout"].map({True: "🚀 Yes", False: "—"})
        vcp_table["Contraction Depths"] = vcp_candidates["contraction_depths"].apply(
            lambda ds: " → ".join(f"{d:.0f}%" for d in ds) if ds else "—"
        )

        st.dataframe(
            vcp_table.style.format(
                {
                    "Price": "${:.2f}",
                    "VCP Score": "{:.1f}",
                    "Pivot": "${:.2f}",
                    "% Below Pivot": "{:+.1f}%",
                    "Vol Dry-up": "{:.2f}x",
                },
                na_rep="—",
            ).background_gradient(subset=["VCP Score"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
            height=min(700, 45 * (len(vcp_table) + 1)),
        )

        st.subheader(f"Inspect a candidate")
        vcp_selected = st.selectbox(
            "Pick a ticker from the VCP list above", vcp_candidates["ticker"].tolist(),
            key=f"{key_prefix}_vcp_select",
        )

        if vcp_selected:
            row = vcp_candidates[vcp_candidates["ticker"] == vcp_selected].iloc[0]
            df = price_data[vcp_selected]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VCP Score", f"{row['vcp_score']:.1f} / 100")
            depths_str = " → ".join(f"{d:.0f}%" for d in row["contraction_depths"])
            c2.metric("Contractions", f"{row['num_contractions']}", help=depths_str)
            c3.metric("Pivot", f"${row['pivot_price']:.2f}" if pd.notna(row["pivot_price"]) else "—")
            c4.metric(
                "% Below Pivot" if not row["is_breakout"] else "% Above Pivot",
                f"{abs(row['pct_below_pivot']):.1f}%" if pd.notna(row["pct_below_pivot"]) else "—",
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#2E7D32")))
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="50-day MA",
                                      line=dict(color="#1976D2", dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(200).mean(), name="200-day MA",
                                      line=dict(color="#E65100", dash="dot")))

            swing_highs = row["swing_highs"]
            swing_lows = row["swing_lows"]
            if swing_highs:
                fig.add_trace(go.Scatter(
                    x=[df.index[p["pos"]] for p in swing_highs],
                    y=[p["price"] for p in swing_highs],
                    mode="markers", name="Swing high",
                    marker=dict(symbol="triangle-down", size=12, color="#EF5350"),
                ))
            if swing_lows:
                fig.add_trace(go.Scatter(
                    x=[df.index[p["pos"]] for p in swing_lows],
                    y=[p["price"] for p in swing_lows],
                    mode="markers", name="Swing low",
                    marker=dict(symbol="triangle-up", size=12, color="#66BB6A"),
                ))
            if pd.notna(row["pivot_price"]):
                fig.add_hline(y=row["pivot_price"], line_dash="dash", line_color="#FFB300",
                              annotation_text="Pivot", annotation_position="top left")

            fig.update_layout(
                title=f"{vcp_selected} — detected contractions and pivot",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("How VCP detection works"):
        st.markdown(
            """
This scanner looks for assets already in an uptrend (a simplified version of the "Trend
Template": above their 50- and 200-day moving averages with a golden cross, not too far below
their 52-week high, and well clear of their 52-week low), then finds their most recent chain of
price pullbacks that keep getting **shallower** — the hallmark of a Volatility Contraction
Pattern.

- **Contractions**: each pullback's depth (the % drop from a swing high to the following swing
  low). A valid chain needs at least 2, each one shallower than the last.
- **Pivot**: the high just before the tightest, most recent pullback — the level a breakout
  needs to clear.
- **Vol Dry-up**: recent (10-day) average volume divided by the trailing 50-day average.
  Below 1.0 means volume is contracting along with price, the classic VCP "quieting down" sign.
- **Breakout**: flagged when price has closed above the pivot on volume at least 30% above its
  50-day average.

There's no single agreed-upon algorithmic definition of a VCP — this is a reasonable
approximation, not a certified pattern match. Always check the annotated chart (swing highs are
red markers, swing lows are green, the pivot is the dashed line) before trusting a result, and
treat this as a starting point for further research, not a trading signal.
            """
        )


def render_unusual_results(*, price_data, bench_ticker, key_prefix, asset_noun="stocks"):
    st.write(
        "Flags assets with an outsized **single-day price move combined with volume well "
        "above their typical level** — the classic 'something's going on here' signal. Based "
        "on the most recent trading day only (not the multi-week patterns the other screeners "
        "look for)."
    )

    uctrl1, uctrl2, uctrl3 = st.columns(3)
    min_price_change = uctrl1.slider(
        "Minimum price change (%)", 1, 20, 5, key=f"{key_prefix}_unusual_min_change",
        help="Flags days where the price moved at least this much, up or down.",
    )
    min_rel_volume = uctrl2.slider(
        "Minimum relative volume", 1.0, 5.0, 2.0, step=0.1, key=f"{key_prefix}_unusual_min_relvol",
        help="Today's volume divided by its trailing 50-day average. 2.0x = twice the usual volume.",
    )
    direction = uctrl3.selectbox(
        "Direction", ["Both", "Gainers only", "Losers only"], key=f"{key_prefix}_unusual_direction"
    )

    with st.spinner("Scanning for unusual volume/price activity..."):
        if bench_ticker == data_fetch.BENCHMARK_TICKER:
            unusual_all = compute_unusual(price_data)
        else:
            unusual_all = compute_crypto_unusual(price_data)

    if unusual_all.empty:
        st.warning(f"No {asset_noun} had enough history to scan for unusual activity.")
        return

    valid = unusual_all[unusual_all["price_change_pct"].notna() & unusual_all["relative_volume"].notna()]
    candidates = valid[valid["relative_volume"] >= min_rel_volume]
    if direction == "Gainers only":
        candidates = candidates[candidates["price_change_pct"] >= min_price_change]
    elif direction == "Losers only":
        candidates = candidates[candidates["price_change_pct"] <= -min_price_change]
    else:
        candidates = candidates[candidates["price_change_pct"].abs() >= min_price_change]

    candidates = candidates.sort_values("unusual_score", ascending=False).reset_index(drop=True)
    if not candidates.empty:
        candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))

    st.caption(f"Scanned {len(unusual_all)} {asset_noun} · {len(candidates)} match current filters")

    if candidates.empty:
        st.warning(
            "No assets match the current filters. Try lowering the minimum price change "
            "or relative volume, or widening the universe size."
        )
        return

    unusual_display_cols = {
        "rank": "Rank",
        "ticker": "Ticker",
        "last_price": "Price",
        "price_change_pct": "Price Change %",
        "relative_volume": "Relative Volume",
        "last_volume": "Volume",
        "unusual_score": "Unusual Score",
    }
    unusual_table = candidates[list(unusual_display_cols.keys())].rename(columns=unusual_display_cols)

    st.dataframe(
        unusual_table.style.format(
            {
                "Price": "${:.2f}",
                "Price Change %": "{:+.1f}%",
                "Relative Volume": "{:.2f}x",
                "Volume": "{:,.0f}",
                "Unusual Score": "{:.1f}",
            },
            na_rep="—",
        ).background_gradient(subset=["Unusual Score"], cmap="Oranges"),
        use_container_width=True,
        hide_index=True,
        height=min(700, 45 * (len(unusual_table) + 1)),
    )

    st.subheader("Inspect a mover")
    unusual_selected = st.selectbox(
        "Pick a ticker from the list above", candidates["ticker"].tolist(),
        key=f"{key_prefix}_unusual_select",
    )

    if unusual_selected:
        row = candidates[candidates["ticker"] == unusual_selected].iloc[0]
        df = price_data[unusual_selected]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price Change", f"{row['price_change_pct']:+.1f}%")
        c2.metric("Relative Volume", f"{row['relative_volume']:.2f}x average")
        c3.metric("Latest Volume", f"{row['last_volume']:,.0f}")
        c4.metric("Unusual Score", f"{row['unusual_score']:.1f}")

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=df.index[-90:], y=df["Close"].iloc[-90:], name="Close",
                                        line=dict(color="#2E7D32")))
        price_fig.update_layout(
            title=f"{unusual_selected} — last 90 trading days",
            height=320, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(price_fig, use_container_width=True)

        vol_fig = go.Figure()
        vol_colors = ["#EF5350" if i == len(df.iloc[-90:]) - 1 else "#90A4AE"
                      for i in range(len(df.iloc[-90:]))]
        vol_fig.add_trace(go.Bar(x=df.index[-90:], y=df["Volume"].iloc[-90:], marker_color=vol_colors,
                                  name="Volume"))
        vol_fig.update_layout(
            title="Volume (today highlighted)",
            height=220, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    st.caption(
        "⚡ A volume/price spike can mean news, an announcement, or nothing more than noise — "
        "this tab flags candidates for you to research, it doesn't tell you why they moved. "
        "Always check for a news catalyst before acting on anything here."
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Momentum Screener")

st.sidebar.subheader("Universe")
universe_source = st.sidebar.radio(
    "Universe source",
    ["S&P 500", "All US Stocks (NYSE + Nasdaq)"],
    help=(
        "All US Stocks is several thousand tickers (NYSE + Nasdaq, ETFs and SPAC "
        "warrants/units excluded) -- much slower to download than the S&P 500 and "
        "more likely to hit an occasional individual-ticker hiccup (skipped automatically). "
        "The Crypto Screener tab has its own separate universe and fetch button."
    ),
)

if universe_source == "S&P 500":
    universe_size = st.sidebar.slider(
        "Number of S&P 500 stocks to scan",
        min_value=50, max_value=500, value=150, step=25,
        help="Smaller = faster to download. Start small to test, then go up to the full 500.",
    )
else:
    universe_size = st.sidebar.slider(
        "Number of stocks to scan",
        min_value=100, max_value=6000, value=500, step=100,
        help=(
            "The full list is 6000+ tickers and can take 30+ minutes to download. "
            "Start smaller (a few hundred) to test before going wide."
        ),
    )
    st.sidebar.caption(
        f"⏱️ Scanning {universe_size} stocks from the full US market will take noticeably "
        "longer than the S&P 500 option -- budget several minutes or more."
    )

st.sidebar.subheader("Factor weights")
st.sidebar.caption("Adjust how much each category contributes to the final Momentum Score (used by both the stock and crypto Momentum screeners).")
w_price_momentum = st.sidebar.slider("Price momentum (12-1 month)", 0, 100, 30)
w_relative_strength = st.sidebar.slider("Relative strength vs benchmark", 0, 100, 30)
w_trend_volume = st.sidebar.slider("Trend + volume confirmation", 0, 100, 25)
w_short_term = st.sidebar.slider("Short-term technical signals", 0, 100, 15)

weights = {
    "price_momentum": w_price_momentum,
    "relative_strength": w_relative_strength,
    "trend_volume": w_trend_volume,
    "short_term_technical": w_short_term,
}
weight_total = sum(weights.values())
if weight_total == 0:
    st.sidebar.warning("At least one weight must be greater than 0.")

st.sidebar.subheader("Stock filters")
min_price = st.sidebar.number_input("Minimum price ($)", min_value=0.0, value=5.0, step=1.0)
require_uptrend = st.sidebar.checkbox("Only show stocks above their 200-day MA", value=True)
top_n = st.sidebar.slider("Show top N stocks", 10, 100, 25)

refresh = st.sidebar.button("🔄 Fetch data / Refresh", type="primary", use_container_width=True)
st.sidebar.caption(
    "Data is cached for 1 hour after each fetch. Prices are end-of-day via Yahoo Finance "
    "(yfinance) and may be delayed — this is not a live/real-time feed."
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("📈 Momentum Stock Screener")
st.write(
    "Scans either the S&P 500 or the full US stock market (your choice in the sidebar) and ranks "
    "stocks by a composite **Momentum Score (0-100)** combining price momentum, relative strength "
    "vs the market, trend/volume confirmation, and short-term technical signals -- plus VCP pattern "
    "and unusual-activity scanners, and a separate **Crypto Screener** tab with all three of those "
    "same screens applied to major cryptocurrencies. This is a research tool, not financial advice."
)

if "price_data" not in st.session_state:
    st.session_state.price_data = None
if "factors_df" not in st.session_state:
    st.session_state.factors_df = None
if "crypto_price_data" not in st.session_state:
    st.session_state.crypto_price_data = None
if "crypto_factors_df" not in st.session_state:
    st.session_state.crypto_factors_df = None

if refresh or st.session_state.price_data is None:
    fetch_label = "Fetching S&P 500 ticker list..." if universe_source == "S&P 500" else "Fetching full US market ticker list..."
    with st.spinner(fetch_label):
        all_tickers = load_tickers() if universe_source == "S&P 500" else load_all_us_tickers()
    tickers = all_tickers[:universe_size]

    progress_bar = st.progress(0.0, text="Downloading price history...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Downloading price history... ({done}/{total})")

    fetch_error = None
    price_data = None
    try:
        price_data = data_fetch.download_price_history(tickers, progress_callback=_progress)
    except ImportError:
        fetch_error = (
            "The `yfinance` package isn't installed. Run `pip install -r requirements.txt` "
            "in your terminal, then restart the app."
        )
    except Exception as e:
        fetch_error = f"Failed to download price data: {e}"

    progress_bar.empty()

    if fetch_error is None and data_fetch.BENCHMARK_TICKER not in (price_data or {}):
        fetch_error = (
            "Couldn't download benchmark (SPY) data, so relative strength can't be computed. "
            "Check your internet connection and try again."
        )

    if fetch_error:
        st.error(fetch_error)
    else:
        st.session_state.price_data = price_data
        st.session_state.factors_df = None  # force recompute below

if st.session_state.price_data:
    if st.session_state.factors_df is None:
        with st.spinner("Computing momentum factors..."):
            st.session_state.factors_df = compute_factors(st.session_state.price_data)
    factors_df = st.session_state.factors_df
    total_downloaded = len(st.session_state.price_data)
else:
    factors_df = None
    total_downloaded = 0

tab_momentum, tab_vcp, tab_unusual, tab_crypto = st.tabs(
    ["📈 Momentum Screener", "🔎 VCP Scanner", "⚡ Unusual Activity", "🪙 Crypto Screener"]
)

# -----------------------------------------------------------------
# Momentum Screener tab (stocks)
# -----------------------------------------------------------------
with tab_momentum:
    if not st.session_state.price_data:
        st.info("Click **🔄 Fetch data / Refresh** in the sidebar to scan the market.")
    elif factors_df.empty:
        st.warning("No stocks had enough history to score. Try increasing the universe size.")
    else:
        render_momentum_results(
            factors_df=factors_df,
            price_data=st.session_state.price_data,
            weights=weights,
            total_downloaded=total_downloaded,
            benchmark_label="the S&P 500",
            key_prefix="stock",
            min_price=min_price,
            require_uptrend=require_uptrend,
            top_n=top_n,
        )
        st.caption(
            "⚠️ Educational tool only, not investment advice. Momentum strategies can reverse "
            "sharply and past performance doesn't predict future results. Data via Yahoo Finance "
            "(yfinance), end-of-day, may contain gaps or delays."
        )

# -----------------------------------------------------------------
# VCP Scanner tab (stocks)
# -----------------------------------------------------------------
with tab_vcp:
    if not st.session_state.price_data:
        st.info("Click **🔄 Fetch data / Refresh** in the sidebar to scan the market.")
    else:
        render_vcp_results(
            price_data=st.session_state.price_data,
            bench_ticker=data_fetch.BENCHMARK_TICKER,
            days_per_year=mc.TRADING_DAYS_PER_YEAR,
            key_prefix="stock",
        )

# -----------------------------------------------------------------
# Unusual Activity tab (stocks)
# -----------------------------------------------------------------
with tab_unusual:
    if not st.session_state.price_data:
        st.info("Click **🔄 Fetch data / Refresh** in the sidebar to scan the market.")
    else:
        render_unusual_results(
            price_data=st.session_state.price_data,
            bench_ticker=data_fetch.BENCHMARK_TICKER,
            key_prefix="stock",
        )

# -----------------------------------------------------------------
# Crypto Screener tab -- self-contained: its own universe, its own
# fetch button, and the same three screens nested in sub-tabs. Doesn't
# auto-fetch on page load (unlike the stock tabs) since most visits to
# the app won't need it -- only downloads once you click its button.
# -----------------------------------------------------------------
with tab_crypto:
    st.write(
        f"Scans a curated list of {len(data_fetch.CRYPTO_TICKERS)} major cryptocurrencies the "
        "same three ways as the stock tabs above, using **Bitcoin (BTC-USD)** as the "
        "relative-strength benchmark instead of the S&P 500 (the same way SPY isn't ranked "
        "against itself, BTC-USD isn't ranked among the coins here). Crypto trades every "
        "calendar day, not just weekdays, so all the 12-month/52-week lookback windows use a "
        "365-day-a-year calendar instead of stocks' 252-day trading-year calendar."
    )
    st.caption(
        "This is a hand-maintained list, not a live \"top coins by market cap\" feed, so it will "
        "drift out of date over time (edit `CRYPTO_TICKERS` in `data_fetch.py` to add/remove "
        "coins — anything yfinance recognizes as a `TICKER-USD` pair works). Stablecoins are "
        "excluded since they're pegged to $1 and \"momentum\" is meaningless for them."
    )

    crypto_fetch_col1, crypto_fetch_col2 = st.columns([3, 1])
    crypto_universe_size = crypto_fetch_col1.slider(
        "Number of cryptocurrencies to scan", 10, len(data_fetch.CRYPTO_TICKERS),
        len(data_fetch.CRYPTO_TICKERS), step=5, key="crypto_universe_size",
    )
    crypto_refresh = crypto_fetch_col2.button(
        "🔄 Fetch Crypto Data", type="primary", use_container_width=True, key="crypto_refresh"
    )

    if crypto_refresh:
        with st.spinner("Fetching crypto ticker list..."):
            crypto_tickers_all = load_crypto_tickers()
        crypto_tickers = crypto_tickers_all[:crypto_universe_size]

        crypto_progress = st.progress(0.0, text="Downloading crypto price history...")

        def _crypto_progress(done, total):
            crypto_progress.progress(done / total, text=f"Downloading crypto price history... ({done}/{total})")

        crypto_fetch_error = None
        crypto_price_data = None
        try:
            crypto_price_data = data_fetch.download_price_history(
                crypto_tickers, progress_callback=_crypto_progress,
                benchmark_ticker=data_fetch.CRYPTO_BENCHMARK_TICKER,
            )
        except ImportError:
            crypto_fetch_error = (
                "The `yfinance` package isn't installed. Run `pip install -r requirements.txt` "
                "in your terminal, then restart the app."
            )
        except Exception as e:
            crypto_fetch_error = f"Failed to download crypto price data: {e}"

        crypto_progress.empty()

        if crypto_fetch_error is None and data_fetch.CRYPTO_BENCHMARK_TICKER not in (crypto_price_data or {}):
            crypto_fetch_error = (
                "Couldn't download benchmark (BTC-USD) data, so relative strength can't be computed. "
                "Check your internet connection and try again."
            )

        if crypto_fetch_error:
            st.error(crypto_fetch_error)
        else:
            st.session_state.crypto_price_data = crypto_price_data
            st.session_state.crypto_factors_df = None  # force recompute below

    if not st.session_state.crypto_price_data:
        st.info("Click **🔄 Fetch Crypto Data** above to scan the crypto market.")
    else:
        if st.session_state.crypto_factors_df is None:
            with st.spinner("Computing crypto momentum factors..."):
                st.session_state.crypto_factors_df = compute_crypto_factors(st.session_state.crypto_price_data)
        crypto_factors_df = st.session_state.crypto_factors_df
        crypto_total_downloaded = len(st.session_state.crypto_price_data)

        crypto_sub_momentum, crypto_sub_vcp, crypto_sub_unusual = st.tabs(
            ["📈 Momentum", "🔎 VCP", "⚡ Unusual Activity"]
        )

        with crypto_sub_momentum:
            if crypto_factors_df.empty:
                st.warning("No cryptocurrencies had enough history to score.")
            else:
                cm1, cm2 = st.columns(2)
                crypto_require_uptrend = cm1.checkbox(
                    "Only show coins above their 200-day MA", value=False, key="crypto_uptrend"
                )
                crypto_top_n = cm2.slider("Show top N coins", 5, 50, 20, key="crypto_top_n")
                render_momentum_results(
                    factors_df=crypto_factors_df,
                    price_data=st.session_state.crypto_price_data,
                    weights=weights,
                    total_downloaded=crypto_total_downloaded,
                    benchmark_label="Bitcoin (BTC-USD)",
                    key_prefix="crypto",
                    min_price=0.0,  # crypto prices span fractions of a cent to $100k+; a $ floor doesn't generalize
                    require_uptrend=crypto_require_uptrend,
                    top_n=crypto_top_n,
                    asset_noun="cryptocurrencies",
                    asset_noun_singular="cryptocurrency",
                )
                st.caption(
                    "⚠️ Educational tool only, not investment advice. Crypto is especially "
                    "volatile — momentum can reverse sharply and past performance doesn't "
                    "predict future results. Data via Yahoo Finance (yfinance), may contain "
                    "gaps or delays."
                )

        with crypto_sub_vcp:
            render_vcp_results(
                price_data=st.session_state.crypto_price_data,
                bench_ticker=data_fetch.CRYPTO_BENCHMARK_TICKER,
                days_per_year=365,
                key_prefix="crypto",
                asset_noun="cryptocurrencies",
                asset_noun_singular="cryptocurrency",
            )

        with crypto_sub_unusual:
            render_unusual_results(
                price_data=st.session_state.crypto_price_data,
                bench_ticker=data_fetch.CRYPTO_BENCHMARK_TICKER,
                key_prefix="crypto",
                asset_noun="cryptocurrencies",
            )
