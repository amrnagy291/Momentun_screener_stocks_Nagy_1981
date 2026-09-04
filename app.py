"""
Momentum Stock Screener -- Streamlit dashboard

Scans the S&P 500 and ranks stocks by a composite momentum score built
from four factor categories:
  1. Price momentum (12 months of return, excluding the most recent month)
  2. Relative strength vs the S&P 500 (3/6/12 month outperformance)
  3. Trend + volume confirmation (above 50/200-day MAs, golden cross, volume)
  4. Short-term technical signals (RSI, MACD crossover, 52-week high proximity)

Run it with:
    streamlit run app.py

This is an educational/research tool, NOT financial advice. Momentum
strategies carry real risk, including sharp reversals ("momentum crashes").
Always do your own research before trading on this or any screener's output.
"""

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_fetch
import momentum_calc as mc

st.set_page_config(page_title="Momentum Stock Screener", layout="wide")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_tickers() -> list[str]:
    return data_fetch.get_sp500_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def load_price_history(tickers: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return data_fetch.download_price_history(list(tickers))


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


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Momentum Screener")

st.sidebar.subheader("Universe")
universe_size = st.sidebar.slider(
    "Number of S&P 500 stocks to scan",
    min_value=50, max_value=500, value=150, step=25,
    help="Smaller = faster to download. Start small to test, then go up to the full 500.",
)

st.sidebar.subheader("Factor weights")
st.sidebar.caption("Adjust how much each category contributes to the final Momentum Score.")
w_price_momentum = st.sidebar.slider("Price momentum (12-1 month)", 0, 100, 30)
w_relative_strength = st.sidebar.slider("Relative strength vs S&P 500", 0, 100, 30)
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

st.sidebar.subheader("Filters")
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
    "Ranks S&P 500 stocks by a composite **Momentum Score (0-100)** combining price momentum, "
    "relative strength vs the market, trend/volume confirmation, and short-term technical signals. "
    "This is a research tool, not financial advice."
)

if "price_data" not in st.session_state:
    st.session_state.price_data = None
if "factors_df" not in st.session_state:
    st.session_state.factors_df = None

if refresh or st.session_state.price_data is None:
    with st.spinner("Fetching S&P 500 ticker list..."):
        all_tickers = load_tickers()
    tickers = all_tickers[:universe_size]

    progress_bar = st.progress(0.0, text="Downloading price history...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Downloading price history... ({done}/{total})")

    try:
        price_data = data_fetch.download_price_history(tickers, progress_callback=_progress)
    except ImportError:
        st.error(
            "The `yfinance` package isn't installed. Run `pip install -r requirements.txt` "
            "in your terminal, then restart the app."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to download price data: {e}")
        st.stop()

    progress_bar.empty()

    if data_fetch.BENCHMARK_TICKER not in price_data:
        st.error(
            "Couldn't download benchmark (SPY) data, so relative strength can't be computed. "
            "Check your internet connection and try again."
        )
        st.stop()

    st.session_state.price_data = price_data
    st.session_state.factors_df = None  # force recompute below

if st.session_state.price_data:
    if st.session_state.factors_df is None:
        with st.spinner("Computing momentum factors..."):
            st.session_state.factors_df = compute_factors(st.session_state.price_data)

    factors_df = st.session_state.factors_df

    if factors_df.empty:
        st.warning("No stocks had enough history to score. Try increasing the universe size.")
        st.stop()

    ranked = mc.rank_universe(factors_df, weights)

    # apply filters
    filtered = ranked[ranked["last_price"] >= min_price]
    if require_uptrend:
        filtered = filtered[filtered["above_200ma"]]
    filtered = filtered.head(top_n)

    total_downloaded = len(st.session_state.price_data) if st.session_state.price_data else 0
    st.caption(
        f"Scanned {len(factors_df)} stocks · {total_downloaded} "
        f"total tickers downloaded · showing top {len(filtered)} after filters"
    )

    if filtered.empty:
        st.warning("No stocks match the current filters. Try lowering the minimum price or unchecking the uptrend filter.")
        st.stop()

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

    st.subheader("Inspect a stock")
    selected = st.selectbox("Pick a ticker from the ranked list above", filtered["ticker"].tolist())

    if selected:
        row = filtered[filtered["ticker"] == selected].iloc[0]
        df = st.session_state.price_data[selected]

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
            """
Each stock gets a **percentile rank (0-100) within the scanned universe** for every underlying
signal, then those are blended into four category scores, which are combined using your
sidebar weights into the final **Momentum Score**:

- **Price momentum**: 12-month return excluding the most recent month (the standard academic
  momentum factor — skipping the last month avoids short-term reversal noise).
- **Relative strength**: average of how much the stock has out/underperformed SPY over 3, 6,
  and 12 months.
- **Trend + volume**: whether price is above its 50-day and 200-day moving averages, whether
  the 50-day MA is above the 200-day MA (a "golden cross" / established uptrend), and whether
  recent (20-day) volume is elevated versus the trailing 100-day average.
- **Short-term technicals**: RSI (favoring the 55–75 "strong but not overbought" zone), a
  recent bullish MACD crossover, and closeness to the 52-week high.

A score of 100 in a category means "best in the scanned universe on that dimension right now" —
scores are relative to the other stocks scanned, not absolute thresholds.
            """
        )

    st.caption(
        "⚠️ Educational tool only, not investment advice. Momentum strategies can reverse "
        "sharply and past performance doesn't predict future results. Data via Yahoo Finance "
        "(yfinance), end-of-day, may contain gaps or delays."
    )
else:
    st.info("Click **Fetch data / Refresh** in the sidebar to scan the market.")
