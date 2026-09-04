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


@st.cache_data(show_spinner=False)
def compute_vcp(price_data: dict[str, pd.DataFrame], window: int) -> pd.DataFrame:
    return vc.compute_universe_vcp(price_data, data_fetch.BENCHMARK_TICKER, window=window)


@st.cache_data(show_spinner=False)
def compute_unusual(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return ua.compute_universe_unusual_activity(price_data, data_fetch.BENCHMARK_TICKER)


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

    total_downloaded = len(st.session_state.price_data) if st.session_state.price_data else 0

    tab_momentum, tab_vcp, tab_unusual = st.tabs(
        ["📈 Momentum Screener", "🔎 VCP Scanner", "⚡ Unusual Activity"]
    )

    # -----------------------------------------------------------------
    # Momentum Screener tab
    # -----------------------------------------------------------------
    with tab_momentum:
        ranked = mc.rank_universe(factors_df, weights)

        # apply filters
        filtered = ranked[ranked["last_price"] >= min_price]
        if require_uptrend:
            filtered = filtered[filtered["above_200ma"]]
        filtered = filtered.head(top_n)

        st.caption(
            f"Scanned {len(factors_df)} stocks · {total_downloaded} "
            f"total tickers downloaded · showing top {len(filtered)} after filters"
        )

        if filtered.empty:
            st.warning("No stocks match the current filters. Try lowering the minimum price or unchecking the uptrend filter.")
        else:
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
            selected = st.selectbox(
                "Pick a ticker from the ranked list above", filtered["ticker"].tolist(), key="momentum_select"
            )

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

    # -----------------------------------------------------------------
    # VCP Scanner tab
    # -----------------------------------------------------------------
    with tab_vcp:
        st.write(
            "Looks for stocks tracing out a **Volatility Contraction Pattern (VCP)** — a series "
            "of pullbacks that get progressively shallower (and quieter, on volume) as a stock "
            "coils beneath a resistance level (the **pivot**), often resolving in a breakout. "
            "This is a heuristic approximation, not a certified pattern match — always look at "
            "the chart before trusting a result."
        )

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        min_contractions = ctrl1.slider(
            "Minimum contractions required", 2, 4, 2,
            help="A classic VCP usually has 2-4 (sometimes more) progressively smaller pullbacks.",
        )
        breakouts_only = ctrl2.checkbox("Show breakout candidates only", value=False)
        swing_window = ctrl3.slider(
            "Swing sensitivity (days each side)", 3, 10, 5,
            help="Smaller = picks up minor wiggles as contractions. Larger = only major swings count.",
        )

        with st.spinner("Scanning for VCP setups..."):
            vcp_all = compute_vcp(st.session_state.price_data, swing_window)

        if vcp_all.empty:
            st.warning("No stocks had enough history to scan for VCP setups.")
        else:
            vcp_candidates = vcp_all[
                vcp_all["is_vcp_candidate"] & (vcp_all["num_contractions"] >= min_contractions)
            ].copy()
            if breakouts_only:
                vcp_candidates = vcp_candidates[vcp_candidates["is_breakout"]]
            vcp_candidates = vcp_candidates.sort_values("vcp_score", ascending=False).reset_index(drop=True)
            if not vcp_candidates.empty:
                vcp_candidates.insert(0, "rank", np.arange(1, len(vcp_candidates) + 1))

            st.caption(
                f"Scanned {len(vcp_all)} stocks · {int(vcp_all['is_vcp_candidate'].sum())} showing at least "
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

                st.subheader("Inspect a candidate")
                vcp_selected = st.selectbox(
                    "Pick a ticker from the VCP list above", vcp_candidates["ticker"].tolist(), key="vcp_select"
                )

                if vcp_selected:
                    row = vcp_candidates[vcp_candidates["ticker"] == vcp_selected].iloc[0]
                    df = st.session_state.price_data[vcp_selected]

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
This scanner looks for stocks already in an uptrend (a simplified version of the "Trend
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

    # -----------------------------------------------------------------
    # Unusual Activity tab
    # -----------------------------------------------------------------
    with tab_unusual:
        st.write(
            "Flags stocks with an outsized **single-day price move combined with volume well "
            "above their typical level** — the classic 'something's going on with this stock' "
            "signal. Based on the most recent trading day only (not the multi-week patterns the "
            "other two tabs look for)."
        )

        uctrl1, uctrl2, uctrl3 = st.columns(3)
        min_price_change = uctrl1.slider(
            "Minimum price change (%)", 1, 20, 5,
            help="Flags days where the stock moved at least this much, up or down.",
        )
        min_rel_volume = uctrl2.slider(
            "Minimum relative volume", 1.0, 5.0, 2.0, step=0.1,
            help="Today's volume divided by its trailing 50-day average. 2.0x = twice the usual volume.",
        )
        direction = uctrl3.selectbox("Direction", ["Both", "Gainers only", "Losers only"])

        with st.spinner("Scanning for unusual volume/price activity..."):
            unusual_all = compute_unusual(st.session_state.price_data)

        if unusual_all.empty:
            st.warning("No stocks had enough history to scan for unusual activity.")
        else:
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

            st.caption(
                f"Scanned {len(unusual_all)} stocks · {len(candidates)} match current filters"
            )

            if candidates.empty:
                st.warning(
                    "No stocks match the current filters. Try lowering the minimum price change "
                    "or relative volume, or widening the universe size."
                )
            else:
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
                    "Pick a ticker from the list above", candidates["ticker"].tolist(), key="unusual_select"
                )

                if unusual_selected:
                    row = candidates[candidates["ticker"] == unusual_selected].iloc[0]
                    df = st.session_state.price_data[unusual_selected]

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
            "⚡ A volume/price spike can mean news, earnings, an analyst call, or nothing more "
            "than noise — this tab flags candidates for you to research, it doesn't tell you why "
            "they moved. Always check for a news catalyst before acting on anything here."
        )
else:
    st.info("Click **Fetch data / Refresh** in the sidebar to scan the market.")
