# Momentum Stock Screener

A local web dashboard that scans the S&P 500 and ranks stocks three ways, in
three tabs:

- **📈 Momentum Screener** — ranks stocks by a composite **Momentum Score
  (0-100)**, combining:

  1. **Price momentum (12-1 month)** — trailing 12-month return, excluding the
     most recent month (the standard academic momentum factor).
  2. **Relative strength vs the S&P 500** — how much a stock has out/underperformed
     SPY over 3, 6, and 12 months.
  3. **Trend + volume confirmation** — price above its 50-day and 200-day moving
     averages, a "golden cross" (50-day MA above 200-day MA), and above-average
     recent trading volume.
  4. **Short-term technical signals** — RSI, a bullish MACD crossover, and
     closeness to the 52-week high.

  You control how much each category counts toward the final score with
  sliders in the sidebar.

- **🔎 VCP Scanner** — looks for stocks tracing out a **Volatility
  Contraction Pattern (VCP)**: the Mark Minervini-style setup where a stock
  already in an uptrend forms a series of pullbacks that get progressively
  shallower (and quieter, on volume) as it coils beneath a resistance level
  (the "pivot"), often ahead of a breakout. There's no single agreed
  algorithmic definition of a VCP — this is a documented heuristic
  approximation, not a certified pattern match. See "How VCP detection
  works" inside that tab for exactly what it checks, and always look at the
  annotated chart (swing highs/lows and the pivot line) before trusting a
  result.

- **⚡ Unusual Activity** — flags stocks with an outsized single-day price
  move (up or down, your threshold — default 5%) combined with volume well
  above their typical recent level (your threshold — default 2x the trailing
  50-day average). Based only on the most recent trading day, unlike the
  other two tabs' multi-week patterns. A price/volume spike can mean real
  news or just noise — this tab surfaces candidates for you to research, it
  doesn't tell you why a stock moved.

**This is an educational/research tool, not financial advice.** Momentum,
VCP, and volume/price-spike strategies can all reverse or fail sharply, past
performance doesn't predict future results, and none of these screeners
account for fees, taxes, slippage, or your personal risk tolerance. Always do
your own research (or talk to a licensed financial advisor) before acting on
anything any tab shows you.

## What's in this folder

| File | Purpose |
|---|---|
| `app.py` | The Streamlit dashboard (what you actually run) |
| `momentum_calc.py` | Pure math: all the momentum/technical factor calculations |
| `vcp_calc.py` | Pure math: the VCP (Volatility Contraction Pattern) detection and scoring |
| `unusual_activity_calc.py` | Pure math: single-day price-change % and relative-volume detection |
| `data_fetch.py` | Gets the ticker list (S&P 500 or all US stocks) and downloads price history via `yfinance` |
| `requirements.txt` | Python packages needed |
| `test_momentum_calc.py` | Sanity checks for the momentum math, using synthetic price data (no internet needed) |
| `test_vcp_calc.py` | Sanity checks for VCP detection, using a hand-built synthetic contraction pattern (no internet needed) |
| `test_unusual_activity_calc.py` | Sanity checks for the unusual-activity math, using synthetic price/volume spikes (no internet needed) |
| `test_app_smoke.py` | End-to-end dry run of the app's logic with fake data (no internet needed) |

## Setup (one-time)

You'll need Python 3.10+ installed. Then, in a terminal, from inside this folder:

```bash
# (optional but recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).
Click **"🔄 Fetch data / Refresh"** in the sidebar the first time — it will
download the ticker list and 2 years of daily price history for each stock
via Yahoo Finance. With the default S&P 500 / 150-stock universe this usually
takes 30-90 seconds; the full 500 takes a few minutes. Results are cached
until your next refresh so you don't re-download every time you tweak a slider.

**Universe source:** choose **S&P 500** (fast, ~500 stocks, good for everyday
use) or **All US Stocks** in the sidebar. The latter pulls essentially every
NYSE + Nasdaq common stock (several thousand tickers, via NASDAQ Trader's
official symbol directory) — ETFs and SPAC warrants/units are filtered out,
but this is a much bigger download: scanning a few hundred still takes a
minute or two, and the full multi-thousand-ticker list can take 30+ minutes.
Start with a smaller number on this slider even when using "All US Stocks."

## How to use it

1. Pick your **universe source** (S&P 500 or All US Stocks) and **universe
   size** (how many stocks to scan — start smaller while you're experimenting,
   then increase).
2. Adjust the **factor weights** to emphasize what matters to you — e.g. crank
   up "Relative strength" if you specifically care about market-beating stocks,
   or "Short-term technical signals" if you're more of a swing trader.
3. Use the **filters** (minimum price, uptrend-only) to cut out penny stocks
   or names that are technically weak despite a decent score.
4. The **ranked table** shows the top stocks by Momentum Score, with a
   breakdown by category so you can see *why* a stock scored well.
5. Pick any ticker from the dropdown to see its price chart with 50/200-day
   moving averages and its individual metrics.
6. Switch to the **VCP Scanner** or **Unusual Activity** tabs for the other
   two screens — see their descriptions above.

## Before you rely on this for real trading

- **Verify the data.** Yahoo Finance data (via `yfinance`) is generally
  reliable but end-of-day and occasionally has gaps, splits/dividend
  adjustment quirks, or short outages. Cross-check anything you're about to
  act on.
- **Momentum ≠ guaranteed continuation.** High-momentum stocks can and do
  reverse sharply ("momentum crashes"), especially in volatile markets.
- **This screens for a factor, not a full strategy.** It doesn't include
  position sizing, stop-losses, sector/correlation risk, fundamentals,
  earnings dates, or transaction costs — all things a real trading plan
  needs.
- **Ticker lists are fetched live** each hour — the S&P 500 list from
  Wikipedia, the All US Stocks list from NASDAQ Trader's official symbol
  directory. If either fails (no internet, page/file format changes), it
  silently falls back to a static list embedded in `data_fetch.py`, which
  will drift out of date over time. If your results look off, check
  `data_fetch.get_sp500_tickers()` / `data_fetch.get_all_us_tickers()`.
- **"All US Stocks" still isn't literally every security.** It excludes
  ETFs, test/placeholder issues, and obvious SPAC warrants/units/rights (by
  name, not by symbol pattern), and it doesn't cover OTC/pink-sheet stocks —
  just NYSE and Nasdaq-listed common stock.

## Customizing further

- **Different universe:** Swap `data_fetch.get_sp500_tickers()` for your own
  ticker list (e.g. Nasdaq-100, a sector, or a personal watchlist) — it just
  needs to return a list of ticker strings.
- **Different weighting logic:** All the scoring math lives in
  `momentum_calc.py`, in `rank_universe()` — the rest of the app doesn't care
  how the score is built, only that you get back a `momentum_score` column.
- **Add a factor:** Write a function in `momentum_calc.py` that takes a
  price/volume DataFrame and returns a value, add it to
  `compute_all_factors()`, then fold it into `rank_universe()`.

## Testing without the internet

Both `test_momentum_calc.py` and `test_app_smoke.py` run entirely on
synthetic, locally-generated price data — no `yfinance`, no `streamlit`, no
network access required. Useful for confirming the math still works after
you change something:

```bash
python3 test_momentum_calc.py
python3 test_app_smoke.py   # requires numpy/pandas only, stubs out streamlit/plotly
```
