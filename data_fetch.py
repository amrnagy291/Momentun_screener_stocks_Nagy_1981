"""
data_fetch.py

Handles getting the S&P 500 ticker universe and pulling price/volume
history for all of them via yfinance. Isolated from the Streamlit UI so
it can be reused or tested independently.

Note: this module needs internet access (Wikipedia for the ticker list,
Yahoo Finance via yfinance for prices) -- it's meant to run on YOUR
machine when you launch the app, not in any restricted sandbox.
"""

from __future__ import annotations

import time
import pandas as pd
import requests

BENCHMARK_TICKER = "SPY"  # liquid, easy proxy for the S&P 500 index itself
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Static fallback list, used only if the live Wikipedia fetch fails
# (e.g. no internet at that moment, or the page structure changes).
# This will drift out of date over time -- it's a safety net, not the
# primary source. Update FALLBACK_SP500 periodically if you rely on it.
FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK.B", "AVGO", "TSLA",
    "LLY", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "NFLX",
    "JNJ", "WMT", "ABBV", "CRM", "BAC", "ORCL", "MRK", "CVX", "KO", "AMD",
    "ADBE", "PEP", "ACN", "TMO", "LIN", "MCD", "CSCO", "ABT", "WFC", "IBM",
    "GE", "DHR", "CAT", "PM", "TXN", "INTU", "VZ", "NOW", "ISRG", "QCOM",
    "DIS", "AMGN", "CMCSA", "SPGI", "AXP", "BKNG", "NEE", "PFE", "UNP", "LOW",
    "HON", "AMAT", "RTX", "COP", "T", "GS", "ETN", "PGR", "SYK", "UBER",
    "BLK", "TJX", "VRTX", "BSX", "ADP", "MU", "LMT", "PANW", "MDT", "NKE",
    "SCHW", "C", "ADI", "ANET", "SBUX", "BX", "GILD", "CB", "MMC", "PLD",
    "DE", "AMT", "SO", "LRCX", "ELV", "KLAC", "MO", "REGN", "FI", "EQIX",
]


def get_sp500_tickers() -> list[str]:
    """
    Fetch the current S&P 500 constituent list from Wikipedia.
    Falls back to a static (and progressively stale) list if that fails.
    yfinance/Yahoo uses '-' instead of '.' in tickers like BRK.B -> BRK-B.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (momentum-screener educational script)"}
        resp = requests.get(WIKI_SP500_URL, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        table = tables[0]
        tickers = table["Symbol"].astype(str).tolist()
    except Exception:
        tickers = FALLBACK_SP500

    return [t.strip().replace(".", "-") for t in tickers]


def download_price_history(tickers: list[str], period: str = "2y", batch_size: int = 50,
                            pause_seconds: float = 1.0, progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV history for a list of tickers (plus the benchmark) via
    yfinance, in batches to stay polite to the API. Returns a dict of
    {ticker: DataFrame} with columns Open/High/Low/Close/Volume, dropping
    any ticker that failed to download or has too little history.

    progress_callback(done_count, total_count) is called after each batch
    if provided, so a UI can show a progress bar.
    """
    import yfinance as yf

    all_tickers = list(dict.fromkeys(tickers + [BENCHMARK_TICKER]))
    results: dict[str, pd.DataFrame] = {}

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            data = None

        if data is not None and not data.empty:
            # Depending on the yfinance version, a batch of exactly one ticker
            # can come back either as a flat DataFrame (columns: Open/High/.../Close)
            # or with the same per-ticker MultiIndex columns used for multi-ticker
            # batches. Handle both shapes instead of assuming single-ticker == flat.
            is_multiindex = isinstance(data.columns, pd.MultiIndex)
            for ticker in batch:
                try:
                    if is_multiindex:
                        df = data[ticker].dropna(how="all")
                    elif len(batch) == 1:
                        df = data.dropna(how="all")
                    else:
                        continue
                except (KeyError, TypeError):
                    continue
                if not df.empty and "Close" in df.columns and df["Close"].notna().sum() >= 210:
                    results[ticker] = df

        if progress_callback:
            progress_callback(min(i + batch_size, len(all_tickers)), len(all_tickers))

        time.sleep(pause_seconds)

    return results
