"""
Smoke test for app.py's control flow, without needing streamlit/plotly/yfinance
actually installed. Stubs out just enough of the streamlit and plotly APIs that
app.py calls, and monkeypatches data_fetch to return synthetic data instead of
hitting the network. Confirms the script runs end-to-end without exceptions,
which catches wiring bugs (wrong dict keys, undefined names, bad column refs)
that plain syntax-checking can't.

Run with: python3 test_app_smoke.py
"""

import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd


class StopExecution(Exception):
    pass


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


# --- stub streamlit -------------------------------------------------------

class FakeSidebar:
    def title(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def slider(self, label, *a, **k):
        # return the default value if present, else the min
        if "value" in k:
            return k["value"]
        return a[2] if len(a) > 2 else (a[0] if a else 0)
    def number_input(self, label, *a, **k):
        return k.get("value", 0.0)
    def checkbox(self, label, *a, **k):
        return k.get("value", False)
    def radio(self, label, options, *a, **k):
        return options[k.get("index", 0)] if len(options) else None
    def button(self, *a, **k):
        return False  # simulate: user hasn't clicked refresh


class FakeColumn:
    def metric(self, *a, **k): pass
    def slider(self, label, *a, **k):
        if "value" in k:
            return k["value"]
        return a[2] if len(a) > 2 else (a[0] if a else 0)
    def checkbox(self, label, *a, **k):
        return k.get("value", False)
    def number_input(self, label, *a, **k):
        return k.get("value", 0.0)
    def selectbox(self, label, options, *a, **k):
        return options[0] if len(options) else None
    def button(self, *a, **k):
        return True  # simulate: user clicked (used for the crypto tab's own fetch button)
    def download_button(self, *a, **k):
        return False
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeTab:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeProgressBar:
    def progress(self, *a, **k): pass
    def empty(self): pass


class FakeExpander:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeSpinner:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False


class FakeSessionState(dict):
    __getattr__ = dict.get
    def __setattr__(self, key, value):
        self[key] = value


class FakeDataFrameStyle:
    def __init__(self, df):
        self.df = df
    def format(self, *a, **k):
        return self
    def background_gradient(self, *a, **k):
        return self


def install_fake_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = FakeSessionState()
    st.sidebar = FakeSidebar()

    st.set_page_config = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.subheader = lambda *a, **k: None
    st.write = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.info = lambda *a, **k: None
    st.warning = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.selectbox = lambda label, options, *a, **k: (options[0] if len(options) else None)
    st.plotly_chart = lambda *a, **k: None
    st.dataframe = lambda *a, **k: None
    st.metric = lambda *a, **k: None
    st.text_input = lambda label, *a, **k: k.get("value", "")
    st.download_button = lambda *a, **k: False

    def _error(msg, *a, **k):
        print(f"  [st.error called]: {msg}")
    st.error = _error

    def _stop():
        raise StopExecution()
    st.stop = _stop

    def _progress(value, text=None):
        return FakeProgressBar()
    st.progress = _progress

    def _spinner(text=""):
        return FakeSpinner(text)
    st.spinner = _spinner

    def _expander(*a, **k):
        return FakeExpander()
    st.expander = _expander

    def _columns(spec, *a, **k):
        # streamlit accepts either an int (equal-width columns) or a list of
        # relative widths (e.g. st.columns([3, 1])) -- handle both.
        n = spec if isinstance(spec, int) else len(spec)
        return [FakeColumn() for _ in range(n)]
    st.columns = _columns

    def _tabs(labels, *a, **k):
        return [FakeTab() for _ in labels]
    st.tabs = _tabs

    def _cache_data(*dargs, **dkwargs):
        # support both @st.cache_data and @st.cache_data(ttl=..., show_spinner=...)
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def decorator(func):
            return func
        return decorator
    st.cache_data = _cache_data

    # patch DataFrame.style.format/background_gradient chain used in app.py
    pd.DataFrame.style = property(lambda self: FakeDataFrameStyle(self))

    sys.modules["streamlit"] = st
    return st


def install_fake_plotly():
    go = types.ModuleType("plotly.graph_objects")

    class FakeFigure:
        def __init__(self, *a, **k): self.traces = []
        def add_trace(self, *a, **k): self.traces.append(a)
        def add_hline(self, *a, **k): pass
        def update_layout(self, *a, **k): pass

    class FakeScatter:
        def __init__(self, *a, **k): pass

    class FakeBar:
        def __init__(self, *a, **k): pass

    go.Figure = FakeFigure
    go.Scatter = FakeScatter
    go.Bar = FakeBar

    plotly_pkg = types.ModuleType("plotly")
    plotly_pkg.graph_objects = go

    sys.modules["plotly"] = plotly_pkg
    sys.modules["plotly.graph_objects"] = go


def make_fake_universe(n=12, n_crypto=6, days=400, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    universe = {}

    def _make_series(start_price):
        drift = rng.uniform(-0.001, 0.0018)
        vol = rng.uniform(0.01, 0.025)
        rets = rng.normal(drift, vol, days)
        close = start_price * np.cumprod(1 + rets)
        volume = rng.integers(500_000, 5_000_000, days).astype(float)
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": volume},
            index=idx,
        )

    for i in range(n):
        universe[f"TICK{i}"] = _make_series(50)

    # stock benchmark
    bench_rets = rng.normal(0.0004, 0.01, days)
    bench_close = 400 * np.cumprod(1 + bench_rets)
    universe["SPY"] = pd.DataFrame(
        {"Open": bench_close, "High": bench_close * 1.01, "Low": bench_close * 0.99,
         "Close": bench_close, "Volume": rng.integers(1e7, 2e7, days).astype(float)},
        index=idx,
    )

    # crypto universe + its own benchmark (BTC-USD), same shape, different tickers
    for i in range(n_crypto):
        universe[f"CTICK{i}-USD"] = _make_series(30)
    btc_rets = rng.normal(0.0006, 0.02, days)
    btc_close = 30_000 * np.cumprod(1 + btc_rets)
    universe["BTC-USD"] = pd.DataFrame(
        {"Open": btc_close, "High": btc_close * 1.01, "Low": btc_close * 0.99,
         "Close": btc_close, "Volume": rng.integers(1e7, 2e7, days).astype(float)},
        index=idx,
    )

    return universe


def main():
    st = install_fake_streamlit()
    install_fake_plotly()

    import data_fetch
    fake_universe = make_fake_universe()
    fake_tickers = [t for t in fake_universe if t.startswith("TICK")]
    fake_crypto_tickers = [t for t in fake_universe if t.startswith("CTICK")]

    data_fetch.get_sp500_tickers = lambda: fake_tickers
    data_fetch.get_crypto_tickers = lambda: fake_crypto_tickers

    call_counts = {"stock": 0, "crypto": 0}

    def _fake_download(tickers, benchmark_ticker=data_fetch.BENCHMARK_TICKER, **kwargs):
        # Mirrors the real download_price_history signature: the benchmark
        # defaults to SPY (stocks) but the crypto tab passes BTC-USD instead --
        # respecting the kwarg (rather than always appending "SPY") is what
        # lets the crypto tab's benchmark check succeed in this smoke test.
        if benchmark_ticker == data_fetch.CRYPTO_BENCHMARK_TICKER:
            call_counts["crypto"] += 1
        else:
            call_counts["stock"] += 1
        return {t: fake_universe[t] for t in list(tickers) + [benchmark_ticker] if t in fake_universe}
    data_fetch.download_price_history = _fake_download

    # Redirect the disk cache to a scratch directory for this test run, so it
    # never touches (or is affected by) a real .cache/ folder from an actual
    # run of the app.
    import cache_store
    cache_scratch_dir = Path(tempfile.mkdtemp())
    cache_store.CACHE_DIR = cache_scratch_dir

    # app.py checks `if refresh or st.session_state.price_data is None:` on first run,
    # and our fake sidebar.button() returns False for refresh, so first run relies on
    # session_state.price_data being None initially -- that's the default path we want to test.
    # The crypto tab's own fetch button (FakeColumn.button) returns True instead, so its
    # fetch-and-render path is exercised too, on every run.

    try:
        with open("app.py") as f:
            code = f.read()

        print("Running app.py end-to-end with synthetic data (no real network/streamlit)...")
        try:
            exec(compile(code, "app.py", "exec"), {"__name__": "__main__"})
        except StopExecution:
            print("  (script called st.stop() -- treated as a controlled stop, not a crash)")
        print("[PASS] app.py ran without raising an unhandled exception.")
        check("first run actually downloaded stock data (no cache existed yet)", call_counts["stock"] > 0)

        # Second pass: a fresh session_state (simulating a new browser session /
        # app restart) but the SAME on-disk cache dir from the first run's save --
        # the stock tabs should load straight from disk instead of re-downloading.
        # (The crypto tab's fetch button is faked to always return True, so crypto
        # keeps "fetching" on every pass regardless -- only the stock count matters here.)
        stock_calls_before = call_counts["stock"]
        st.session_state = FakeSessionState()
        print("\nRunning app.py again with a fresh session (same on-disk cache)...")
        try:
            exec(compile(code, "app.py", "exec"), {"__name__": "__main__"})
        except StopExecution:
            print("  (script called st.stop() -- treated as a controlled stop, not a crash)")
        check("second run reused the on-disk cache instead of re-downloading stock data",
              call_counts["stock"] == stock_calls_before)
        print("[PASS] disk cache prevented a redundant stock download on the second run.")
    finally:
        shutil.rmtree(cache_scratch_dir, ignore_errors=True)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
