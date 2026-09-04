"""
Smoke test for app.py's control flow, without needing streamlit/plotly/yfinance
actually installed. Stubs out just enough of the streamlit and plotly APIs that
app.py calls, and monkeypatches data_fetch to return synthetic data instead of
hitting the network. Confirms the script runs end-to-end without exceptions,
which catches wiring bugs (wrong dict keys, undefined names, bad column refs)
that plain syntax-checking can't.

Run with: python3 test_app_smoke.py
"""

import sys
import types
import numpy as np
import pandas as pd


class StopExecution(Exception):
    pass


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
    def button(self, *a, **k):
        return False  # simulate: user hasn't clicked refresh


class FakeColumn:
    def metric(self, *a, **k): pass
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

    def _columns(n, *a, **k):
        return [FakeColumn() for _ in range(n)]
    st.columns = _columns

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
        def update_layout(self, *a, **k): pass

    class FakeScatter:
        def __init__(self, *a, **k): pass

    go.Figure = FakeFigure
    go.Scatter = FakeScatter

    plotly_pkg = types.ModuleType("plotly")
    plotly_pkg.graph_objects = go

    sys.modules["plotly"] = plotly_pkg
    sys.modules["plotly.graph_objects"] = go


def make_fake_universe(n=12, days=400, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days, freq="B")
    universe = {}
    for i in range(n):
        drift = rng.uniform(-0.001, 0.0018)
        vol = rng.uniform(0.01, 0.025)
        rets = rng.normal(drift, vol, days)
        close = 50 * np.cumprod(1 + rets)
        volume = rng.integers(500_000, 5_000_000, days).astype(float)
        df = pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": volume},
            index=idx,
        )
        universe[f"TICK{i}"] = df
    # benchmark
    bench_rets = rng.normal(0.0004, 0.01, days)
    bench_close = 400 * np.cumprod(1 + bench_rets)
    universe["SPY"] = pd.DataFrame(
        {"Open": bench_close, "High": bench_close * 1.01, "Low": bench_close * 0.99,
         "Close": bench_close, "Volume": rng.integers(1e7, 2e7, days).astype(float)},
        index=idx,
    )
    return universe


def main():
    install_fake_streamlit()
    install_fake_plotly()

    import data_fetch
    fake_universe = make_fake_universe()
    fake_tickers = [t for t in fake_universe if t != "SPY"]

    data_fetch.get_sp500_tickers = lambda: fake_tickers
    data_fetch.download_price_history = lambda tickers, **kwargs: {
        t: fake_universe[t] for t in list(tickers) + ["SPY"] if t in fake_universe
    }

    # app.py checks `if refresh or st.session_state.price_data is None:` on first run,
    # and our fake sidebar.button() returns False for refresh, so first run relies on
    # session_state.price_data being None initially -- that's the default path we want to test.

    print("Running app.py end-to-end with synthetic data (no real network/streamlit)...")
    try:
        with open("app.py") as f:
            code = f.read()
        exec(compile(code, "app.py", "exec"), {"__name__": "__main__"})
    except StopExecution:
        print("  (script called st.stop() -- treated as a controlled stop, not a crash)")

    print("\n[PASS] app.py ran without raising an unhandled exception.")


if __name__ == "__main__":
    main()
