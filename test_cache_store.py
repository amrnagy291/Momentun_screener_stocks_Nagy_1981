"""
Sanity tests for cache_store.py's disk cache. Uses a temporary directory
(monkeypatching CACHE_DIR) so it doesn't touch the real .cache/ folder and
needs no network access.
Run with: python3 test_cache_store.py
"""

import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

import cache_store as cs


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    assert cond, name


def main():
    tmp_dir = Path(tempfile.mkdtemp())
    cs.CACHE_DIR = tmp_dir  # redirect the cache to a scratch dir for this test run
    try:
        # --- cache miss ---
        price_data, ts = cs.load("nonexistent_key")
        check("load() on a missing key returns (None, None)", price_data is None and ts is None)
        check("age_seconds(None) is None", cs.age_seconds(None) is None)
        check("format_age(None) says 'never'", cs.format_age(None) == "never")

        # --- save + load round-trip ---
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [100, 200, 300]})
        fake_price_data = {"AAPL": df, "MSFT": df}
        saved_ts = cs.save("stocks_test", fake_price_data)
        loaded_data, loaded_ts = cs.load("stocks_test")

        check("loaded data round-trips with the same tickers", set(loaded_data.keys()) == {"AAPL", "MSFT"})
        check("loaded DataFrame content matches what was saved", loaded_data["AAPL"].equals(df))
        check("loaded timestamp matches what save() returned", loaded_ts == saved_ts)

        # --- age calculations ---
        age = cs.age_seconds(loaded_ts)
        check("a freshly-saved cache is under a few seconds old", age is not None and age < 5)
        check("format_age of a fresh save says 'just now'", cs.format_age(age) == "just now")

        old_ts = time.time() - 3 * 3600  # 3 hours ago
        check("format_age of a 3-hour-old timestamp mentions hours", "hour" in cs.format_age(cs.age_seconds(old_ts)))

        # --- keys with spaces/parens/slashes don't break the filesystem path ---
        weird_key = "stocks/All US Stocks (NYSE + Nasdaq)/500"
        cs.save(weird_key, fake_price_data)
        weird_loaded, _ = cs.load(weird_key)
        check("a key with slashes/spaces/parens still round-trips", weird_loaded is not None)

        # --- overwriting an existing key replaces it, doesn't merge ---
        cs.save("stocks_test", {"NVDA": df})
        overwritten, _ = cs.load("stocks_test")
        check("saving again under the same key replaces the old contents", set(overwritten.keys()) == {"NVDA"})

        # --- a corrupted cache file is treated as a miss, not a crash ---
        corrupt_path = cs._cache_path("corrupt_test")
        cs.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid pickle")
        corrupt_data, corrupt_ts = cs.load("corrupt_test")
        check("a corrupted cache file is treated as a cache miss, not an exception",
              corrupt_data is None and corrupt_ts is None)

        print("\nAll checks passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
