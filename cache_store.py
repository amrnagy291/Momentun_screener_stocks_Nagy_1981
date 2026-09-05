"""
cache_store.py

Lightweight disk cache for downloaded price history, so relaunching the app
doesn't force a full re-download every time you open it -- only
data_fetch.download_price_history's actual network call is slow; everything
else (ranking, scoring) is fast and recomputes instantly.

Cached as one pickle file per "universe key" (e.g. a specific stock universe
+ size, or crypto) under a .cache/ folder next to this file. Each cache entry
also records when it was saved, so the app can show "last updated: X ago"
and let a manual refresh always get fresh data regardless of the cache.

Deliberately simple (a single pickle per key, no eviction policy, no locking)
since this is a single-user local app, not a shared service. No dependency
on Streamlit, so it can be unit tested standalone.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _cache_path(key: str) -> Path:
    # Cache keys are built from things like universe name/size, which can
    # contain spaces, parens, "&", etc. -- sanitize down to a safe filename.
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return CACHE_DIR / f"{safe_key}.pkl"


def save(key: str, price_data: dict[str, pd.DataFrame]) -> float:
    """Persist price_data to disk, tagged with the current time. Returns that timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.time()
    path = _cache_path(key)
    tmp_path = path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump({"timestamp": timestamp, "price_data": price_data}, f)
    tmp_path.replace(path)  # atomic rename -- avoids ever reading a half-written cache file
    return timestamp


def load(key: str) -> tuple[dict[str, pd.DataFrame] | None, float | None]:
    """
    Returns (price_data, timestamp), or (None, None) if there's no cache yet
    for this key, or the cache file can't be read (corrupted, wrong pickle
    protocol, etc). A bad cache is treated as a cache miss, not an error --
    the app just re-downloads instead of crashing.
    """
    path = _cache_path(key)
    if not path.exists():
        return None, None
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        return blob.get("price_data"), blob.get("timestamp")
    except Exception:
        return None, None


def age_seconds(timestamp: float | None) -> float | None:
    """How many seconds old a cache timestamp is, or None if there's no timestamp."""
    if timestamp is None:
        return None
    return max(0.0, time.time() - timestamp)


def format_age(seconds: float | None) -> str:
    """Human-readable age string for display, e.g. '3 minutes ago', '2.1 hours ago'."""
    if seconds is None:
        return "never"
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} minute{'s' if minutes >= 1.5 else ''} ago"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hour{'s' if hours >= 1.5 else ''} ago"
    days = hours / 24
    return f"{days:.1f} day{'s' if days >= 1.5 else ''} ago"
