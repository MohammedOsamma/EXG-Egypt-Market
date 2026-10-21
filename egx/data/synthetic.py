"""SYNTHETIC random-walk data for tests/smoke runs ONLY. Not market data."""
import numpy as np
import pandas as pd


def make_synthetic(n=600, seed=1, start_price=10.0):
    rng = np.random.default_rng(seed)
    drift = np.where(np.sin(np.arange(n) / 60) > 0, 0.0015, -0.001)
    ret = drift + rng.normal(0, 0.02, n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = np.r_[start_price, close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(200_000, 900_000, n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
