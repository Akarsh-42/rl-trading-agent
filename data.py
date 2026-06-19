from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_market_data(n_steps: int = 2500, seed: int = 7) -> pd.DataFrame:
    """Create a noisy trending price series with simple technical features."""
    rng = np.random.default_rng(seed)

    trend = np.linspace(0, 1.4, n_steps)
    cycle = 0.08 * np.sin(np.linspace(0, 22 * np.pi, n_steps))
    noise = rng.normal(0, 0.012, n_steps)
    log_returns = 0.0004 + cycle / 100 + noise

    price = 100 * np.exp(np.cumsum(log_returns + trend / 10000))
    df = pd.DataFrame({"close": price})
    df["return"] = df["close"].pct_change().fillna(0)
    df["ma_fast"] = df["close"].rolling(8).mean().bfill()
    df["ma_slow"] = df["close"].rolling(32).mean().bfill()
    df["ma_ratio"] = df["ma_fast"] / df["ma_slow"] - 1
    df["volatility"] = df["return"].rolling(20).std().fillna(0)
    return df

