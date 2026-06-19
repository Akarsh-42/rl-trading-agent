from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Simple long-only trading environment for RL experiments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 30,
        initial_cash: float = 10_000.0,
        transaction_fee: float = 0.001,
    ) -> None:
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.feature_columns = ["return", "ma_ratio", "volatility"]

        self.action_space = spaces.Discrete(3)
        obs_size = window_size * len(self.feature_columns) + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares = 0.0
        self.portfolio_value = self.initial_cash
        self.history = [self.portfolio_value]
        return self._get_observation(), {}

    def step(self, action: int):
        previous_value = self.portfolio_value
        price = float(self.data.loc[self.current_step, "close"])

        if action == 1 and self.cash > 0:
            spendable_cash = self.cash * (1 - self.transaction_fee)
            self.shares = spendable_cash / price
            self.cash = 0.0
        elif action == 2 and self.shares > 0:
            self.cash = self.shares * price * (1 - self.transaction_fee)
            self.shares = 0.0

        self.current_step += 1
        next_price = float(self.data.loc[self.current_step, "close"])
        self.portfolio_value = self.cash + self.shares * next_price
        self.history.append(self.portfolio_value)

        reward = (self.portfolio_value - previous_value) / previous_value
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        info = {"portfolio_value": self.portfolio_value}
        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self):
        print(
            f"step={self.current_step} cash={self.cash:.2f} "
            f"shares={self.shares:.4f} value={self.portfolio_value:.2f}"
        )

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        window = self.data.loc[start : self.current_step - 1, self.feature_columns]
        features = window.to_numpy(dtype=np.float32).flatten()
        position = np.array(
            [
                self.cash / self.initial_cash,
                self.shares
                * float(self.data.loc[self.current_step, "close"])
                / self.initial_cash,
            ],
            dtype=np.float32,
        )
        return np.concatenate([features, position]).astype(np.float32)

