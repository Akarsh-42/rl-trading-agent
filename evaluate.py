from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO

from data import make_synthetic_market_data
from trading_env import TradingEnv


def sharpe_ratio(values: list[float]) -> float:
    returns = np.diff(values) / values[:-1]
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def max_drawdown(values: list[float]) -> float:
    arr = np.array(values)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(drawdowns.min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    data = make_synthetic_market_data(seed=args.seed)
    env = TradingEnv(data)

    model_cls = PPO if "ppo" in args.model.lower() else DQN
    model = model_cls.load(args.model, env=env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    values = env.history
    total_return = values[-1] / values[0] - 1
    print(f"Final portfolio value: ${values[-1]:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Sharpe ratio: {sharpe_ratio(values):.2f}")
    print(f"Max drawdown: {max_drawdown(values):.2%}")

    Path("reports").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(values)
    plt.title("RL Trading Agent Portfolio Value")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    output_path = Path("reports/equity_curve.png")
    plt.savefig(output_path)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()

