from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from data import make_synthetic_market_data
from trading_env import TradingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = make_synthetic_market_data(seed=args.seed)
    env = TradingEnv(data)
    check_env(env, warn=True)

    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
        model_name = "ppo_trading_agent"
    else:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            seed=args.seed,
            learning_starts=1000,
            buffer_size=50_000,
        )
        model_name = "dqn_trading_agent"

    model.learn(total_timesteps=args.timesteps)
    Path("models").mkdir(exist_ok=True)
    model.save(f"models/{model_name}")
    print(f"Saved model to models/{model_name}.zip")


if __name__ == "__main__":
    main()

