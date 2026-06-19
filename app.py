from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from stable_baselines3 import DQN, PPO

from data import make_synthetic_market_data
from trading_env import TradingEnv


MODEL_OPTIONS = {
    "PPO": Path("models/ppo_trading_agent.zip"),
    "DQN": Path("models/dqn_trading_agent.zip"),
}


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


@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_OPTIONS[model_name]
    if not model_path.exists():
        return None

    model_cls = PPO if model_name == "PPO" else DQN
    return model_cls.load(model_path)


def evaluate_model(model_name: str, seed: int):
    data = make_synthetic_market_data(seed=seed)
    env = TradingEnv(data)
    model = load_model(model_name)

    if model is None:
        return None, None

    obs, _ = env.reset()
    done = False
    actions: list[int] = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        actions.append(action_int)
        obs, _, terminated, truncated, _ = env.step(action_int)
        done = terminated or truncated

    return env.history, actions


def main() -> None:
    st.set_page_config(
        page_title="RL Trading Agent",
        page_icon="📈",
        layout="wide",
    )

    st.title("RL Trading Agent")
    st.caption("Gymnasium environment trained with PPO and DQN using Stable-Baselines3")

    st.warning(
        "Educational project only. This demo uses synthetic market data and is not financial advice."
    )

    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
        seed = st.slider("Synthetic market seed", min_value=1, max_value=250, value=99)
        run_button = st.button("Run Evaluation", type="primary")

    st.subheader("Environment")
    col1, col2, col3 = st.columns(3)
    col1.metric("Actions", "Hold / Buy / Sell")
    col2.metric("Initial Cash", "$10,000")
    col3.metric("Transaction Fee", "0.1%")

    if run_button:
        values, actions = evaluate_model(model_name, seed)

        if values is None or actions is None:
            st.error(
                f"Model file not found: {MODEL_OPTIONS[model_name]}. "
                "Train the model locally and commit the model file before deploying."
            )
            return

        total_return = values[-1] / values[0] - 1
        sharpe = sharpe_ratio(values)
        drawdown = max_drawdown(values)

        st.subheader("Evaluation Results")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Final Portfolio Value", f"${values[-1]:,.2f}")
        metric_cols[1].metric("Total Return", f"{total_return:.2%}")
        metric_cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
        metric_cols[3].metric("Max Drawdown", f"{drawdown:.2%}")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(values, linewidth=1.8)
        ax.set_title(f"{model_name} Portfolio Value")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        action_counts = {
            "Hold": actions.count(0),
            "Buy": actions.count(1),
            "Sell": actions.count(2),
        }
        st.subheader("Action Distribution")
        st.bar_chart(action_counts)
    else:
        st.info("Choose a model and click Run Evaluation to generate live results.")


if __name__ == "__main__":
    main()

