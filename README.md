# RL Trading Agent using Gymnasium, PPO, DQN, and Stable-Baselines3

This project implements a reinforcement learning trading agent that learns to make buy, sell, and hold decisions in a custom market environment. The goal is to demonstrate reinforcement learning environment design, reward shaping, model training, and evaluation using financial performance metrics.

> Educational project only. This is not financial advice and should not be used for live trading.

## Project Demo

The agent is evaluated by tracking portfolio value over time.

![RL Trading Agent Equity Curve](reports/equity_curve.png)

## Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- PPO
- DQN
- NumPy
- Pandas
- Matplotlib

## How It Works

The project uses a custom `TradingEnv` environment that follows the Gymnasium API.

The agent observes market features such as:

- recent returns
- moving-average ratio
- volatility
- current cash allocation
- current invested allocation

The action space is discrete:

- `0`: hold
- `1`: buy / go long
- `2`: sell / exit position

The reward is based on the percentage change in portfolio value after each step, including transaction fees.

## Results

The trained agent is evaluated using:

- final portfolio value
- total return
- Sharpe ratio
- max drawdown
- equity curve visualization

Sample evaluation output:

```text
Final portfolio value: generated during evaluation
Total return: generated during evaluation
Sharpe ratio: generated during evaluation
Max drawdown: generated during evaluation
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train PPO

```bash
python train.py --algo ppo --timesteps 50000
```

## Train DQN

```bash
python train.py --algo dqn --timesteps 50000
```

## Evaluate A Model

```bash
python evaluate.py --model models/ppo_trading_agent.zip
```

or:

```bash
python evaluate.py --model models/dqn_trading_agent.zip
```

The evaluation script saves the chart to:

```text
reports/equity_curve.png
```

## Project Structure

```text
rl-trading-agent/
  data.py
  trading_env.py
  train.py
  evaluate.py
  requirements.txt
  README.md
  reports/
    equity_curve.png
```

## Future Improvements

- Replace synthetic data with real stock or crypto data.
- Add a buy-and-hold benchmark.
- Add short selling and position sizing.
- Use train/test splits and walk-forward validation.
- Add TensorBoard logging.
- Build a Streamlit dashboard for interactive evaluation.
