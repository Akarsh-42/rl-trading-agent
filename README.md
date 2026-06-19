# RL Trading Agent with Gymnasium, PPO, DQN, and Stable-Baselines3

This is a portfolio-ready reinforcement learning project where an agent learns to trade a synthetic price series. It includes:

- A custom `Gymnasium` trading environment
- PPO and DQN training with `Stable-Baselines3`
- Evaluation metrics such as total return, Sharpe ratio, and max drawdown
- Reproducible scripts suitable for GitHub and resume discussion

> Educational project only. This is not financial advice and should not be used for live trading.

## 1. Project Idea

The agent observes recent market features and chooses one of three actions:

- `0`: hold
- `1`: buy / go long
- `2`: sell / exit position

The goal is to maximize portfolio value over time while accounting for transaction fees.

## 2. Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Train PPO

```bash
python train.py --algo ppo --timesteps 50000
```

## 4. Train DQN

```bash
python train.py --algo dqn --timesteps 50000
```

## 5. Evaluate

```bash
python evaluate.py --model models/ppo_trading_agent.zip
python evaluate.py --model models/dqn_trading_agent.zip
```

Evaluation creates a portfolio chart in `reports/`.

## 6. Recommended GitHub Structure

```text
rl-trading-agent/
  trading_env.py
  data.py
  train.py
  evaluate.py
  requirements.txt
  README.md
  models/
  reports/
```

Do not commit large model files unless they are small. For a clean portfolio repo, commit the code, README, and sample evaluation plots.

## 7. What To Say On Your Resume

**Reinforcement Learning Trading Agent**

- Built a custom Gymnasium trading environment with discrete buy/sell/hold actions, transaction costs, rolling market observations, and portfolio-based rewards.
- Trained and compared PPO and DQN agents using Stable-Baselines3.
- Evaluated agent performance using cumulative return, Sharpe ratio, max drawdown, and equity curve visualizations.
- Packaged the project with reproducible scripts and documentation for GitHub deployment.

## 8. Interview Explanation

In interviews, explain the project like this:

> I built a reinforcement learning environment where the state contains recent market features, the action space is hold/buy/sell, and the reward is based on portfolio value changes after transaction costs. I trained PPO and DQN agents using Stable-Baselines3, then compared them using financial metrics like Sharpe ratio and max drawdown. The goal was not to build a production trading bot, but to demonstrate RL environment design, reward shaping, model training, and evaluation.

## 9. Next Improvements

- Replace synthetic prices with real stock or crypto data.
- Add short selling.
- Add position sizing instead of all-in/all-out trades.
- Compare against buy-and-hold.
- Use walk-forward validation.
- Add TensorBoard charts.
- Deploy results with Streamlit or GitHub Pages.

