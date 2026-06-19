# Step-by-Step Learning Plan

## Step 1: Understand the RL Loop

Reinforcement learning has five core ideas:

- **Agent**: the learner, here PPO or DQN.
- **Environment**: the world the agent interacts with, here `TradingEnv`.
- **State / observation**: what the agent sees, such as returns and moving-average features.
- **Action**: what the agent does, such as hold, buy, or sell.
- **Reward**: the feedback signal, here portfolio percentage change.

The loop is:

```text
observe state -> choose action -> environment updates -> receive reward -> repeat
```

## Step 2: Learn Why Gymnasium Is Used

Gymnasium gives a standard interface:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Stable-Baselines3 can train on any environment that follows this interface.

## Step 3: Understand This Trading Environment

The environment is in `trading_env.py`.

Observation:

- recent returns
- moving-average ratio
- volatility
- current cash percentage
- current invested percentage

Actions:

- `0`: hold
- `1`: buy
- `2`: sell

Reward:

```text
new portfolio value - old portfolio value
----------------------------------------
old portfolio value
```

This makes the reward equal to portfolio percentage return at each step.

## Step 4: Understand PPO

PPO stands for Proximal Policy Optimization.

Use PPO when:

- you want a strong general-purpose RL baseline
- training should be stable
- the policy directly learns which action to choose

In this project:

```bash
python train.py --algo ppo --timesteps 50000
```

## Step 5: Understand DQN

DQN stands for Deep Q-Network.

Use DQN when:

- the action space is discrete
- you want the agent to learn action values
- actions are simple, like hold/buy/sell

In this project:

```bash
python train.py --algo dqn --timesteps 50000
```

## Step 6: Evaluate Properly

Do not only say the model "trained successfully." Use metrics:

- **Total return**: how much the portfolio grew.
- **Sharpe ratio**: return adjusted for volatility.
- **Max drawdown**: worst peak-to-bottom drop.
- **Equity curve**: chart of portfolio value over time.

Run:

```bash
python evaluate.py --model models/ppo_trading_agent.zip
```

## Step 7: Put It On GitHub

```bash
git init
git add .
git commit -m "Initial RL trading agent project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rl-trading-agent.git
git push -u origin main
```

Before pushing, update the README with:

- your name
- screenshots from `reports/equity_curve.png`
- results from your best PPO and DQN runs
- lessons learned

## Step 8: Resume Version

Use this bullet:

```text
Built a reinforcement learning trading agent using Gymnasium and Stable-Baselines3, implementing a custom market environment with PPO/DQN training, transaction-cost-aware rewards, and evaluation via Sharpe ratio, max drawdown, and equity curve analysis.
```

## Step 9: Internship Interview Questions To Prepare

Be ready to answer:

- Why did you choose PPO and DQN?
- What is the observation space?
- What is the action space?
- How is reward calculated?
- How did you avoid unrealistic trading assumptions?
- Why is train/test separation important for trading?
- What would you improve before using real data?

## Step 10: Best Next Upgrade

The best next upgrade is to add real data and compare the agent against buy-and-hold.

Suggested datasets:

- Yahoo Finance daily stock data
- crypto OHLCV data
- Kaggle stock datasets

Suggested package:

```bash
pip install yfinance
```

