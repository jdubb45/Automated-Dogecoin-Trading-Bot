
# 🚀 AI Crypto Trading Bot for Gemini

A real-time, reinforcement learning–powered trading bot built in Python, designed to trade DOGE/USD on the Gemini exchange using a Deep Q-Network (DQN) strategy and rich terminal UI.

---

## 📌 Features

- ✅ Real Gemini API integration for live trading
- 💰 Trades DOGE/USD using a rule-based and AI-reinforced strategy
- 🧠 Deep Q-Network agent learns optimal actions (buy/sell/hold)
- 📉 Dynamic reward structure based on profit and portfolio growth
- 📊 Live updating terminal dashboard via `rich`
- 🧮 Trade management includes:
  - Profit and Loss (PnL)
  - Holdings and liquidity checks
  - Stop loss and goal tracking
- 💻 GPU-accelerated training via PyTorch (auto-detects CUDA)

---

## ⚙️ Requirements

- Python 3.10+
- Gemini account with API key/secret
- GPU (optional, but recommended for RL training)

### 🔧 Install dependencies

```bash
pip install requests torch numpy rich
```

---

## 🔑 Configuration

Update your API credentials and settings in the script:

```python
API_KEY = "your-api-key"
API_SECRET = "your-secret"
TRADE_PAIR = "dogeusd"
TRADE_USD = 1.00
MIN_PURCHASE_USD = 0.50
TARGET_AMOUNT = 1000
```

---

## 🧠 How It Works

1. **Market Data Fetching**: Pulls live DOGE/USD price and account balances from Gemini.
2. **DQN Agent**: Learns using past market states to optimize trade decisions.
3. **Trade Execution**: Buys and sells DOGE using limit orders with precision rounding.
4. **Terminal UI**: Displays an interactive trading dashboard including:
   - Current price
   - Profit/Loss (PnL)
   - Liquidity
   - Holdings
   - Action taken (Buy/Sell/Hold)
   - Trade outcome and safety triggers

---

## 📈 Agent Behavior

- Input State: `[price, holdings, usd_balance]`
- Actions: `BUY`, `SELL`, `HOLD`
- Reward Strategy: Incremental profit milestones trigger rewards
- Memory Replay: Uses experience replay for better generalization

---

## 📉 Risk Controls

- `STOP_LOSS_PERCENT`: Halts trading logic if simulated PnL drops too far
- `MIN_PURCHASE_USD`: Avoids micro-trades below this threshold
- `MIN_DOGE_TO_HOLD`: Maintains a small reserve of DOGE

---

## 🖥 Live Terminal Display

| Pair   | Price    | PnL     | Liquidity | Holdings | Profit | Last Trade       | Failsafe | Action |
|--------|----------|---------|-----------|----------|--------|------------------|----------|--------|
| DOGEUSD| 0.12345  | +0.0012 | 22500     | 10.0000  | $0.50  | BUY @ 0.12200... | OK       | BUY    |

---

## 🔐 Security Warning

> Never commit your real `API_KEY` or `API_SECRET` to GitHub. Use `.env` or config files in production.



---

## 📜 License

MIT License – Use at your own risk. Live trading involves real financial loss potential. Backtest before use.
