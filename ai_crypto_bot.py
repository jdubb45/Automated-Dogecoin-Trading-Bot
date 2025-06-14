import time
import json
import hmac
import hashlib
import base64
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from rich.live import Live

# === CONFIG ===
API_KEY = "account-MFBPbzBWLsjWWCsmEqrH"  # Replace with your API key
API_SECRET = "27DqUsph29zoPr7AJ9892tncinee"  # Replace with your API secret
GEMINI_URL = "https://api.gemini.com"
TRADE_PAIR = "dogeusd"
TRADE_USD = 1.00
MIN_PURCHASE_USD = 0.50  # Set minimum USD balance to 50 cents
MIN_DOGE_TO_HOLD = 0.1
TARGET_GROWTH = 0.04
STOP_LOSS_PERCENT = 0.02
RATE_LIMIT = 600
REQUEST_INTERVAL = 60 / RATE_LIMIT
TARGET_AMOUNT = 1000  # Long-term goal
INCREMENTAL_GOAL = 100  # Incremental goal to break down the target
console = Console()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.print(f"Using device: {device}")
if torch.cuda.is_available():
    console.print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# === GEMINI API REQUEST ===
def gemini_request(endpoint, payload=None):
    url = f"{GEMINI_URL}{endpoint}"
    nonce = str(int(time.time() * 1000))
    payload = payload or {}
    payload.update({"request": endpoint, "nonce": nonce})
    b64 = base64.b64encode(json.dumps(payload).encode())
    signature = hmac.new(API_SECRET.encode(), b64, hashlib.sha384).hexdigest()
    headers = {
        "X-GEMINI-APIKEY": API_KEY,
        "X-GEMINI-PAYLOAD": b64.decode(),
        "X-GEMINI-SIGNATURE": signature,
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(url, headers=headers, timeout=10)
        res.raise_for_status()
        console.print(f"[INFO] Request to {endpoint} succeeded with response: {res.json()}")
        return res.json()
    except requests.exceptions.HTTPError as http_err:
        console.print(f"[ERROR] HTTP error occurred: {http_err}")
        console.print(f"[ERROR] Response content: {res.content}")
    except Exception as e:
        console.print(f"[ERROR] Gemini API request failed: {e}")
    return None

def get_usd_balance():
    data = gemini_request("/v1/balances")
    if isinstance(data, list):
        for item in data:
            if item.get('currency', '').upper() == 'USD':
                return float(item.get('available', 0))
    return 0.0

def get_single_holding(symbol):
    data = gemini_request("/v1/balances")
    if isinstance(data, list):
        for item in data:
            if item.get("currency", "").upper() == symbol.upper():
                return float(item.get("available", 0))
    return 0.0

def get_live_price(pair):
    try:
        data = requests.get(f"{GEMINI_URL}/v1/pubticker/{pair}", timeout=5).json()
        return float(data.get('last', 0))
    except Exception as e:
        console.print(f"[ERROR] Failed to get price: {e}")
        return None

def get_symbol_details(pair):
    try:
        d = requests.get(f"{GEMINI_URL}/v1/symbols/details/{pair}", timeout=5).json()
        return float(d["min_order_size"]), float(d["quote_increment"])
    except Exception as e:
        console.print(f"[ERROR] Failed to get symbol details: {e}")
        return 0.0, 0.0001

def execute_trade(pair, price, side, tick_size, min_size, amount):
    symbol = pair.replace("usd", "").upper()
    side = side.lower()
    usd_balance = get_usd_balance()
    crypto_balance = get_single_holding(symbol)
    console.print(f"[INFO] Available USD balance: {usd_balance}")
    console.print(f"[INFO] Available {symbol} balance: {crypto_balance}")

    if side == "buy" and usd_balance < amount * price:
        return f"[red]Not enough USD to buy {amount} {symbol}[/red]"
    if side == "sell" and crypto_balance < amount:
        return f"[red]Not enough {symbol} to sell {amount}[/red]"

    # Adjust amount and price to match Gemini's precision requirements
    adj_amount = max(round(amount / min_size) * min_size, min_size)
    adj_price = round(price / tick_size) * tick_size
    adj_amount = float(f"{adj_amount:.6f}")  # 6 decimal places for amount
    adj_price = float(f"{adj_price:.5f}")  # 5 decimal places for price

    payload = {
        "symbol": pair.lower(),
        "amount": f"{adj_amount}",
        "price": f"{adj_price}",
        "side": side,
        "type": "exchange limit"
    }

    console.print(f"[blue]Sending {side.upper()} order: {payload}[/blue]")
    res = gemini_request("/v1/order/new", payload)
    if res and "order_id" in res:
        return f"{side.upper()} at {adj_price} (ID: {res['order_id']})"
    else:
        console.print(f"[red]Trade failed: {res}[/red]")
        if res and 'message' in res:
            console.print(f"[red]Error message: {res['message']}[/red]")
        return f"[red]Trade failed: {res}[/red]"

# === REINFORCEMENT LEARNING AGENT ===
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model().to(device)  # Move model to GPU
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_amount = TARGET_AMOUNT
        self.incremental_goal = INCREMENTAL_GOAL

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).to(device)  # Move state to GPU
        act_values = self.model(state)
        return np.argmax(act_values.detach().cpu().numpy())  # Move back to CPU for NumPy conversion

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            state = torch.FloatTensor(state).to(device)  # Move state to GPU
            next_state = torch.FloatTensor(next_state).to(device)  # Move next_state to GPU
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, target.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_reward(self, current_balance, holdings, price):
        # Calculate the reward based on the current state and the long-term goal
        total_value = current_balance + holdings * price
        incremental_reward = 0
        if total_value >= self.incremental_goal:
            self.incremental_goal += INCREMENTAL_GOAL
            incremental_reward = 1
        return incremental_reward

# === MAIN LOOP ===
last_trade_log, profit_log = "", 0.0
holdings = 0.0  # Initialize holdings

# Initialize the DQN agent
state_size = 3  # Example: price, holdings, usd_balance
action_size = 3  # Example: buy, sell, hold
agent = DQNAgent(state_size, action_size)

with Live(console=console, refresh_per_second=2) as live:
    while True:
        price = get_live_price(TRADE_PAIR)
        if price is not None:  # Ensure price is not None
            min_size, tick_size = get_symbol_details(TRADE_PAIR)
            if min_size == 0 or tick_size == 0:
                console.print("[ERROR] Invalid symbol details received.")
                time.sleep(REQUEST_INTERVAL)
                continue

            pnl = np.random.uniform(-0.001, 0.001)
            liquidity = np.random.randint(10000, 50000)
            failsafe = pnl < -STOP_LOSS_PERCENT or liquidity < 10000
            usd_balance = get_usd_balance()
            profit_log += pnl * TRADE_USD
            action = "HOLD"

            # Update holdings
            holdings = get_single_holding(TRADE_PAIR.replace("usd", "").upper())
            console.print(f"[INFO] Current holdings: {holdings} DOGE")

            # Prepare the state for the agent
            state = np.array([[price, holdings, usd_balance]])

            # Let the agent decide the action
            action_index = agent.act(state)
            if action_index == 0:
                action = "BUY"
                max_buyable_amount = usd_balance / price
                amount_to_buy = min(max_buyable_amount, MIN_DOGE_TO_HOLD - holdings)
                if amount_to_buy > 0:
                    last_trade_log = execute_trade(TRADE_PAIR, price, "buy", tick_size, min_size, amount_to_buy)
            elif action_index == 1:
                action = "SELL"
                amount_to_sell = min(holdings - MIN_DOGE_TO_HOLD, TRADE_USD / price)
                if amount_to_sell >= min_size:
                    last_trade_log = execute_trade(TRADE_PAIR, price, "sell", tick_size, min_size, amount_to_sell)
            else:
                action = "HOLD"

            # Get the reward based on the long-term goal
            reward = agent.get_reward(usd_balance, holdings, price)

            # Prepare the next state
            next_state = np.array([[price, holdings, usd_balance]])

            # Remember the experience
            agent.remember(state, action_index, reward, next_state, done=False)

            # Replay the experience
            if len(agent.memory) > 32:
                agent.replay(32)

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Pair", style="cyan")
            table.add_column("Price", justify="right")
            table.add_column("PnL", justify="right")
            table.add_column("Liquidity", justify="right")
            table.add_column("Holdings", justify="right")
            table.add_column("Profit", justify="right")
            table.add_column("Last Trade", justify="left")
            table.add_column("Failsafe", justify="center")
            table.add_column("Action", justify="center")
            fs = "[red]ACTIVE[/red]" if failsafe else "[green]OK[/green]"
            pnl_color = "green" if pnl > 0 else "red"
            profit_color = "green" if profit_log >= 0 else "red"
            table.add_row(
                TRADE_PAIR.upper(), f"{price:.5f}", f"[{pnl_color}]{pnl:.4f}[/{pnl_color}]",
                str(liquidity), f"{holdings:.6f}", f"[{profit_color}]{profit_log:.2f}[/{profit_color}]",
                last_trade_log, fs, action
            )
            live.update(table)
        time.sleep(REQUEST_INTERVAL)
