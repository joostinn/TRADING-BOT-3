# Quant Backtests (research)

This folder contains a small, reproducible backtesting harness for a few widely studied quant strategy families:

- Time-series momentum / trend following (trend filter + volatility targeting)
- Cross-asset momentum rotation (12-1 momentum, monthly rebalance)
- Simple benchmark (buy & hold SPY)

## Setup

In PowerShell:

```powershell
cd C:\Users\Justin\Downloads\quant_backtests
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\run_backtests.py
```

## Semi-automated live workflow (recommended first)

Generate a **trade list CSV** using **Alpaca market data**, then execute manually.

```powershell
cd C:\Users\Justin\Downloads\quant_backtests
.\.venv\Scripts\Activate.ps1

$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"
$env:ALPACA_PAPER="true"

python .\alpaca_trade_list.py
```

Outputs land in `output_live\`:
- `orders_*.csv`: buy/sell quantities to place
- `watchlist_*.csv`: the current algorithm-selected names (your “boom candidates” list)

## Build a larger universe (Alpaca, top by dollar volume)

If you want to test on a much larger set of US stocks, generate a universe from Alpaca:

```powershell
cd C:\Users\Justin\Downloads\quant_backtests
.\.venv\Scripts\Activate.ps1

$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"
$env:ALPACA_PAPER="true"

python .\build_universe_alpaca.py
```

This writes `universe_us_top_dollar_volume.csv` (default top 300 by recent average dollar volume).

Then re-run backtests using it:

```powershell
$env:UNIVERSE_CSV="universe_us_top_dollar_volume.csv"
python .\us_stocks_vectorbt.py
```

## Visualize the strategy (equity, drawdown, rolling Sharpe)

After running the backtests, generate plots for `xs_mom_ls`:

```powershell
python .\visualize_xs_mom_ls.py
```

Plots are saved to `output_us_stocks\plots\`.

## 5-stock live trade demo (CSV + Telegram)

This demonstrates the full live flow with a tiny universe (5 stocks), including Telegram alerts:

```powershell
cd C:\Users\Justin\Downloads\quant_backtests
.\.venv\Scripts\Activate.ps1

$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"
$env:ALPACA_PAPER="true"

$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"

.\demo_live_trade_5_stocks.ps1
```

## Telegram alerts (Option A)

This project can send you a Telegram message whenever a new trade list is generated.

### 1) Create the bot and get your token

- In Telegram, open `@BotFather`
- Run `/newbot`
- Copy the token (looks like `123456789:ABCDEF...`)

Set it in PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
```

### 2) Get your chat id

Simplest:

- Send a message to your new bot (e.g. “hi”)
- In a browser, open:
  - `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
- Find `"chat":{"id": ... }` in the JSON

Set it in PowerShell:

```powershell
$env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
```

### 3) Run the trade list generator

```powershell
python .\alpaca_trade_list.py
```

If `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set, it will send a message including:
- strategy name
- number of orders
- filenames written
- top picks (your “boom candidates” list per the algorithm)

### Add Telegram alerts to your phone

- Install the **Telegram** app on iOS/Android
- Log in to the same Telegram account
- Search for your bot username and tap **Start**
- Once you run `alpaca_trade_list.py`, you’ll get alerts on your phone like any other Telegram message

## Notes

- Results are **not** investment advice and do **not** guarantee future performance.
- Backtests here are daily-bar, end-of-day signals, and include basic transaction costs and slippage assumptions.
