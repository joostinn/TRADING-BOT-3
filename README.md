# Quant Backtests (research)

This folder contains a small, reproducible backtesting harness for a few widely studied quant strategy families with enhanced risk management features:

- **Enhanced Time-series momentum / trend following** with:
  - Market regime filter (SPY above 200-day MA)
  - Tightened stop loss during drawdowns
  - Sharpe ratio-based position scaling
  - Volatility scaling for position sizing
  - Multiple lookback windows (63-day and 252-day)
- Cross-asset momentum rotation (12-1 momentum, monthly rebalance)
- Simple benchmark (buy & hold SPY)

## Recent Enhancements

### Risk Management Features
- **Market Regime Filter**: Stops trading momentum when SPY is below its 200-day moving average
- **Dynamic Stop Loss**: Reduces position sizes faster during drawdowns (up to 80% reduction)
- **Sharpe Scaling**: Scales exposure down when rolling Sharpe ratio drops below 0.5
- **Volatility Scaling**: Sizes positions inversely to recent volatility (21-day lookback)
- **Multiple Lookbacks**: Tests both 63-day (3-month) and 252-day (12-month) momentum windows

### Telegram Bot Integration
- Real-time trade execution notifications
- Portfolio rebalance alerts
- Performance updates
- Error monitoring and alerts
- Market regime change notifications

## Quick Setup (Recommended)

Run the automated setup script:

```powershell
cd C:\Users\Justin\Downloads\quant_backtest_bot
python setup_bot.py
```

This will:
- Check dependencies
- Configure your Telegram bot
- Set up Alpaca API (optional)
- Test everything works
- Provide final commands to start trading

## Manual Setup

### 1. Environment Setup

```powershell
cd C:\Users\Justin\Downloads\quant_backtest_bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Telegram Bot Setup

1. Create bot with @BotFather on Telegram
2. Get your chat ID by messaging your bot, then visit:
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`

3. Set environment variables:
```powershell
$env:TELEGRAM_BOT_TOKEN="8637252922:AAHXntggeFcb8VWk77yJIAzmwU4q7Z8eYGA"
$env:TELEGRAM_CHAT_ID="1192686660"  # Your chat ID
$env:TELEGRAM_ENABLED="true"
```

4. Test the bot:
```powershell
python test_telegram.py
```

### 3. Alpaca API Setup

#### **Get Your Alpaca API Keys**

1. **Sign up for Alpaca** (if you haven't already):
   - Go to: https://alpaca.markets/
   - Click "Get Started" → "Individual" → "Paper Trading"
   - Complete account verification (takes ~5 minutes)

2. **Generate API Keys**:
   - Log into your Alpaca dashboard
   - Go to "Account" → "API Keys" 
   - Click "Generate New Key"
   - **Save both keys immediately** (secret key won't be shown again!)

3. **Set Environment Variables**:
```powershell
$env:ALPACA_API_KEY="YOUR_API_KEY_HERE"
$env:ALPACA_SECRET_KEY="YOUR_SECRET_KEY_HERE"
$env:ALPACA_PAPER="true"  # Always start with paper trading!
```

4. **Test Your Connection**:
```powershell
python test_alpaca.py
```

#### **⚠️ Important Notes:**
- **Start with Paper Trading**: `$env:ALPACA_PAPER="true"` (fake money)
- **Never share your secret key**
- **Live trading** requires additional verification and uses real money
- **Paper trading** is perfect for testing your bot safely

### 4. Start Automated Trading

Once you have both Telegram and Alpaca set up:

```powershell
# Set all environment variables
$env:TELEGRAM_BOT_TOKEN="8637252922:AAHXntggeFcb8VWk77yJIAzmwU4q7Z8eYGA"
$env:TELEGRAM_CHAT_ID="1192686660"
$env:TELEGRAM_ENABLED="true"
$env:ALPACA_API_KEY="YOUR_API_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET_KEY"
$env:ALPACA_PAPER="true"

# Start the bot
python .\alpaca_paper_trade.py
```

The bot will:
- Send a startup message to Telegram
- Monitor markets and rebalance every 30 minutes
- Send trade notifications for every order
- Alert you to any errors or issues

## Semi-automated live workflow (recommended first)

Generate a **trade list CSV** using **Alpaca market data**, then execute manually.

```powershell
cd C:\Users\Justin\Downloads\quant_backtest_bot
.\.venv\Scripts\Activate.ps1

$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"
$env:ALPACA_PAPER="true"

python .\alpaca_trade_list.py
```

Outputs land in `output_live\`:
- `orders_*.csv`: buy/sell quantities to place
- `watchlist_*.csv`: the current algorithm-selected names (your "boom candidates" list)

## Automated Live Trading with Telegram Notifications

For fully automated paper trading with Telegram alerts:

```powershell
cd C:\Users\Justin\Downloads\quant_backtest_bot
.\.venv\Scripts\Activate.ps1

$env:ALPACA_API_KEY="YOUR_KEY"
$env:ALPACA_SECRET_KEY="YOUR_SECRET"
$env:ALPACA_PAPER="true"
$env:TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN"
$env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"

python .\alpaca_paper_trade.py
```

The bot will:
- Send startup confirmation
- Rebalance portfolio every 30 minutes
- Notify on all trades executed
- Alert on errors or issues
- Send performance summaries

## Strategy Configuration

Edit `PaperConfig` in `alpaca_paper_trade.py` to customize:

```python
@dataclass(frozen=True)
class PaperConfig:
    tickers: Tuple[str, ...] = ("SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV")
    strategy: str = "enhanced_momentum"  # Use enhanced momentum strategy
    allow_short: bool = True
    max_positions: int = 5
    target_gross_exposure: float = 0.95
    rebalance_cooldown_sec: int = 30 * 60  # 30 minutes
```

## Build a larger universe (Alpaca, top by dollar volume)

If you want to test on a much larger set of US stocks, generate a universe from Alpaca:

```powershell
cd C:\Users\Justin\Downloads\quant_backtest_bot
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
cd C:\Users\Justin\Downloads\quant_backtest_bot
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
