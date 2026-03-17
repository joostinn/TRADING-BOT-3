$ErrorActionPreference = "Stop"

cd "C:\Users\Justin\Downloads\quant_backtests"
.\.venv\Scripts\Activate.ps1

# Set these before running:
# $env:ALPACA_API_KEY="..."
# $env:ALPACA_SECRET_KEY="..."
# $env:ALPACA_PAPER="true"
# $env:TELEGRAM_BOT_TOKEN="..."
# $env:TELEGRAM_CHAT_ID="..."

# Demo: 5 widely followed tickers
@"
symbol
AAPL
MSFT
NVDA
AMZN
META
"@ | Out-File -Encoding utf8 ".\universe_demo_5.csv"

$env:UNIVERSE_CSV="universe_demo_5.csv"

python .\alpaca_trade_list.py

Write-Host ""
Write-Host "Demo completed. Check output_live\ for orders_*.csv and watchlist_*.csv"
