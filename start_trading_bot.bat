@echo off
cd /d C:\Users\Justin\Downloads\quant_backtest_bot
set ALPACA_API_KEY=PKPZOFWR3MAJOUEVKVS7O7744S
set ALPACA_SECRET_KEY=3TrJX6xZXBqnzsmoucJVWuRzMVKhMtRqsuBxfaYEgBLj
set TELEGRAM_BOT_TOKEN=8637252922:AAHXntggeFcb8VWk77yJIAzmwU4q7Z8eYGA
set TELEGRAM_CHAT_ID=1192686660
set ALPACA_PAPER=true
C:\Users\Justin\Downloads\quant_backtest_bot\.venv\Scripts\python.exe alpaca_paper_trade.py