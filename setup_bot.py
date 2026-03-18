#!/usr/bin/env python3
"""
Complete setup script for the trading bot with Telegram notifications.
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run a command and return success."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🚀 Trading Bot Setup with Telegram Notifications")
    print("=" * 50)

    # Check if virtual environment exists
    if not os.path.exists(".venv"):
        print("❌ Virtual environment not found!")
        print("Run: python -m venv .venv")
        return

    # Check if packages are installed
    print("📦 Checking dependencies...")
    try:
        import pandas
        import numpy
        import yfinance
        import matplotlib
        import vectorbt
        import alpaca_py
        import requests
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return

    # Get Telegram configuration
    print("\n🤖 Telegram Bot Setup")
    print("-" * 20)

    token = input("Enter your Telegram bot token: ").strip()
    if not token:
        print("❌ No token provided")
        return

    chat_id = input("Enter your Telegram chat ID: ").strip()
    if not chat_id:
        print("❌ No chat ID provided")
        return

    # Set environment variables
    os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    os.environ["TELEGRAM_ENABLED"] = "true"

    print("\n✅ Telegram configuration set!")

    # Test the bot
    print("\n📤 Testing Telegram bot...")
    from telegram_bot import get_notifier, init_telegram_bot

    init_telegram_bot(token, chat_id, True)
    notifier = get_notifier()

    success = notifier._send_message("🤖 *Trading Bot Setup Complete!*\n\n✅ Telegram integration is working!\n\nYour bot will send:\n• Trade notifications\n• Portfolio updates\n• Error alerts\n• Performance summaries")

    if success:
        print("✅ Telegram test successful!")
    else:
        print("❌ Telegram test failed. Check your token and chat ID.")
        return

    # Get Alpaca configuration
    print("\n📊 Alpaca API Setup")
    print("-" * 15)

    alpaca_key = input("Enter your Alpaca API key (optional for testing): ").strip()
    alpaca_secret = input("Enter your Alpaca API secret (optional for testing): ").strip()

    if alpaca_key and alpaca_secret:
        os.environ["ALPACA_API_KEY"] = alpaca_key
        os.environ["ALPACA_SECRET_KEY"] = alpaca_secret
        os.environ["ALPACA_PAPER"] = "true"
        print("✅ Alpaca API configured")
    else:
        print("⚠️  Alpaca API not configured (you can set it later)")

    print("\n🎉 Setup Complete!")
    print("\nTo start the bot:")
    print("1. Set environment variables (or use the commands below):")
    print(f"   $env:TELEGRAM_BOT_TOKEN='{token}'")
    print(f"   $env:TELEGRAM_CHAT_ID='{chat_id}'")
    print("   $env:TELEGRAM_ENABLED='true'")
    if alpaca_key:
        print(f"   $env:ALPACA_API_KEY='{alpaca_key}'")
        print(f"   $env:ALPACA_SECRET_KEY='{alpaca_secret}'")
        print("   $env:ALPACA_PAPER='true'")

    print("\n2. Start live trading:")
    print("   python .\\alpaca_paper_trade.py")

    print("\n3. Or run backtests:")
    print("   python .\\run_backtests.py")

if __name__ == "__main__":
    main()