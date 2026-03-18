#!/usr/bin/env python3
"""
Test script for the Telegram bot functionality.
Run this to verify your Telegram setup is working.
"""

import os
from telegram_bot import get_notifier, init_telegram_bot

def main():
    # Check if environment variables are set
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("1", "true", "yes", "y")

    print("🔍 Checking Telegram configuration...")
    print(f"Token set: {'✅' if token else '❌'}")
    print(f"Chat ID set: {'✅' if chat_id else '❌'}")
    print(f"Enabled: {'✅' if enabled else '❌'}")

    if not token:
        print("\n❌ TELEGRAM_BOT_TOKEN not found!")
        print("Set it with: $env:TELEGRAM_BOT_TOKEN='your_bot_token'")
        return

    if not chat_id:
        print("\n❌ TELEGRAM_CHAT_ID not found!")
        print("To get your Chat ID:")
        print("1. Message your bot on Telegram")
        print("2. Visit: https://api.telegram.org/bot" + token + "/getUpdates")
        print("3. Find your chat ID in the JSON response")
        print("4. Set it with: $env:TELEGRAM_CHAT_ID='your_chat_id'")
        return

    print("\n✅ All environment variables found!")

    # Initialize the bot
    init_telegram_bot(token, chat_id, enabled)
    notifier = get_notifier()

    # Send test messages
    print("📤 Sending test messages...")

    try:
        notifier.send_message_sync("🤖 *Trading Bot Test*\n\n✅ Telegram integration is working!\n\nFeatures:\n• Trade notifications\n• Portfolio updates\n• Error alerts\n• Performance summaries")

        # Simulate some notifications
        notifier.notify_trade_execution("SPY", "BUY", 10.5, 450.25)
        notifier.notify_trade_execution("QQQ", "SELL", 5.2, 380.80)

        # Simulate portfolio update
        positions = {
            "SPY": 0.3,
            "QQQ": 0.2,
            "IWM": -0.15,
            "XLK": 0.25
        }
        notifier.notify_rebalance(positions, 125000.50)

        print("✅ Test messages sent! Check your Telegram for notifications.")
        print("\n🎉 Telegram bot is ready for live trading!")

    except Exception as e:
        print(f"❌ Error sending messages: {e}")
        print("Check your bot token and chat ID are correct.")

if __name__ == "__main__":
    main()