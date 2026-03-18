#!/usr/bin/env python3
"""
Quick setup script for Telegram bot configuration.
"""

import os

def main():
    print("🤖 Telegram Bot Setup")
    print("=" * 30)

    # Get bot token
    token = input("Enter your Telegram bot token: ").strip()
    if not token:
        print("❌ No token provided")
        return

    # Get chat ID
    chat_id = input("Enter your Telegram chat ID: ").strip()
    if not chat_id:
        print("❌ No chat ID provided")
        return

    # Set environment variables
    os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    os.environ["TELEGRAM_ENABLED"] = "true"

    print("\n✅ Environment variables set!")
    print(f"Token: {token[:20]}...")
    print(f"Chat ID: {chat_id}")

    # Test the bot
    print("\n📤 Testing bot...")
    from telegram_bot import get_notifier, init_telegram_bot

    init_telegram_bot(token, chat_id, True)
    notifier = get_notifier()

    try:
        notifier.send_message_sync("🤖 *Bot Setup Complete!*\n\n✅ Telegram integration is working!\n\nYour trading bot will now send:\n• Trade notifications\n• Portfolio updates\n• Error alerts\n• Performance summaries")
        print("✅ Test message sent! Check your Telegram.")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()