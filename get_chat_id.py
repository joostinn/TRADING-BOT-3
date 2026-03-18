#!/usr/bin/env python3
"""
Get your Telegram Chat ID for bot setup.
"""

import os
import requests

def get_chat_id():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not set!")
        print("First set: $env:TELEGRAM_BOT_TOKEN='your_bot_token'")
        return

    print("🔍 Checking for recent messages to your bot...")
    print("Make sure you've sent a message to your bot on Telegram first!")

    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        response = requests.get(url, timeout=10)
        data = response.json()

        if data.get("ok") and data.get("result"):
            for update in data["result"]:
                if "message" in update:
                    chat_id = update["message"]["chat"]["id"]
                    print(f"✅ Found Chat ID: {chat_id}")
                    print(f"Set it with: $env:TELEGRAM_CHAT_ID='{chat_id}'")
                    return chat_id
            print("❌ No messages found. Send a message to your bot first!")
        else:
            print("❌ Failed to get updates. Check your bot token.")
            print(f"Response: {data}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_chat_id()