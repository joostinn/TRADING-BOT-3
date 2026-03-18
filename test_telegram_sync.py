#!/usr/bin/env python3
"""
Simple synchronous test for Telegram bot using direct HTTP requests.
"""

import os
import requests

def send_telegram_message(token, chat_id, message):
    """Send a message using direct HTTP request."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, data=data, timeout=10)
        return response.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    # Check environment variables
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    print("🔍 Checking Telegram configuration...")
    print(f"Token set: {'✅' if token else '❌'}")
    print(f"Chat ID set: {'✅' if chat_id else '❌'}")

    if not token or not chat_id:
        print("\n❌ Missing environment variables!")
        return

    print("\n✅ Configuration found!")
    print("📤 Sending test messages...")

    # Test basic message
    result1 = send_telegram_message(token, chat_id, "🤖 *Trading Bot Test*\n\n✅ Telegram integration is working!\n\nFeatures:\n• Trade notifications\n• Portfolio updates\n• Error alerts\n• Performance summaries")
    print(f"Test message: {'✅' if result1.get('ok') else '❌'}")

    # Test trade notification
    result2 = send_telegram_message(token, chat_id, "🔄 *Trade Executed*\nSymbol: SPY\nSide: BUY\nQuantity: 10.50\nPrice: $450.25\nValue: $4727.63")
    print(f"Trade notification: {'✅' if result2.get('ok') else '❌'}")

    # Test portfolio update
    result3 = send_telegram_message(token, chat_id, "📊 *Portfolio Rebalanced*\nTotal Value: $125000.50\n\n*Positions:*\nSPY: 30.0%\nQQQ: 20.0%\nIWM: -15.0%\nXLK: 25.0%")
    print(f"Portfolio update: {'✅' if result3.get('ok') else '❌'}")

    if result1.get('ok') and result2.get('ok') and result3.get('ok'):
        print("\n🎉 All tests passed! Telegram bot is working perfectly!")
        print("Check your Telegram for the test messages.")
    else:
        print("\n❌ Some tests failed. Check your token and chat ID.")
        if not result1.get('ok'):
            print(f"Error: {result1}")
        if not result2.get('ok'):
            print(f"Error: {result2}")
        if not result3.get('ok'):
            print(f"Error: {result3}")

if __name__ == "__main__":
    main()