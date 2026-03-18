#!/usr/bin/env python3
"""
Test Alpaca API connection and account details.
"""

import os
from alpaca.trading.client import TradingClient

def test_alpaca_connection():
    """Test Alpaca API connection and get account info."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

    print("🔍 Testing Alpaca API connection...")
    print(f"API Key set: {'✅' if api_key else '❌'}")
    print(f"Secret Key set: {'✅' if secret_key else '❌'}")
    print(f"Paper trading: {'✅' if paper else '❌ LIVE TRADING'}")

    if not api_key or not secret_key:
        print("\n❌ Missing Alpaca API credentials!")
        print("Get them from: https://alpaca.markets/docs/trading/api/")
        return False

    try:
        # Create client
        tc = TradingClient(api_key, secret_key, paper=paper)

        # Get account info
        account = tc.get_account()

        print("\n✅ Alpaca connection successful!")
        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
        print(f"Equity: ${float(account.equity):.2f}")
        print(f"Buying Power: ${float(account.buying_power):.2f}")
        print(f"Cash: ${float(account.cash):.2f}")

        # Test market data (optional)
        try:
            from alpaca.data.client import StockHistoricalDataClient
            data_client = StockHistoricalDataClient(api_key, secret_key)
            # Quick test - get SPY data
            bars = data_client.get_stock_bars("SPY", timeframe="1Day", limit=1)
            if bars:
                print("✅ Market data access working")
            else:
                print("⚠️  Market data access may have issues")
        except Exception as e:
            print(f"⚠️  Market data test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
        print("Check your API keys and internet connection.")
        return False

if __name__ == "__main__":
    success = test_alpaca_connection()
    if success:
        print("\n🎉 Ready to start automated trading!")
        print("Run: python alpaca_paper_trade.py")
    else:
        print("\n❌ Fix the issues above before trading.")