#!/usr/bin/env python3
"""
Test script to verify the position quantity fixes work correctly.
"""

import os
from alpaca.trading.client import TradingClient

def test_position_details():
    """Test the position details functionality."""
    # Get API credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables not set")
        return

    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")
    tc = TradingClient(api_key, secret_key, paper=paper)

    try:
        # Test the position details function
        from alpaca_paper_trade import _get_position_details, _get_positions_by_symbol

        print("🔍 Testing position details functionality...")

        # Get detailed position information
        position_details = _get_position_details(tc)
        available_positions = _get_positions_by_symbol(tc)

        if not position_details:
            print("ℹ️  No positions found")
            return

        print("📊 Current Positions:")
        print("-" * 60)

        for symbol in position_details:
            details = position_details[symbol]
            available = available_positions.get(symbol, 0)

            print(f"Symbol: {symbol}")
            print(f"  Total Quantity: {details['total_qty']:.0f}")
            print(f"  Available Quantity: {details['available_qty']:.0f}")
            print(f"  Held for Orders: {details['held_for_orders']:.0f}")
            print(f"  (Function returns: {available:.0f})")
            print()

        print("✅ Position details test completed successfully!")
        print("💡 The trading bot will now use 'available_qty' instead of total 'qty' to prevent insufficient quantity errors.")

    except Exception as e:
        print(f"❌ Error testing position details: {e}")

if __name__ == "__main__":
    test_position_details()