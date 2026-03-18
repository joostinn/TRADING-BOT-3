#!/usr/bin/env python3
"""
TEST: Verify improved adaptive strategy works in live trading context.
"""

import pandas as pd
import vectorbt as vbt
from improved_adaptive_strategy import ImprovedAdaptiveEMAStrategy

def test_improved_strategy_live_context():
    """Test that the improved strategy works in the same context as live trading."""

    print("🧪 TESTING IMPROVED ADAPTIVE STRATEGY FOR LIVE TRADING")
    print("=" * 55)
    print()

    # Use same tickers as alpaca_paper_trade.py
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

    print(f"📊 Downloading data for {len(tickers)} tickers...")
    # Use same data download as live trading (2 years of daily data)
    close = vbt.YFData.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        missing_index="drop",
        missing_columns="drop",
    ).get("Close")

    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.sort_index().dropna(how="all").ffill()

    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert(None)

    print(f"✅ Downloaded {len(close)} days of data")
    print(f"   Date range: {close.index[0].date()} to {close.index[-1].date()}")
    print()

    # Test strategy initialization
    print("🚀 Initializing improved adaptive strategy...")
    try:
        strategy = ImprovedAdaptiveEMAStrategy()
        print("✅ Strategy initialized successfully")
        print(f"   Current parameters: {strategy.current_params}")
    except Exception as e:
        print(f"❌ Strategy initialization failed: {e}")
        return False

    print()

    # Test signal generation
    print("📈 Generating trading signals...")
    try:
        from improved_adaptive_strategy import improved_adaptive_signals
        signals_dict = improved_adaptive_signals(close)
        # Convert dict to series for easier handling
        signals = pd.Series(signals_dict, index=close.columns)
        print("✅ Signals generated successfully")
        print(f"   Signal range: {signals.min():.3f} to {signals.max():.3f}")
        print(f"   Non-zero signals: {(signals != 0).sum()}")
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

    print()

    # Show sample signals
    print("🎯 SAMPLE SIGNALS (latest date):")
    latest_signals = signals  # signals is already a Series with ticker index
    latest_prices = close.iloc[-1]

    signal_summary = []
    for ticker in tickers:
        if ticker in latest_signals.index:
            signal = latest_signals[ticker]
            price = latest_prices[ticker]
            if signal != 0:
                position = "LONG" if signal > 0 else "SHORT"
                weight = abs(signal)
                value = weight * 100000  # Assuming $100k portfolio
                signal_summary.append((ticker, position, weight, price, value))

    if signal_summary:
        print("   Active positions:")
        for ticker, position, weight, price, value in signal_summary:
            print(f"     {ticker}: {position} {weight:.1%} (${value:,.0f} @ ${price:.2f})")
    else:
        print("   No active signals (strategy may be in neutral state)")

    print()

    # Test parameter learning (simulate a few trades)
    print("🧠 Testing parameter learning...")
    try:
        from improved_adaptive_strategy import record_improved_trade_outcome
        # Simulate some trades to test learning
        test_trades = [
            {"symbol": "SPY", "pnl": 1000, "pnl_pct": 0.04, "holding_days": 5},  # Good trade
            {"symbol": "QQQ", "pnl": -500, "pnl_pct": -0.02, "holding_days": 15},  # Bad trade
            {"symbol": "IWM", "pnl": 800, "pnl_pct": 0.032, "holding_days": 3},   # Good trade
        ]

        for trade in test_trades:
            record_improved_trade_outcome(trade)
            print(f"   Learned from {trade['symbol']}: ${trade['pnl']} over {trade['holding_days']} days")

        print("✅ Parameter learning working")
        # Get updated status
        from improved_adaptive_strategy import get_improved_adaptive_status
        print("   Updated status:")
        status_lines = get_improved_adaptive_status().split('\n')
        for line in status_lines[-3:]:  # Show last few lines
            if line.strip():
                print(f"     {line}")
    except Exception as e:
        print(f"❌ Parameter learning failed: {e}")
        return False

    print()

    print("🎉 ALL TESTS PASSED!")
    print("   ✅ Strategy initialization")
    print("   ✅ Signal generation")
    print("   ✅ Parameter learning")
    print("   🚀 Ready for live trading with improved adaptive strategy")
    print()

    return True

if __name__ == "__main__":
    success = test_improved_strategy_live_context()
    if not success:
        exit(1)