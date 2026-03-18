"""
PHASE 1 PAPER TRADING MONITOR
Track performance and fine-tune parameters for higher profit and positive net returns.
"""

import os
import time
import pandas as pd
from datetime import datetime
from alpaca.trading.client import TradingClient
import vectorbt as vbt
import pandas_ta as ta
from adaptive_ema_strategy import get_adaptive_strategy, get_adaptive_status, adaptive_ema_signals

def get_current_signals():
    """Get current ADAPTIVE EMA crossover signals for all tickers."""

    # Get 2 years of data for adaptive EMA calculations
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]
    close = vbt.YFData.download(tickers, period='2y', interval='1d').get('Close')
    if isinstance(close, pd.Series):
        close = close.to_frame()

    # Use adaptive strategy signals
    signals = adaptive_ema_signals(close)

    # Get detailed signal information for reporting using current adaptive parameters
    strategy = get_adaptive_strategy()
    params = strategy.current_params

    details = {}
    for symbol in close.columns:
        prices = close[symbol].dropna()
        if len(prices) >= params['slow_ema'] + 10:
            # Calculate current technical indicators for reporting
            fast_ema = ta.ema(prices, length=params['fast_ema'])
            slow_ema = ta.ema(prices, length=params['slow_ema'])
            ema_separation = abs(fast_ema - slow_ema) / prices
            rsi = ta.rsi(prices, length=14)

            details[symbol] = {
                'signal': 1 if signals.get(symbol, 0) > 0 else -1 if signals.get(symbol, 0) < 0 else 0,
                'weight': signals.get(symbol, 0),
                'rsi': rsi.iloc[-1] if len(rsi) > 0 else 0,
                'separation': ema_separation.iloc[-1] if len(ema_separation) > 0 else 0,
                'fast_ema': fast_ema.iloc[-1] if len(fast_ema) > 0 else 0,
                'slow_ema': slow_ema.iloc[-1] if len(slow_ema) > 0 else 0,
                'price': prices.iloc[-1] if len(prices) > 0 else 0,
                'valid_filters': signals.get(symbol, 0) != 0
            }
        else:
            details[symbol] = {
                'signal': 0,
                'weight': 0,
                'rsi': 0,
                'separation': 0,
                'fast_ema': 0,
                'slow_ema': 0,
                'price': 0,
                'valid_filters': False
            }

    return signals, details

def get_portfolio_status():
    """Get current portfolio status from Alpaca."""
    api_key = os.getenv('ALPACA_API_KEY', 'PKPZOFWR3MAJOUEVKVS7O7744S')
    secret_key = os.getenv('ALPACA_SECRET_KEY', '3TrJX6xZXBqnzsmoucJVWuRzMVKhMtRqsuBxfaYEgBLj')

    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    try:
        account = client.get_account()
        positions = client.get_all_positions()

        portfolio_data = {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'positions': {}
        }

        total_value = 0
        total_pnl = 0

        for pos in positions:
            try:
                qty = float(pos.qty)
                entry_price = float(pos.avg_entry_price) if pos.avg_entry_price else 0
                current_price = float(pos.current_price) if pos.current_price else 0
                market_value = float(pos.market_value) if pos.market_value else 0
                unrealized_pl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
                unrealized_plpc = float(pos.unrealized_plpc) if pos.unrealized_plpc else 0

                portfolio_data['positions'][pos.symbol] = {
                    'qty': qty,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'market_value': market_value,
                    'pnl': unrealized_pl,
                    'pnl_pct': unrealized_plpc
                }

                total_value += market_value
                total_pnl += unrealized_pl

            except (ValueError, TypeError) as e:
                print(f"Error parsing position for {pos.symbol}: {e}")
                continue

        portfolio_data['total_position_value'] = total_value
        portfolio_data['total_pnl'] = total_pnl

        return portfolio_data

    except Exception as e:
        print(f"Error getting portfolio status: {e}")
        return None

def monitor_phase1_performance():
    """Monitor Phase 1 paper trading performance and suggest parameter adjustments."""

    print("🚀 PHASE 1 PAPER TRADING MONITOR")
    print("=" * 50)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Get current signals
    print("📊 CURRENT STRATEGY SIGNALS:")
    signals, details = get_current_signals()

    active_signals = {k: v for k, v in signals.items() if v != 0}
    if active_signals:
        for symbol, weight in active_signals.items():
            detail = details[symbol]
            signal_type = "LONG" if weight > 0 else "SHORT"
            print(f"  {symbol}: {signal_type} ({weight:.1f}) - RSI: {detail['rsi']:.1f}, Sep: {detail['separation']:.3f}")
    else:
        print("  No active signals (all positions filtered out)")
    print()

    # Get portfolio status
    print("💰 PORTFOLIO STATUS:")
    portfolio = get_portfolio_status()

    if portfolio:
        print(f"  Equity: ${portfolio['equity']:,.2f}")
        print(f"  Cash: ${portfolio['cash']:,.2f}")
        print(f"  Buying Power: ${portfolio['buying_power']:,.2f}")

        if portfolio['positions']:
            print(f"  Positions ({len(portfolio['positions'])}):")
            for symbol, pos in portfolio['positions'].items():
                pnl_color = "🟢" if pos['pnl'] >= 0 else "🔴"
                print(f"    {symbol}: {pos['qty']:.0f} shares @ ${pos['entry_price']:.2f} | {pnl_color} P&L: ${pos['pnl']:,.2f} ({pos['pnl_pct']:.2f}%)")

            print(f"  Total Position Value: ${portfolio['total_position_value']:,.2f}")
            print(f"  Total P&L: ${portfolio['total_pnl']:,.2f}")
        else:
            print("  No open positions")
    else:
        print("  Unable to retrieve portfolio data")
    print()

    # Adaptive Strategy Status
    print("🧠 ADAPTIVE LEARNING STATUS:")
    adaptive_status = get_adaptive_status()
    print(adaptive_status)
    print()

    # Performance analysis and recommendations
    print("🎯 PHASE 1 FINE-TUNING ANALYSIS:")
    print("-" * 40)

    if portfolio and portfolio['positions']:
        # Analyze current performance
        winning_positions = sum(1 for pos in portfolio['positions'].values() if pos['pnl'] > 0)
        total_positions = len(portfolio['positions'])
        win_rate = winning_positions / total_positions if total_positions > 0 else 0

        print(f"  Win Rate: {win_rate:.1f} ({winning_positions}/{total_positions})")
        print(f"  Total P&L: ${portfolio['total_pnl']:,.2f}")

        # Recommendations based on performance
        if portfolio['total_pnl'] > 0:
            print("  ✅ Positive net returns - strategy working!")
            print("  💡 Consider: Increasing position sizes or adding more tickers")
        else:
            print("  ⚠️ Negative returns - tighter filters applied")
            print("  💡 Current tightened parameters:")
            print("     - EMA separation threshold: >2.5% (was >1.5%)")
            print("     - RSI filter: 35-65 (was 30-70)")
            print("     - Volatility threshold: 1.1x (was 1.2x)")
            print("     - Position size: 15% (was 20%)")
            print("  📊 Monitor next 24 hours for improved performance")

        # Signal efficiency
        active_signals_count = len(active_signals)
        positions_count = len(portfolio['positions'])
        print(f"  Signal Efficiency: {positions_count}/{active_signals_count + positions_count} positions from signals")

    else:
        print("  No positions yet - waiting for first rebalance")
        print("  Bot rebalances every 30 minutes during market hours")

    print()
    print("🔄 NEXT STEPS:")
    print("  1. Monitor adaptive learning over next 24-48 hours")
    print("  2. System automatically adjusts parameters based on trade outcomes")
    print("  3. Target: Sharpe > 2.0 through continuous learning")
    print("  4. Watch for parameter evolution and performance improvement")

if __name__ == "__main__":
    monitor_phase1_performance()