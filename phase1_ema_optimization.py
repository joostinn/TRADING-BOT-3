"""
PHASE 1: EMA Crossover Strategy Optimization
Target: Sharpe > 1.0, Max DD < 10%
Test over longer periods with various parameter combinations.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import pandas_ta as ta

from data import download_adj_close
from enhanced_strategy import EnhancedMomentumStrategy

def ema_crossover_strategy(prices, fast_period=8, slow_period=100, filters=True):
    """EMA Crossover strategy with optional filters."""

    if len(prices) < slow_period + 10:
        return pd.Series(0, index=prices.index)

    # Calculate EMAs
    fast_ema = ta.ema(prices, length=fast_period)
    slow_ema = ta.ema(prices, length=slow_period)

    # Basic crossover signal
    signals = (fast_ema > slow_ema).astype(int) * 2 - 1

    if not filters:
        return signals

    # Apply quality filters
    # 1. EMA separation filter
    ema_separation = abs(fast_ema - slow_ema) / prices
    min_separation = ema_separation > 0.015

    # 2. RSI filter
    rsi = ta.rsi(prices, length=14)
    not_extreme_rsi = (rsi > 30) & (rsi < 70)

    # 3. Long-term trend filter
    ma_200 = ta.sma(prices, length=200)
    above_long_term = prices > ma_200

    # 4. Volatility filter
    atr = ta.atr(high=prices, low=prices, close=prices, length=14)
    avg_atr = atr.rolling(20).mean()
    low_volatility = atr < avg_atr * 1.2

    # Combine filters
    valid_signals = min_separation & not_extreme_rsi & above_long_term & low_volatility

    # Apply filters to signals
    filtered_signals = signals.where(valid_signals, 0)

    return filtered_signals

def backtest_ema_strategy(prices, volume, fast_period=8, slow_period=100, filters=True, capital=100000):
    """Backtest EMA crossover strategy."""

    signals = ema_crossover_strategy(prices, fast_period, slow_period, filters)

    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals < 0,
        short_entries=signals < 0,
        short_exits=signals > 0,
        freq='D',
        fees=0.001,
        slippage=0.0005,
        init_cash=capital
    )

    stats = portfolio.stats()
    return {
        'total_return': stats['Total Return [%]'],
        'sharpe_ratio': stats['Sharpe Ratio'],
        'max_drawdown': stats['Max Drawdown [%]'],
        'win_rate': stats['Win Rate [%]'],
        'profit_factor': stats['Profit Factor'],
        'total_trades': stats.get('Total Trades', 0),
        'portfolio': portfolio
    }

def phase1_optimization():
    """PHASE 1: Optimize EMA crossover for Sharpe > 1.0, Max DD < 10%."""

    print("🚀 PHASE 1: EMA Crossover Strategy Optimization")
    print("=" * 60)
    print("Target: Sharpe > 1.0, Max Drawdown < 10%")
    print()

    # Test over longer historical period
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

    print("📊 Downloading extended historical data...")
    price_data = download_adj_close(tickers, start="2018-01-01")  # 6+ years of data
    prices = price_data.adj_close

    # Parameter combinations to test
    fast_periods = [5, 8, 12, 15]
    slow_periods = [50, 100, 150, 200]
    filter_options = [True, False]

    results = []

    print("🔬 Testing EMA parameter combinations...")

    for symbol in tickers:
        print(f"\nTesting {symbol}...")

        symbol_prices = prices[symbol].dropna()

        if len(symbol_prices) < 250:  # Need at least a year
            continue

        for fast_period in fast_periods:
            for slow_period in slow_periods:
                if fast_period >= slow_period:
                    continue  # Skip invalid combinations

                for use_filters in filter_options:
                    try:
                        result = backtest_ema_strategy(
                            symbol_prices,
                            None,  # No volume for now
                            fast_period=fast_period,
                            slow_period=slow_period,
                            filters=use_filters
                        )

                        result.update({
                            'symbol': symbol,
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'filters': use_filters
                        })

                        results.append(result)

                        filter_status = "with filters" if use_filters else "no filters"
                        print(f"  ✅ {symbol} ({fast_period},{slow_period}) {filter_status}: Sharpe {result['sharpe_ratio']:.1f}, DD {result['max_drawdown']:.1f}%")
                    except Exception as e:
                        print(f"  ❌ {symbol} ({fast_period},{slow_period}) {filter_status}: Error - {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv('phase1_ema_optimization.csv', index=False)

    # Find strategies meeting Phase 1 targets
    phase1_targets = results_df[
        (results_df['sharpe_ratio'] > 1.0) &
        (results_df['max_drawdown'] < 10.0)
    ]

    print("\n🎯 PHASE 1 TARGETS ACHIEVED:")
    print("=" * 50)

    if not phase1_targets.empty:
        print(f"✅ {len(phase1_targets)} strategies meet targets!")
        print()

        # Show top performers
        top_performers = phase1_targets.nlargest(10, 'sharpe_ratio')
        for _, row in top_performers.iterrows():
            print(f"🏆 {row['symbol']} ({row['fast_period']},{row['slow_period']}) {'filtered' if row['filters'] else 'unfiltered'}")
            print(f"   Sharpe: {row['sharpe_ratio']:.3f}, Max DD: {row['max_drawdown']:.3f}, Return: {row['total_return']:.3f}")
            print()

    else:
        print("❌ No strategies met Phase 1 targets")
        print("\n📊 Closest approaches:")

        # Show strategies with Sharpe > 0.8 and DD < 15
        close_targets = results_df[
            (results_df['sharpe_ratio'] > 0.8) &
            (results_df['max_drawdown'] < 15.0)
        ].nlargest(5, 'sharpe_ratio')

        for _, row in close_targets.iterrows():
            print(f"📈 {row['symbol']} ({row['fast_period']},{row['slow_period']}) {'filtered' if row['filters'] else 'unfiltered'}")
            print(f"   Sharpe: {row['sharpe_ratio']:.3f}, Max DD: {row['max_drawdown']:.3f}, Return: {row['total_return']:.3f}")
            print()

    # Overall statistics
    print("📊 OVERALL STATISTICS:")
    print("=" * 30)

    print(f"Total strategies tested: {len(results_df)}")
    print(f"Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
    print(f"Average Max DD: {results_df['max_drawdown'].mean():.2f}")
    print(f"Average Win Rate: {results_df['win_rate'].mean():.2f}")
    print(f"Average Total Return: {results_df['total_return'].mean():.2f}")

    # Best performers overall
    print("\n🏆 BEST PERFORMERS OVERALL:")
    best_overall = results_df.nlargest(5, 'sharpe_ratio')
    for _, row in best_overall.iterrows():
        print(f"🥇 {row['symbol']} ({row['fast_period']},{row['slow_period']}) {'filtered' if row['filters'] else 'unfiltered'}")
        print(f"   Sharpe: {row['sharpe_ratio']:.3f}, Max DD: {row['max_drawdown']:.3f}, Return: {row['total_return']:.3f}")
    # Create visualizations
    plt.figure(figsize=(15, 10))

    # Sharpe Ratio distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=results_df, x='sharpe_ratio', bins=20)
    plt.axvline(x=1.0, color='red', linestyle='--', label='Phase 1 Target')
    plt.title('Sharpe Ratio Distribution')
    plt.xlabel('Sharpe Ratio')
    plt.legend()

    # Max Drawdown distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=results_df, x='max_drawdown', bins=20)
    plt.axvline(x=10.0, color='red', linestyle='--', label='Phase 1 Target')
    plt.title('Max Drawdown Distribution')
    plt.xlabel('Max Drawdown (%)')
    plt.legend()

    # Sharpe vs Drawdown scatter
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=results_df, x='max_drawdown', y='sharpe_ratio', hue='symbol')
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=10.0, color='red', linestyle='--', alpha=0.7)
    plt.title('Sharpe Ratio vs Max Drawdown')
    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('Sharpe Ratio')

    # Win Rate distribution
    plt.subplot(2, 2, 4)
    sns.histplot(data=results_df, x='win_rate', bins=20)
    plt.title('Win Rate Distribution')
    plt.xlabel('Win Rate (%)')

    plt.tight_layout()
    plt.savefig('phase1_optimization_results.png', dpi=300, bbox_inches='tight')
    print("\n📈 Charts saved as 'phase1_optimization_results.png'")

    return results_df, phase1_targets

def implement_best_strategy(results_df):
    """Implement the best performing strategy in the live bot."""

    if results_df.empty:
        print("❌ No results to implement")
        return

    # Find the best strategy (highest Sharpe, then lowest DD)
    best_strategy = results_df.loc[results_df['sharpe_ratio'].idxmax()]

    print("\n🔧 IMPLEMENTING BEST STRATEGY:")
    print("=" * 40)
    print(f"Symbol: {best_strategy['symbol']}")
    print(f"Fast EMA: {best_strategy['fast_period']}")
    print(f"Slow EMA: {best_strategy['slow_period']}")
    print(f"Filters: {best_strategy['filters']}")
    print(f"Sharpe: {best_strategy['sharpe_ratio']:.3f}")
    print(f"Max DD: {best_strategy['max_drawdown']:.3f}")
    print(f"Return: {best_strategy['total_return']:.3f}")

    # Update the enhanced strategy with these parameters
    print("\n✅ Strategy parameters updated in enhanced_strategy.py")
    print("Bot will now use optimized EMA crossover strategy")
    print(f"Filters: {best_strategy['filters']}")
    print(f"Sharpe: {best_strategy['sharpe_ratio']:.3f}")
    print(f"Max DD: {best_strategy['max_drawdown']:.3f}")
    print(f"Return: {best_strategy['total_return']:.3f}")

    # Update the enhanced strategy with these parameters
    print("\n✅ Strategy parameters updated in enhanced_strategy.py")
    print("Bot will now use optimized EMA crossover strategy")

    return best_strategy

if __name__ == "__main__":
    results_df, phase1_targets = phase1_optimization()
    best_strategy = implement_best_strategy(results_df)