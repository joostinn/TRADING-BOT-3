"""
Backtest comparison of different trend identification strategies.
Tests EMA, VWAP, and Hull Moving Average with various parameters.
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
from strategies import strategy_time_series_momentum_trend_filter

def create_trend_strategy_signals(prices, strategy_name, params):
    """Create signals based on different trend strategies."""

    if strategy_name == 'ema_crossover':
        fast_ema = ta.ema(prices, length=params['fast_period'])
        slow_ema = ta.ema(prices, length=params['slow_period'])
        signals = (fast_ema > slow_ema).astype(int) * 2 - 1  # 1 for long, -1 for short

    elif strategy_name == 'vwap_breakout':
        # Need volume data for VWAP
        volume = params.get('volume', pd.Series(1, index=prices.index))
        vwap = ta.vwap(high=prices, low=prices, close=prices, volume=volume)
        signals = (prices > vwap).astype(int) * 2 - 1

    elif strategy_name == 'hull_trend':
        hma = ta.hma(prices, length=params['period'])
        signals = (prices > hma).astype(int) * 2 - 1

    elif strategy_name == 'ema_vwap_combo':
        ema = ta.ema(prices, length=params['ema_period'])
        volume = params.get('volume', pd.Series(1, index=prices.index))
        vwap = ta.vwap(high=prices, low=prices, close=prices, volume=volume)
        # Long when price > EMA and price > VWAP, short otherwise
        long_condition = (prices > ema) & (prices > vwap)
        signals = long_condition.astype(int) * 2 - 1

    elif strategy_name == 'original_momentum':
        # Use the existing momentum strategy
        if len(prices) > 252:
            signal = strategy_time_series_momentum_trend_filter(prices, lookback=252)
            signals = pd.Series(signal, index=[prices.index[-1]])
            signals = signals.reindex(prices.index).fillna(method='ffill')
        else:
            signals = pd.Series(0, index=prices.index)

    else:
        signals = pd.Series(0, index=prices.index)

    return signals

def backtest_trend_strategy(prices, volume, strategy_name, params, capital=100000):
    """Backtest a single trend strategy."""

    signals = create_trend_strategy_signals(prices, strategy_name, {**params, 'volume': volume})

    # Create portfolio
    portfolio = vbt.Portfolio.from_signals(
        close=prices,
        entries=signals > 0,
        exits=signals < 0,
        short_entries=signals < 0,
        short_exits=signals > 0,
        freq='D',
        fees=0.001,  # 10 bps round trip
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
        'portfolio': portfolio
    }

def run_trend_strategy_comparison():
    """Run comprehensive comparison of trend strategies."""

    # Define universe
    tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

    print("📊 Downloading data...")
    try:
        price_data = download_adj_close(tickers, start="2020-01-01")
        prices = price_data.adj_close
        print(f"✅ Downloaded price data: {prices.shape}")
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("Falling back to direct yfinance download...")
        import yfinance as yf
        prices = yf.download(tickers, start="2020-01-01", auto_adjust=True)['Close']
        print(f"✅ Downloaded price data: {prices.shape}")

    # Get volume data for VWAP strategies
    volume_data = {}
    for ticker in tickers:
        try:
            import yfinance as yf
            vol_data = yf.download(ticker, start="2020-01-01", auto_adjust=True)['Volume']
            volume_data[ticker] = vol_data
        except:
            volume_data[ticker] = pd.Series(1, index=prices.index)

    # Define strategies and their parameter grids
    strategies = {
        'ema_crossover': {
            'fast_period': [5, 8, 12, 15],
            'slow_period': [20, 26, 50, 100]
        },
        'vwap_breakout': {
            # VWAP doesn't need parameters, just uses price > VWAP
        },
        'hull_trend': {
            'period': [9, 16, 21, 25, 50]
        },
        'ema_vwap_combo': {
            'ema_period': [10, 20, 50]
        },
        'original_momentum': {
            # No parameters needed
        }
    }

    results = []

    print("🔬 Testing trend strategies...")

    for symbol in tickers:
        print(f"\nTesting {symbol}...")

        symbol_prices = prices[symbol].dropna()
        symbol_volume = volume_data.get(symbol, pd.Series(1, index=symbol_prices.index))

        if len(symbol_prices) < 100:  # Skip if not enough data
            continue

        for strategy_name, param_grid in strategies.items():
            if not param_grid:
                # No parameters to test
                try:
                    result = backtest_trend_strategy(symbol_prices, symbol_volume, strategy_name, {})
                    result['symbol'] = symbol
                    result['strategy'] = strategy_name
                    result['params'] = {}
                    results.append(result)
                    print(".1f")
                except Exception as e:
                    print(f"  ❌ {strategy_name}: Error - {e}")
            else:
                # Test all parameter combinations
                param_names = list(param_grid.keys())
                param_values = list(param_grid.values())

                for param_combo in product(*param_values):
                    params = dict(zip(param_names, param_combo))

                    try:
                        result = backtest_trend_strategy(symbol_prices, symbol_volume, strategy_name, params)
                        result['symbol'] = symbol
                        result['strategy'] = strategy_name
                        result['params'] = params
                        results.append(result)

                        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                        print(".1f")
                    except Exception as e:
                        print(f"  ❌ {strategy_name} ({param_str}): Error - {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    results_df.to_csv('trend_strategy_comparison.csv', index=False)

    # Create summary statistics
    summary = results_df.groupby('strategy').agg({
        'total_return': ['mean', 'std', 'max'],
        'sharpe_ratio': ['mean', 'std', 'max'],
        'max_drawdown': ['mean', 'std', 'min'],  # Min because lower (less negative) is better
        'win_rate': ['mean', 'std', 'max'],
        'profit_factor': ['mean', 'std', 'max']
    }).round(3)

    print("\n" + "="*80)
    print("📊 TREND STRATEGY COMPARISON SUMMARY")
    print("="*80)

    for strategy in summary.index:
        print(f"\n🔬 {strategy.upper()}")
        print("-" * 40)
        stats = summary.loc[strategy]

        print(f"  Total Return: {stats['total_return']['mean']:.1f}% (σ={stats['total_return']['std']:.1f}%, max={stats['total_return']['max']:.1f}%)")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']['mean']:.3f} (σ={stats['sharpe_ratio']['std']:.3f}, max={stats['sharpe_ratio']['max']:.3f})")
        print(f"  Max Drawdown: {stats['max_drawdown']['mean']:.1f}% (σ={stats['max_drawdown']['std']:.1f}%, best={stats['max_drawdown']['min']:.1f}%)")
        print(f"  Win Rate: {stats['win_rate']['mean']:.1f}% (σ={stats['win_rate']['std']:.1f}%, max={stats['win_rate']['max']:.1f}%)")
        print(f"  Profit Factor: {stats['profit_factor']['mean']:.3f} (σ={stats['profit_factor']['std']:.3f}, max={stats['profit_factor']['max']:.3f})")

    # Find best performing strategies
    print("\n🏆 TOP PERFORMING STRATEGIES:")
    top_sharpe = results_df.nlargest(5, 'sharpe_ratio')[['strategy', 'symbol', 'sharpe_ratio', 'total_return', 'params']]
    for _, row in top_sharpe.iterrows():
        param_str = str(row['params']) if row['params'] else 'default'
        print(".3f")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Sharpe Ratio comparison
    plt.subplot(2, 2, 1)
    sns.boxplot(data=results_df, x='strategy', y='sharpe_ratio')
    plt.title('Sharpe Ratio Distribution by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Total Return comparison
    plt.subplot(2, 2, 2)
    sns.boxplot(data=results_df, x='strategy', y='total_return')
    plt.title('Total Return Distribution by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Max Drawdown comparison
    plt.subplot(2, 2, 3)
    sns.boxplot(data=results_df, x='strategy', y='max_drawdown')
    plt.title('Max Drawdown Distribution by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Win Rate comparison
    plt.subplot(2, 2, 4)
    sns.boxplot(data=results_df, x='strategy', y='win_rate')
    plt.title('Win Rate Distribution by Strategy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trend_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📈 Charts saved as 'trend_strategy_comparison.png'")

    # Save summary to CSV
    summary.to_csv('trend_strategy_summary.csv')

    return results_df, summary

if __name__ == "__main__":
    results_df, summary = run_trend_strategy_comparison()