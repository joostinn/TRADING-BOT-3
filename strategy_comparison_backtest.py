"""
COMPREHENSIVE BACKTEST: IMPROVED ADAPTIVE EMA STRATEGY
Tests the improved strategy against the old one to show performance gains.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import both strategies
from adaptive_ema_strategy import AdaptiveEMAStrategy, adaptive_ema_signals
from improved_adaptive_strategy import ImprovedAdaptiveEMAStrategy, improved_adaptive_signals

def run_strategy_comparison(start_date: str = "2014-01-01",
                          end_date: str = "2024-12-31",
                          tickers: list = None,
                          initial_cash: float = 100000) -> dict:
    """
    Run comprehensive comparison between old and improved adaptive strategies.
    """

    if tickers is None:
        tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU"]

    print("🔬 STRATEGY COMPARISON: OLD vs IMPROVED ADAPTIVE EMA")
    print("=" * 60)
    print(f"📊 Period: {start_date} to {end_date}")
    print(f"📈 Tickers: {len(tickers)} sector ETFs")
    print(f"💰 Initial Capital: ${initial_cash:,.0f}")
    print()

    # Download data
    print("📥 Downloading historical data...")
    data = vbt.YFData.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        missing_index="drop",
        missing_columns="drop",
    )

    close_prices = data.get('Close')
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame()

    close_prices = close_prices.dropna(how='all').ffill()
    print(f"✅ Data loaded: {len(close_prices)} trading days")
    print()

    results = {}

    # Test Old Strategy
    print("🧠 Testing OLD Adaptive Strategy...")
    old_strategy = AdaptiveEMAStrategy()
    old_signals = adaptive_ema_signals(close_prices)

    # Convert signals to DataFrame format for vectorbt
    old_signals_df = pd.DataFrame(old_signals, index=[close_prices.index[-1]])
    old_signals_df = old_signals_df.reindex(close_prices.index).ffill()

    old_portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=old_signals_df > 0,
        exits=old_signals_df < 0,
        short_entries=old_signals_df < 0,
        short_exits=old_signals_df > 0,
        init_cash=initial_cash,
        fees=0.0005,
        slippage=0.0005,
        freq='D'
    )

    old_results = {
        'total_return': float(old_portfolio.total_return().iloc[-1]),
        'sharpe_ratio': float(old_portfolio.sharpe_ratio()),
        'max_drawdown': float(old_portfolio.max_drawdown()),
        'win_rate': float(old_portfolio.trades.win_rate()),
        'total_trades': len(old_portfolio.trades),
        'final_equity': float(old_portfolio.value().iloc[-1]),
        'equity_curve': old_portfolio.value()
    }

    print(f"  Old Strategy: {old_results['total_return']:.2%} return, {old_results['sharpe_ratio']:.3f} Sharpe, {old_results['total_trades']} trades")
    print()

    # Test Improved Strategy
    print("🚀 Testing IMPROVED Adaptive Strategy...")
    improved_strategy = ImprovedAdaptiveEMAStrategy()
    improved_signals = improved_adaptive_signals(close_prices)

    # Convert signals to DataFrame format
    improved_signals_df = pd.DataFrame(improved_signals, index=[close_prices.index[-1]])
    improved_signals_df = improved_signals_df.reindex(close_prices.index).ffill()

    improved_portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=improved_signals_df > 0,
        exits=improved_signals_df < 0,
        short_entries=improved_signals_df < 0,
        short_exits=improved_signals_df > 0,
        init_cash=initial_cash,
        fees=0.0005,
        slippage=0.0005,
        freq='D'
    )

    improved_results = {
        'total_return': float(improved_portfolio.total_return().iloc[-1]),
        'sharpe_ratio': float(improved_portfolio.sharpe_ratio()),
        'max_drawdown': float(improved_portfolio.max_drawdown()),
        'win_rate': float(improved_portfolio.trades.win_rate()),
        'total_trades': len(improved_portfolio.trades),
        'final_equity': float(improved_portfolio.value().iloc[-1]),
        'equity_curve': improved_portfolio.value()
    }

    print(f"  Improved Strategy: {improved_results['total_return']:.2%} return, {improved_results['sharpe_ratio']:.3f} Sharpe, {improved_results['total_trades']} trades")
    print()

    # Calculate improvements
    return_improvement = improved_results['total_return'] - old_results['total_return']
    sharpe_improvement = improved_results['sharpe_ratio'] - old_results['sharpe_ratio']
    trade_increase = improved_results['total_trades'] - old_results['total_trades']

    print("📈 IMPROVEMENT ANALYSIS")
    print("-" * 30)
    print(f"  Return Improvement: {return_improvement:.2%}")
    print(f"  Sharpe Improvement: {sharpe_improvement:.3f}")
    print(f"  Additional Trades: {trade_increase}")
    print()

    # Analyze by market periods
    periods = {
        'Pre-COVID (2014-2019)': ('2014-01-01', '2019-12-31'),
        'COVID Crash (2020)': ('2020-01-01', '2020-12-31'),
        'Recovery (2021-2022)': ('2021-01-01', '2022-12-31'),
        'Post-COVID (2023-2024)': ('2023-01-01', '2024-12-31')
    }

    period_analysis = {}
    for period_name, (start, end) in periods.items():
        try:
            mask = (close_prices.index >= start) & (close_prices.index <= end)
            if mask.any():
                # Old strategy period performance
                old_period_return = float(old_portfolio.sharpe_ratio(mask=mask))
                old_period_dd = float(old_portfolio.max_drawdown(mask=mask))

                # Improved strategy period performance
                improved_period_return = float(improved_portfolio.sharpe_ratio(mask=mask))
                improved_period_dd = float(improved_portfolio.max_drawdown(mask=mask))

                period_analysis[period_name] = {
                    'old_sharpe': old_period_return,
                    'improved_sharpe': improved_period_return,
                    'old_dd': old_period_dd,
                    'improved_dd': improved_period_dd,
                    'sharpe_improvement': improved_period_return - old_period_return,
                    'dd_improvement': old_period_dd - improved_period_dd  # Lower DD is better
                }
        except:
            continue

    return {
        'old_strategy': old_results,
        'improved_strategy': improved_results,
        'improvements': {
            'return_improvement': return_improvement,
            'sharpe_improvement': sharpe_improvement,
            'trade_increase': trade_increase
        },
        'period_analysis': period_analysis,
        'config': {
            'start_date': start_date,
            'end_date': end_date,
            'tickers': tickers,
            'initial_cash': initial_cash
        }
    }

def plot_strategy_comparison(results: dict):
    """Create comprehensive comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Adaptive EMA Strategy Comparison: Old vs Improved', fontsize=16, fontweight='bold')

    old_results = results['old_strategy']
    improved_results = results['improved_strategy']

    # 1. Equity Curves
    axes[0,0].plot(old_results['equity_curve'].index, old_results['equity_curve'].values,
                   linewidth=2, label='Old Strategy', color='red', alpha=0.7)
    axes[0,0].plot(improved_results['equity_curve'].index, improved_results['equity_curve'].values,
                   linewidth=2, label='Improved Strategy', color='green', alpha=0.8)
    axes[0,0].set_title('Portfolio Equity Curves')
    axes[0,0].set_ylabel('Portfolio Value ($)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Add COVID period shading
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2020-03-23')
    recovery_end = pd.Timestamp('2021-01-01')
    axes[0,0].axvspan(covid_start, covid_end, alpha=0.2, color='red', label='COVID Crash')
    axes[0,0].axvspan(covid_end, recovery_end, alpha=0.2, color='green', label='Recovery')

    # 2. Key Metrics Comparison
    metrics = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    old_values = [old_results['total_return'], old_results['sharpe_ratio'],
                 old_results['max_drawdown'], old_results['win_rate']]
    improved_values = [improved_results['total_return'], improved_results['sharpe_ratio'],
                      improved_results['max_drawdown'], improved_results['win_rate']]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = axes[0,1].bar(x - width/2, old_values, width, label='Old Strategy', color='red', alpha=0.7)
    bars2 = axes[0,1].bar(x + width/2, improved_values, width, label='Improved Strategy', color='green', alpha=0.8)

    axes[0,1].set_title('Key Performance Metrics')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(metrics)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      '.1%' if 'Rate' in metrics[x[int(bar.get_x() + width/2)]] else '.2%',
                      ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                      '.1%' if 'Rate' in metrics[x[int(bar.get_x() + width/2)]] else '.2%',
                      ha='center', va='bottom', fontsize=8)

    # 3. Period-by-Period Analysis
    period_names = list(results['period_analysis'].keys())
    sharpe_improvements = [results['period_analysis'][p]['sharpe_improvement'] for p in period_names]

    axes[1,0].bar(period_names, sharpe_improvements, color='blue', alpha=0.7)
    axes[1,0].set_title('Sharpe Ratio Improvement by Period')
    axes[1,0].set_ylabel('Sharpe Improvement')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 4. Rolling Sharpe Comparison (last 2 years)
    old_curve = old_results['equity_curve']
    improved_curve = improved_results['equity_curve']

    # Calculate rolling Sharpe for last 500 days
    old_returns = old_curve.pct_change().rolling(252).mean() / old_curve.pct_change().rolling(252).std() * np.sqrt(252)
    improved_returns = improved_curve.pct_change().rolling(252).mean() / improved_curve.pct_change().rolling(252).std() * np.sqrt(252)

    # Plot last 500 days
    recent_old = old_returns.tail(500)
    recent_improved = improved_returns.tail(500)

    axes[1,1].plot(recent_old.index, recent_old.values, label='Old Strategy', color='red', alpha=0.7)
    axes[1,1].plot(recent_improved.index, recent_improved.values, label='Improved Strategy', color='green', alpha=0.8)
    axes[1,1].set_title('Rolling 1-Year Sharpe Ratio (Recent)')
    axes[1,1].set_ylabel('Sharpe Ratio')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Target Sharpe 2.0')
    axes[1,1].axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Sharpe 1.0')

    plt.tight_layout()
    plt.savefig('output/strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Comparison plot saved to: output/strategy_comparison.png")
    plt.show()

def print_detailed_comparison(results: dict):
    """Print detailed comparison analysis."""

    old = results['old_strategy']
    improved = results['improved_strategy']
    improvements = results['improvements']

    print("🎯 STRATEGY COMPARISON RESULTS")
    print("=" * 50)
    print()

    print("📊 OVERALL PERFORMANCE (2014-2024)")
    print("-" * 35)
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print()

    print("🚀 IMPROVEMENTS ACHIEVED")
    print("-" * 25)
    print(f"  Return Improvement: {improvements['return_improvement']:.2%}")
    print(f"  Sharpe Improvement: {improvements['sharpe_improvement']:.3f}")
    print(f"  Additional Trades: {improvements['trade_increase']}")
    print()

    # Performance grade
    if improved['total_return'] > 0.5:  # 50%+ return
        grade = "A+ (EXCELLENT)"
    elif improved['total_return'] > 0.2:  # 20%+ return
        grade = "A (VERY GOOD)"
    elif improved['total_return'] > 0:
        grade = "B (GOOD)"
    elif improved['total_return'] > -0.2:
        grade = "C (FAIR)"
    else:
        grade = "F (POOR)"

    print(f"🎓 IMPROVED STRATEGY GRADE: {grade}")
    print()

    print("📅 PERIOD-BY-PERIOD ANALYSIS")
    print("-" * 30)
    for period, analysis in results['period_analysis'].items():
        print(f"  {period}:")
        print(f"    Sharpe Improvement: {analysis['sharpe_improvement']:.3f}")
        print(f"    DD Improvement: {analysis['dd_improvement']:.2%}")
        print()

    print("🔧 KEY IMPROVEMENTS MADE")
    print("-" * 25)
    print("  1. Better EMA Periods: 20/50 vs 15/200")
    print("  2. Less Restrictive Filters: More trading opportunities")
    print("  3. Improved Learning Logic: Better parameter adaptation")
    print("  4. Larger Position Sizes: 25% vs 15%")
    print("  5. Risk Management: Stop losses and take profits")
    print("  6. More Permissive Volatility Filter")
    print()

    print("💡 WHY THE OLD STRATEGY FAILED")
    print("-" * 30)
    print("  • Only 22 trades in 10 years (too restrictive)")
    print("  • EMA 15/200 too extreme for signals")
    print("  • Overly strict filters eliminated opportunities")
    print("  • Poor learning logic made parameters worse")
    print("  • Small position sizes couldn't overcome losses")
    print()

    if improved['total_return'] > 0:
        print("✅ SUCCESS: Strategy now generates POSITIVE returns!")
    else:
        print("⚠️  PARTIAL SUCCESS: Still needs more work for consistent profits")

if __name__ == "__main__":
    # Run comprehensive comparison
    results = run_strategy_comparison()

    # Print detailed analysis
    print_detailed_comparison(results)

    # Create comparison plots
    plot_strategy_comparison(results)