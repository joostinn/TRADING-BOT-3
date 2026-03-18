"""
LONG-TERM ADAPTIVE EMA STRATEGY BACKTEST
Tests the adaptive EMA strategy over extended periods including economic crises like COVID-19.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our adaptive strategy
from adaptive_ema_strategy import AdaptiveEMAStrategy, adaptive_ema_signals

def run_adaptive_ema_backtest(start_date: str = "2014-01-01",
                             end_date: str = "2024-12-31",
                             tickers: list = None,
                             initial_cash: float = 100000) -> dict:
    """
    Run comprehensive backtest of adaptive EMA strategy over long period including COVID.

    Args:
        start_date: Start date for backtest (YYYY-MM-DD)
        end_date: End date for backtest (YYYY-MM-DD)
        tickers: List of tickers to test (default: major ETFs)
        initial_cash: Starting capital

    Returns:
        Dictionary with backtest results and analysis
    """

    if tickers is None:
        # Major liquid ETFs including sectors that performed differently during COVID
        tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI", "XLB", "XLY", "XLP", "XLU"]

    print(f"🚀 Starting Adaptive EMA Backtest: {start_date} to {end_date}")
    print(f"📊 Tickers: {', '.join(tickers)}")
    print(f"💰 Initial Capital: ${initial_cash:,.0f}")
    print()

    # Download data - extended period to include COVID and recovery
    print("📥 Downloading historical data...")
    data = vbt.YFData.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        missing_index="drop"
    )

    close_prices = data.get('Close')
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame()

    # Clean data
    close_prices = close_prices.dropna(how='all').ffill()

    print(f"✅ Data loaded: {len(close_prices)} trading days")
    print()

    # Initialize adaptive strategy
    strategy = AdaptiveEMAStrategy()

    # Generate signals for the entire period
    print("🧠 Generating adaptive signals for full period...")

    # Process in chunks to simulate real-time adaptation
    signals_history = []
    chunk_size = 252  # 1 year chunks

    for i in range(0, len(close_prices), chunk_size):
        chunk_end = min(i + chunk_size, len(close_prices))
        chunk_data = close_prices.iloc[i:chunk_end]

        if len(chunk_data) < 50:  # Skip if chunk too small
            continue

        # Get signals for this chunk
        chunk_signals = adaptive_ema_signals(chunk_data)

        # Convert to DataFrame format for vectorbt
        chunk_signals_df = pd.DataFrame(chunk_signals, index=[chunk_data.index[-1]])
        signals_history.append(chunk_signals_df)

        # Simulate learning from trades in this chunk (simplified)
        # In reality, this would happen after each trade execution
        if i > 0 and len(signals_history) > 1:
            # Simulate some trade outcomes for learning
            for symbol in tickers:
                if np.random.random() > 0.6:  # 40% win rate simulation
                    pnl_pct = np.random.normal(0.02, 0.05)  # Mean 2% return, 5% std
                    # Calculate approximate dollar pnl based on position size
                    position_value = 10000  # Assume $10k position
                    pnl = position_value * pnl_pct

                    trade_result = {
                        'symbol': symbol,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'holding_period': np.random.randint(1, 10)
                    }
                    strategy.update_from_trade_outcome(trade_result)

    # Combine all signals
    if signals_history:
        signals_df = pd.concat(signals_history).reindex(close_prices.index).ffill()
    else:
        signals_df = pd.DataFrame(0, index=close_prices.index, columns=tickers)

    print("✅ Signals generated and strategy adapted")
    print()

    # Run vectorbt backtest
    print("📈 Running vectorbt backtest...")

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,
        entries=signals_df > 0,
        exits=signals_df < 0,
        short_entries=signals_df < 0,
        short_exits=signals_df > 0,
        init_cash=initial_cash,
        fees=0.0005,  # 5 bps
        slippage=0.0005,  # 5 bps
        freq='D'
    )

    # Calculate performance metrics
    total_return = portfolio.total_return()
    sharpe_ratio = portfolio.sharpe_ratio()
    max_drawdown = portfolio.max_drawdown()
    win_rate = portfolio.trades.win_rate()
    total_trades = len(portfolio.trades)

    # Get equity curve
    equity_curve = portfolio.value()

    # Analyze performance by periods
    periods = {
        'Pre-COVID (2014-2019)': ('2014-01-01', '2019-12-31'),
        'COVID Crash (2020)': ('2020-01-01', '2020-12-31'),
        'Recovery (2021-2022)': ('2021-01-01', '2022-12-31'),
        'Post-COVID (2023-2024)': ('2023-01-01', '2024-12-31')
    }

    period_performance = {}
    for period_name, (start, end) in periods.items():
        try:
            mask = (equity_curve.index >= start) & (equity_curve.index <= end)
            if mask.any():
                period_equity = equity_curve[mask]
                period_return = (period_equity.iloc[-1] / period_equity.iloc[0] - 1) if len(period_equity) > 1 else 0
                period_sharpe = portfolio.sharpe_ratio(mask=mask)
                period_max_dd = portfolio.max_drawdown(mask=mask)
                period_performance[period_name] = {
                    'return': period_return,
                    'sharpe': period_sharpe,
                    'max_drawdown': period_max_dd,
                    'days': len(period_equity)
                }
        except:
            period_performance[period_name] = {'return': 0, 'sharpe': 0, 'max_drawdown': 0, 'days': 0}

    # Results summary
    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'final_equity': equity_curve.iloc[-1],
        'period_performance': period_performance,
        'equity_curve': equity_curve,
        'signals': signals_df,
        'strategy_params': strategy.current_params,
        'backtest_config': {
            'start_date': start_date,
            'end_date': end_date,
            'tickers': tickers,
            'initial_cash': initial_cash
        }
    }

    return results

def plot_adaptive_backtest_results(results: dict, save_path: str = None):
    """Create comprehensive plots for the adaptive backtest results."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Adaptive EMA Strategy Long-Term Backtest Results', fontsize=16, fontweight='bold')

    # 1. Equity Curve
    equity_curve = results['equity_curve']
    axes[0,0].plot(equity_curve.index, equity_curve.values, linewidth=2, label='Adaptive EMA')
    axes[0,0].set_title('Portfolio Equity Curve')
    axes[0,0].set_ylabel('Portfolio Value ($)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()

    # Add COVID period shading
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2020-03-23')  # Market bottom
    recovery_end = pd.Timestamp('2021-01-01')
    axes[0,0].axvspan(covid_start, covid_end, alpha=0.2, color='red', label='COVID Crash')
    axes[0,0].axvspan(covid_end, recovery_end, alpha=0.2, color='green', label='Recovery')

    # 2. Period Performance Comparison
    periods = list(results['period_performance'].keys())
    returns = [results['period_performance'][p]['return'] for p in periods]
    sharpes = [results['period_performance'][p]['sharpe'] for p in periods]

    x = np.arange(len(periods))
    width = 0.35

    axes[0,1].bar(x - width/2, returns, width, label='Total Return', alpha=0.8)
    axes[0,1].bar(x + width/2, sharpes, width, label='Sharpe Ratio', alpha=0.8)
    axes[0,1].set_title('Performance by Economic Period')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(periods, rotation=45, ha='right')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Rolling Sharpe Ratio
    rolling_sharpe = results['equity_curve'].pct_change().rolling(252).mean() / results['equity_curve'].pct_change().rolling(252).std() * np.sqrt(252)
    axes[1,0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
    axes[1,0].set_title('Rolling 1-Year Sharpe Ratio')
    axes[1,0].axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Target Sharpe 2.0')
    axes[1,0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Sharpe 1.0')
    axes[1,0].set_ylabel('Sharpe Ratio')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()

    # 4. Strategy Parameters Evolution (if available)
    params = results['strategy_params']
    param_names = ['fast_ema', 'slow_ema', 'min_separation', 'position_size']
    param_values = [params.get(name, 0) for name in param_names]

    axes[1,1].bar(param_names, param_values)
    axes[1,1].set_title('Final Adaptive Parameters')
    axes[1,1].set_ylabel('Parameter Value')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")

    plt.show()

def print_adaptive_backtest_summary(results: dict):
    """Print comprehensive summary of adaptive backtest results."""

    print("🎯 ADAPTIVE EMA STRATEGY LONG-TERM BACKTEST RESULTS")
    print("=" * 60)
    print()

    # Overall Performance
    print("📊 OVERALL PERFORMANCE:")
    print(f"  Total Return: {float(results['total_return']):.2%}")
    print(f"  Sharpe Ratio: {float(results['sharpe_ratio']):.3f}")
    print(f"  Max Drawdown: {float(results['max_drawdown']):.2%}")
    print(f"  Win Rate: {float(results['win_rate']):.1%}")
    print(f"  Total Trades: {int(results['total_trades'])}")
    print(f"  Final Equity: ${float(results['final_equity']):,.0f}")
    print()

    # Performance by Economic Period
    print("📅 PERFORMANCE BY ECONOMIC PERIOD:")
    print("-" * 40)
    for period, perf in results['period_performance'].items():
        print(f"  {period}:")
        print(".2%")
        print(f"    Sharpe: {perf['sharpe']:.3f}")
        print(".2%")
        print(f"    Days: {perf['days']}")
        print()

    # Strategy Adaptation
    print("🧠 STRATEGY ADAPTATION:")
    params = results['strategy_params']
    print(f"  Final EMA Periods: {params['fast_ema']}/{params['slow_ema']}")
    print(".3f")
    print(f"  RSI Range: {params['rsi_lower']}-{params['rsi_upper']}")
    print(".2f")
    print(".1%")
    print()

    # COVID-19 Specific Analysis
    covid_perf = results['period_performance'].get('COVID Crash (2020)', {})
    recovery_perf = results['period_performance'].get('Recovery (2021-2022)', {})

    if covid_perf and recovery_perf:
        print("🦠 COVID-19 CRISIS ANALYSIS:")
        print("-" * 30)
        print(f"  Crash Performance: {covid_perf.get('return', 0):.2%}")
        print(f"  Recovery Performance: {recovery_perf.get('return', 0):.2%}")
        print(f"  Round-trip Return: {(1 + covid_perf.get('return', 0)) * (1 + recovery_perf.get('return', 0)) - 1:.2%}")
        print()

    # Risk-Adjusted Performance Assessment
    sharpe = results['sharpe_ratio']
    max_dd = results['max_drawdown']

    print("🎯 RISK-ADJUSTED PERFORMANCE ASSESSMENT:")
    print("-" * 40)

    if sharpe >= 2.0:
        print("  ✅ EXCELLENT: Achieved Sharpe > 2.0 target!")
    elif sharpe >= 1.5:
        print("  🟡 GOOD: Sharpe > 1.5, approaching target")
    elif sharpe >= 1.0:
        print("  🟠 FAIR: Sharpe > 1.0, needs improvement")
    else:
        print("  🔴 POOR: Sharpe < 1.0, significant adjustments needed")

    if abs(max_dd) <= 0.10:
        print("  ✅ EXCELLENT: Max DD ≤ 10%")
    elif abs(max_dd) <= 0.15:
        print("  🟡 GOOD: Max DD ≤ 15%")
    elif abs(max_dd) <= 0.20:
        print("  🟠 FAIR: Max DD ≤ 20%")
    else:
        print("  🔴 POOR: Max DD > 20%")

    print()
    print("💡 KEY INSIGHTS:")
    print("-" * 15)

    # Analyze COVID performance
    if covid_perf.get('return', 0) > 0:
        print("  • Strategy performed well during COVID crash - captured downside protection")
    elif covid_perf.get('return', 0) > -0.20:
        print("  • Strategy contained losses during COVID crash")
    else:
        print("  • Strategy experienced significant losses during COVID - may need crash protection")

    if recovery_perf.get('return', 0) > 0.20:
        print("  • Strong recovery performance - good participation in rebound")
    elif recovery_perf.get('return', 0) > 0:
        print("  • Moderate recovery - captured some upside")
    else:
        print("  • Weak recovery performance - missed rebound opportunities")

    if sharpe > 1.5 and abs(max_dd) < 0.15:
        print("  • Overall strong risk-adjusted performance across economic cycles")
    else:
        print("  • Performance needs improvement for consistent risk-adjusted returns")

if __name__ == "__main__":
    # Run comprehensive backtest including COVID period
    results = run_adaptive_ema_backtest(
        start_date="2014-01-01",
        end_date="2024-12-31",
        initial_cash=100000
    )

    # Print detailed summary
    print_adaptive_backtest_summary(results)

    # Create and save plots
    plot_adaptive_backtest_results(results, save_path="output/adaptive_ema_long_term_backtest.png")