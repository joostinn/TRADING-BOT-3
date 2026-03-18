from adaptive_long_term_backtest import run_adaptive_ema_backtest
import pandas as pd

# Run comprehensive backtest including COVID period
print("🚀 Running Adaptive EMA Long-Term Backtest (2014-2024 including COVID)")
print("=" * 70)

results = run_adaptive_ema_backtest(
    start_date='2014-01-01',
    end_date='2024-12-31',
    initial_cash=100000
)

print("\n🎯 FINAL RESULTS")
print("=" * 50)

# Overall Performance
print("📊 OVERALL PERFORMANCE (2014-2024):")
total_return = float(results['total_return'].iloc[0]) if hasattr(results['total_return'], 'iloc') else float(results['total_return'])
sharpe_ratio = float(results['sharpe_ratio'].iloc[0]) if hasattr(results['sharpe_ratio'], 'iloc') else float(results['sharpe_ratio'])
max_drawdown = float(results['max_drawdown'].iloc[0]) if hasattr(results['max_drawdown'], 'iloc') else float(results['max_drawdown'])
win_rate = float(results['win_rate'].iloc[0]) if hasattr(results['win_rate'], 'iloc') else float(results['win_rate'])
total_trades = int(results['total_trades'].iloc[0]) if hasattr(results['total_trades'], 'iloc') else int(results['total_trades'])
final_equity = float(results['final_equity'].iloc[-1]) if hasattr(results['final_equity'], 'iloc') else float(results['final_equity'])

print(f"  Total Return: {total_return:.2%}")
print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"  Max Drawdown: {max_drawdown:.2%}")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Total Trades: {total_trades}")
print(f"  Final Equity: ${final_equity:,.0f}")

# Performance by Economic Period
print("\n📅 PERFORMANCE BY ECONOMIC PERIOD:")
print("-" * 40)
for period, perf in results['period_performance'].items():
    if perf['days'] > 0:
        print(f"  {period}:")
        print(f"    Return: {perf['return']:.2%}")
        print(f"    Sharpe: {perf['sharpe']:.3f}")
        print(f"    Max DD: {perf['max_drawdown']:.2%}")
        print(f"    Days: {perf['days']}")

# COVID-19 Specific Analysis
covid_perf = results['period_performance'].get('COVID Crash (2020)', {})
recovery_perf = results['period_performance'].get('Recovery (2021-2022)', {})

if covid_perf and recovery_perf:
    print("\n🦠 COVID-19 CRISIS ANALYSIS:")
    print("-" * 30)
    print(f"  Crash Performance: {covid_perf.get('return', 0):.2%}")
    print(f"  Recovery Performance: {recovery_perf.get('return', 0):.2%}")
    print(f"  Round-trip Return: {(1 + covid_perf.get('return', 0)) * (1 + recovery_perf.get('return', 0)) - 1:.2%}")

# Strategy Evolution
print("\n🧠 STRATEGY EVOLUTION:")
params = results['strategy_params']
print(f"  Final EMA Periods: {params['fast_ema']}/{params['slow_ema']}")
print(f"  Min Separation: {params['min_separation']:.3f}")
print(f"  RSI Range: {params['rsi_lower']}-{params['rsi_upper']}")
print(f"  Volatility Threshold: {params['volatility_threshold']:.2f}")
print(f"  Position Size: {params['position_size']:.1%}")

# Risk-Adjusted Assessment
print("\n🎯 RISK-ADJUSTED ASSESSMENT:")
print("-" * 40)

if sharpe_ratio >= 2.0:
    print("  ✅ ACHIEVED TARGET: Sharpe ≥ 2.0!")
elif sharpe_ratio >= 1.5:
    print("  🟡 GOOD: Sharpe ≥ 1.5, approaching target")
elif sharpe_ratio >= 1.0:
    print("  🟠 FAIR: Sharpe ≥ 1.0, needs improvement")
else:
    print("  🔴 POOR: Sharpe < 1.0, needs significant adjustments")

if abs(max_drawdown) <= 0.10:
    print("  ✅ EXCELLENT: Max DD ≤ 10%")
elif abs(max_drawdown) <= 0.15:
    print("  🟡 GOOD: Max DD ≤ 15%")
elif abs(max_drawdown) <= 0.20:
    print("  🟠 FAIR: Max DD ≤ 20%")
else:
    print("  🔴 POOR: Max DD > 20%")

print("\n💡 KEY INSIGHTS:")
print("-" * 15)

# Analyze COVID performance
if covid_perf.get('return', 0) > 0:
    print("  • Strategy performed well during COVID crash")
elif covid_perf.get('return', 0) > -0.20:
    print("  • Strategy contained losses during COVID crash")
else:
    print("  • Strategy experienced significant losses during COVID")

if recovery_perf.get('return', 0) > 0.20:
    print("  • Strong recovery performance")
elif recovery_perf.get('return', 0) > 0:
    print("  • Moderate recovery participation")
else:
    print("  • Weak recovery performance")

if sharpe_ratio > 1.5 and abs(max_drawdown) < 0.15:
    print("  • Strong risk-adjusted performance across economic cycles")
else:
    print("  • Performance needs improvement for consistent returns")

print("\n✅ Backtest completed successfully!")