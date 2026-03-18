from adaptive_long_term_backtest import run_adaptive_ema_backtest

# Run quick backtest
results = run_adaptive_ema_backtest(
    start_date='2014-01-01',
    end_date='2024-12-31',
    initial_cash=100000
)

print("🎯 ADAPTIVE EMA LONG-TERM BACKTEST RESULTS")
print("=" * 50)
print(f"Total Return: {float(results['total_return']):.2%}")
print(f"Sharpe Ratio: {float(results['sharpe_ratio']):.3f}")
print(f"Max Drawdown: {float(results['max_drawdown']):.2%}")
print(f"Win Rate: {float(results['win_rate']):.1%}")
print(f"Total Trades: {int(results['total_trades'])}")
print(f"Final Equity: ${float(results['final_equity']):,.0f}")

# COVID Analysis
covid_perf = results['period_performance'].get('COVID Crash (2020)', {})
recovery_perf = results['period_performance'].get('Recovery (2021-2022)', {})

print("\n🦠 COVID-19 ANALYSIS:")
print(f"  Crash Performance: {covid_perf.get('return', 0):.2%}")
print(f"  Recovery Performance: {recovery_perf.get('return', 0):.2%}")

print("\n🧠 FINAL PARAMETERS:")
params = results['strategy_params']
print(f"  EMA: {params['fast_ema']}/{params['slow_ema']}")
print(f"  Min Separation: {params['min_separation']:.3f}")
print(f"  Position Size: {params['position_size']:.1%}")