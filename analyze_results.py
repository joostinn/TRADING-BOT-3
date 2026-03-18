import pandas as pd

df = pd.read_csv('trend_strategy_comparison.csv')

print('Strategy Performance Summary:')
print('=' * 50)

strategies = df.groupby('strategy').agg({
    'total_return': ['mean', 'max'],
    'sharpe_ratio': ['mean', 'max'],
    'max_drawdown': ['mean', 'min'],
    'win_rate': ['mean'],
    'profit_factor': ['mean']
}).round(2)

for strategy in strategies.index:
    stats = strategies.loc[strategy]
    print(f'\n{strategy.upper()}:')
    print(f'  Avg Return: {stats[("total_return", "mean")]}% (Best: {stats[("total_return", "max")]}%)')
    print(f'  Avg Sharpe: {stats[("sharpe_ratio", "mean")]} (Best: {stats[("sharpe_ratio", "max")]})')
    print(f'  Avg Max DD: {stats[("max_drawdown", "mean")]}% (Best: {stats[("max_drawdown", "min")]}%)')
    print(f'  Win Rate: {stats[("win_rate", "mean")]}%')
    print(f'  Profit Factor: {stats[("profit_factor", "mean")]}')

print('\n🏆 TOP PERFORMING INDIVIDUAL STRATEGIES:')
print('=' * 50)

top_strategies = df.nlargest(10, 'sharpe_ratio')[['strategy', 'symbol', 'sharpe_ratio', 'total_return', 'params']]
for _, row in top_strategies.iterrows():
    print(f'{row.strategy} ({row.symbol}): Sharpe={row.sharpe_ratio:.3f}, Return={row.total_return:.1f}%, Params={row.params}')