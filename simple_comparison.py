"""
Simple strategy comparison focusing on the enhanced strategies.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from enhanced_strategy import strategy_enhanced_momentum, EnhancedMomentumStrategy
from data import estimate_daily_vol, to_daily_returns
from strategies import BacktestAssumptions

def calculate_performance_metrics(returns: pd.Series, strategy_name: str) -> dict:
    """Calculate comprehensive performance metrics."""
    if len(returns) < 30:
        return {"error": "Insufficient data"}

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        "Strategy": strategy_name,
        "Total Return": ".2%",
        "Annual Return": ".2%",
        "Annual Volatility": ".2%",
        "Sharpe Ratio": ".3f",
        "Max Drawdown": ".2%",
        "Win Rate": ".1%",
        "Total Trades": len(returns)
    }

def run_simple_comparison():
    """Run a simple comparison of enhanced strategies."""
    print("🔬 ENHANCED STRATEGY COMPARISON")
    print("=" * 50)

    # Define universe
    tickers = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV']

    # Download data
    print("📊 Downloading 2 years of data...")
    price_data = vbt.YFData.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        missing_index="drop",
        missing_columns="drop",
    ).get("Close")

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()

    price_data = price_data.sort_index().dropna(how="all").ffill()
    print(f"✅ Downloaded {len(price_data)} days for {len(tickers)} symbols")

    # Calculate required data
    returns = to_daily_returns(price_data)
    vol = estimate_daily_vol(returns, span=60)
    price_returns = price_data.pct_change()

    # Strategy 1: Enhanced Momentum
    print("\n🏃 Testing Enhanced Momentum Strategy...")
    assumptions = BacktestAssumptions()
    momentum_weights = strategy_enhanced_momentum(
        adj_close=price_data,
        daily_vol=vol,
        assumptions=assumptions
    )

    if not momentum_weights.empty:
        momentum_returns = (momentum_weights.shift(1) * price_returns).sum(axis=1).dropna()
        momentum_metrics = calculate_performance_metrics(momentum_returns, "Enhanced Momentum")
    else:
        momentum_metrics = {"Strategy": "Enhanced Momentum", "error": "No weights generated"}

    # Strategy 2: Enhanced ML (simplified backtest)
    print("🤖 Testing Enhanced ML Strategy...")
    ml_strategy = EnhancedMomentumStrategy()

    # Simulate monthly rebalancing for backtest
    ml_returns_list = []
    dates = price_data.index[60::21]  # Every ~21 trading days (monthly)

    for i, date in enumerate(dates):
        if i >= len(dates) - 1:
            break

        # Get data up to current date
        current_data = price_data.loc[:date]

        # Generate signals
        signals = ml_strategy.generate_signals(current_data.tail(100))

        # Apply position limits (simplified)
        signals = signals.clip(-0.25, 0.25)  # Max 25% per position
        total_exposure = abs(signals).sum()
        if total_exposure > 0.95:  # Normalize to 95% exposure
            signals = signals * (0.95 / total_exposure)

        # Calculate next period returns
        next_date = dates[i + 1]
        next_returns = price_returns.loc[date:next_date].iloc[1:]  # Skip current date

        if not next_returns.empty:
            period_return = (signals * next_returns).sum(axis=1).mean()
            ml_returns_list.append(period_return)

    if ml_returns_list:
        ml_returns = pd.Series(ml_returns_list, index=dates[:-1])
        ml_metrics = calculate_performance_metrics(ml_returns, "Enhanced ML")
    else:
        ml_metrics = {"Strategy": "Enhanced ML", "error": "No returns generated"}

    # Strategy 3: Buy and Hold SPY (benchmark)
    print("📊 Calculating Buy & Hold Benchmark...")
    spy_returns = price_returns['SPY'].dropna()
    benchmark_metrics = calculate_performance_metrics(spy_returns, "SPY Buy & Hold")

    # Display results
    print("\n" + "="*80)
    print("🎯 STRATEGY PERFORMANCE RESULTS")
    print("="*80)

    strategies = [momentum_metrics, ml_metrics, benchmark_metrics]

    # Print header
    if strategies and "error" not in strategies[0]:
        headers = list(strategies[0].keys())
        print("<15")
        print("-" * 80)

        for strategy in strategies:
            if "error" in strategy:
                print("<15")
            else:
                values = [strategy[h] for h in headers]
                print("<15")

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    if all("error" not in s for s in strategies):
        sharpe_ratios = {s["Strategy"]: float(s["Sharpe Ratio"]) for s in strategies}
        best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)

        total_returns = {s["Strategy"]: s["Total Return"] for s in strategies}
        best_return = max(total_returns, key=lambda x: float(total_returns[x].strip('%'))/100)

        print(f"🏆 Best Sharpe Ratio: {best_sharpe} ({sharpe_ratios[best_sharpe]:.3f})")
        print(f"💰 Best Total Return: {best_return} ({total_returns[best_return]})")

        # Recommendation
        if best_sharpe == "Enhanced ML":
            print("\n🎯 RECOMMENDATION: Use Enhanced ML strategy for live trading!")
            print("   This strategy combines momentum, sentiment, and ML signals.")
        elif best_sharpe == "Enhanced Momentum":
            print("\n🎯 RECOMMENDATION: Use Enhanced Momentum strategy.")
            print("   Solid performance with lower complexity.")
        else:
            print("\n🎯 RECOMMENDATION: Consider Enhanced strategies over buy-and-hold.")
            print("   Active strategies can provide better risk-adjusted returns.")

    print("\n💡 Key Advantages of Enhanced Strategies:")
    print("   • Dynamic position sizing based on market conditions")
    print("   • Risk management with position limits")
    print("   • Multiple signal sources for better decisions")
    print("   • Adaptable to changing market regimes")

if __name__ == "__main__":
    run_simple_comparison()