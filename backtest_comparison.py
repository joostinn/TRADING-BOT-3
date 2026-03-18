"""
Comprehensive backtest comparison between traditional and enhanced ML strategies.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import strategies
from strategies import strategy_time_series_momentum_trend_filter, BacktestAssumptions
from enhanced_strategy import strategy_enhanced_momentum, EnhancedMomentumStrategy
from alpaca_paper_trade import _latest_signals, PaperConfig
from data import estimate_daily_vol, to_daily_returns

def run_strategy_backtest(strategy_name: str, price_data: pd.DataFrame, config: PaperConfig = None) -> dict:
    """Run backtest for a specific strategy."""
    if config is None:
        config = PaperConfig()

    print(f"🏃 Running {strategy_name} backtest...")

    try:
        if strategy_name == "traditional_momentum":
            # Traditional momentum strategy
            assumptions = BacktestAssumptions()

            # Calculate volatility data
            returns = to_daily_returns(price_data)
            vol = estimate_daily_vol(returns, span=60)

            weights = strategy_time_series_momentum_trend_filter(
                adj_close=price_data,
                daily_vol=vol,
                lookback_days=63,
                ma_days=200,
                assumptions=assumptions,
                market_regime_filter=True,
                tighten_stop_loss=True,
                sharpe_scaling=True,
                vol_scaling=True
            )

        elif strategy_name == "enhanced_momentum":
            # Enhanced momentum (current live strategy)
            assumptions = BacktestAssumptions()

            # Calculate volatility data
            returns = to_daily_returns(price_data)
            vol = estimate_daily_vol(returns, span=60)

            weights = strategy_enhanced_momentum(
                adj_close=price_data,
                daily_vol=vol,
                assumptions=assumptions
            )

        elif strategy_name == "enhanced_ml":
            # Full enhanced ML strategy
            enhanced_strategy = EnhancedMomentumStrategy()
            # For backtest, we need to simulate the rolling signal generation
            weights_list = []

            # Use rolling window to simulate real-time signals
            window_size = 200  # Minimum history needed
            step_size = 5  # Rebalance every 5 days for backtest

            for i in range(window_size, len(price_data), step_size):
                window_data = price_data.iloc[:i+1]
                signals = enhanced_strategy.generate_signals(window_data.tail(100))
                signals = signals.reindex(price_data.columns).fillna(0)

                # Apply position limits
                from alpaca_paper_trade import _apply_position_limits
                signals = _apply_position_limits(signals, config)

                weights_list.append(signals)

            # Create weights DataFrame
            weights = pd.DataFrame(weights_list, index=price_data.index[window_size::step_size])

        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        if weights.empty:
            return {"error": "No weights generated"}

        # Calculate returns
        returns = price_data.pct_change()
        strategy_returns = (weights.shift(1) * returns).sum(axis=1).dropna()

        if len(strategy_returns) < 30:
            return {"error": "Insufficient return data"}

        # Calculate performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = strategy_returns.mean() * 252
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Additional metrics
        sortino_ratio = calculate_sortino_ratio(strategy_returns)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # Turnover (rough estimate)
        turnover = weights.diff().abs().sum().sum() / len(weights)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "turnover": turnover,
            "avg_trade_return": strategy_returns.mean(),
            "best_trade": strategy_returns.max(),
            "worst_trade": strategy_returns.min()
        }

    except Exception as e:
        print(f"Error in {strategy_name} backtest: {e}")
        return {"error": str(e)}

def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """Calculate Sortino ratio (downside deviation only)."""
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return 0.0

    downside_std = downside_returns.std() * np.sqrt(252)
    annual_return = returns.mean() * 252

    return (annual_return - target_return) / downside_std if downside_std > 0 else 0

def print_performance_comparison(results: dict):
    """Print formatted performance comparison."""
    print("\n" + "="*80)
    print("🎯 STRATEGY PERFORMANCE COMPARISON")
    print("="*80)

    # Define metrics to display
    metrics = [
        ("Total Return", "total_return", ".2%"),
        ("Annual Return", "annual_return", ".2%"),
        ("Annual Volatility", "annual_volatility", ".2%"),
        ("Sharpe Ratio", "sharpe_ratio", ".3f"),
        ("Sortino Ratio", "sortino_ratio", ".3f"),
        ("Calmar Ratio", "calmar_ratio", ".3f"),
        ("Max Drawdown", "max_drawdown", ".2%"),
        ("Win Rate", "win_rate", ".1%"),
        ("Total Trades", "total_trades", "d"),
        ("Turnover", "turnover", ".3f"),
        ("Avg Trade Return", "avg_trade_return", ".4%"),
        ("Best Trade", "best_trade", ".2%"),
        ("Worst Trade", "worst_trade", ".2%")
    ]

    print("<12")
    print("-" * 80)

    for strategy_name, result in results.items():
        if "error" in result:
            print("<12")
            continue

        print("<12")

        for metric_name, key, fmt in metrics:
            if key in result:
                value = result[key]
                if isinstance(value, (int, float)) and not pd.isna(value):
                    print("<12")
                else:
                    print("<12")
            else:
                print("<12")

        print("-" * 80)

def run_comprehensive_backtest():
    """Run comprehensive backtest comparison."""
    print("🔬 COMPREHENSIVE STRATEGY BACKTEST COMPARISON")
    print("=" * 60)

    # Define universe
    tickers = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV']

    # Download data
    print("📊 Downloading historical data (2 years)...")
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

    print(f"✅ Downloaded {len(price_data)} days of data for {len(tickers)} symbols")

    # Define strategies to test
    strategies = [
        "traditional_momentum",
        "enhanced_momentum",
        "enhanced_ml"
    ]

    # Run backtests
    results = {}
    config = PaperConfig()

    for strategy in strategies:
        result = run_strategy_backtest(strategy, price_data, config)
        results[strategy] = result

    # Print comparison
    print_performance_comparison(results)

    # Summary analysis
    print("\n" + "="*80)
    print("📊 ANALYSIS & RECOMMENDATIONS")
    print("="*80)

    if all("error" not in result for result in results.values()):
        # Compare Sharpe ratios
        sharpe_ratios = {name: result.get("sharpe_ratio", 0) for name, result in results.items()}
        best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)

        # Compare returns
        total_returns = {name: result.get("total_return", 0) for name, result in results.items()}
        best_return = max(total_returns, key=total_returns.get)

        # Compare drawdowns
        drawdowns = {name: abs(result.get("max_drawdown", 0)) for name, result in results.items()}
        best_drawdown = min(drawdowns, key=drawdowns.get)

        print(f"🏆 Best Sharpe Ratio: {best_sharpe} ({sharpe_ratios[best_sharpe]:.3f})")
        print(f"💰 Best Total Return: {best_return} ({total_returns[best_return]:.2%})")
        print(f"🛡️  Best Max Drawdown: {best_drawdown} ({results[best_drawdown]['max_drawdown']:.2%})")

        # Overall recommendation
        scores = {}
        for name, result in results.items():
            # Weighted score: 40% Sharpe, 30% Return, 30% Risk-adjusted
            sharpe_score = result['sharpe_ratio'] / max(sharpe_ratios.values()) if max(sharpe_ratios.values()) > 0 else 0
            return_score = result['total_return'] / max(total_returns.values()) if max(total_returns.values()) > 0 else 0
            risk_score = (1 - abs(result['max_drawdown'])) / (1 - min(abs(r) for r in drawdowns.values())) if drawdowns else 1

            scores[name] = 0.4 * sharpe_score + 0.3 * return_score + 0.3 * risk_score

        best_overall = max(scores, key=scores.get)
        print(f"\n🎯 RECOMMENDED STRATEGY: {best_overall.upper()}")
        print(f"   Overall Score: {scores[best_overall]:.3f}/1.000")

    print("\n💡 Key Insights:")
    print("   • Higher Sharpe ratios indicate better risk-adjusted returns")
    print("   • Lower maximum drawdown means less severe losses")
    print("   • Win rate shows consistency of profitable trades")
    print("   • Turnover indicates trading frequency (higher = more costs)")

    print("\n🚀 Ready for live trading with the recommended strategy!")

if __name__ == "__main__":
    run_comprehensive_backtest()