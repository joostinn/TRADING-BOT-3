"""
Test script for enhanced ML and sentiment features.
"""

import os
import pandas as pd
import vectorbt as vbt
from datetime import datetime

# Set up environment
os.environ['ALPACA_API_KEY'] = 'PKPZOFWR3MAJOUEVKVS7O7744S'
os.environ['ALPACA_SECRET_KEY'] = '3TrJX6xZXBqnzsmoucJVWuRzMVKhMtRqsuBxfaYEgBLj'

from enhanced_strategy import EnhancedMomentumStrategy, strategy_enhanced_momentum
from strategies import strategy_time_series_momentum_trend_filter, BacktestAssumptions
from news_sentiment import get_market_sentiment_index
from feature_engineering import FeatureEngineer
from ml_system import MLSignalGenerator

def test_enhanced_features():
    """Test the enhanced features and strategies."""
    print("🔬 Testing Enhanced Trading Features")
    print("=" * 50)

    # Define universe
    tickers = ['SPY', 'QQQ', 'IWM', 'XLK', 'XLF', 'XLE', 'XLV']

    # Download data
    print("📊 Downloading market data...")
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

    print(f"✅ Downloaded data for {len(tickers)} symbols, {len(price_data)} trading days")

    # Test news sentiment
    print("\n📰 Testing News Sentiment Analysis...")
    try:
        sentiment = get_market_sentiment_index(tickers[:3], days=7)  # Test with first 3 symbols
        print("📈 Market Sentiment Scores:")
        for symbol, score in sentiment.items():
            sentiment_type = "🐂 Bullish" if score > 0.1 else "🧸 Bearish" if score < -0.1 else "⚪ Neutral"
            print(".3f")
    except Exception as e:
        print(f"⚠️  Sentiment analysis failed: {e}")

    # Test feature engineering
    print("\n⚙️  Testing Feature Engineering...")
    try:
        engineer = FeatureEngineer()
        features = engineer.create_features(price_data)

        if not features.empty:
            print(f"✅ Created {features.shape[1]} features for {len(features.columns.levels[0])} symbols")
            print(f"📊 Sample features: {list(features.columns.levels[1])[:10]}...")
        else:
            print("⚠️  Feature engineering returned empty results")
    except Exception as e:
        print(f"⚠️  Feature engineering failed: {e}")

    # Test ML signal generation
    print("\n🤖 Testing ML Signal Generation...")
    try:
        ml_generator = MLSignalGenerator()

        # Train models (this will take time)
        print("🏋️  Training ML models...")
        training_results = ml_generator.train_models(features, price_data)

        if training_results:
            print("📊 Model Training Results:")
            for symbol, accuracy in training_results.items():
                print(".3f")
            print(".3f")

            # Generate signals
            signals = ml_generator.generate_signals(features.tail(1))
            if not signals.empty:
                print("🎯 ML Signals Generated:")
                for symbol, signal in signals.items():
                    direction = "📈 BUY" if signal > 0.1 else "📉 SELL" if signal < -0.1 else "⏹️  HOLD"
                    print(".3f")
        else:
            print("⚠️  No models were trained successfully")

    except Exception as e:
        print(f"⚠️  ML training failed: {e}")

    # Test enhanced strategy
    print("\n🚀 Testing Enhanced Strategy...")
    try:
        enhanced_strategy = EnhancedMomentumStrategy()
        signals = enhanced_strategy.generate_signals(price_data.tail(100))

        if not signals.empty:
            print("🎯 Enhanced Strategy Signals:")
            for symbol, signal in signals.items():
                direction = "📈 LONG" if signal > 0.1 else "📉 SHORT" if signal < -0.1 else "⏹️  HOLD"
                print(".3f")
        else:
            print("⚠️  Enhanced strategy returned no signals")

    except Exception as e:
        print(f"⚠️  Enhanced strategy failed: {e}")

    # Compare strategies
    print("\n⚖️  Comparing Strategy Performance...")
    try:
        assumptions = BacktestAssumptions()

        # Traditional momentum
        trad_weights = strategy_time_series_momentum_trend_filter(
            adj_close=price_data,
            daily_vol=None,
            lookback_days=63,
            assumptions=assumptions,
            market_regime_filter=True,
            sharpe_scaling=True,
            vol_scaling=True
        )

        # Enhanced strategy
        enhanced_weights = strategy_enhanced_momentum(
            adj_close=price_data,
            assumptions=assumptions
        )

        # Calculate returns
        returns = price_data.pct_change()

        if not trad_weights.empty:
            trad_returns = (trad_weights.shift(1) * returns).sum(axis=1).dropna()
            trad_sharpe = trad_returns.mean() / trad_returns.std() * (252 ** 0.5) if trad_returns.std() > 0 else 0
            trad_total_return = (1 + trad_returns).prod() - 1
            print(".3f")
            print(".3f")

        if not enhanced_weights.empty:
            enhanced_returns = (enhanced_weights.shift(1) * returns).sum(axis=1).dropna()
            enhanced_sharpe = enhanced_returns.mean() / enhanced_returns.std() * (252 ** 0.5) if enhanced_returns.std() > 0 else 0
            enhanced_total_return = (1 + enhanced_returns).prod() - 1
            print(".3f")
            print(".3f")

    except Exception as e:
        print(f"⚠️  Strategy comparison failed: {e}")

    print("\n🎉 Enhanced Features Test Complete!")
    print("\n💡 To use enhanced features in live trading:")
    print("   1. Set strategy to 'enhanced_ml' in PaperConfig")
    print("   2. Get API keys for news data (optional)")
    print("   3. Models will train automatically on first run")

if __name__ == "__main__":
    test_enhanced_features()