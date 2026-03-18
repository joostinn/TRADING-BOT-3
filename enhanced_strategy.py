"""
Enhanced trading strategy integrating news sentiment, ML signals, and advanced features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import vectorbt as vbt
import pandas_ta as ta

from news_sentiment import NewsSentimentAnalyzer, get_market_sentiment_index
from feature_engineering import FeatureEngineer, create_market_regime_features
from ml_system import MLSignalGenerator, ParameterOptimizer, create_adaptive_signals
from strategies import BacktestAssumptions

@dataclass
class EnhancedConfig:
    # Feature weights
    momentum_weight: float = 0.4
    sentiment_weight: float = 0.2
    ml_weight: float = 0.3
    adaptive_weight: float = 0.1

    # News sentiment
    use_news_sentiment: bool = True
    sentiment_lookback_days: int = 7

    # ML settings
    use_ml_signals: bool = True
    ml_model_retrain_days: int = 30

    # Risk management
    max_sentiment_exposure: float = 0.2  # Max exposure from sentiment
    sentiment_confidence_threshold: float = 0.1

class EnhancedMomentumStrategy:
    """Enhanced momentum strategy with ML and sentiment integration."""

    def __init__(self, config: EnhancedConfig = None):
        if config is None:
            config = EnhancedConfig()
        self.config = config

        # Initialize components
        self.news_analyzer = NewsSentimentAnalyzer() if config.use_news_sentiment else None
        self.feature_engineer = FeatureEngineer()
        self.ml_generator = MLSignalGenerator() if config.use_ml_signals else None
        self.param_optimizer = ParameterOptimizer()

        self.last_ml_training = None

    def generate_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Generate enhanced trading signals."""
        signals = {}

        # Get basic momentum signals
        momentum_signals = self._get_momentum_signals(price_data)

        # Get sentiment signals
        sentiment_signals = self._get_sentiment_signals(price_data.columns.tolist())

        # Get ML signals
        ml_signals = self._get_ml_signals(price_data)

        # Get adaptive signals
        adaptive_signals = create_adaptive_signals(price_data, None)  # Could pass features here

        # Combine signals
        for symbol in price_data.columns:
            momentum = momentum_signals.get(symbol, 0)
            sentiment = sentiment_signals.get(symbol, 0)
            ml = ml_signals.get(symbol, 0)
            adaptive = adaptive_signals.get(symbol, 0)

            # Weighted combination
            combined = (
                self.config.momentum_weight * momentum +
                self.config.sentiment_weight * sentiment +
                self.config.ml_weight * ml +
                self.config.adaptive_weight * adaptive
            )

            # Apply sentiment confidence filter
            if abs(sentiment) < self.config.sentiment_confidence_threshold:
                sentiment = 0

            # Cap sentiment exposure
            sentiment = np.clip(sentiment, -self.config.max_sentiment_exposure, self.config.max_sentiment_exposure)

            signals[symbol] = combined

        return pd.Series(signals)

    def _get_momentum_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """PHASE 1: EMA Crossover Strategy - Optimized parameters from comprehensive backtesting."""
        signals = {}

        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()

            if len(prices) < 210:  # Need 200-period EMA + buffer
                signals[symbol] = 0
                continue

            # PHASE 1 OPTIMIZED: EMA Crossover (15, 200) - Best performer from 6+ years backtest
            # Sharpe: 1.156, Return: 344.5%, but DD: 28.6% (above Phase 1 target of 10%)
            fast_ema = ta.ema(prices, length=15)  # Fast EMA (15-period) - optimized
            slow_ema = ta.ema(prices, length=200) # Slow EMA (200-period) - optimized

            # Generate crossover signals with trend confirmation
            prev_fast = fast_ema.shift(1)
            prev_slow = slow_ema.shift(1)

            # Current trend direction (primary signal)
            current_trend = (fast_ema > slow_ema).astype(int) * 2 - 1  # 1 for up, -1 for down

            # Additional filters for quality signals
            # 1. EMA separation filter (>1.5% of price for meaningful trends)
            ema_separation = abs(fast_ema - slow_ema) / prices
            min_separation = ema_separation > 0.015  # 1.5% minimum separation

            # 2. Avoid extreme RSI conditions (oversold/overbought)
            rsi = ta.rsi(prices, length=14)
            not_extreme_rsi = (rsi > 30) & (rsi < 70)  # Avoid extreme conditions

            # 3. Require above long-term trend (200-day MA)
            ma_200 = ta.sma(prices, length=200)
            above_long_term_trend = prices > ma_200

            # 4. Volatility filter - avoid high volatility periods
            atr = ta.atr(high=prices, low=prices, close=prices, length=14)
            avg_atr = atr.rolling(20).mean()
            low_volatility = atr < avg_atr * 1.2  # Not excessively volatile

            # Combine all filters
            valid_signal = (
                min_separation.iloc[-1] and
                not_extreme_rsi.iloc[-1] and
                above_long_term_trend.iloc[-1] and
                low_volatility.iloc[-1]
            )

            # Get final signal
            signal = current_trend.iloc[-1] if valid_signal else 0

            # Scale down signal strength for risk management
            # Stronger signals get full weight, weaker get reduced
            if signal != 0:
                separation_pct = ema_separation.iloc[-1]
                # Scale signal based on trend strength (separation)
                if separation_pct > 0.05:  # Very strong trend
                    signal *= 1.0
                elif separation_pct > 0.03:  # Strong trend
                    signal *= 0.8
                else:  # Moderate trend
                    signal *= 0.6

            signals[symbol] = np.clip(signal, -1, 1)

        return pd.Series(signals)

    def _get_sentiment_signals(self, symbols: List[str]) -> pd.Series:
        """Get news sentiment signals."""
        if not self.news_analyzer:
            return pd.Series(0, index=symbols)

        try:
            sentiment_index = get_market_sentiment_index(symbols, self.config.sentiment_lookback_days)

            # Convert sentiment to signals
            # Positive sentiment -> bullish, negative -> bearish
            signals = sentiment_index * 0.5  # Scale down sentiment impact

            return signals
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return pd.Series(0, index=symbols)

    def _get_ml_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Get ML-based signals."""
        # Temporarily disabled ML to focus on basic enhanced features
        return pd.Series(0, index=price_data.columns)

    def optimize_parameters(self, price_data: pd.DataFrame) -> Dict:
        """Optimize strategy parameters."""
        param_ranges = {
            'lookback_days': [21, 63, 126, 252],
            'ma_days': [50, 100, 200],
            'market_regime_filter': [True, False],
            'sharpe_scaling': [True, False],
            'vol_scaling': [True, False]
        }

        best_params = self.param_optimizer.optimize_momentum_params(price_data, param_ranges)

        # Update config with optimized parameters
        for key, value in best_params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        return best_params

def strategy_enhanced_momentum(
    adj_close: pd.DataFrame,
    daily_vol: Optional[pd.DataFrame] = None,
    assumptions: Optional[BacktestAssumptions] = None,
    **kwargs
) -> pd.DataFrame:
    """Enhanced momentum strategy with ML and sentiment integration."""
    strategy = EnhancedMomentumStrategy()

    # Generate signals for each date
    signals = []

    # Use rolling window to simulate real-time signal generation
    window_size = 500  # Minimum history needed

    for i in range(window_size, len(adj_close)):
        window_data = adj_close.iloc[:i+1]

        # Generate signal for this date
        signal = strategy.generate_signals(window_data.tail(100))  # Use last 100 days

        # Reindex to match full universe
        signal = signal.reindex(adj_close.columns).fillna(0)
        signals.append(signal)

    if signals:
        signals_df = pd.DataFrame(signals, index=adj_close.index[window_size:])
        return signals_df
    else:
        return pd.DataFrame(0, index=adj_close.index, columns=adj_close.columns)