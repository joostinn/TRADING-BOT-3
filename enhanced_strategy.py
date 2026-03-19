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
from strategies import BacktestAssumptions, strategy_cross_asset_momentum_rotation

@dataclass
class EnhancedConfig:
    # Feature weights (sum to 1.0)
    momentum_weight: float = 0.28
    swing_weight: float = 0.27
    momentum_burst_weight: float = 0.20
    sentiment_weight: float = 0.05
    ml_weight: float = 0.0
    adaptive_weight: float = 0.0
    xs_momentum_weight: float = 0.20

    # News sentiment
    use_news_sentiment: bool = True
    sentiment_lookback_days: int = 7

    # ML settings
    use_ml_signals: bool = False  # Disabled intentionally
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
        """Generate enhanced trading signals with multiple strategy components."""
        signals = {}

        # Get strategy signals
        momentum_signals = self._get_momentum_signals(price_data)
        swing_signals = self._get_swing_signals(price_data)
        momentum_burst_signals = self._get_momentum_burst_signals(price_data)
        sentiment_signals = self._get_sentiment_signals(price_data.columns.tolist())
        ml_signals = self._get_ml_signals(price_data)
        adaptive_signals = create_adaptive_signals(price_data, None)
        xs_signals = self._get_xs_momentum_signals(price_data)

        # Combine signals with new weights
        for symbol in price_data.columns:
            momentum = momentum_signals.get(symbol, 0)
            swing = swing_signals.get(symbol, 0)
            momentum_burst = momentum_burst_signals.get(symbol, 0)
            sentiment = sentiment_signals.get(symbol, 0)
            ml = ml_signals.get(symbol, 0)
            adaptive = adaptive_signals.get(symbol, 0)
            xs = xs_signals.get(symbol, 0)

            # Apply sentiment confidence filter
            if abs(sentiment) < self.config.sentiment_confidence_threshold:
                sentiment = 0

            # Cap sentiment exposure
            sentiment = np.clip(sentiment, -self.config.max_sentiment_exposure, self.config.max_sentiment_exposure)

            # Weighted combination
            combined = (
                self.config.momentum_weight * momentum +
                self.config.swing_weight * swing +
                self.config.momentum_burst_weight * momentum_burst +
                self.config.sentiment_weight * sentiment +
                self.config.ml_weight * ml +
                self.config.adaptive_weight * adaptive +
                self.config.xs_momentum_weight * xs
            )

            signals[symbol] = np.clip(combined, -1, 1)

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

            # 4. Volatility filter - use ATR properly if high/low available
            if isinstance(price_data, pd.DataFrame):
                # If we have OHLC data, use proper ATR calculation
                # Reconstruct the OHLC structure for this symbol
                try:
                    if hasattr(price_data, 'columns') and isinstance(price_data.columns, pd.MultiIndex):
                        # MultiIndex structure: (High, Low, Close) per symbol
                        high = price_data[('High', symbol)] if ('High', symbol) in price_data.columns else prices
                        low = price_data[('Low', symbol)] if ('Low', symbol) in price_data.columns else prices
                        close = prices
                    else:
                        # Single-level columns, use close for all
                        high, low, close = prices, prices, prices
                except:
                    high, low, close = prices, prices, prices
            else:
                high, low, close = prices, prices, prices

            atr = ta.atr(high=high, low=low, close=close, length=14)
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

    def _get_swing_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Swing trading signals: EMA 8/21 crossover with ADX/RSI filters."""
        signals = {}

        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()

            if len(prices) < 100:
                signals[symbol] = 0
                continue

            try:
                # EMA 8/21 crossover
                ema8 = ta.ema(prices, length=8)
                ema21 = ta.ema(prices, length=21)
                ema_signal = (ema8 > ema21).astype(int) * 2 - 1  # 1 for up, -1 for down

                # ADX trend strength filter (need ADX >= 20)
                adx = ta.adx(high=prices, low=prices, close=prices, length=14)
                if adx is None or isinstance(adx, (int, float)):
                    adx_strength = pd.Series(0, index=prices.index)
                else:
                    adx_strength = adx['ADX_14'] if 'ADX_14' in adx.columns else adx.iloc[:, 0]
                    if isinstance(adx_strength, pd.DataFrame):
                        adx_strength = adx_strength.iloc[:, 0]

                has_adx = adx_strength >= 20

                # RSI in middle range (40-60) for swing trading
                rsi = ta.rsi(prices, length=14)
                rsi_ok = (rsi >= 40) & (rsi <= 60)

                # Volume filter (> 1.3x 20-day average)
                if not isinstance(price_data, pd.Series):
                    try:
                        vol = price_data.get(('Volume', symbol), pd.Series(1, index=prices.index)) if isinstance(price_data.columns, pd.MultiIndex) else pd.Series(1, index=prices.index)
                    except:
                        vol = pd.Series(1, index=prices.index)
                else:
                    vol = pd.Series(1, index=prices.index)

                avg_vol = vol.rolling(20).mean()
                vol_ok = vol > (avg_vol * 1.3)

                # Combine filters with partial scoring
                score = 0
                if ema_signal.iloc[-1] > 0:  # Uptrend
                    score += 0.4
                    if has_adx.iloc[-1]:
                        score += 0.3
                    if rsi_ok.iloc[-1]:
                        score += 0.2
                    if vol_ok.iloc[-1]:
                        score += 0.1
                elif ema_signal.iloc[-1] < 0:  # Downtrend
                    score -= 0.4
                    if has_adx.iloc[-1]:
                        score -= 0.3
                    if rsi_ok.iloc[-1]:
                        score -= 0.2
                    if vol_ok.iloc[-1]:
                        score -= 0.1

                signals[symbol] = np.clip(score, -1, 1)
            except Exception as e:
                signals[symbol] = 0

        return pd.Series(signals)

    def _get_momentum_burst_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """Momentum burst signals: Strong intraday moves with volume confirmation (LONG ONLY)."""
        signals = {}

        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()

            if len(prices) < 60:
                signals[symbol] = 0
                continue

            try:
                # Price up > 1.5% from prior close
                price_change = (prices / prices.shift(1) - 1) * 100
                strong_up = price_change > 1.5

                # Volume > 2x average
                try:
                    if isinstance(price_data.columns, pd.MultiIndex):
                        vol = price_data.get(('Volume', symbol), pd.Series(1, index=prices.index))
                    else:
                        vol = pd.Series(1, index=prices.index)
                except:
                    vol = pd.Series(1, index=prices.index)

                avg_vol = vol.rolling(20).mean()
                vol_spike = vol > (avg_vol * 2)

                # RSI 45-70 (momentum without overbought extremes)
                rsi = ta.rsi(prices, length=14)
                rsi_ok = (rsi >= 45) & (rsi <= 70)

                # Price above EMA50
                ema50 = ta.ema(prices, length=50)
                above_ema50 = prices > ema50

                # LONG ONLY: Combine filters with partial scoring (0 to +1.0)
                score = 0
                if strong_up.iloc[-1]:
                    score += 0.3
                if vol_spike.iloc[-1]:
                    score += 0.3
                if rsi_ok.iloc[-1]:
                    score += 0.2
                if above_ema50.iloc[-1]:
                    score += 0.2

                # Only return positive scores
                signals[symbol] = max(0, np.clip(score, 0, 1))
            except Exception as e:
                signals[symbol] = 0

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
        # Intentionally disabled ML to focus on proven enhanced features
        return pd.Series(0, index=price_data.columns)

    def _get_xs_momentum_signals(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Cross-sectional momentum long-short (S5).
        Ranks all tickers by 2-year momentum.
        Top 5 get score +1.0 (go long).
        Bottom 5 get score -1.0 (go short).
        Middle tickers get score 0.0 (no position).
        Rebalances weekly not monthly.
        """
        if price_data is None or len(price_data) < 63:
            return pd.Series(0.0, index=price_data.columns 
                            if price_data is not None else [])
        
        try:
            lookback = min(504, len(price_data) - 22)
            skip = 21
            
            momentum_scores = {}
            for ticker in price_data.columns:
                prices = price_data[ticker].dropna()
                if len(prices) < lookback + skip:
                    momentum_scores[ticker] = 0.0
                    continue
                # 2-year return skipping most recent month
                ret = float(
                    (prices.iloc[-skip] / prices.iloc[-(lookback + skip)]) - 1
                )
                momentum_scores[ticker] = ret
            
            if not momentum_scores:
                return pd.Series(0.0, index=price_data.columns)
            
            scores = pd.Series(momentum_scores)
            
            # Rank to percentile 0.0 to 1.0 then scale to -1.0 to +1.0
            ranked = scores.rank(pct=True)
            scaled = (ranked * 2.0) - 1.0
            
            # Only keep signal for top 5 longs and bottom 5 shorts
            # Zero out everything in the middle
            n_keep = min(5, max(1, len(scaled) // 5))
            top_threshold = scaled.nlargest(n_keep).min()
            bottom_threshold = scaled.nsmallest(n_keep).max()
            
            final = pd.Series(0.0, index=scaled.index)
            final[scaled >= top_threshold] = scaled[scaled >= top_threshold]
            final[scaled <= bottom_threshold] = scaled[scaled <= bottom_threshold]
            
            return final.fillna(0.0)
        
        except Exception as e:
            print(f"[xs_momentum] Signal error: {e}")
            return pd.Series(0.0, index=price_data.columns)

    def scan_all(self, price_data_dict: Dict[str, pd.DataFrame], etf_list: List[str]) -> pd.DataFrame:
        """Scan all tickers and return comprehensive signal analysis."""
        import os
        from datetime import datetime

        results = []

        for ticker in etf_list:
            if ticker not in price_data_dict:
                continue

            price_data = price_data_dict[ticker]
            prices = price_data if isinstance(price_data, pd.Series) else price_data.iloc[:, 0]

            if len(prices) < 50:
                continue

            try:
                # Get individual signal components
                momentum_signal = self._get_momentum_signals(
                    pd.DataFrame({ticker: prices})
                ).get(ticker, 0)

                swing_signal = self._get_swing_signals(
                    pd.DataFrame({ticker: prices})
                ).get(ticker, 0)

                burst_signal = self._get_momentum_burst_signals(
                    pd.DataFrame({ticker: prices})
                ).get(ticker, 0)

                sentiment_signal = self._get_sentiment_signals([ticker]).get(ticker, 0)
                
                # Get xs momentum signal (need full price data dict for cross-sectional ranking)
                # Reconstruct price data dict view for xs calculation
                price_data_full = pd.DataFrame({
                    t: prices_series.iloc[:, 0] if isinstance(prices_series, pd.DataFrame) else prices_series
                    for t, prices_series in price_data_dict.items()
                    if t in etf_list
                })
                xs_signal = self._get_xs_momentum_signals(price_data_full).get(ticker, 0)

                # Calculate composite score
                composite = (
                    self.config.momentum_weight * momentum_signal +
                    self.config.swing_weight * swing_signal +
                    self.config.momentum_burst_weight * burst_signal +
                    self.config.sentiment_weight * sentiment_signal +
                    self.config.xs_momentum_weight * xs_signal
                )

                # Determine signal and strength
                if composite > 0.50:
                    signal = "BUY"
                    strength = "strong"
                elif composite > 0.25:
                    signal = "BUY"
                    strength = "moderate"
                elif composite < -0.25:
                    signal = "SELL"
                    strength = "moderate" if composite > -0.50 else "strong"
                else:
                    signal = "HOLD"
                    strength = "weak"

                # Build reason string
                reason_parts = []
                if abs(momentum_signal) > 0.3:
                    reason_parts.append(f"Momentum: {momentum_signal:.2f}")
                if abs(swing_signal) > 0.3:
                    reason_parts.append(f"Swing: {swing_signal:.2f}")
                if burst_signal > 0.3:
                    reason_parts.append(f"Burst: {burst_signal:.2f}")
                if abs(sentiment_signal) > 0.1:
                    reason_parts.append(f"Sentiment: {sentiment_signal:.2f}")
                if abs(xs_signal) > 0.3:
                    reason_parts.append(f"XS Momentum: {xs_signal:.2f}")

                reason = " | ".join(reason_parts) if reason_parts else "Neutral conditions"

                results.append({
                    'ticker': ticker,
                    's_trend': momentum_signal,
                    's_swing': swing_signal,
                    's_burst': burst_signal,
                    's_sentiment': sentiment_signal,
                    's5_xs': xs_signal,
                    'composite': composite,
                    'signal': signal,
                    'strength': strength,
                    'timeframe_days': len(prices),
                    'reason': reason
                })
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Log to signal_log.csv
        if len(results_df) > 0:
            log_file = 'signal_log.csv'
            timestamp = datetime.now().isoformat()

            # Add timestamp column
            results_df['timestamp'] = timestamp

            # Reorder columns
            cols = ['timestamp', 'ticker', 's_trend', 's_swing', 's_burst', 's_sentiment', 's5_xs',
                    'composite', 'signal', 'strength', 'timeframe_days', 'reason']
            results_df = results_df[cols]

            # Append to log file
            if os.path.exists(log_file):
                existing = pd.read_csv(log_file)
                results_df = pd.concat([existing, results_df], ignore_index=True)

            results_df.to_csv(log_file, index=False)

        return results_df if len(results_df) > 0 else pd.DataFrame()

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