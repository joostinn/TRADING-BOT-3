"""
Enhanced feature engineering for trading signals.
Includes technical indicators, volatility measures, and market microstructure features.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineer:
    """Enhanced feature engineering for trading."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance

    def create_features(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create comprehensive feature set from price and volume data."""
        features = {}

        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()
            if len(prices) < 50:  # Need minimum data for indicators
                continue

            symbol_features = self._create_symbol_features(prices, volume_data[symbol] if volume_data is not None else None)
            features[symbol] = symbol_features

        # Combine all features
        if features:
            feature_df = pd.concat(features, axis=1, keys=features.keys())
            feature_df = feature_df.dropna()

            # Add cross-sectional features
            feature_df = self._add_cross_sectional_features(feature_df)

            return feature_df
        else:
            return pd.DataFrame()

    def _create_symbol_features(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create features for a single symbol."""
        features = {}

        # Basic price features
        features['returns'] = prices.pct_change()
        features['log_returns'] = np.log(prices / prices.shift(1))

        # Moving averages - Enhanced with professional indicators
        for window in [5, 10, 20, 50, 200]:
            features[f'sma_{window}'] = ta.sma(prices, length=window)
            features[f'ema_{window}'] = ta.ema(prices, length=window)

        # Hull Moving Average - reduces lag significantly
        features['hma_9'] = ta.hma(prices, length=9)
        features['hma_21'] = ta.hma(prices, length=21)

        # VWAP (Volume Weighted Average Price) - institutional benchmark
        if volume is not None and not volume.empty:
            features['vwap'] = ta.vwap(high=prices, low=prices, close=prices, volume=volume)
            # Price relative to VWAP
            features['price_to_vwap'] = prices / features['vwap']

        # Momentum indicators
        features['rsi_14'] = ta.rsi(prices, length=14)
        # Stochastic needs high, low, close - using close as approximation
        features['stoch_k'] = ta.stoch(close=prices, high=prices, low=prices, length=14)['STOCHk_14_3_3']
        features['stoch_d'] = ta.stoch(close=prices, high=prices, low=prices, length=14)['STOCHd_14_3_3']
        features['williams_r'] = ta.willr(high=prices, low=prices, close=prices, length=14)

        # Trend indicators
        macd_data = ta.macd(prices)
        features['macd'] = macd_data['MACD_12_26_9']
        features['macd_signal'] = macd_data['MACDs_12_26_9']
        features['macd_hist'] = macd_data['MACDh_12_26_9']

        # Volatility indicators
        bb_data = ta.bbands(close=prices, length=20)
        features['bb_upper'] = bb_data['BBU_20_2.0']
        features['bb_middle'] = bb_data['BBM_20_2.0']
        features['bb_lower'] = bb_data['BBL_20_2.0']
        features['atr_14'] = ta.atr(high=prices, low=prices, close=prices, length=14)
        features['natr'] = ta.natr(high=prices, low=prices, close=prices, length=14)  # Normalized ATR

        # Volume indicators (if available)
        if volume is not None and not volume.empty:
            features['volume_sma_20'] = ta.sma(volume, length=20)
            features['volume_ratio'] = volume / ta.sma(volume, length=20)
            features['obv'] = ta.obv(prices, volume)
            features['cmf'] = ta.cmf(high=prices, low=prices, close=prices, volume=volume, length=20)  # Chaikin Money Flow

        # Statistical features
        for window in [20, 50, 100]:
            roll_returns = features['returns'].rolling(window=window)
            features[f'skew_{window}'] = roll_returns.skew()
            features[f'kurt_{window}'] = roll_returns.kurt()
            features[f'volatility_{window}'] = roll_returns.std() * np.sqrt(252)  # Annualized

        # Price patterns
        features['price_to_sma_200'] = prices / features['sma_200']
        features['price_to_sma_50'] = prices / features['sma_50']
        features['sma_50_to_200'] = features['sma_50'] / features['sma_200']

        # Momentum features
        for window in [1, 3, 5, 10, 20]:
            features[f'momentum_{window}'] = prices / prices.shift(window) - 1

        # Mean reversion features
        features['zscore_20'] = (prices - ta.sma(prices, length=20)) / ta.stdev(prices, length=20)
        features['zscore_50'] = (prices - ta.sma(prices, length=50)) / ta.stdev(prices, length=50)

        # Combine into DataFrame
        feature_df = pd.DataFrame(features, index=prices.index)

        # Add lagged features
        for col in ['returns', 'rsi_14', 'macd']:
            if col in feature_df.columns:
                for lag in [1, 2, 3, 5]:
                    feature_df[f'{col}_lag_{lag}'] = feature_df[col].shift(lag)

        return feature_df

    def _add_cross_sectional_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features across all symbols."""
        # Rank features
        for col in feature_df.columns.levels[1]:  # Feature names
            if col in feature_df.columns:
                feature_df[f'{col}_rank'] = feature_df[col].rank(axis=1, pct=True)

        # Relative strength features
        if 'returns' in feature_df.columns.levels[1]:
            returns_cols = [col for col in feature_df.columns if col[1] == 'returns']
            if returns_cols:
                returns_df = feature_df[returns_cols]
                feature_df['market_returns'] = returns_df.mean(axis=1)
                for col in returns_cols:
                    feature_df[f'{col[0]}_relative_strength'] = feature_df[col] - feature_df['market_returns']

        return feature_df

    def reduce_dimensions(self, feature_df: pd.DataFrame, n_components: Optional[int] = None) -> pd.DataFrame:
        """Reduce feature dimensions using PCA."""
        if n_components:
            self.pca = PCA(n_components=n_components)

        # Flatten multi-index columns for PCA
        flat_df = feature_df.copy()
        flat_df.columns = [f'{col[0]}_{col[1]}' for col in flat_df.columns]

        # Remove NaN values
        flat_df = flat_df.dropna()

        if len(flat_df) < 10:
            return feature_df

        # Scale features
        scaled_features = self.scaler.fit_transform(flat_df)

        # Apply PCA
        pca_features = self.pca.fit_transform(scaled_features)

        # Create new DataFrame with PCA components
        pca_df = pd.DataFrame(
            pca_features,
            index=flat_df.index,
            columns=[f'pca_{i}' for i in range(pca_features.shape[1])]
        )

        return pca_df

def create_market_regime_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create market regime classification features."""
    features = {}

    # Calculate market returns (using first symbol as proxy)
    market_prices = price_data.iloc[:, 0]
    market_returns = market_prices.pct_change()

    # Volatility regimes
    vol_20 = market_returns.rolling(20).std()
    vol_60 = market_returns.rolling(60).std()
    features['vol_regime'] = (vol_20 > vol_60).astype(int)  # High vol regime

    # Trend regimes
    sma_50 = ta.sma(market_prices, length=50)
    sma_200 = ta.sma(market_prices, length=200)
    features['trend_regime'] = (sma_50 > sma_200).astype(int)  # Bull market

    # Momentum regimes
    mom_20 = market_prices / market_prices.shift(20) - 1
    features['mom_regime'] = (mom_20 > 0).astype(int)  # Positive momentum

    return pd.DataFrame(features, index=price_data.index)