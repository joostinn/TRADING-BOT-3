"""
Machine Learning system for trading signal generation and parameter optimization.
Uses reinforcement learning and supervised learning approaches.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLConfig:
    model_type: str = 'random_forest'  # 'random_forest', 'gradient_boosting', 'xgboost'
    lookback_window: int = 60  # Days of history to use for prediction
    prediction_horizon: int = 5  # Days ahead to predict
    min_samples_split: int = 50
    max_depth: int = 10
    n_estimators: int = 100
    learning_rate: float = 0.1
    cv_folds: int = 5
    model_path: str = 'models'

class MLSignalGenerator:
    """Machine learning-based signal generation."""

    def __init__(self, config: MLConfig = None):
        if config is None:
            config = MLConfig()
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

        # Create model directory
        os.makedirs(self.config.model_path, exist_ok=True)

    def train_models(self, feature_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Train ML models for each symbol."""
        results = {}

        for symbol in feature_data.columns.levels[0]:
            try:
                # Prepare data for this symbol
                symbol_features = feature_data[symbol]
                symbol_prices = price_data[symbol]

                # Create target variable (future returns)
                future_returns = symbol_prices.shift(-self.config.prediction_horizon) / symbol_prices - 1
                target = (future_returns > 0).astype(int)  # Binary classification

                # Align data
                combined_data = pd.concat([symbol_features, target], axis=1).dropna()
                if len(combined_data) < 100:  # Need minimum data
                    continue

                X = combined_data.iloc[:, :-1]
                y = combined_data.iloc[:, -1]

                # Train model
                accuracy = self._train_symbol_model(symbol, X, y)
                results[symbol] = accuracy

                print(f"Trained {symbol} model with {accuracy:.3f} accuracy")

            except Exception as e:
                print(f"Error training {symbol}: {e}")
                continue

        return results

    def _train_symbol_model(self, symbol: str, X: pd.DataFrame, y: pd.Series) -> float:
        """Train model for a single symbol."""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

        # Model selection
        if self.config.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Cross-validation scores
        cv_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores.append(accuracy_score(y_test, y_pred))

        # Train final model on all data
        model.fit(X_scaled, y)

        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler

        # Store feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[symbol] = dict(zip(X.columns, model.feature_importances_))

        # Save model
        model_path = os.path.join(self.config.model_path, f'{symbol}_model.pkl')
        scaler_path = os.path.join(self.config.model_path, f'{symbol}_scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        return np.mean(cv_scores)

    def generate_signals(self, feature_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using trained models."""
        signals = {}

        for symbol in feature_data.columns.levels[0]:
            if symbol not in self.models:
                continue

            try:
                # Get latest features
                symbol_features = feature_data[symbol].iloc[-1:]

                # Scale features
                scaler = self.scalers[symbol]
                features_scaled = scaler.transform(symbol_features)

                # Predict
                model = self.models[symbol]
                prediction = model.predict_proba(features_scaled)[0]

                # Convert to signal (-1 to 1)
                # prediction[1] is probability of positive return
                signal = (prediction[1] - 0.5) * 2  # Scale to -1 to 1
                signals[symbol] = signal

            except Exception as e:
                print(f"Error generating signal for {symbol}: {e}")
                continue

        return pd.Series(signals)

    def load_models(self) -> bool:
        """Load pre-trained models from disk."""
        loaded_count = 0

        if not os.path.exists(self.config.model_path):
            return False

        for filename in os.listdir(self.config.model_path):
            if filename.endswith('_model.pkl'):
                symbol = filename.replace('_model.pkl', '')
                try:
                    model_path = os.path.join(self.config.model_path, filename)
                    scaler_path = os.path.join(self.config.model_path, symbol + '_scaler.pkl')

                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    loaded_count += 1

                except Exception as e:
                    print(f"Error loading {symbol} model: {e}")

        print(f"Loaded {loaded_count} pre-trained models")
        return loaded_count > 0

class ParameterOptimizer:
    """Optimizes trading strategy parameters using ML."""

    def __init__(self):
        self.best_params = {}
        self.optimization_history = []

    def optimize_momentum_params(self, price_data: pd.DataFrame, param_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Optimize momentum strategy parameters."""
        from strategies import strategy_time_series_momentum_trend_filter, BacktestAssumptions

        # Define parameter grid
        param_grid = self._create_param_grid(param_ranges)

        best_score = -np.inf
        best_params = None

        assumptions = BacktestAssumptions()

        for params in param_grid:
            try:
                # Run strategy with these parameters
                weights = strategy_time_series_momentum_trend_filter(
                    adj_close=price_data,
                    daily_vol=None,  # Could add volatility if available
                    lookback_days=params.get('lookback_days', 252),
                    ma_days=params.get('ma_days', 200),
                    assumptions=assumptions,
                    market_regime_filter=params.get('market_regime_filter', True),
                    tighten_stop_loss=params.get('tighten_stop_loss', True),
                    sharpe_scaling=params.get('sharpe_scaling', True),
                    vol_scaling=params.get('vol_scaling', True)
                )

                # Calculate performance metric (Sharpe ratio proxy)
                if not weights.empty:
                    returns = self._calculate_strategy_returns(weights, price_data)
                    sharpe = self._calculate_sharpe_ratio(returns)

                    if sharpe > best_score:
                        best_score = sharpe
                        best_params = params

            except Exception as e:
                continue

        self.best_params = best_params or {}
        return self.best_params

    def _create_param_grid(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Create parameter grid for optimization."""
        import itertools

        keys = list(param_ranges.keys())
        values = list(param_ranges.values())

        grid = []
        for combination in itertools.product(*values):
            grid.append(dict(zip(keys, combination)))

        return grid

    def _calculate_strategy_returns(self, weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns from weights and prices."""
        # Simple calculation - could be more sophisticated
        daily_returns = prices.pct_change()
        strategy_returns = (weights.shift(1) * daily_returns).sum(axis=1)
        return strategy_returns.dropna()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 30:
            return -np.inf

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if excess_returns.std() == 0:
            return -np.inf

        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return sharpe

def create_adaptive_signals(price_data: pd.DataFrame, feature_data: pd.DataFrame) -> pd.Series:
    """Create adaptive signals using multiple ML approaches."""
    signals = {}

    # Simple momentum + mean reversion combination
    for symbol in price_data.columns:
        prices = price_data[symbol]

        # Momentum signal
        mom_20 = prices / prices.shift(20) - 1
        mom_60 = prices / prices.shift(60) - 1

        # Mean reversion signal
        sma_20 = prices.rolling(20).mean()
        zscore = (prices - sma_20) / prices.rolling(20).std()

        # Volatility adjustment
        vol = prices.pct_change().rolling(20).std()

        # Combine signals with adaptive weights
        mom_weight = 0.6 if vol.iloc[-1] < vol.quantile(0.3) else 0.4  # More momentum in low vol
        mr_weight = 0.4 if vol.iloc[-1] < vol.quantile(0.3) else 0.6   # More mean reversion in high vol

        combined_signal = mom_weight * mom_20.iloc[-1] - mr_weight * zscore.iloc[-1]

        # Normalize to reasonable range
        signals[symbol] = np.clip(combined_signal, -0.5, 0.5)

    return pd.Series(signals)