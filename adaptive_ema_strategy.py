"""
ADAPTIVE PHASE 1: Learning EMA Strategy for Sharpe > 2.0
Dynamically adjusts parameters based on trade outcomes to achieve superior risk-adjusted returns.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
import pandas_ta as ta
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdaptiveEMAStrategy:
    """Adaptive EMA crossover strategy that learns from trade outcomes."""

    def __init__(self, learning_file: str = "adaptive_params.json"):
        self.learning_file = learning_file
        self.performance_history = []
        self.current_params = self._load_parameters()
        self.target_sharpe = 2.0
        self.learning_rate = 0.05  # How aggressively to adjust parameters

    def _load_parameters(self) -> Dict:
        """Load current parameters from file or use defaults."""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    params = json.load(f)
                print(f"📚 Loaded learned parameters: Sharpe {params.get('best_sharpe', 0):.3f}")
                return params
            except:
                pass

        # Default parameters (Phase 1 optimized)
        return {
            'fast_ema': 15,
            'slow_ema': 200,
            'min_separation': 0.025,
            'rsi_lower': 35,
            'rsi_upper': 65,
            'volatility_threshold': 1.1,
            'position_size': 0.15,
            'best_sharpe': 1.156,  # From backtest
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'last_update': datetime.now().isoformat()
        }

    def _save_parameters(self):
        """Save current parameters to file."""
        self.current_params['last_update'] = datetime.now().isoformat()
        with open(self.learning_file, 'w') as f:
            json.dump(self.current_params, f, indent=2)

    def update_from_trade_outcome(self, trade_result: Dict):
        """
        Learn from trade outcome and adjust parameters.

        trade_result format:
        {
            'symbol': 'SPY',
            'side': 'long'/'short',
            'entry_price': 100.0,
            'exit_price': 105.0,
            'pnl': 5.0,
            'pnl_pct': 0.05,
            'holding_period': 5,  # days
            'entry_signal_strength': 0.8,  # 0-1 scale
            'market_conditions': 'bull'/'bear'/'sideways'
        }
        """

        self.performance_history.append(trade_result)
        self.current_params['total_trades'] += 1

        # Update win/loss count
        if trade_result['pnl'] > 0:
            self.current_params['winning_trades'] += 1

        self.current_params['total_pnl'] += trade_result['pnl']

        # Calculate current win rate and Sharpe proxy
        win_rate = self.current_params['winning_trades'] / self.current_params['total_trades']
        avg_win = sum(t['pnl'] for t in self.performance_history if t['pnl'] > 0) / max(1, self.current_params['winning_trades'])
        avg_loss = abs(sum(t['pnl'] for t in self.performance_history if t['pnl'] < 0)) / max(1, self.current_params['total_trades'] - self.current_params['winning_trades'])

        # Ensure no division by zero
        avg_win = max(avg_win, 0.001)
        avg_loss = max(avg_loss, 0.001)

        # Sharpe proxy (simplified Kelly criterion approximation)
        kelly_f = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        current_sharpe_proxy = kelly_f * 2  # Rough approximation

        print(f"📊 Trade Learning: Win Rate {win_rate:.1f}, Sharpe Proxy {current_sharpe_proxy:.3f}")

        # Adaptive parameter adjustment based on trade outcome
        self._adapt_parameters(trade_result, win_rate, current_sharpe_proxy)

        # Save updated parameters
        self._save_parameters()

        # Report progress toward Sharpe 2.0
        progress = min(1.0, current_sharpe_proxy / self.target_sharpe)
        print(f"🎯 Progress to Sharpe 2.0: {progress:.1%} ({current_sharpe_proxy:.3f}/{self.target_sharpe:.3f})")

    def _adapt_parameters(self, trade_result: Dict, win_rate: float, sharpe_proxy: float):
        """Adapt parameters based on trade outcome and performance."""

        pnl = trade_result['pnl']
        pnl_pct = trade_result['pnl_pct']
        signal_strength = trade_result.get('entry_signal_strength', 0.5)

        # Learning logic based on trade outcome
        if pnl > 0:  # Winning trade
            self._learn_from_win(trade_result, win_rate, sharpe_proxy)
        else:  # Losing trade
            self._learn_from_loss(trade_result, win_rate, sharpe_proxy)

        # General adaptation based on overall performance
        if sharpe_proxy < 1.5:  # Below target
            # Be more conservative
            self.current_params['min_separation'] = min(0.04, self.current_params['min_separation'] * 1.02)
            self.current_params['position_size'] = max(0.08, self.current_params['position_size'] * 0.98)
            self.current_params['rsi_lower'] = min(40, self.current_params['rsi_lower'] + 0.5)
            self.current_params['rsi_upper'] = max(60, self.current_params['rsi_upper'] - 0.5)

        elif sharpe_proxy > 1.8:  # Approaching target
            # Be slightly more aggressive
            self.current_params['min_separation'] = max(0.015, self.current_params['min_separation'] * 0.98)
            self.current_params['position_size'] = min(0.20, self.current_params['position_size'] * 1.02)

        # Ensure parameters stay within reasonable bounds
        self._clamp_parameters()

    def _learn_from_win(self, trade_result: Dict, win_rate: float, sharpe_proxy: float):
        """Learn from winning trade - reinforce successful patterns."""

        signal_strength = trade_result.get('entry_signal_strength', 0.5)
        holding_period = trade_result.get('holding_period', 5)

        # If strong signal led to win, be more aggressive
        if signal_strength > 0.7:
            self.current_params['min_separation'] *= 0.99  # Accept slightly weaker signals
            self.current_params['position_size'] = min(0.18, self.current_params['position_size'] * 1.01)

        # If quick win, consider shorter EMA for faster signals
        if holding_period < 3 and win_rate > 0.6:
            self.current_params['fast_ema'] = max(8, int(self.current_params['fast_ema'] * 0.98))
            self.current_params['slow_ema'] = max(100, int(self.current_params['slow_ema'] * 0.99))

    def _learn_from_loss(self, trade_result: Dict, win_rate: float, sharpe_proxy: float):
        """Learn from losing trade - avoid similar patterns."""

        signal_strength = trade_result.get('entry_signal_strength', 0.5)
        pnl_pct = trade_result['pnl_pct']

        # If strong signal led to loss, be more selective
        if signal_strength > 0.7:
            self.current_params['min_separation'] = min(0.035, self.current_params['min_separation'] * 1.03)
            self.current_params['rsi_lower'] = min(42, self.current_params['rsi_lower'] + 1)
            self.current_params['rsi_upper'] = max(58, self.current_params['rsi_upper'] - 1)

        # If large loss, reduce position size and tighten filters
        if pnl_pct < -0.02:  # >2% loss
            self.current_params['position_size'] = max(0.10, self.current_params['position_size'] * 0.95)
            self.current_params['volatility_threshold'] = max(1.05, self.current_params['volatility_threshold'] * 0.98)

        # If win rate dropping, use longer EMA for more reliable signals
        if win_rate < 0.5:
            self.current_params['fast_ema'] = min(25, int(self.current_params['fast_ema'] * 1.02))
            self.current_params['slow_ema'] = min(300, int(self.current_params['slow_ema'] * 1.01))

    def _clamp_parameters(self):
        """Ensure parameters stay within reasonable bounds."""
        self.current_params['fast_ema'] = int(np.clip(self.current_params['fast_ema'], 5, 30))
        self.current_params['slow_ema'] = int(np.clip(self.current_params['slow_ema'], 50, 400))
        self.current_params['min_separation'] = np.clip(self.current_params['min_separation'], 0.01, 0.05)
        self.current_params['rsi_lower'] = int(np.clip(self.current_params['rsi_lower'], 25, 45))
        self.current_params['rsi_upper'] = int(np.clip(self.current_params['rsi_upper'], 55, 75))
        self.current_params['volatility_threshold'] = np.clip(self.current_params['volatility_threshold'], 1.0, 1.3)
        self.current_params['position_size'] = np.clip(self.current_params['position_size'], 0.08, 0.20)

    def get_current_signals(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals using current learned parameters."""

        signals = {}
        params = self.current_params

        for symbol in price_data.columns:
            prices = price_data[symbol].dropna()

            if len(prices) < params['slow_ema'] + 10:
                signals[symbol] = 0
                continue

            # Use learned EMA periods
            fast_ema = ta.ema(prices, length=params['fast_ema'])
            slow_ema = ta.ema(prices, length=params['slow_ema'])

            current_trend = (fast_ema > slow_ema).astype(int) * 2 - 1
            ema_separation = abs(fast_ema - slow_ema) / prices
            rsi = ta.rsi(prices, length=14)
            atr = ta.atr(high=prices, low=prices, close=prices, length=14)
            avg_atr = atr.rolling(20).mean()

            # Apply learned filters
            valid_signal = (
                ema_separation.iloc[-1] > params['min_separation'] and
                rsi.iloc[-1] > params['rsi_lower'] and rsi.iloc[-1] < params['rsi_upper'] and
                atr.iloc[-1] < avg_atr.iloc[-1] * params['volatility_threshold']
            )

            signal = current_trend.iloc[-1] if valid_signal else 0
            weight = params['position_size'] if signal == 1 else -params['position_size'] if signal == -1 else 0

            signals[symbol] = weight

        return signals

    def get_status_report(self) -> str:
        """Generate status report on learning progress."""
        params = self.current_params
        total_trades = params['total_trades']
        win_rate = params['winning_trades'] / max(1, total_trades)
        total_pnl = params['total_pnl']

        # Calculate current Sharpe proxy
        if self.performance_history:
            returns = [t['pnl_pct'] for t in self.performance_history]
            if len(returns) > 1:
                sharpe_proxy = np.mean(returns) / max(0.001, np.std(returns)) * np.sqrt(252)
            else:
                sharpe_proxy = 0
        else:
            sharpe_proxy = params.get('best_sharpe', 1.156)

        progress = min(1.0, sharpe_proxy / self.target_sharpe)

        report = f"""
🤖 ADAPTIVE EMA STRATEGY STATUS
{'='*40}
🎯 Target: Sharpe > 2.0 (Current: {sharpe_proxy:.3f})
📊 Progress: {progress:.1%}

📈 Performance:
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1%}
• Total P&L: ${total_pnl:.2f}
• Best Sharpe: {params['best_sharpe']:.3f}

🔧 Current Parameters:
• EMA: {params['fast_ema']}/{params['slow_ema']}
• Min Separation: {params['min_separation']:.3f}
• RSI Range: {params['rsi_lower']}-{params['rsi_upper']}
• Volatility Threshold: {params['volatility_threshold']:.2f}
• Position Size: {params['position_size']:.1%}

🧠 Learning Active: Adapting parameters based on trade outcomes
        """

        return report.strip()

# Global instance for the adaptive strategy
_adaptive_strategy = None

def get_adaptive_strategy() -> AdaptiveEMAStrategy:
    """Get or create the global adaptive strategy instance."""
    global _adaptive_strategy
    if _adaptive_strategy is None:
        _adaptive_strategy = AdaptiveEMAStrategy()
    return _adaptive_strategy

def adaptive_ema_signals(price_data: pd.DataFrame) -> Dict[str, float]:
    """Generate adaptive EMA signals for live trading."""
    strategy = get_adaptive_strategy()
    return strategy.get_current_signals(price_data)

def record_trade_outcome(trade_result: Dict):
    """Record a trade outcome for learning."""
    strategy = get_adaptive_strategy()
    strategy.update_from_trade_outcome(trade_result)

def get_adaptive_status() -> str:
    """Get current adaptive strategy status."""
    strategy = get_adaptive_strategy()
    return strategy.get_status_report()

if __name__ == "__main__":
    # Test the adaptive strategy
    strategy = AdaptiveEMAStrategy()

    # Simulate some trades for testing
    test_trades = [
        {'symbol': 'SPY', 'side': 'long', 'entry_price': 100, 'exit_price': 105, 'pnl': 5, 'pnl_pct': 0.05, 'holding_period': 3, 'entry_signal_strength': 0.8},
        {'symbol': 'QQQ', 'side': 'short', 'entry_price': 200, 'exit_price': 195, 'pnl': 5, 'pnl_pct': 0.025, 'holding_period': 2, 'entry_signal_strength': 0.9},
        {'symbol': 'SPY', 'side': 'long', 'entry_price': 100, 'exit_price': 97, 'pnl': -3, 'pnl_pct': -0.03, 'holding_period': 5, 'entry_signal_strength': 0.6},
    ]

    print("🧠 Testing Adaptive EMA Strategy Learning...")
    for i, trade in enumerate(test_trades, 1):
        print(f"\nTrade {i}: {trade['symbol']} {trade['side']} - P&L: ${trade['pnl']}")
        strategy.update_from_trade_outcome(trade)

    print(f"\n{sstrategy.get_status_report()}")