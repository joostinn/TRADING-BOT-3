#!/usr/bin/env python3
"""
PAPER TRADING VALIDATION SYSTEM
Comprehensive live paper trading with ML data collection and optimization.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from improved_adaptive_strategy import record_improved_trade_outcome, get_improved_adaptive_status
from telegram_bot import get_notifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTradingValidator:
    """Comprehensive paper trading validation system with ML data collection."""

    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables required")

        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        self.notifier = get_notifier()

        # Trading universe
        self.tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV"]

        # Risk parameters
        self.max_positions = 5
        self.target_gross_exposure = 0.95
        self.max_position_weight = 0.25
        self.min_trade_size_dollars = 100.0

        # Performance tracking
        self.trade_history = []
        self.daily_performance = []
        self.starting_equity = None

        # ML data collection
        self.ml_data_file = "paper_trading_ml_data.json"

        logger.info("🚀 Paper Trading Validator initialized")
        logger.info(f"📊 Paper trading mode: {self.paper}")
        logger.info(f"📈 Universe: {len(self.tickers)} tickers")

    def get_account_info(self) -> Dict:
        """Get current account information."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'daytrade_count': int(account.daytrade_count)
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions with detailed information."""
        positions = {}
        try:
            for position in self.trading_client.get_all_positions():
                positions[position.symbol] = {
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price)
                }
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        return positions

    def get_market_data(self, days: int = 30) -> pd.DataFrame:
        """Get recent market data for signal generation."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            request = StockBarsRequest(
                symbol_or_symbols=self.tickers,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.data_client.get_stock_bars(request)
            data = bars.df

            if isinstance(data, pd.DataFrame) and not data.empty:
                # Reshape to wide format
                data = data.reset_index()
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.pivot(index='timestamp', columns='symbol', values='close')
                data = data.ffill().dropna()
                return data

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")

        return pd.DataFrame()

    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals using improved adaptive strategy."""
        try:
            from improved_adaptive_strategy import improved_adaptive_signals
            signals = improved_adaptive_signals(market_data)
            return signals
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return {}

    def apply_position_limits(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Apply position sizing and risk limits."""
        if not signals:
            return {}

        # Convert to series for easier manipulation
        signal_series = pd.Series(signals)

        # Cap individual position weights
        signal_series = signal_series.clip(-self.max_position_weight, self.max_position_weight)

        # Select top positions by absolute weight
        long_signals = signal_series[signal_series > 0].nlargest(self.max_positions // 2)
        short_signals = signal_series[signal_series < 0].nsmallest(self.max_positions // 2)
        selected_signals = pd.concat([long_signals, short_signals])

        # Zero out unselected positions
        signal_series = signal_series.where(signal_series.index.isin(selected_signals.index), 0.0)

        # Normalize to target gross exposure
        gross_exposure = signal_series.abs().sum()
        if gross_exposure > 0:
            scale_factor = self.target_gross_exposure / gross_exposure
            signal_series = signal_series * scale_factor

        return signal_series.to_dict()

    def execute_trades(self, target_weights: Dict[str, float], current_positions: Dict[str, Dict], account_info: Dict):
        """Execute trades to reach target weights."""
        executed_trades = []
        equity = account_info.get('equity', 0)

        if equity <= 0:
            logger.warning("No equity available for trading")
            return executed_trades

        # Get latest prices
        try:
            market_data = self.get_market_data(days=5)
            if market_data.empty:
                logger.error("No market data available for pricing")
                return executed_trades

            latest_prices = market_data.iloc[-1]
        except Exception as e:
            logger.error(f"Failed to get latest prices: {e}")
            return executed_trades

        # Calculate target dollar values
        target_dollars = {symbol: weight * equity for symbol, weight in target_weights.items()}

        # Calculate target shares
        target_shares = {}
        for symbol, dollars in target_dollars.items():
            if symbol in latest_prices.index:
                price = latest_prices[symbol]
                shares = dollars / price
                target_shares[symbol] = round(shares)

        # Get current shares
        current_shares = {symbol: pos['qty'] for symbol, pos in current_positions.items()}

        # Calculate trades needed
        for symbol in set(target_shares.keys()) | set(current_shares.keys()):
            target_qty = target_shares.get(symbol, 0)
            current_qty = current_shares.get(symbol, 0)
            diff = target_qty - current_qty

            if abs(diff) < 1:  # Skip tiny trades
                continue

            if symbol not in latest_prices.index:
                logger.warning(f"No price available for {symbol}")
                continue

            price = latest_prices[symbol]
            trade_value = abs(diff) * price

            if trade_value < self.min_trade_size_dollars:
                continue

            # Execute trade
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)

            try:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )

                order = self.trading_client.submit_order(order_data=order_request)

                trade_info = {
                    'symbol': symbol,
                    'side': side.value,
                    'qty': qty,
                    'price': price,
                    'value': trade_value,
                    'order_id': order.id,
                    'timestamp': datetime.now().isoformat()
                }

                executed_trades.append(trade_info)
                logger.info(f"✅ Executed {side.value} {qty} {symbol} @ ${price:.2f}")

            except Exception as e:
                logger.error(f"❌ Failed to execute trade for {symbol}: {e}")

        return executed_trades

    def collect_ml_data(self, account_info: Dict, positions: Dict[str, Dict], signals: Dict[str, float]):
        """Collect data for machine learning optimization."""
        try:
            ml_data_point = {
                'timestamp': datetime.now().isoformat(),
                'account_info': account_info,
                'positions': positions,
                'signals': signals,
                'market_conditions': self.get_market_conditions()
            }

            # Load existing data
            existing_data = []
            if os.path.exists(self.ml_data_file):
                try:
                    with open(self.ml_data_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []

            # Add new data point
            existing_data.append(ml_data_point)

            # Keep only last 1000 data points to prevent file from growing too large
            if len(existing_data) > 1000:
                existing_data = existing_data[-1000:]

            # Save updated data
            with open(self.ml_data_file, 'w') as f:
                json.dump(existing_data, f, indent=2, default=str)

            logger.info(f"📊 Collected ML data point ({len(existing_data)} total)")

        except Exception as e:
            logger.error(f"Failed to collect ML data: {e}")

    def get_market_conditions(self) -> Dict:
        """Get current market conditions for ML features."""
        try:
            market_data = self.get_market_data(days=30)
            if market_data.empty:
                return {}

            # Calculate market metrics
            returns = market_data.pct_change().dropna()

            conditions = {
                'volatility_30d': returns.std().mean(),
                'trend_strength': (market_data.iloc[-1] / market_data.iloc[0] - 1).mean(),
                'correlation_avg': returns.corr().mean().mean(),
                'volume_regime': 'normal'  # Could be enhanced with volume data
            }

            return conditions

        except Exception as e:
            logger.error(f"Failed to get market conditions: {e}")
            return {}

    def log_performance_summary(self):
        """Log comprehensive performance summary."""
        try:
            account_info = self.get_account_info()
            positions = self.get_current_positions()

            if not account_info:
                return

            equity = account_info.get('equity', 0)
            if self.starting_equity is None:
                self.starting_equity = equity

            total_return = (equity / self.starting_equity - 1) if self.starting_equity > 0 else 0

            # Calculate position P&L
            total_unrealized_pl = sum(pos.get('unrealized_pl', 0) for pos in positions.values())

            logger.info("=" * 60)
            logger.info("📊 PERFORMANCE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"💰 Equity: ${equity:,.2f}")
            logger.info(f"📈 Total Return: {total_return:.2%}")
            logger.info(f"🏦 Cash: ${account_info.get('cash', 0):,.2f}")
            logger.info(f"📊 Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            logger.info(f"🎯 Unrealized P&L: ${total_unrealized_pl:,.2f}")
            logger.info(f"📊 Positions: {len(positions)}")
            logger.info(f"🎯 Day Trades: {account_info.get('daytrade_count', 0)}")

            if positions:
                logger.info("📋 CURRENT POSITIONS:")
                for symbol, pos in positions.items():
                    pl_color = "🟢" if pos['unrealized_pl'] >= 0 else "🔴"
                    logger.info(f"  {symbol}: {pos['qty']:.0f} @ ${pos['avg_entry_price']:.2f} | "
                              f"Current: ${pos['current_price']:.2f} | "
                              f"P&L: {pl_color} ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2%})")

            # Strategy status
            try:
                strategy_status = get_improved_adaptive_status()
                logger.info("🧠 STRATEGY STATUS:")
                for line in strategy_status.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            except Exception as e:
                logger.error(f"Failed to get strategy status: {e}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Failed to log performance summary: {e}")

    def run_validation_cycle(self):
        """Run one complete validation cycle."""
        try:
            logger.info("🔄 Starting validation cycle...")

            # Get account and position data
            account_info = self.get_account_info()
            current_positions = self.get_current_positions()

            if not account_info:
                logger.error("Could not get account information")
                return

            # Get market data and generate signals
            market_data = self.get_market_data(days=60)  # Need enough history for adaptive strategy
            if market_data.empty:
                logger.error("No market data available")
                return

            signals = self.generate_signals(market_data)
            if not signals:
                logger.warning("No signals generated")
                return

            # Apply position limits
            target_weights = self.apply_position_limits(signals)

            # Execute trades
            executed_trades = self.execute_trades(target_weights, current_positions, account_info)

            # Collect ML data
            self.collect_ml_data(account_info, current_positions, signals)

            # Log performance
            self.log_performance_summary()

            # Send notification
            if executed_trades:
                trade_summary = f"Executed {len(executed_trades)} trades"
                self.notifier.notify_info(f"📊 Paper Trading Update: {trade_summary}")
            else:
                self.notifier.notify_info("📊 Paper Trading Update: No trades executed")

            logger.info("✅ Validation cycle completed")

        except Exception as e:
            logger.error(f"❌ Validation cycle failed: {e}")
            self.notifier.notify_error(f"Paper trading cycle failed: {e}")

def main():
    """Main paper trading validation function."""
    print("🚀 STARTING PAPER TRADING VALIDATION")
    print("=" * 50)

    # Check environment
    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set them before running:")
        print("   $env:ALPACA_API_KEY = 'your_api_key'")
        print("   $env:ALPACA_SECRET_KEY = 'your_secret_key'")
        print("   $env:ALPACA_PAPER = 'true'")
        return

    try:
        validator = PaperTradingValidator()

        print("✅ Paper trading validator initialized")
        print("🔄 Running initial validation cycle...")

        # Run initial cycle
        validator.run_validation_cycle()

        print("✅ Initial cycle completed")
        print("📊 Check paper_trading.log for detailed logs")
        print("📊 ML data being collected in paper_trading_ml_data.json")
        print("🔄 Run again during market hours for continuous validation")

    except Exception as e:
        print(f"❌ Failed to initialize paper trading: {e}")
        return

if __name__ == "__main__":
    main()