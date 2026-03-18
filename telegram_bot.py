from __future__ import annotations

import os
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str
    enabled: bool = True


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable {name}")
    return v


class TradingBotNotifier:
    def __init__(self, config: Optional[TelegramConfig] = None):
        if config is None:
            token = _get_env("TELEGRAM_BOT_TOKEN")
            chat_id = _get_env("TELEGRAM_CHAT_ID")
            enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("1", "true", "yes", "y")
            config = TelegramConfig(token=token, chat_id=chat_id, enabled=enabled)

        self.config = config

    def _send_message(self, message: str) -> bool:
        """Send a message using direct HTTP request."""
        if not self.config.enabled:
            print(f"Telegram disabled. Would send: {message}")
            return True

        url = f"https://api.telegram.org/bot{self.config.token}/sendMessage"
        data = {
            "chat_id": self.config.chat_id,
            "text": message
            # "parse_mode": "Markdown"  # Temporarily disabled due to parsing issues
        }

        try:
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            if result.get("ok"):
                return True
            else:
                print(f"Telegram API error: {result}")
                return False
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False

    def send_message_sync(self, message: str) -> None:
        """Synchronous wrapper for sending messages."""
        self._send_message(message)

    def notify_trade_execution(self, symbol: str, side: str, quantity: float, price: float) -> None:
        """Notify about a trade execution."""
        message = f"🔄 *Trade Executed*\n" \
                 f"Symbol: {symbol}\n" \
                 f"Side: {side.upper()}\n" \
                 f"Quantity: {quantity:.2f}\n" \
                 f"Price: ${price:.2f}\n" \
                 f"Value: ${quantity * price:.2f}"
        self.send_message_sync(message)

    def notify_rebalance(self, positions: dict[str, float], total_value: float) -> None:
        """Notify about portfolio rebalance."""
        message = f"📊 *Portfolio Rebalanced*\n" \
                 f"Total Value: ${total_value:.2f}\n\n" \
                 f"*Positions:*\n"

        for symbol, weight in sorted(positions.items()):
            if abs(weight) > 0.001:  # Only show meaningful positions
                message += f"{symbol}: {weight:.1%}\n"

        self.send_message_sync(message)

    def notify_performance_update(self, returns: pd.Series, sharpe: float, max_dd: float) -> None:
        """Notify about performance metrics."""
        recent_return = returns.tail(30).sum()  # Last 30 days
        ytd_return = returns[returns.index.year == pd.Timestamp.now().year].sum()

        message = f"📈 *Performance Update*\n" \
                 f"30-Day Return: {recent_return:.1%}\n" \
                 f"YTD Return: {ytd_return:.1%}\n" \
                 f"Sharpe Ratio: {sharpe:.2f}\n" \
                 f"Max Drawdown: {max_dd:.1%}"

        self.send_message_sync(message)

    def notify_market_regime_change(self, regime: str, spy_price: float, spy_ma: float) -> None:
        """Notify about market regime changes."""
        status = "🟢 BULL" if regime == "bull" else "🔴 BEAR"
        message = f"📊 *Market Regime Change*\n" \
                 f"Status: {status}\n" \
                 f"SPY Price: ${spy_price:.2f}\n" \
                 f"SPY 200MA: ${spy_ma:.2f}"

        self.send_message_sync(message)

    def notify_error(self, error_msg: str) -> None:
        """Notify about errors."""
        message = f"🚨 *Error Alert*\n{error_msg}"
        self.send_message_sync(message)

    def notify_daily_summary(self, pnl: float, trades_count: int, top_positions: list[tuple[str, float]]) -> None:
        """Send daily summary."""
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        message = f"📊 *Daily Summary*\n" \
                 f"P&L: {pnl_emoji} ${pnl:.2f}\n" \
                 f"Trades: {trades_count}\n\n" \
                 f"*Top Positions:*\n"

        for symbol, weight in top_positions[:5]:
            message += f"{symbol}: {weight:.1%}\n"

        self.send_message_sync(message)


# Global notifier instance
_notifier: Optional[TradingBotNotifier] = None


def get_notifier() -> TradingBotNotifier:
    """Get the global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TradingBotNotifier()
    return _notifier


def init_telegram_bot(token: str, chat_id: str, enabled: bool = True) -> None:
    """Initialize the telegram bot with custom config."""
    global _notifier
    config = TelegramConfig(token=token, chat_id=chat_id, enabled=enabled)
    _notifier = TradingBotNotifier(config)


# Example usage and testing
async def test_bot():
    """Test the telegram bot functionality."""
    notifier = get_notifier()
    await notifier.send_message("🤖 *Trading Bot Started*\nBot is now active and monitoring markets.")


if __name__ == "__main__":
    # Test the bot
    asyncio.run(test_bot())