from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from strategies import strategy_time_series_momentum_trend_filter, BacktestAssumptions
from enhanced_strategy import strategy_enhanced_momentum, EnhancedMomentumStrategy
from telegram_bot import get_notifier


@dataclass(frozen=True)
class PaperConfig:
    # Universe to trade
    tickers: Tuple[str, ...] = (
        "SPY",
        "QQQ",
        "IWM",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
    )
    # Strategy choice (keep it simple for live paper)
    strategy: str = "enhanced_ml"  # "ma_crossover", "rsi_mr", "xs_mom_ls", "enhanced_momentum", "enhanced_ml"
    allow_short: bool = True
    # Parameters (set after backtesting)
    ma_fast: int = 10
    ma_slow: int = 200
    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 60.0
    rsi_short_entry: float = 70.0
    rsi_short_exit: float = 50.0
    xs_lookback_days: int = 252
    xs_skip_days: int = 21
    xs_top_n: int = 10
    # Risk / sizing
    max_positions: int = 5
    target_gross_exposure: float = 0.95  # 95% of equity allocated when fully invested
    max_position_weight: float = 0.25
    min_trade_size_dollars: float = 100.0  # Minimum trade size to avoid excessive trading
    rebalance_cooldown_sec: int = 60 * 30  # don't rebalance more often than this


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable {name}")
    return v


def _latest_signals(close: pd.DataFrame, cfg: PaperConfig) -> pd.Series:
    """Return desired target weights (long/short) for the *latest* date."""
    if cfg.strategy == "ma_crossover":
        fast_ma = vbt.MA.run(close, window=cfg.ma_fast).ma
        slow_ma = vbt.MA.run(close, window=cfg.ma_slow).ma
        fast_ma = pd.DataFrame(np.asarray(fast_ma), index=close.index, columns=close.columns)
        slow_ma = pd.DataFrame(np.asarray(slow_ma), index=close.index, columns=close.columns)
        long_mask = (fast_ma > slow_ma).iloc[-1]
        short_mask = (fast_ma < slow_ma).iloc[-1] if cfg.allow_short else pd.Series(False, index=close.columns)
    elif cfg.strategy == "rsi_mr":
        rsi = vbt.RSI.run(close, window=cfg.rsi_window).rsi
        rsi = pd.DataFrame(np.asarray(rsi), index=close.index, columns=close.columns)
        long_mask = (rsi.iloc[-1] < cfg.rsi_entry)
        short_mask = (rsi.iloc[-1] > cfg.rsi_short_entry) if cfg.allow_short else pd.Series(False, index=close.columns)
    elif cfg.strategy == "xs_mom_ls":
        # Cross-sectional momentum: rank by (lookback return) skipping most recent period.
        score = close.pct_change(cfg.xs_lookback_days).shift(cfg.xs_skip_days).iloc[-1]
        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        if score.empty:
            return pd.Series(0.0, index=close.columns)
        n = int(min(cfg.xs_top_n, max(len(score) // 2, 1)))
        longs = score.sort_values(ascending=False).head(n).index.tolist()
        shorts = score.sort_values(ascending=True).head(n).index.tolist() if cfg.allow_short else []
        w = pd.Series(0.0, index=close.columns)
        if longs:
            w.loc[longs] = min(cfg.target_gross_exposure / 2.0 / len(longs), cfg.max_position_weight)
        if shorts:
            w.loc[shorts] = -min(cfg.target_gross_exposure / 2.0 / len(shorts), cfg.max_position_weight)
        return w
    elif cfg.strategy == "enhanced_momentum":
        # Use the enhanced time series momentum strategy
        from data import estimate_daily_vol, to_daily_returns

        # Get more data for calculations
        extended_close = vbt.YFData.download(
            list(cfg.tickers),
            period="5y",  # Need more history for calculations
            interval="1d",
            auto_adjust=True,
            missing_index="drop",
            missing_columns="drop",
        ).get("Close")
        if isinstance(extended_close, pd.Series):
            extended_close = extended_close.to_frame()
        extended_close = extended_close.sort_index().dropna(how="all").ffill()
        if getattr(extended_close.index, "tz", None) is not None:
            extended_close.index = extended_close.index.tz_convert(None)

        extended_rets = to_daily_returns(extended_close)
        extended_vol = estimate_daily_vol(extended_rets, span=60)

        assumptions = BacktestAssumptions(
            fee_bps=1.0, slippage_bps=2.0, annual_target_vol=0.10, max_leverage=2.0
        )

        # Use 63-day lookback for more responsive signals
        weights = strategy_time_series_momentum_trend_filter(
            adj_close=extended_close,
            daily_vol=extended_vol,
            lookback_days=63,  # More responsive than 252
            ma_days=200,
            assumptions=assumptions,
            market_regime_filter=True,
            tighten_stop_loss=True,
            sharpe_scaling=True,
            vol_scaling=True
        )

        # Return the latest weights
        latest_weights = weights.iloc[-1] if not weights.empty else pd.Series(0.0, index=close.columns)
        latest_weights = latest_weights.reindex(close.columns).fillna(0.0)

        # Apply position sizing constraints
        latest_weights = _apply_position_limits(latest_weights, cfg)

        return latest_weights
    elif cfg.strategy == "enhanced_ml":
        # Use the enhanced ML strategy
        from data import estimate_daily_vol, to_daily_returns

        # Get more data for calculations
        extended_close = vbt.YFData.download(
            list(cfg.tickers),
            period="2y",  # Need more history for ML training
            interval="1d",
            auto_adjust=True,
            missing_index="drop",
            missing_columns="drop",
        ).get("Close")
        if isinstance(extended_close, pd.Series):
            extended_close = extended_close.to_frame()
        extended_close = extended_close.sort_index().dropna(how="all").ffill()
        if getattr(extended_close.index, "tz", None) is not None:
            extended_close.index = extended_close.index.tz_convert(None)

        # Generate enhanced signals
        enhanced_strategy = EnhancedMomentumStrategy()
        signals = enhanced_strategy.generate_signals(extended_close)

        # Apply position sizing constraints
        signals = _apply_position_limits(signals, cfg)

        return signals.reindex(close.columns).fillna(0.0)
    else:
        raise ValueError(f"Unknown strategy {cfg.strategy}")

    longs = long_mask[long_mask].index.tolist()
    shorts = short_mask[short_mask].index.tolist()

    # Keep gross exposure bounded by splitting slots
    half = max(cfg.max_positions // 2, 1) if cfg.allow_short else cfg.max_positions
    longs = longs[: cfg.max_positions] if not cfg.allow_short else longs[:half]
    shorts = shorts[:half] if cfg.allow_short else []

    if not longs and not shorts:
        return pd.Series(0.0, index=close.columns)

    w = pd.Series(0.0, index=close.columns)
    n = len(longs) + len(shorts)
    per = min(cfg.target_gross_exposure / max(n, 1), cfg.max_position_weight)
    if longs:
        w.loc[longs] = per
    if shorts:
        w.loc[shorts] = -per
    return w


def _apply_position_limits(weights: pd.Series, cfg: PaperConfig) -> pd.Series:
    """Apply position sizing limits to strategy weights."""
    if weights.empty:
        return weights

    # Cap individual position weights
    weights = weights.clip(-cfg.max_position_weight, cfg.max_position_weight)

    # Select top positions by absolute weight
    if cfg.allow_short:
        # For long/short strategies, select top positions on each side
        long_weights = weights[weights > 0].nlargest(cfg.max_positions // 2)
        short_weights = weights[weights < 0].nsmallest(cfg.max_positions // 2)
        selected_weights = pd.concat([long_weights, short_weights])
    else:
        # For long-only strategies, select top long positions
        selected_weights = weights[weights > 0].nlargest(cfg.max_positions)

    # Zero out unselected positions
    weights = weights.where(weights.index.isin(selected_weights.index), 0.0)

    # Normalize to target gross exposure
    gross_exposure = weights.abs().sum()
    if gross_exposure > 0:
        scale_factor = cfg.target_gross_exposure / gross_exposure
        weights = weights * scale_factor

    return weights


def _get_positions_by_symbol(tc: TradingClient) -> Dict[str, float]:
    pos = {}
    for p in tc.get_all_positions():
        try:
            pos[p.symbol] = float(p.qty)
        except Exception:
            continue
    return pos


def _submit_market_order(tc: TradingClient, symbol: str, qty: float, side: OrderSide) -> None:
    if qty <= 0:
        return
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    tc.submit_order(order_data=req)


def rebalance_once(cfg: PaperConfig) -> None:
    api_key = _get_env("ALPACA_API_KEY")
    secret_key = _get_env("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

    tc = TradingClient(api_key, secret_key, paper=paper)
    acct = tc.get_account()
    equity = float(acct.equity)
    buying_power = float(acct.buying_power)

    notifier = get_notifier()

    # Use yfinance daily bars for the signal (free). For production, use Alpaca data.
    close = vbt.YFData.download(
        list(cfg.tickers),
        period="3y",
        interval="1d",
        auto_adjust=True,
        missing_index="drop",
        missing_columns="drop",
    ).get("Close")
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.sort_index().dropna(how="all").ffill()
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert(None)

    target_w = _latest_signals(close, cfg)
    last_px = close.iloc[-1]
    target_dollars = target_w * equity
    target_shares = (target_dollars / last_px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Round to whole shares since Alpaca doesn't allow fractional shares for short sales
    target_shares = target_shares.round().astype(int)

    current_qty = _get_positions_by_symbol(tc)

    trades_executed = []

    # Compute deltas and place market orders (simple; no smart execution).
    for sym in target_shares.index:
        tgt = float(target_shares.loc[sym])
        cur = float(current_qty.get(sym, 0.0))
        diff = tgt - cur
        if abs(diff) < 1e-6:
            continue

        # For selling, don't sell more than we have
        if diff < 0:
            qty_to_trade = min(abs(diff), cur)  # Don't sell more than current position
        else:
            # For buying, check buying power
            trade_cost = abs(diff) * last_px.loc[sym]
            if trade_cost > buying_power:
                # Scale down the buy to fit buying power
                max_qty = buying_power // last_px.loc[sym]
                qty_to_trade = min(abs(diff), max_qty)
                if qty_to_trade < 1:
                    continue  # Skip if can't buy even 1 share
            else:
                qty_to_trade = abs(diff)

        # Check minimum trade size
        trade_value = qty_to_trade * last_px.loc[sym]
        if trade_value < cfg.min_trade_size_dollars:
            continue

        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        try:
            print(f"Trading {sym}: {side.value} {qty_to_trade} shares (target: {tgt}, current: {cur}, diff: {diff})")
            _submit_market_order(tc, sym, qty=qty_to_trade, side=side)
            trades_executed.append((sym, side.value, qty_to_trade, last_px.loc[sym]))
            # Update buying power after successful buy
            if side == OrderSide.BUY:
                buying_power -= qty_to_trade * last_px.loc[sym]
        except Exception as e:
            notifier.notify_error(f"Failed to execute trade for {sym}: {e}")

    # If a symbol is held but not in target universe/weight, flatten it.
    for sym, cur in current_qty.items():
        if sym not in target_shares.index:
            # Sell closes longs; buy closes shorts.
            side = OrderSide.SELL if cur > 0 else OrderSide.BUY
            trade_value = abs(cur) * (last_px.get(sym, 0))
            if trade_value >= cfg.min_trade_size_dollars:
                try:
                    _submit_market_order(tc, sym, qty=abs(cur), side=side)
                    trades_executed.append((sym, side.value, abs(cur), last_px.get(sym, 0)))
                except Exception as e:
                    notifier.notify_error(f"Failed to close position for {sym}: {e}")
        else:
            tgt = float(target_shares.loc[sym])
            if tgt == 0 and cur != 0:
                side = OrderSide.SELL if cur > 0 else OrderSide.BUY
                trade_value = abs(cur) * last_px.loc[sym]
                if trade_value >= cfg.min_trade_size_dollars:
                    try:
                        _submit_market_order(tc, sym, qty=abs(cur), side=side)
                        trades_executed.append((sym, side.value, abs(cur), last_px.loc[sym]))
                    except Exception as e:
                        notifier.notify_error(f"Failed to close position for {sym}: {e}")

    # Send notifications
    if trades_executed:
        for sym, side, qty, price in trades_executed:
            notifier.notify_trade_execution(sym, side, qty, price)

    # Notify about rebalance
    positions = {sym: float(target_shares.loc[sym]) * last_px.loc[sym] / equity
                for sym in target_shares.index if abs(target_shares.loc[sym]) > 0}
    notifier.notify_rebalance(positions, equity)

    print("Rebalance submitted.")
    print("Target weights:")
    print(target_w[target_w != 0].round(4))


def main() -> None:
    cfg = PaperConfig()
    notifier = get_notifier()

    # Format the startup message
    strategy_name = cfg.strategy
    tickers_list = ', '.join(cfg.tickers)
    hours = cfg.rebalance_cooldown_sec // 3600
    minutes = (cfg.rebalance_cooldown_sec % 3600) // 60

    startup_message = "Trading Bot Started - Now monitoring markets"

    notifier.send_message_sync(startup_message)

    last = 0.0
    while True:
        now = time.time()
        if now - last >= cfg.rebalance_cooldown_sec:
            try:
                rebalance_once(cfg)
                last = now
            except Exception as e:
                notifier.notify_error(f"Rebalance failed: {e}")
                print(f"Rebalance failed: {e}")
                time.sleep(60)  # Wait a minute before retrying
        time.sleep(5)


if __name__ == "__main__":
    main()

