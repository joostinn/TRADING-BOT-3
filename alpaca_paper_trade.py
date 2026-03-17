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
    strategy: str = "ma_crossover"  # "ma_crossover", "rsi_mr", or "xs_mom_ls"
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

    current_qty = _get_positions_by_symbol(tc)

    # Compute deltas and place market orders (simple; no smart execution).
    for sym in target_shares.index:
        tgt = float(target_shares.loc[sym])
        cur = float(current_qty.get(sym, 0.0))
        diff = tgt - cur
        if abs(diff) < 1e-6:
            continue
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        _submit_market_order(tc, sym, qty=abs(diff), side=side)

    # If a symbol is held but not in target universe/weight, flatten it.
    for sym, cur in current_qty.items():
        if sym not in target_shares.index:
            # Sell closes longs; buy closes shorts.
            side = OrderSide.SELL if cur > 0 else OrderSide.BUY
            _submit_market_order(tc, sym, qty=abs(cur), side=side)
        else:
            tgt = float(target_shares.loc[sym])
            if tgt == 0 and cur != 0:
                side = OrderSide.SELL if cur > 0 else OrderSide.BUY
                _submit_market_order(tc, sym, qty=abs(cur), side=side)

    print("Rebalance submitted.")
    print("Target weights:")
    print(target_w[target_w != 0].round(4))


def main() -> None:
    cfg = PaperConfig()
    last = 0.0
    while True:
        now = time.time()
        if now - last >= cfg.rebalance_cooldown_sec:
            rebalance_once(cfg)
            last = now
        time.sleep(5)


if __name__ == "__main__":
    main()

