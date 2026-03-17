from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestAssumptions:
    fee_bps: float = 1.0  # per 1x notional traded
    slippage_bps: float = 2.0  # per 1x notional traded
    annual_target_vol: float = 0.10
    max_leverage: float = 2.0


def _bps_cost_per_dollar(fee_bps: float, slippage_bps: float) -> float:
    return (fee_bps + slippage_bps) / 10_000.0


def _monthly_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Use last available trading day of each month in the series.
    months = index.to_period("M")
    s = pd.Series(index, index=index)
    month_ends = s.groupby(months).max().sort_values()
    return pd.DatetimeIndex(month_ends.values)


def strategy_buy_and_hold(
    returns: pd.DataFrame, ticker: str = "SPY"
) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    if ticker not in w.columns:
        raise ValueError(f"Ticker {ticker} not in returns columns.")
    w[ticker] = 1.0
    return w


def strategy_time_series_momentum_trend_filter(
    adj_close: pd.DataFrame,
    daily_vol: pd.DataFrame,
    lookback_days: int = 252,
    ma_days: int = 200,
    assumptions: Optional[BacktestAssumptions] = None,
) -> pd.DataFrame:
    """
    Simple trend following:
    - signal = sign(12m return) * 1{price > MA}
    - position sized to hit target annual vol using EWMA daily vol
    - monthly rebalance
    """
    if assumptions is None:
        assumptions = BacktestAssumptions()

    px = adj_close
    mom = px.pct_change(lookback_days)
    ma = px.rolling(ma_days, min_periods=ma_days).mean()
    signal = np.sign(mom) * (px > ma).astype(float)

    # Vol targeting: dollar weight per asset
    vol = daily_vol.copy()
    daily_target = assumptions.annual_target_vol / np.sqrt(252.0)
    raw_w = signal.mul(daily_target).div(vol)
    raw_w = raw_w.clip(lower=-assumptions.max_leverage, upper=assumptions.max_leverage)

    rebalance_days = _monthly_rebalance_dates(px.index)
    w = raw_w.loc[rebalance_days].reindex(px.index).ffill().fillna(0.0)
    return w


def strategy_cross_asset_momentum_rotation(
    adj_close: pd.DataFrame,
    daily_vol: pd.DataFrame,
    lookback_days: int = 252,
    skip_recent_days: int = 21,
    top_k: int = 3,
    assumptions: Optional[BacktestAssumptions] = None,
) -> pd.DataFrame:
    """
    Cross-asset (ETF basket) momentum:
    - score = 12m return excluding last month (12-1)
    - hold equal weight of top_k
    - optional vol targeting on the whole portfolio via scaling
    - monthly rebalance
    """
    if assumptions is None:
        assumptions = BacktestAssumptions()

    px = adj_close
    score = px.pct_change(lookback_days).shift(skip_recent_days)
    rebalance_days = _monthly_rebalance_dates(px.index)

    w_reb = pd.DataFrame(0.0, index=rebalance_days, columns=px.columns)
    for dt in rebalance_days:
        row = score.loc[dt].dropna()
        if row.empty:
            continue
        winners = row.sort_values(ascending=False).head(top_k).index
        w_reb.loc[dt, winners] = 1.0 / float(len(winners))

    # Vol scale by inverse realized vol of portfolio (approx using asset vols, assuming zero corr)
    vol = daily_vol.reindex(px.index)
    w = w_reb.reindex(px.index).ffill().fillna(0.0)
    port_vol_est = np.sqrt(((w**2) * (vol**2)).sum(axis=1))
    daily_target = assumptions.annual_target_vol / np.sqrt(252.0)
    scale = (daily_target / port_vol_est).clip(upper=assumptions.max_leverage).replace(
        [np.inf, -np.inf], np.nan
    )
    w = w.mul(scale, axis=0).fillna(0.0)
    return w


def backtest_from_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    assumptions: Optional[BacktestAssumptions] = None,
) -> pd.Series:
    """
    Vectorized backtest with linear costs:
    - portfolio return = sum(w_{t-1} * r_t) - cost * turnover_t
    where turnover_t = sum(|w_t - w_{t-1}|)
    """
    if assumptions is None:
        assumptions = BacktestAssumptions()

    returns, weights = returns.align(weights, join="inner", axis=0)
    weights = weights.reindex(columns=returns.columns).fillna(0.0)

    w_prev = weights.shift(1).fillna(0.0)
    gross = (w_prev * returns).sum(axis=1)

    turnover = (weights - w_prev).abs().sum(axis=1)
    cost_rate = _bps_cost_per_dollar(assumptions.fee_bps, assumptions.slippage_bps)
    net = gross - cost_rate * turnover
    return net.rename("portfolio_return")

