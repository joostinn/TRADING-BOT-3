from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerfStats:
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0)
    return (1.0 + r).cumprod() * start_value


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(returns: pd.Series) -> PerfStats:
    r = returns.dropna()
    if r.empty:
        return PerfStats(cagr=np.nan, ann_vol=np.nan, sharpe=np.nan, max_drawdown=np.nan)

    eq = equity_curve(r, 1.0)
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)

    ann_vol = float(r.std(ddof=0) * math.sqrt(252.0))
    ann_ret = float(r.mean() * 252.0)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else np.nan

    mdd = max_drawdown(eq)
    return PerfStats(cagr=cagr, ann_vol=ann_vol, sharpe=sharpe, max_drawdown=mdd)


def stats_table(strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, r in strategy_returns.items():
        s = perf_stats(r)
        rows[name] = {
            "CAGR": s.cagr,
            "AnnVol": s.ann_vol,
            "Sharpe": s.sharpe,
            "MaxDD": s.max_drawdown,
        }
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values(by="Sharpe", ascending=False)

