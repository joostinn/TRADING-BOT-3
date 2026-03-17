from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class PriceData:
    adj_close: pd.DataFrame  # rows: dates, cols: tickers


def download_adj_close(
    tickers: Iterable[str],
    start: str = "2005-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> PriceData:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
        threads=True,
    )

    # yfinance returns different shapes depending on count of tickers.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)) and (not auto_adjust):
            px = df["Adj Close"].copy()
        else:
            px = df["Close"].copy()
    else:
        col = "Adj Close" if ("Adj Close" in df.columns) else "Close"
        px = df[[col]].copy()
        px.columns = tickers[:1]

    px = px.sort_index()
    px = px.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    px = px.ffill().dropna(how="all")

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        raise RuntimeError(f"Missing tickers in downloaded data: {missing}")

    return PriceData(adj_close=px)


def to_daily_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    rets = adj_close.pct_change().replace([np.inf, -np.inf], np.nan)
    return rets


def estimate_daily_vol(returns: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    # EWMA volatility (annualization happens later)
    return returns.ewm(span=span, adjust=False, min_periods=span).std()

