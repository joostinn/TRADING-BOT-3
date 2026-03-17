from __future__ import annotations

import math
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient


@dataclass(frozen=True)
class UniverseConfig:
    out_csv: str = "universe_us_top_dollar_volume.csv"
    top_n: int = 300
    lookback_days: int = 60  # recent trading days window (calendar days approximation)
    min_price: float = 5.0
    batch_size: int = 150  # Alpaca multi-symbol request size


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable {name}")
    return v


def _chunked(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    cfg = UniverseConfig()
    root = pathlib.Path(__file__).resolve().parent

    api_key = _get_env("ALPACA_API_KEY")
    secret_key = _get_env("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

    tc = TradingClient(api_key, secret_key, paper=paper)
    assets = tc.get_all_assets()

    # Filter to active, tradable US equities with simple symbols (avoid OTC / weird share classes here)
    symbols = []
    for a in assets:
        try:
            if not a.tradable:
                continue
            if getattr(a, "status", None) not in ("active", "ACTIVE", None):
                continue
            if getattr(a, "asset_class", None) not in ("us_equity", "US_EQUITY", None):
                continue
            sym = str(a.symbol).upper()
            # Skip symbols with spaces; keep '-' share classes (e.g. BRK-B)
            if " " in sym:
                continue
            symbols.append(sym)
        except Exception:
            continue
    symbols = sorted(set(symbols))
    if not symbols:
        raise RuntimeError("No symbols found from Alpaca assets.")

    dc = StockHistoricalDataClient(api_key, secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(cfg.lookback_days * 1.6))  # buffer to capture ~N trading days

    rows = []
    for batch in _chunked(symbols, cfg.batch_size):
        req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, end=end)
        df = dc.get_stock_bars(req).df
        if df is None or df.empty:
            continue
        # df: MultiIndex (symbol, timestamp)
        df = df.reset_index()
        df = df.sort_values(["symbol", "timestamp"])
        df["dollar_vol"] = df["close"] * df["volume"]

        # recent average dollar volume and last close
        for sym, g in df.groupby("symbol", sort=False):
            g2 = g.tail(cfg.lookback_days)
            if g2.empty:
                continue
            last_close = float(g2["close"].iloc[-1])
            if not math.isfinite(last_close) or last_close < cfg.min_price:
                continue
            adv = float(g2["dollar_vol"].mean())
            if not math.isfinite(adv) or adv <= 0:
                continue
            rows.append({"symbol": sym, "avg_dollar_vol": adv, "last_close": last_close})

    uni = pd.DataFrame(rows)
    if uni.empty:
        raise RuntimeError("No universe rows computed (bars may be empty or filters too strict).")

    uni = uni.sort_values("avg_dollar_vol", ascending=False).head(cfg.top_n)
    out_path = root / cfg.out_csv
    uni[["symbol"]].to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(uni)} symbols)")


if __name__ == "__main__":
    main()

