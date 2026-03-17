from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

from notify import get_telegram_config_from_env, send_telegram_message


@dataclass(frozen=True)
class LiveConfig:
    universe_csv: str = "universe_us_liquid.csv"
    strategy: str = "xs_mom_ls"  # "xs_mom_ls", "ma_crossover", "rsi_mr"
    allow_short: bool = True

    # Signal parameters
    xs_lookback_days: int = 252
    xs_skip_days: int = 21
    xs_top_n: int = 10

    ma_fast: int = 50
    ma_slow: int = 200

    rsi_window: int = 14
    rsi_entry: float = 30.0
    rsi_exit: float = 60.0
    rsi_short_entry: float = 70.0
    rsi_short_exit: float = 50.0

    # Sizing / risk
    target_gross_exposure: float = 1.0  # e.g. 1.0 = 100% gross (50/50 long/short)
    max_position_weight: float = 0.10

    # Data window for indicators
    history_years: int = 5

    # Output
    out_dir: str = "output_live"


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable {name}")
    return v


def _load_universe_csv(path: pathlib.Path) -> List[str]:
    df = pd.read_csv(path)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    tickers = (
        df[col]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .str.upper()
        .tolist()
    )
    return list(dict.fromkeys(tickers))


def _download_alpaca_daily_close(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    api_key = _get_env("ALPACA_API_KEY")
    secret_key = _get_env("ALPACA_SECRET_KEY")

    dc = StockHistoricalDataClient(api_key, secret_key)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end)
    bars = dc.get_stock_bars(req).df

    # bars is a MultiIndex: (symbol, timestamp)
    if bars.empty:
        raise RuntimeError("No bars returned from Alpaca.")

    close = (
        bars.reset_index()
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .dropna(how="all")
        .ffill()
    )
    # Normalize timestamps to tz-naive dates for indicator logic
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert(None)
    return close


def _target_weights(close: pd.DataFrame, cfg: LiveConfig) -> pd.Series:
    if cfg.strategy == "xs_mom_ls":
        score = close.pct_change(cfg.xs_lookback_days).shift(cfg.xs_skip_days).iloc[-1]
        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        if score.empty:
            return pd.Series(0.0, index=close.columns)
        n = int(min(cfg.xs_top_n, max(len(score) // 2, 1)))
        longs = score.sort_values(ascending=False).head(n).index.tolist()
        shorts = score.sort_values(ascending=True).head(n).index.tolist() if cfg.allow_short else []

        w = pd.Series(0.0, index=close.columns)
        if longs:
            per = min(cfg.target_gross_exposure / 2.0 / len(longs), cfg.max_position_weight)
            w.loc[longs] = per
        if shorts:
            per = min(cfg.target_gross_exposure / 2.0 / len(shorts), cfg.max_position_weight)
            w.loc[shorts] = -per
        return w

    if cfg.strategy == "ma_crossover":
        fast_ma = vbt.MA.run(close, window=cfg.ma_fast).ma
        slow_ma = vbt.MA.run(close, window=cfg.ma_slow).ma
        fast_ma = pd.DataFrame(np.asarray(fast_ma), index=close.index, columns=close.columns)
        slow_ma = pd.DataFrame(np.asarray(slow_ma), index=close.index, columns=close.columns)
        long_mask = (fast_ma > slow_ma).iloc[-1]
        short_mask = (fast_ma < slow_ma).iloc[-1] if cfg.allow_short else pd.Series(False, index=close.columns)

        longs = long_mask[long_mask].index.tolist()
        shorts = short_mask[short_mask].index.tolist()
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

    if cfg.strategy == "rsi_mr":
        rsi = vbt.RSI.run(close, window=cfg.rsi_window).rsi
        rsi = pd.DataFrame(np.asarray(rsi), index=close.index, columns=close.columns)
        long_mask = (rsi.iloc[-1] < cfg.rsi_entry)
        short_mask = (rsi.iloc[-1] > cfg.rsi_short_entry) if cfg.allow_short else pd.Series(False, index=close.columns)

        longs = long_mask[long_mask].index.tolist()
        shorts = short_mask[short_mask].index.tolist()
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

    raise ValueError(f"Unknown strategy {cfg.strategy}")


def _current_positions(tc: TradingClient) -> Dict[str, float]:
    pos = {}
    for p in tc.get_all_positions():
        try:
            pos[p.symbol] = float(p.qty)
        except Exception:
            continue
    return pos


def main() -> None:
    cfg = LiveConfig()
    root = pathlib.Path(__file__).resolve().parent
    out_dir = root / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Alpaca clients (trading is used ONLY to read equity and positions)
    api_key = _get_env("ALPACA_API_KEY")
    secret_key = _get_env("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")
    tc = TradingClient(api_key, secret_key, paper=paper)
    acct = tc.get_account()
    equity = float(acct.equity)
    cur_pos = _current_positions(tc)

    universe_file = os.getenv("UNIVERSE_CSV", cfg.universe_csv)
    universe = _load_universe_csv(root / universe_file)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(cfg.history_years * 365.25))
    close = _download_alpaca_daily_close(universe, start=start, end=end)

    w = _target_weights(close, cfg)
    last_px = close.iloc[-1]
    target_dollars = w * equity
    target_shares = (target_dollars / last_px).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    rows = []
    for sym in target_shares.index:
        tgt = float(target_shares.loc[sym])
        cur = float(cur_pos.get(sym, 0.0))
        diff = tgt - cur
        if abs(diff) < 1e-6:
            continue
        side = "BUY" if diff > 0 else "SELL"
        rows.append(
            {
                "symbol": sym,
                "side": side,
                "qty": round(abs(diff), 6),
                "current_qty": cur,
                "target_qty": tgt,
                "target_weight": float(w.loc[sym]),
                "last_close": float(last_px.loc[sym]) if sym in last_px.index else np.nan,
            }
        )

    orders = pd.DataFrame(rows).sort_values(["side", "symbol"])

    # "Boom candidates": strongest-ranked names by the algorithm (watchlist)
    watch = pd.DataFrame(
        {
            "symbol": w.index,
            "target_weight": w.values,
            "last_close": last_px.reindex(w.index).values,
        }
    )
    watch = watch[watch["target_weight"] != 0].copy()
    watch["abs_weight"] = watch["target_weight"].abs()
    watch = watch.sort_values(["abs_weight", "symbol"], ascending=[False, True]).drop(columns=["abs_weight"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    orders_path = out_dir / f"orders_{cfg.strategy}_{ts}.csv"
    watch_path = out_dir / f"watchlist_{cfg.strategy}_{ts}.csv"
    orders.to_csv(orders_path, index=False)
    watch.to_csv(watch_path, index=False)

    print(f"Equity: {equity}")
    print(f"Saved orders: {orders_path}")
    print(f"Saved watchlist: {watch_path}")
    print("")
    print("Preview orders:")
    print(orders.head(30).to_string(index=False))

    tg = get_telegram_config_from_env()
    if tg is not None:
        n_orders = int(len(orders))
        n_watch = int(len(watch))
        top_watch = watch.copy()
        top_watch = top_watch.assign(side=np.where(top_watch["target_weight"] > 0, "LONG", "SHORT"))
        top_watch = top_watch.sort_values("target_weight", ascending=False)
        top_lines = []
        for _, r in top_watch.head(10).iterrows():
            top_lines.append(f"- {r['side']} `{r['symbol']}` w={float(r['target_weight']):.3f} px={float(r['last_close']):.2f}")

        msg = "\n".join(
            [
                "*Trade list generated*",
                f"- Strategy: `{cfg.strategy}`",
                f"- Orders: *{n_orders}*",
                f"- Watchlist: *{n_watch}*",
                f"- Orders file: `{orders_path.name}`",
                f"- Watchlist file: `{watch_path.name}`",
                "",
                "*Top picks (by target weight)*",
                *top_lines,
            ]
        )
        send_telegram_message(tg, msg)


if __name__ == "__main__":
    main()

