from __future__ import annotations

import itertools
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2005-01-01"
    end: str | None = None
    timeframe: str = "1d"
    init_cash: float = 100_000.0
    fees: float = 0.0005  # 5 bps
    slippage: float = 0.0005  # 5 bps
    freq: str = "D"
    allow_short: bool = True


def _download_yf(tickers: List[str], start: str, end: str | None, interval: str) -> pd.DataFrame:
    # Use vectorbt's yfinance wrapper to keep format consistent.
    data = vbt.YFData.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        missing_index="drop",
        missing_columns="drop",
    )
    close = data.get("Close")
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.sort_index().dropna(how="all").ffill()
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_convert(None)
    return close


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


def _to_scalar(x) -> float:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = np.asarray(x).reshape(-1)
        x = x[~np.isnan(x)]
        return float(x[0]) if x.size else float("nan")
    return float(x)


def _trade_metrics_from_trades(trades: vbt.portfolio.trades.Trades) -> Dict[str, float]:
    # Profit factor = gross profits / gross losses (abs)
    pnl = trades.pnl.values
    pnl = pnl[~np.isnan(pnl)]
    if pnl.size == 0:
        return {"win_rate": np.nan, "profit_factor": np.nan, "trades": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float((-losses).sum()) if losses.size else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    win_rate = float((pnl > 0).mean())
    return {"win_rate": win_rate, "profit_factor": float(profit_factor), "trades": float(pnl.size)}


def run_ma_crossover_sweep(
    close: pd.DataFrame,
    fast_windows: Iterable[int],
    slow_windows: Iterable[int],
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], vbt.Portfolio]]:
    portfolios: Dict[Tuple[int, int], vbt.Portfolio] = {}
    rows = []

    for fast, slow in itertools.product(fast_windows, slow_windows):
        if fast >= slow:
            continue
        # Normalize outputs to plain DataFrames to avoid column metadata mismatches
        fast_ma = vbt.MA.run(close, window=fast).ma
        slow_ma = vbt.MA.run(close, window=slow).ma
        fast_ma = pd.DataFrame(np.asarray(fast_ma), index=close.index, columns=close.columns)
        slow_ma = pd.DataFrame(np.asarray(slow_ma), index=close.index, columns=close.columns)
        long_entries = fast_ma > slow_ma
        long_exits = fast_ma < slow_ma

        short_entries = None
        short_exits = None
        if cfg.allow_short:
            short_entries = fast_ma < slow_ma
            short_exits = fast_ma > slow_ma

        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            freq="D",
            fees=cfg.fees,
            slippage=cfg.slippage,
            init_cash=cfg.init_cash,
            direction="both" if cfg.allow_short else "longonly",
            cash_sharing=True,
            group_by=True,
        )
        portfolios[(fast, slow)] = pf

        trades = pf.trades
        tm = _trade_metrics_from_trades(trades)
        rows.append(
            {
                "strategy": "ma_crossover",
                "fast": fast,
                "slow": slow,
                "sharpe": _to_scalar(pf.sharpe_ratio()),
                "maxdd": _to_scalar(pf.max_drawdown()),
                "cagr": _to_scalar(pf.annualized_return()),
                "win_rate": tm["win_rate"],
                "profit_factor": tm["profit_factor"],
                "trades": tm["trades"],
            }
        )

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return df, portfolios


def run_rsi_mean_reversion_sweep(
    close: pd.DataFrame,
    rsi_windows: Iterable[int],
    entry_thresholds: Iterable[float],
    exit_thresholds: Iterable[float],
    short_entry_thresholds: Optional[Iterable[float]] = None,
    short_exit_thresholds: Optional[Iterable[float]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, float, float], vbt.Portfolio]]:
    portfolios: Dict[Tuple[int, float, float], vbt.Portfolio] = {}
    rows = []

    if short_entry_thresholds is None:
        short_entry_thresholds = [70.0]
    if short_exit_thresholds is None:
        short_exit_thresholds = [50.0]

    for win, ent, ext, sent, sext in itertools.product(
        rsi_windows, entry_thresholds, exit_thresholds, short_entry_thresholds, short_exit_thresholds
    ):
        if ent >= ext:
            continue
        if sent <= sext:
            continue
        rsi = vbt.RSI.run(close, window=win).rsi
        rsi = pd.DataFrame(np.asarray(rsi), index=close.index, columns=close.columns)

        long_entries = rsi < ent
        long_exits = rsi > ext

        short_entries = None
        short_exits = None
        if cfg.allow_short:
            short_entries = rsi > sent
            short_exits = rsi < sext

        pf = vbt.Portfolio.from_signals(
            close,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            freq="D",
            fees=cfg.fees,
            slippage=cfg.slippage,
            init_cash=cfg.init_cash,
            direction="both" if cfg.allow_short else "longonly",
            cash_sharing=True,
            group_by=True,
        )
        portfolios[(win, ent, ext)] = pf

        trades = pf.trades
        tm = _trade_metrics_from_trades(trades)
        rows.append(
            {
                "strategy": "rsi_mr",
                "rsi_window": win,
                "entry": ent,
                "exit": ext,
                "short_entry": sent,
                "short_exit": sext,
                "sharpe": _to_scalar(pf.sharpe_ratio()),
                "maxdd": _to_scalar(pf.max_drawdown()),
                "cagr": _to_scalar(pf.annualized_return()),
                "win_rate": tm["win_rate"],
                "profit_factor": tm["profit_factor"],
                "trades": tm["trades"],
            }
        )

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return df, portfolios


def _month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    me = s.groupby(index.to_period("M")).max().sort_values()
    return pd.DatetimeIndex(me.values)


def _weights_xs_mom_ls(
    close: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
    top_n: int = 10,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Dollar-neutral long/short momentum target weights, rebalanced monthly.

    Returns a DataFrame indexed like `close` but with weights only on rebalance days,
    and NaN elsewhere (so `from_orders` will only trade on rebalance days).
    """
    score = close.pct_change(lookback_days).shift(skip_days)
    rebal_dates = _month_end_index(close.index)
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    for dt in rebal_dates:
        s = score.loc[dt].dropna()
        if s.empty:
            continue
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        n = int(min(top_n, max(len(s) // 2, 1)))
        longs = s.sort_values(ascending=False).head(n).index
        shorts = s.sort_values(ascending=True).head(n).index
        per_side = gross_exposure / 2.0
        w.loc[dt, :] = 0.0
        w.loc[dt, longs] = per_side / float(len(longs))
        w.loc[dt, shorts] = -per_side / float(len(shorts))
    return w


def run_xs_mom_ls_sweep(
    close: pd.DataFrame,
    lookbacks: Iterable[int],
    skips: Iterable[int],
    top_ns: Iterable[int],
    gross_exposures: Iterable[float] = (1.0,),
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int, int], vbt.Portfolio]]:
    portfolios: Dict[Tuple[int, int, int], vbt.Portfolio] = {}
    rows = []

    for lb, sk, n, ge in itertools.product(lookbacks, skips, top_ns, gross_exposures):
        w = _weights_xs_mom_ls(close, lookback_days=int(lb), skip_days=int(sk), top_n=int(n), gross_exposure=float(ge))
        pf = vbt.Portfolio.from_orders(
            close,
            size=w,
            size_type="targetpercent",
            fees=cfg.fees,
            slippage=cfg.slippage,
            init_cash=cfg.init_cash,
            cash_sharing=True,
            group_by=True,
            freq=cfg.freq,
        )
        portfolios[(int(lb), int(sk), int(n))] = pf

        tm = _trade_metrics_from_trades(pf.trades)
        rows.append(
            {
                "strategy": "xs_mom_ls",
                "lookback": int(lb),
                "skip": int(sk),
                "top_n": int(n),
                "gross_exposure": float(ge),
                "sharpe": _to_scalar(pf.sharpe_ratio()),
                "maxdd": _to_scalar(pf.max_drawdown()),
                "cagr": _to_scalar(pf.annualized_return()),
                "win_rate": tm["win_rate"],
                "profit_factor": tm["profit_factor"],
                "trades": tm["trades"],
            }
        )

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    return df, portfolios


def gate(df: pd.DataFrame) -> pd.DataFrame:
    # Your targets (can tweak): Sharpe > 1.5, MaxDD < 20%, Win rate > 45%, PF > 1.5
    return df[
        (df["sharpe"] > 1.5)
        & (df["maxdd"] > -0.20)
        & (df["win_rate"] > 0.45)
        & (df["profit_factor"] > 1.5)
    ].copy()


def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_years: int = 8,
    test_years: int = 2,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    # Rolling non-overlapping splits by calendar time.
    start = index.min()
    end = index.max()
    splits = []
    cur = pd.Timestamp(start.year, start.month, start.day)

    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(years=test_years) - pd.DateOffset(days=1)
        if test_end > end:
            break
        splits.append((train_start, train_end, test_start, test_end))
        cur = cur + pd.DateOffset(years=test_years)

    # snap to available trading days
    snapped = []
    for a, b, c, d in splits:
        a2 = index[index >= a][0]
        b2 = index[index <= b][-1]
        c2 = index[index >= c][0]
        d2 = index[index <= d][-1]
        snapped.append((a2, b2, c2, d2))
    return snapped


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parent / "output_us_stocks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Universe: editable CSV. You can swap to a larger file generated by build_universe_alpaca.py
    universe_file = os.getenv("UNIVERSE_CSV", "universe_us_liquid.csv")
    uni_path = pathlib.Path(__file__).resolve().parent / universe_file
    tickers = _load_universe_csv(uni_path)

    close = _download_yf(tickers, start=cfg.start, end=cfg.end, interval=cfg.timeframe)
    close = close.dropna(axis=1, how="all")

    # Parameter grids (keep moderate for speed).
    ma_fast = [10, 20, 50]
    ma_slow = [100, 150, 200]
    rsi_win = [14, 21]
    rsi_entry = [20.0, 25.0, 30.0]
    rsi_exit = [60.0, 70.0]
    rsi_short_entry = [70.0, 75.0, 80.0] if cfg.allow_short else [70.0]
    rsi_short_exit = [55.0, 50.0] if cfg.allow_short else [50.0]
    xs_lb = [63, 126, 252]
    xs_skip = [0, 21]
    xs_topn = [5, 10, 20]
    xs_ge = [0.5, 1.0]

    ma_df, _ = run_ma_crossover_sweep(close, ma_fast, ma_slow)
    rsi_df, _ = run_rsi_mean_reversion_sweep(
        close, rsi_win, rsi_entry, rsi_exit, short_entry_thresholds=rsi_short_entry, short_exit_thresholds=rsi_short_exit
    )
    xs_df, _ = run_xs_mom_ls_sweep(close, xs_lb, xs_skip, xs_topn, gross_exposures=xs_ge)

    combined = pd.concat([ma_df, rsi_df, xs_df], ignore_index=True)
    combined.to_csv(out_dir / "all_results.csv", index=False)

    passed = gate(combined)
    passed.to_csv(out_dir / "passed_gate.csv", index=False)

    print("Saved:")
    print(f"- {out_dir / 'all_results.csv'}")
    print(f"- {out_dir / 'passed_gate.csv'}")
    print("")
    print("Top 15 by Sharpe (in-sample, not walk-forward):")
    print(combined.head(15).round(4))
    print("")
    print("Passed gate:")
    print(passed.head(25).round(4))

    # Walk-forward: pick best params on train, evaluate on test.
    splits = walk_forward_splits(close.index, train_years=8, test_years=2)
    wf_rows = []
    for tr_s, tr_e, te_s, te_e in splits:
        train_close = close.loc[tr_s:tr_e]
        test_close = close.loc[te_s:te_e]

        ma_train, _ = run_ma_crossover_sweep(train_close, ma_fast, ma_slow)
        rsi_train, _ = run_rsi_mean_reversion_sweep(
            train_close,
            rsi_win,
            rsi_entry,
            rsi_exit,
            short_entry_thresholds=rsi_short_entry,
            short_exit_thresholds=rsi_short_exit,
        )
        xs_train, _ = run_xs_mom_ls_sweep(train_close, xs_lb, xs_skip, xs_topn, gross_exposures=xs_ge)
        train_all = pd.concat([ma_train, rsi_train, xs_train], ignore_index=True)
        best = train_all.sort_values("sharpe", ascending=False).head(1).iloc[0].to_dict()

        # Evaluate best config on test period
        if best["strategy"] == "xs_mom_ls":
            lb = int(best["lookback"])
            sk = int(best["skip"])
            n = int(best["top_n"])
            ge = float(best.get("gross_exposure", 1.0))
            w = _weights_xs_mom_ls(test_close, lookback_days=lb, skip_days=sk, top_n=n, gross_exposure=ge)
            pf = vbt.Portfolio.from_orders(
                test_close,
                size=w,
                size_type="targetpercent",
                fees=cfg.fees,
                slippage=cfg.slippage,
                init_cash=cfg.init_cash,
                cash_sharing=True,
                group_by=True,
                freq=cfg.freq,
            )
        elif best["strategy"] == "ma_crossover":
            fast = int(best["fast"])
            slow = int(best["slow"])
            fast_ma = vbt.MA.run(test_close, window=fast).ma
            slow_ma = vbt.MA.run(test_close, window=slow).ma
            fast_ma = pd.DataFrame(np.asarray(fast_ma), index=test_close.index, columns=test_close.columns)
            slow_ma = pd.DataFrame(np.asarray(slow_ma), index=test_close.index, columns=test_close.columns)
            long_entries = fast_ma > slow_ma
            long_exits = fast_ma < slow_ma
            short_entries = (fast_ma < slow_ma) if cfg.allow_short else None
            short_exits = (fast_ma > slow_ma) if cfg.allow_short else None
            pf = vbt.Portfolio.from_signals(
                test_close,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                freq=cfg.freq,
                fees=cfg.fees,
                slippage=cfg.slippage,
                init_cash=cfg.init_cash,
                direction="both" if cfg.allow_short else "longonly",
                cash_sharing=True,
                group_by=True,
            )
        else:
            win = int(best["rsi_window"])
            ent = float(best["entry"])
            ext = float(best["exit"])
            sent = float(best.get("short_entry", 70.0))
            sext = float(best.get("short_exit", 50.0))
            rsi = vbt.RSI.run(test_close, window=win).rsi
            rsi = pd.DataFrame(np.asarray(rsi), index=test_close.index, columns=test_close.columns)
            long_entries = rsi < ent
            long_exits = rsi > ext
            short_entries = (rsi > sent) if cfg.allow_short else None
            short_exits = (rsi < sext) if cfg.allow_short else None
            pf = vbt.Portfolio.from_signals(
                test_close,
                entries=long_entries,
                exits=long_exits,
                short_entries=short_entries,
                short_exits=short_exits,
                freq=cfg.freq,
                fees=cfg.fees,
                slippage=cfg.slippage,
                init_cash=cfg.init_cash,
                direction="both" if cfg.allow_short else "longonly",
                cash_sharing=True,
                group_by=True,
            )
        tm = _trade_metrics_from_trades(pf.trades)
        wf_rows.append(
            {
                "train_start": tr_s,
                "train_end": tr_e,
                "test_start": te_s,
                "test_end": te_e,
                "picked_strategy": best["strategy"],
                "picked_params": {k: best[k] for k in best.keys() if k not in ["sharpe", "maxdd", "cagr", "win_rate", "profit_factor", "trades"]},
                "test_sharpe": _to_scalar(pf.sharpe_ratio()),
                "test_maxdd": _to_scalar(pf.max_drawdown()),
                "test_cagr": _to_scalar(pf.annualized_return()),
                "test_win_rate": tm["win_rate"],
                "test_profit_factor": tm["profit_factor"],
                "test_trades": tm["trades"],
            }
        )

    wf = pd.DataFrame(wf_rows)
    wf.to_csv(out_dir / "walk_forward.csv", index=False)
    print("")
    print(f"Saved: {out_dir / 'walk_forward.csv'}")
    print("Walk-forward results:")
    print(wf.round(4))


cfg = BacktestConfig()

if __name__ == "__main__":
    main()

