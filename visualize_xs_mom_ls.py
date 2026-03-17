from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass(frozen=True)
class VizConfig:
    universe_csv: str = "universe_us_liquid.csv"
    start: str = "2013-01-01"
    lookback_days: int = 126
    skip_days: int = 21
    top_n: int = 5
    gross_exposure: float = 1.0  # 1.0 => 50/50 long/short
    fees: float = 0.0005
    slippage: float = 0.0005
    init_cash: float = 100_000.0


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


def _month_end_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    me = s.groupby(index.to_period("M")).max().sort_values()
    return pd.DatetimeIndex(me.values)


def weights_xs_mom_ls(
    close: pd.DataFrame,
    lookback_days: int,
    skip_days: int,
    top_n: int,
    gross_exposure: float,
) -> pd.DataFrame:
    score = close.pct_change(lookback_days).shift(skip_days)
    rebal_dates = _month_end_index(close.index)
    w = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for dt in rebal_dates:
        s = score.loc[dt].replace([np.inf, -np.inf], np.nan).dropna()
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


def main() -> None:
    cfg = VizConfig()
    root = pathlib.Path(__file__).resolve().parent
    out_dir = root / "output_us_stocks" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = _load_universe_csv(root / cfg.universe_csv)

    close = vbt.YFData.download(
        tickers,
        start=cfg.start,
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

    w = weights_xs_mom_ls(
        close,
        lookback_days=cfg.lookback_days,
        skip_days=cfg.skip_days,
        top_n=cfg.top_n,
        gross_exposure=cfg.gross_exposure,
    )

    pf = vbt.Portfolio.from_orders(
        close,
        size=w,
        size_type="targetpercent",
        fees=cfg.fees,
        slippage=cfg.slippage,
        init_cash=cfg.init_cash,
        cash_sharing=True,
        group_by=True,
        freq="D",
    )

    value = pf.value().astype(float)
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    value = value.rename("equity").dropna()

    dd = (value / value.cummax() - 1.0).rename("drawdown")

    daily_ret = value.pct_change().dropna()
    roll = 252
    rolling_sharpe = (daily_ret.rolling(roll).mean() / daily_ret.rolling(roll).std(ddof=0)) * np.sqrt(252.0)

    # Plot: equity + drawdown + rolling sharpe
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(value.index, value.values)
    axes[0].set_title(f"xs_mom_ls equity (lb={cfg.lookback_days}, skip={cfg.skip_days}, top_n={cfg.top_n})")
    axes[0].set_ylabel("Equity")
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, color="red", alpha=0.25)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("DD")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, color="purple")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title(f"Rolling Sharpe ({roll}d)")
    axes[2].set_ylabel("Sharpe")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    eq_path = out_dir / "xs_mom_ls_equity_dd_rolling_sharpe.png"
    plt.savefig(eq_path, dpi=160)
    plt.close(fig)

    # Plot: xs parameter sweep heatmap (from saved CSV if present)
    res_path = root / "output_us_stocks" / "all_results.csv"
    if res_path.exists():
        df = pd.read_csv(res_path)
        xs = df[df["strategy"] == "xs_mom_ls"].copy()
        if not xs.empty:
            pivot = xs.pivot_table(index="lookback", columns="top_n", values="sharpe", aggfunc="mean")
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
            im = ax2.imshow(pivot.values, aspect="auto")
            ax2.set_title("xs_mom_ls Sharpe heatmap (from all_results.csv)")
            ax2.set_xlabel("top_n")
            ax2.set_ylabel("lookback")
            ax2.set_xticks(range(len(pivot.columns)))
            ax2.set_xticklabels([str(int(x)) for x in pivot.columns])
            ax2.set_yticks(range(len(pivot.index)))
            ax2.set_yticklabels([str(int(x)) for x in pivot.index])
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            plt.tight_layout()
            hm_path = out_dir / "xs_mom_ls_sharpe_heatmap.png"
            plt.savefig(hm_path, dpi=160)
            plt.close(fig2)

    print("Saved plots to:")
    print(f"- {out_dir}")


if __name__ == "__main__":
    main()

