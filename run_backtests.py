from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from data import download_adj_close, estimate_daily_vol, to_daily_returns
from report import equity_curve, stats_table
from strategies import (
    BacktestAssumptions,
    backtest_from_weights,
    strategy_buy_and_hold,
    strategy_cross_asset_momentum_rotation,
    strategy_time_series_momentum_trend_filter,
)


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Liquid ETF proxies across major asset classes (imperfect vs futures, but accessible).
    tickers = [
        "SPY",  # US equities
        "EFA",  # developed ex-US equities
        "EEM",  # emerging equities
        "TLT",  # long Treasuries
        "IEF",  # intermediate Treasuries
        "GLD",  # gold
        "DBC",  # broad commodities
        "UUP",  # USD index proxy
    ]

    px = download_adj_close(tickers, start="2005-01-01").adj_close
    rets = to_daily_returns(px)
    vol = estimate_daily_vol(rets, span=60)

    assumptions = BacktestAssumptions(fee_bps=1.0, slippage_bps=2.0, annual_target_vol=0.10)

    # Strategies -> weights -> returns
    w_bh = strategy_buy_and_hold(rets, ticker="SPY")
    r_bh = backtest_from_weights(rets, w_bh, assumptions=assumptions)

    # Test different lookback windows
    w_ts_252 = strategy_time_series_momentum_trend_filter(
        adj_close=px, daily_vol=vol, lookback_days=252, ma_days=200, assumptions=assumptions,
        market_regime_filter=True, tighten_stop_loss=True, sharpe_scaling=True, vol_scaling=True
    )
    r_ts_252 = backtest_from_weights(rets, w_ts_252, assumptions=assumptions)

    w_ts_63 = strategy_time_series_momentum_trend_filter(
        adj_close=px, daily_vol=vol, lookback_days=63, ma_days=200, assumptions=assumptions,
        market_regime_filter=True, tighten_stop_loss=True, sharpe_scaling=True, vol_scaling=True
    )
    r_ts_63 = backtest_from_weights(rets, w_ts_63, assumptions=assumptions)

    w_xs = strategy_cross_asset_momentum_rotation(
        adj_close=px,
        daily_vol=vol,
        lookback_days=252,
        skip_recent_days=21,
        top_k=3,
        assumptions=assumptions,
    )
    r_xs = backtest_from_weights(rets, w_xs, assumptions=assumptions)

    strat_returns = {
        "BuyHold_SPY": r_bh,
        "TSMOM_252d_Enhanced": r_ts_252,
        "TSMOM_63d_Enhanced": r_ts_63,
        "XAsset_MomRot": r_xs
    }
    table = stats_table(strat_returns)
    table.to_csv(out_dir / "stats.csv")

    eq = pd.DataFrame({k: equity_curve(v) for k, v in strat_returns.items()}).dropna()
    eq.to_csv(out_dir / "equity_curves.csv")

    ax = eq.plot(figsize=(10, 6), logy=True, title="Equity curves (log scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (log)")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curves.png", dpi=160)

    print("Saved:")
    print(f"- {out_dir / 'stats.csv'}")
    print(f"- {out_dir / 'equity_curves.csv'}")
    print(f"- {out_dir / 'equity_curves.png'}")
    print("")
    print(table.round(4))


if __name__ == "__main__":
    main()

