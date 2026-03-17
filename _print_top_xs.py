import pandas as pd

df = pd.read_csv("output_us_stocks/all_results.csv")
xs = df[df["strategy"] == "xs_mom_ls"].sort_values("sharpe", ascending=False).head(10)
cols = [
    "lookback",
    "skip",
    "top_n",
    "gross_exposure",
    "sharpe",
    "maxdd",
    "cagr",
    "profit_factor",
    "trades",
]
print(xs[cols].to_string(index=False))

