# dispersion_strategy.py
# Minimal dispersion signal (realized correlation only) + tiny backtest (hit rate)
# Saves figures/CSVs to ./reports next to this script.

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ---------- data ----------
def get_prices(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    """
    Robust adjusted prices using Yahoo (yfinance) with auto_adjust=True.
    Returns DataFrame with columns=tickers, index=dates (UTC-naive).
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,   # adjusted prices live in "Close"
        group_by="column",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data (check tickers/dates/connection).")

    # Multi-ticker => MultiIndex columns; single-ticker => flat columns
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise KeyError("Expected 'Close' in downloaded columns.")
        prices = df["Close"].copy()
    else:
        if "Close" not in df.columns:
            raise KeyError("Expected 'Close' in downloaded columns.")
        # ensure a 2D frame with the ticker name as column
        name = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
        prices = df[["Close"]].copy()
        prices.columns = [name]

    # drop all-NaN rows
    return prices.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Business-day log returns from adjusted prices."""
    ret = np.log(prices).diff()
    ret = ret.asfreq("B").dropna(how="any")
    return ret


# ---------- signal ----------
def rolling_avg_pairwise_corr(returns: pd.DataFrame, cols, window=21) -> pd.Series:
    """
    window-day realized average pairwise correlation among component columns.
    """
    out, idx = [], []
    for i in range(window, len(returns) + 1):
        R = returns[cols].iloc[i - window : i]
        C = R.corr().values
        iu = np.triu_indices_from(C, k=1)
        out.append(np.nanmean(C[iu]))
        idx.append(R.index[-1])   # <-- was R.index[i-1] (bug). Use the last date of the window.
    return pd.Series(out, index=idx, name=f"avg_pairwise_corr_{window}d")



def past_only_z(series: pd.Series, lookback_days=252 * 3) -> pd.Series:
    """
    z_t = (x_t - mean_{t-1 over L}) / std_{t-1 over L}
    Uses rolling past-only mean/std to avoid look-ahead.
    """
    mu = series.rolling(lookback_days, min_periods=100).mean().shift(1)
    sd = series.rolling(lookback_days, min_periods=100).std().shift(1)
    return ((series - mu) / sd).rename(series.name + "_z")


def regime_label(z: float) -> str:
    if np.isnan(z):
        return "n/a"
    if z <= -1.0:
        return "LOW (expect corr ↑ → short-dispersion stance)"
    if z >= 1.0:
        return "HIGH (expect corr ↓ → long-dispersion stance)"
    return "NORMAL"


# ---------- figures ----------
def save_figures(rcorr: pd.Series, zcorr: pd.Series, reports_dir: Path, window: int):
    reports_dir.mkdir(exist_ok=True)

    # Figure 1: correlation history
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    rcorr.plot(ax=ax1)
    ax1.set_title(f"{window}d Average Pairwise Correlation (components)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Correlation")
    fig1.tight_layout()
    f1 = reports_dir / "fig_corr.png"
    fig1.savefig(f1, dpi=160)
    plt.close(fig1)

    # Figure 2: z-score with ±1 lines
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    zcorr.plot(ax=ax2)
    ax2.axhline(1.0, linestyle="--")
    ax2.axhline(-1.0, linestyle="--")
    ax2.set_title("Past-only z-score of correlation")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("z")
    fig2.tight_layout()
    f2 = reports_dir / "fig_zscore.png"
    fig2.savefig(f2, dpi=160)
    plt.close(fig2)

    print(f"Saved figures: {f1}, {f2}")


# ---------- main snapshot ----------
def main(index_ticker: str, components: list[str], start: str, window: int, lookback: int):
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    tickers = [index_ticker] + components
    prices = get_prices(tickers, start=start)
    returns = log_returns(prices)

    # realized correlation uses components only
    rcorr = rolling_avg_pairwise_corr(returns, components, window=window)
    zcorr = past_only_z(rcorr, lookback_days=lookback)

    if zcorr.dropna().empty:
        raise RuntimeError("Not enough data to compute z-scores (increase lookback or start earlier).")

    last = zcorr.dropna().index[-1]
    zval = float(zcorr.loc[last])
    lab = regime_label(zval)

    print(f"Date: {last.date()}")
    print(f"{window}d avg pairwise corr: {rcorr.loc[last]:.3f}")
    print(f"z-score (past-only, {lookback}d lookback): {zval:.2f}")
    print(f"Regime: {lab}")

    # Save tiny CSV snapshot
    snap = pd.DataFrame(
        {
            "date": [last.date()],
            "realized_corr": [float(rcorr.loc[last])],
            "realized_corr_z": [zval],
            "regime": [lab],
        }
    )
    out_csv = reports / f"realized_corr_snapshot_{last.date()}.csv"
    snap.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Figures
    save_figures(rcorr, zcorr, reports, window)


# ---------- tiny backtest (signal validity only) ----------
def backtest_signal_validity(
    index_ticker: str,
    components: list[str],
    start: str,
    window: int = 21,
    lookback: int = 252 * 3,
    freq: str = "W-FRI",
):
    """
    Lightweight event-study (no P&L):
      - Build realized avg pairwise corr (window)
      - Compute past-only z-score (lookback)
      - Sample endpoints (weekly Friday default)
      - Label LOW if z<-1 (expect corr up), HIGH if z>+1 (expect corr down)
      - Check if corr moved as expected over the next <window> business days
    Outputs: reports/backtest_signal_validity.csv (rows = trade dates)
    """
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    tickers = [index_ticker] + components
    prices = get_prices(tickers, start=start)
    returns = log_returns(prices)

    rcorr = rolling_avg_pairwise_corr(returns, components, window=window)
    zcorr = past_only_z(rcorr, lookback_days=lookback)

    df = pd.DataFrame({"corr": rcorr, "z": zcorr}).dropna()
    if df.empty:
        raise RuntimeError("Not enough history to run the backtest.")

    # sample weekly endpoints
    sample = df.resample(freq).last().dropna()

    # forward correlation after ~window business days
    corr_fwd = rcorr.shift(-window)
    sample["corr_fwd"] = corr_fwd.reindex(sample.index)
    sample = sample.dropna()

    def lab(z):
        if z < -1.0:
            return "LOW"   # expect UP
        if z > 1.0:
            return "HIGH"  # expect DOWN
        return "NORMAL"

    sample["label"] = sample["z"].apply(lab)
    sample["delta"] = sample["corr_fwd"] - sample["corr"]

    trades = sample[sample["label"] != "NORMAL"].copy()
    trades["correct"] = ((trades["label"] == "LOW") & (trades["delta"] > 0)) | (
        (trades["label"] == "HIGH") & (trades["delta"] < 0)
    )

    n = len(trades)
    hit = float(trades["correct"].mean()) if n > 0 else float("nan")
    up_moves = float((trades["delta"] > 0).mean()) if n > 0 else float("nan")

    print("\nBacktest (signal validity, no P&L)")
    print(f"  Frequency: {freq}, forward horizon: {window} business days")
    print(f"  Trades: {n}")
    if n > 0:
        print(f"  Hit rate: {hit:.2%}")
        print(f"  Mean Δcorr on trades: {trades['delta'].mean():.4f}")
    else:
        print("  No trades under these thresholds.")

    out = trades[["z", "label", "corr", "corr_fwd", "delta", "correct"]].copy()
    out.index.name = "date"
    out_path = reports / "backtest_signal_validity.csv"
    out.to_csv(out_path)
    print(f"Saved: {out_path}")


# ---------- cli ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimal dispersion signal (realized corr only)")
    ap.add_argument("--index", default="SPY")
    ap.add_argument(
        "--components",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        help="component tickers (exclude the index)",
    )
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--window", type=int, default=21, help="rolling window for realized corr (bdays)")
    ap.add_argument("--lookback", type=int, default=252 * 3, help="days for past-only z-score")
    ap.add_argument("--backtest", action="store_true", help="run tiny signal-validity backtest")
    args = ap.parse_args()

    if args.backtest:
        backtest_signal_validity(args.index, args.components, args.start, args.window, args.lookback, freq="W-FRI")
    else:
        main(args.index, args.components, args.start, args.window, args.lookback)
