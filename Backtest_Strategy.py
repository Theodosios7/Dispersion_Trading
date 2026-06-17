"""
Dispersion Strategy Backtest

Signal:  past-only z-score of rolling avg pairwise realized correlation
─────────────────────────────────────────────────────────────────────────
  LOW  (z < −entry_threshold) →  SHORT dispersion
       long ETF  +  short each constituent (beta-adjusted notional)

  HIGH (z > +entry_threshold) →  LONG dispersion
       short ETF  +  long each constituent (beta-adjusted notional)

  Exit when |z| < exit_threshold  OR  full-trade stop-loss fires
─────────────────────────────────────────────────────────────────────────

Key design choices (vs the old Backtrader version):
  1. Trade-level stop-loss (not per-leg): the hedge stays intact until we
     decide the whole trade is wrong. Per-leg stops break the structure.
  2. Broker fills at today's close on the day the signal fires (slightly
     optimistic; fine for a signal that changes ~4-6 times per year).
  3. Commissions deducted from cash on every fill (both sides).
  4. run_sweep() lets you grid-search thresholds without re-downloading.
"""

import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

from Dispersion_Strategy import (
    get_prices, log_returns,
    rolling_avg_pairwise_corr, past_only_z,
)


# ── Sector constituents ───────────────────────────────────────────────────────
XLF_COMPONENTS = ["BRK-B", "JPM", "V", "BAC", "WFC", "MA", "GS", "MS", "C", "AXP"]
XLE_COMPONENTS = ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "WMB", "OKE"]


# ── Signal helpers ────────────────────────────────────────────────────────────

def _compute_signal_series(prices, components, window, lookback):
    """
    Precompute the full (corr, z) series.
    past_only_z uses shift(1) internally — no lookahead.
    """
    rets  = log_returns(prices)
    rcorr = rolling_avg_pairwise_corr(rets, components, window=window)
    zcorr = past_only_z(rcorr, lookback_days=lookback)
    return pd.DataFrame({"corr": rcorr, "z": zcorr}).dropna()


def _classify(z, prev, entry, exit_band):
    """
    Hysteresis regime label.
    Once in a regime we stay there until |z| crosses exit_band inward.
    This prevents whipsaw when z hovers just above/below the threshold.
    """
    if np.isnan(z):         return "NEUTRAL"
    if z < -entry:          return "LOW"
    if z >  entry:          return "HIGH"
    if abs(z) <= exit_band: return "NEUTRAL"
    return prev


def _rolling_betas(returns, components, index_ticker, window):
    """
    Rolling OLS beta of each constituent vs the sector ETF.
    β = cov(R_i, R_idx) / var(R_idx), clipped to [0.2, 5.0].
    Without beta-adjustment, high-beta stocks would dominate the
    component leg and introduce unintended directional bias.
    """
    idx = returns[index_ticker]
    return pd.DataFrame({
        c: (returns[c].rolling(window, min_periods=20).cov(idx) /
            idx.rolling(window, min_periods=20).var()
            ).clip(0.2, 5.0).fillna(1.0)
        for c in components
    })


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    index_ticker:    str   = "XLF",
    components:      list  = None,
    start:           str   = "2018-01-01",
    window:          int   = 21,
    lookback:        int   = 252 * 3,
    cash:            float = 100_000.0,
    allocation:      float = 0.8,
    entry_threshold: float = 1.2,
    exit_threshold:  float = 0.5,
    beta_window:     int   = 60,
    stop_loss_pct:   float = 0.10,
    commission_bps:  float = 10.0,
    reports_dir:     str   = "reports",
    _prices:         Optional[pd.DataFrame] = None,   # pass cached prices for sweeps
    verbose:         bool  = True,
) -> dict:
    """
    Simulate the dispersion strategy and return a results dict.

    Stop-loss is TRADE-LEVEL: the entire position is closed if total
    portfolio value falls more than stop_loss_pct below where it was
    when we entered the trade. This preserves the hedge structure.

    Returns
    -------
    dict with keys: equity, benchmark, drawdown, trades, metrics, signals
    """
    if components is None:
        components = XLF_COMPONENTS

    comm_rate = commission_bps / 1e4
    rep = Path(reports_dir)
    rep.mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    all_tickers = [index_ticker] + components
    prices = _prices if _prices is not None else get_prices(all_tickers, start=start).dropna()
    rets   = log_returns(prices)

    if verbose:
        print(f"\n{'='*62}")
        print(f"  DISPERSION BACKTEST: {index_ticker}")
        print(f"  {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} days)")
        print(f"  Entry ±{entry_threshold}σ | Exit ±{exit_threshold}σ | "
              f"SL {stop_loss_pct:.0%} (trade) | Comm {commission_bps:.0f}bp/side")
        print(f"{'='*62}")

    # ── 2. Pre-compute signals & betas ────────────────────────────────────────
    sig   = _compute_signal_series(prices, components, window, lookback)
    betas = _rolling_betas(rets, components, index_ticker, beta_window)

    if verbose:
        print(f"  Signal from {sig.index[0].date()} ({len(sig)} days)\n")

    # ── 3. Simulation state ───────────────────────────────────────────────────
    cash_now          = cash
    positions         = {}       # ticker → {size, entry_price, entry_date}
    eq_curve          = {}       # date  → portfolio value
    trade_log         = []       # one row per closed leg
    prev_sig          = "NEUTRAL"
    total_comm        = 0.0
    trade_entry_value = None     # portfolio value when current trade was opened
    trade_entry_date  = None
    trade_entry_sig   = None

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def mtm(date):
        return cash_now + sum(
            p["size"] * float(prices.loc[date, t])
            for t, p in positions.items()
        )

    def _pay_comm(notional):
        nonlocal cash_now, total_comm
        c = abs(notional) * comm_rate
        cash_now -= c; total_comm += c

    def open_leg(ticker, date, notional, direction):
        nonlocal cash_now
        px = float(prices.loc[date, ticker])
        if not (np.isfinite(px) and px > 0): return
        size = int(abs(notional) / px) * direction
        if size == 0: return
        cash_now -= size * px
        _pay_comm(size * px)
        positions[ticker] = {"size": size, "entry_price": px, "entry_date": date}

    def close_leg(ticker, date, reason, signal_at_entry):
        nonlocal cash_now
        if ticker not in positions: return
        p  = positions.pop(ticker)
        px = float(prices.loc[date, ticker])
        cash_now += p["size"] * px
        _pay_comm(p["size"] * px)
        trade_log.append({
            "ticker":       ticker,
            "signal":       signal_at_entry,
            "direction":    "long" if p["size"] > 0 else "short",
            "entry_date":   p["entry_date"],
            "exit_date":    date,
            "days_held":    (pd.Timestamp(date) - pd.Timestamp(p["entry_date"])).days,
            "entry_price":  p["entry_price"],
            "exit_price":   px,
            "size":         p["size"],
            "pnl_gross":    p["size"] * (px - p["entry_price"]),
            "exit_reason":  reason,
        })

    def close_all(date, reason):
        for t in list(positions):
            close_leg(t, date, reason, trade_entry_sig)

    # ── 4. Day loop ───────────────────────────────────────────────────────────
    sim_dates = prices.index[prices.index >= sig.index[0]]

    for date in sim_dates:

        # ── Trade-level stop-loss ─────────────────────────────────────────────
        # Check the TOTAL portfolio P&L since we entered this trade.
        # Only fires when we have open positions.
        if positions and trade_entry_value is not None:
            current_pv = mtm(date)
            trade_pnl_pct = (current_pv - trade_entry_value) / trade_entry_value
            if trade_pnl_pct < -stop_loss_pct:
                close_all(date, "stop_loss")
                prev_sig          = "NEUTRAL"   # re-qualify before re-entering
                trade_entry_value = None
                eq_curve[date] = mtm(date)
                continue

        # ── Signal ───────────────────────────────────────────────────────────
        if date not in sig.index:
            eq_curve[date] = mtm(date); continue

        z      = float(sig.loc[date, "z"])
        signal = _classify(z, prev_sig, entry_threshold, exit_threshold)

        # ── Execute on regime change ─────────────────────────────────────────
        if signal != prev_sig:
            close_all(date, "signal_exit")
            pv = mtm(date)

            b_row = (betas.reindex([date]).iloc[0]
                     if date in betas.index
                     else pd.Series(1.0, index=components))
            per_comp = (pv * allocation) / len(components)

            if signal == "LOW":
                # Short dispersion: buy ETF, short beta-adjusted components
                open_leg(index_ticker, date, pv * allocation, +1)
                for c in components:
                    b = max(float(b_row.get(c, 1.0)), 0.2)
                    open_leg(c, date, per_comp / b, -1)

            elif signal == "HIGH":
                # Long dispersion: short ETF, buy beta-adjusted components
                open_leg(index_ticker, date, pv * allocation, -1)
                for c in components:
                    b = max(float(b_row.get(c, 1.0)), 0.2)
                    open_leg(c, date, per_comp / b, +1)

            # Record trade entry reference point (used for trade-level SL)
            if signal in ("LOW", "HIGH"):
                trade_entry_value = mtm(date)
                trade_entry_date  = date
                trade_entry_sig   = signal
            else:
                trade_entry_value = None
                trade_entry_sig   = None

            prev_sig = signal

        eq_curve[date] = mtm(date)

    close_all(sim_dates[-1], "end_of_backtest")

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    equity = pd.Series(eq_curve, dtype=float).sort_index()
    bm_px  = prices[index_ticker].reindex(equity.index)
    bm_eq  = bm_px / bm_px.iloc[0] * cash
    dr     = equity.pct_change().dropna()

    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    tot = equity.iloc[-1] / cash - 1
    ann = (1 + tot) ** (1 / yrs) - 1
    vol = dr.std() * 252**0.5
    sh  = ann / vol if vol > 0 else float("nan")
    dd  = (equity - equity.cummax()) / equity.cummax()
    mdd = dd.min()
    cal = ann / abs(mdd) if mdd != 0 else float("nan")

    tdf = pd.DataFrame(trade_log)
    n   = len(tdf)
    bm_r = bm_px.iloc[-1] / bm_px.iloc[0] - 1

    # Direction breakdown: how did LOW vs HIGH regime trades perform?
    dir_stats = {}
    if n > 0:
        for sig_name in ("LOW", "HIGH"):
            sub = tdf[tdf["signal"] == sig_name]
            if len(sub):
                dir_stats[sig_name] = {
                    "legs":    len(sub),
                    "pnl":     sub["pnl_gross"].sum(),
                    "win_pct": (sub["pnl_gross"] > 0).mean(),
                }

    metrics = {
        "Ticker":            index_ticker,
        "Period":            f"{equity.index[0].date()} → {equity.index[-1].date()}",
        "Initial Capital":   f"${cash:,.0f}",
        "Final Value":       f"${equity.iloc[-1]:,.2f}",
        "Total Return":      f"{tot:.2%}",
        "Benchmark Return":  f"{bm_r:.2%}",
        "Annualized Return": f"{ann:.2%}",
        "Annualized Vol":    f"{vol:.2%}",
        "Sharpe Ratio":      f"{sh:.2f}",
        "Max Drawdown":      f"{mdd:.2%}",
        "Calmar Ratio":      f"{cal:.2f}",
        "Trade Legs Total":  n,
        "Win Rate (legs)":   f"{(tdf['pnl_gross']>0).mean():.1%}" if n else "N/A",
        "Avg Leg P&L":       f"${tdf['pnl_gross'].mean():,.0f}" if n else "N/A",
        "Stop-Loss Exits":   int((tdf['exit_reason']=='stop_loss').sum()) if n else 0,
        "Total Commission":  f"${total_comm:,.0f}",
        "LOW regime P&L":    f"${dir_stats.get('LOW',{}).get('pnl',0):,.0f}",
        "HIGH regime P&L":   f"${dir_stats.get('HIGH',{}).get('pnl',0):,.0f}",
    }

    if verbose:
        print()
        for k, v in metrics.items():
            print(f"  {k:<24} {str(v):>16}")
        print()

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    pd.DataFrame([metrics]).to_csv(rep / f"perf_{index_ticker}.csv", index=False)
    if n:
        tdf.to_csv(rep / f"trades_{index_ticker}.csv", index=False)
    if verbose:
        _plot_results(equity, bm_eq, dd, dr, sig, index_ticker,
                      cash, entry_threshold, rep)

    return {
        "equity":    equity,
        "benchmark": bm_eq,
        "drawdown":  dd,
        "trades":    tdf,
        "metrics":   metrics,
        "signals":   sig,
        "tot":       tot,
        "ann":       ann,
        "sh":        sh,
        "mdd":       mdd,
    }


# ── Parameter sweep ───────────────────────────────────────────────────────────

def run_sweep(
    index_ticker: str  = "XLF",
    components:   list = None,
    start:        str  = "2018-01-01",
    entry_thresholds: list = [0.8, 1.0, 1.2, 1.5],
    exit_thresholds:  list = [0.3, 0.5, 0.7],
    commission_bps:   list = [5, 10],
    stop_loss_pct:    float = 0.10,
    reports_dir:      str   = "reports",
):
    """
    Grid search over entry/exit thresholds and commission rates.
    Downloads prices ONCE, then re-runs the backtest for each combo.
    Prints a sorted summary table and saves it to CSV.
    """
    if components is None:
        components = XLF_COMPONENTS

    print(f"\nDownloading price data for {index_ticker} sweep...")
    prices = get_prices([index_ticker] + components, start=start).dropna()
    print(f"Data ready. Running {len(entry_thresholds) * len(exit_thresholds) * len(commission_bps)} combinations...\n")

    rows = []
    for entry in entry_thresholds:
        for exit_ in exit_thresholds:
            if exit_ >= entry:
                continue  # exit must be tighter than entry
            for comm in commission_bps:
                r = run_backtest(
                    index_ticker=index_ticker,
                    components=components,
                    start=start,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    commission_bps=comm,
                    stop_loss_pct=stop_loss_pct,
                    _prices=prices,
                    verbose=False,
                    reports_dir=reports_dir,
                )
                rows.append({
                    "entry_thr": entry,
                    "exit_thr":  exit_,
                    "comm_bps":  comm,
                    "total_ret": f"{r['tot']:.2%}",
                    "ann_ret":   f"{r['ann']:.2%}",
                    "sharpe":    f"{r['sh']:.2f}",
                    "max_dd":    f"{r['mdd']:.2%}",
                    # raw floats for sorting
                    "_ann": r["ann"],
                    "_sh":  r["sh"],
                })

    df = pd.DataFrame(rows).sort_values("_sh", ascending=False)

    print("═" * 72)
    print(f"{'entry':>7}  {'exit':>5}  {'comm':>5}  "
          f"{'tot_ret':>8}  {'ann_ret':>8}  {'sharpe':>7}  {'max_dd':>8}")
    print("─" * 72)
    for _, row in df.iterrows():
        print(f"{row['entry_thr']:>7.1f}  {row['exit_thr']:>5.1f}  {row['comm_bps']:>5.0f}  "
              f"{row['total_ret']:>8}  {row['ann_ret']:>8}  "
              f"{row['sharpe']:>7}  {row['max_dd']:>8}")
    print("═" * 72)

    out = df.drop(columns=["_ann", "_sh"])
    out.to_csv(Path(reports_dir) / f"sweep_{index_ticker}.csv", index=False)
    print(f"\nSaved → reports/sweep_{index_ticker}.csv")
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(equity, benchmark, drawdown, daily_rets,
                  sig_df, name, cash, entry_thr, rep):
    """4-panel: equity vs benchmark | drawdown | monthly heatmap | z-score."""

    fig = plt.figure(figsize=(14, 14))
    gs  = gridspec.GridSpec(4, 1, hspace=0.52, height_ratios=[3, 2, 2, 2])

    # 1 — Equity curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(equity.index,    equity,    color="steelblue", lw=1.6, label="Strategy")
    ax1.plot(benchmark.index, benchmark, color="gray", ls="--", lw=1.2,
             label=f"{name} Buy & Hold")
    ax1.axhline(cash, color="black", ls=":", lw=0.7, alpha=0.5, label="Initial capital")
    ax1.set_title(f"{name} Dispersion Strategy — Equity Curve",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    # 2 — Drawdown
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color="crimson", alpha=0.45)
    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("DD (%)"); ax2.grid(alpha=0.3)

    # 3 — Monthly returns heatmap
    ax3 = fig.add_subplot(gs[2])
    monthly = daily_rets.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    mdf = pd.DataFrame({
        "ret":   monthly.values,
        "year":  monthly.index.year,
        "month": monthly.index.strftime("%b"),
    }, index=monthly.index)
    pivot = mdf.pivot_table(values="ret", index="year", columns="month", aggfunc="first")
    month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    if not pivot.empty:
        flat    = pivot.values.ravel()
        finite  = flat[~np.isnan(flat)]
        abs_max = max(np.abs(finite).max(), 1e-4)
        norm    = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im = ax3.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns, fontsize=8)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=8)
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax3.text(c, r, f"{v:.1%}", ha="center", va="center",
                             fontsize=7,
                             color="white" if abs(v) > abs_max * 0.5 else "black")
        plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.01)
    ax3.set_title("Monthly Returns", fontsize=11)

    # 4 — Z-score with entry/exit bands
    ax4 = fig.add_subplot(gs[3])
    z = sig_df["z"].dropna()
    ax4.plot(z.index, z, color="navy", lw=0.9, label="Corr z-score")
    ax4.axhline( entry_thr, color="green", ls="--", lw=1.0,
                 label=f"+{entry_thr:.1f}σ  →  LONG dispersion")
    ax4.axhline(-entry_thr, color="red",   ls="--", lw=1.0,
                 label=f"−{entry_thr:.1f}σ  →  SHORT dispersion")
    ax4.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax4.fill_between(z.index, z,  entry_thr,
                     where=z >  entry_thr, color="green", alpha=0.12)
    ax4.fill_between(z.index, z, -entry_thr,
                     where=z < -entry_thr, color="red",   alpha=0.12)
    ax4.set_title("Realized Correlation Z-Score (Signal)", fontsize=11)
    ax4.set_ylabel("z-score"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    path = rep / f"backtest_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Main runs
    run_backtest(index_ticker="XLF", components=XLF_COMPONENTS)
    run_backtest(index_ticker="XLE", components=XLE_COMPONENTS)

    # Parameter sweep — uncomment to run
    # run_sweep(index_ticker="XLF", components=XLF_COMPONENTS)
    # run_sweep(index_ticker="XLE", components=XLE_COMPONENTS)