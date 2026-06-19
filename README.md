# Dispersion Trading — Realized Correlation Signal

A from-scratch study of a sector-dispersion strategy: trade the mean reversion of
realized average pairwise correlation among the constituents of a sector ETF
(XLF financials, XLE energy), hedged against the ETF itself.

## The question

Dispersion trading bets on the gap between index volatility and the volatility of
its components — a gap driven by correlation. The standard version is an options
trade (sell index vol, buy component vol). This project asks a narrower, testable
question first:

**Does a realized-correlation z-score actually predict the direction of future
correlation — and if so, can you make money trading it with cash equities?**

## The finding

Two-part result:

1. **The signal predicts direction, but direction is mechanical and doesn't monetize.**                                                                               When the rolling average pairwise correlation is unusually
   low (z < −1) it tends to rise, and when unusually high (z > +1) it tends to fall.
   On non-overlapping monthly samples the z-score predicts the direction of the next
   ~21-day correlation move **83–92% of the time**, symmetric across both regimes and
   replicated on two independent sectors (XLF and XLE).

2. **The cash-equity implementation still loses money.** Backtested 2018–2026, both
   sectors lose (−11% XLF, −19% XLE). The strategy is roughly **breakeven before costs**
   — commissions account for essentially the entire loss. Annualized volatility is only
   ~3% because the long-ETF / short-components structure is nearly market-neutral, which
   is exactly why the captured edge is too small to pay for itself.

**Why:** the dispersion premium lives in *implied* volatility (options), not realized
correlation traded through stock. A valid signal traded with the wrong instrument
doesn't monetize. The full discussion is in the writeup PDF.

## Files

| File | What it is |
|------|------------|
| `Dispersion_Strategy.py` | Signal construction + signal-validity test (the hit-rate study) |
| `Backtest_Strategy.py`   | Pure-pandas P&L backtest (equity curve, drawdown, metrics) |
| `dispersion_writeup.pdf` | Full writeup: theory, results, limitations |
| `requirements.txt`       | Dependencies |

## Running it

```bash
pip install -r requirements.txt

# Signal validity test (does the z-score predict correlation direction?)
python Dispersion_Strategy.py --backtest --index XLF \
  --components BRK-B JPM V BAC WFC MA GS MS C AXP

# Weekly instead of monthly sampling:
python Dispersion_Strategy.py --backtest --index XLF \
  --components BRK-B JPM V BAC WFC MA GS MS C AXP --freq W-FRI

# Full P&L backtest (XLF and XLE), saves charts + CSVs to ./reports
python Backtest_Strategy.py
```

## Limitations

Small independent sample on the validity test (~25–28 monthly observations per sector);
costs modeled as a flat 10 bp/side with no slippage, market impact, or short borrow;
fills at the signal-day close; constituent lists held fixed at current membership; two
sectors over a largely bull-market period. Realized correlation is a proxy for the
tradeable (implied) quantity.

## Future work

Extend to an implied-vs-realized volatility implementation — start with the index leg
(ETF implied vol via options or a VIX-style proxy vs realized), then component-level
implied correlation if options data is available.

---

*Built as an independent research project. Not investment advice.*
