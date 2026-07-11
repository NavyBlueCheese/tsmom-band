# Preregistration

Filed before any backtest was run. Amendments require a new dated section; the original
text is never edited.

**Date filed:** 2026-07-10

## Hypothesis
Time-series momentum earns a positive risk premium across asset classes.
At $20 with a 200bp round-trip cost, a wide no-trade band converts a strategy with
negative net expected return into one with positive net expected return.

## Universe
SPY, EFA, IEF, GLD.

## Signal
sign(P[t-21] / P[t-252] - 1), long or flat.

## Sizing
Inverse-vol, 60d, Ledoit-Wolf shrunk covariance, 15% annualised vol target,
leverage capped at 1.0.

## Rebalance
Last trading session of each calendar quarter, 15:45 America/New_York.

## Trading rule
Trade leg i only if |w_target(i) - w_current(i)| > 0.20.

## Costs
IBKR Pro tiered, US stocks.
commission = max(0.01, min(0.01 * notional, max(0.35, 0.0035 * shares))).
Plus SEC fee and FINRA TAF on sells.

## Holdout
2019-01-01 onward. Sealed.

## Configurations I intend to test
9 (the BAND sweep: BAND in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}).
Any additional counts.

## Success criterion
Live PnL after one year lies within the 80% interval predicted by the backtest.
NOT: live PnL is positive.
