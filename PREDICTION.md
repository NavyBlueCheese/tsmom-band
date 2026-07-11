# Prediction

Date: 2026-07-10
Git SHA: 737e1a3
Source: block bootstrap (4-quarter blocks, 10,000 draws) of the train-period backtest at BAND = 0.20, on $20.

| Quantity | Value |
|---|---|
| Expected one-year PnL | $+0.85 |
| Median one-year PnL | $+0.73 |
| Standard deviation | $0.97 |
| 10th percentile | $-0.26 |
| 90th percentile | $+2.37 |
| P(account is up after one year) | 71% |
| Expected leg-trades per year | 1.6 |

Success criterion (preregistered): live PnL after one year lies inside the 80% interval [$-0.26, $+2.37]. Not: live PnL is positive.

Commit this file before funding the account. In twelve months, compare.
