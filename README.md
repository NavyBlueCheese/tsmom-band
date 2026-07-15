# tsmom-band


Long/flat time-series momentum across SPY, EFA, IEF and GLD (running on a $20
IBKR Pro cash account)

At that size, IBKR's 1%-of-trade-value commission cap binds on every order and a round
trip costs 2% of the position. The entire design is therefore organized around *not trading* the signal is a
commodity, and the controller (a wide no-trade band) is the substance of the project.

## The strategy

Preregistered in [PREREGISTRATION.md](PREREGISTRATION.md) before any backtest was run:

- **Signal:** `sign(P[t-21] / P[t-252] - 1)`, long or flat, per asset.
- **Sizing:** inverse-vol, 60d Ledoit-Wolf shrunk covariance, 15% annualised vol
  target, leverage capped at 1.0.
- **Rebalance:** last trading session of each calendar quarter, 15:45 New York.
- **No-trade band:** trade leg *i* only if `|w_target(i) - w_current(i)| > 0.20`.
- **Costs:** IBKR Pro tiered US-stock schedule, plus SEC fee and FINRA TAF on sells.

The band width, vol target, lookback and rebalance frequency are fixed by the
preregistration. They are never tuned to improve a backtest number; sweeps are logged
as trials and the defaults stay put.

[PREDICTION.md](PREDICTION.md) was committed before the account was funded: a block
bootstrap of the train-period backtest predicts one-year PnL of about **+$0.85**
(80% interval −$0.26 to +$2.37, ~1.6 leg-trades per year). The success criterion is
calibration, which includes live PnL landing inside that interval.

## Design rules

- **No lookahead by construction.** There is no vectorized backtest anywhere. All
  backtests run through the event engine: the strategy receives a read-only snapshot
  of history up to and including bar *t* and is structurally incapable of reading
  *t+1*. See [docs/ANTI_LOOKAHEAD_REVIEW.md](docs/ANTI_LOOKAHEAD_REVIEW.md).
- **The same strategy code runs in backtest and live.** Only the broker behind it
  changes.
- **`decimal.Decimal`** for anything that becomes an order quantity, price or cash
  amount. Float is for market data and research statistics only.
- **No bare market orders.** Every order is protected.
- **Sealed holdout.** Data from 2019-01-01 onward is held out; its hash is committed
  ([scripts/check_holdout_hash.py](scripts/check_holdout_hash.py)).
- Secrets come from environment variables via pydantic-settings; `.env` is gitignored.

## Layout

```
src/twenty/
  strategies/    tsmom_band.py — signal, sizing, band controller
  backtest/      event engine, fill simulation, ledger
  costs/         IBKR tiered commissions, slippage, breakeven analysis
  risk/          position sizing, limit checks, kill switch
  execution/     ib_async broker, order journal, reconciliation, scheduler
  evaluation/    metrics, deflated Sharpe, trial registry, bootstrap prediction
  data/          sources, parquet store, quality checks, holdout bootstrap
research/        band sweep (reported, never used for tuning)
scripts/         preflight checks, holdout hash verification
docs/            anti-lookahead review
```

Live safety rails: a `HALT` file stops the runner within 30 seconds, and six
kill-switch conditions (reconciliation mismatch, stale data, daily loss, drawdown,
consecutive rejections, halt file) each require a human to clear them. Operations are
documented in [RUNBOOK.md](RUNBOOK.md).

## Running it

```bash
uv sync
cp .env.example .env          # fill in; never committed

uv run pytest                 # tests (hypothesis property tests included)
uv run mypy --strict src      # must pass
uv run ruff check

uv run python -m twenty.execution.runner   # paper trading via IB Gateway (port 4002)
```

Going live additionally requires `scripts/preflight.py` fully green, `IB_PORT=4001`,
`IB_IS_PAPER=false` and `I_UNDERSTAND_THIS_IS_REAL_MONEY=yes`. The runner refuses
otherwise.

Python 3.11+, fully typed (`mypy --strict`), pydantic v2 at module boundaries, polars
for research (no pandas in `src/execution/` or `src/strategies/`), ib_async for the
broker connection.

-------------------------------
**(My paper runner is still doing its weekly cycles, and I'm still in a 30-day cooldown for fractional shares. Will publish more releases as future amendments are made)**

