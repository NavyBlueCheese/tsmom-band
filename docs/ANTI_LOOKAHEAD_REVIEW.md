# Skeptical review: where is the lookahead?

Each check answers with the line that prevents it, per the engine-review
protocol. Line numbers as of the initial commit.

**Does Snapshot hold any object (view, index, lazy frame, closure) whose
lifetime extends past ts?**
No. `Catalog.as_of` (src/twenty/data/store.py:170-180) slices with
`searchsorted` and calls `.copy()` on every array before `SymbolHistory` is
constructed; the snapshot dict is wrapped in `MappingProxyType` and every
array has `writeable=False` (`_ro`, store.py:118-121). Nothing in the
Snapshot references the catalog's full arrays, and there are no lambdas or
lazy frames anywhere in the class.

**Does any rolling statistic use a centered window?**
No rolling-window helpers are used at all. The only windows are trailing
slices: `prices[-(VOL_WINDOW + 1):]` (src/twenty/strategies/tsmom_band.py,
`target_weights`) and the momentum indices `P[-1-SKIP]` / `P[-1-SKIP-LOOKBACK+21]`
(`momentum`). Both end at the snapshot's last bar.

**Is the covariance estimated on data ending at t, or at t+1?**
At t. The returns matrix is built from `snapshot.adjusted_close`, which by
construction contains nothing after t (store.py:170-180); LedoitWolf sees
its last `VOL_WINDOW` rows.

**Are dividends applied on ex-date or pay-date, and does the backtest know
the difference?**
Ex-date, deliberately, and fills at the ex-date open do not receive the
dividend: corporate actions are applied to positions held coming into the
session, before that session's fills (src/twenty/backtest/engine.py, loop
step 1 vs step 2 and the module docstring). Known simplification: cash is
credited at ex-date rather than pay-date, so the backtest sees dividend cash
two to four weeks early. At quarterly rebalancing with ~2% ETF yields this
is immaterial, but it is a simplification and not an oversight.

**Does the fill model read the open of bar t+1 to decide whether an order
placed at t would have been marketable?**
No. `_execute` (src/twenty/backtest/fills.py) fills every surviving order
unconditionally at the fill session's reference price plus half the spread;
its docstring states the rule. The t+1 price sets the fill price; it never
gates *whether* the fill occurs — that gate would be the limit-order
lookahead.

**Does anything call .drop_nulls() in a way that shifts alignment between
two series?**
`drop_nulls`, `dropna` and friends appear nowhere in src/ (grep clean).
Per-symbol arrays are sliced independently by timestamp, never joined and
null-dropped.

**Is `initial_capital` ever read by the strategy?**
No. It lives only in the engine (engine.py:70-98,181-185) for the accounting
invariant. The strategy receives a `PortfolioView` whose `cash` is current
cash — legitimate portfolio state — and the `Strategy` protocol
(src/twenty/backtest/types.py) offers no path to the engine object.

**Found and fixed during construction:** the future-poisoning test caught a
real bug before any results existed > polars datetimes are microsecond-unit
by default and the catalog compared them against nanosecond timestamps, so
every `as_of` silently returned the full history. Fixed by explicit
`.dt.timestamp("ns")` (store.py:141-143). This is why the anti-lookahead
suite runs before the engine is trusted.
