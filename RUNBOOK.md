# Runbook

## Start

Paper (default; IB Gateway on port 4002, `IB_IS_PAPER=true` in `.env`):

    uv run python -m twenty.execution.runner              # quarterly
    uv run python -m twenty.execution.runner --force-weekly   # paper testing only

Live (only after `scripts/preflight.py` is fully green and PREDICTION.md is
committed): set `IB_PORT=4001`, `IB_IS_PAPER=false`, and
`I_UNDERSTAND_THIS_IS_REAL_MONEY=yes`, then run the same command. It refuses
otherwise.

## Stop it right now

    touch HALT

That is the whole procedure. It requires no other subsystem to be working:
the runner checks for the file every 30 seconds and every kill-switch pass
checks it first. Delete the file only after you know why you created it.

## The six halt reasons

1. **HALT_FILE** — a human asked for a stop. Do whatever you stopped for,
   then delete the file.
2. **RECONCILIATION_MISMATCH** — local books disagree with the broker by more
   than $0.02. Nothing automatic. Read `journal.sqlite`, read `ib.trades()`
   in a Python shell, find the missing or duplicated fill, fix positions by
   hand in Client Portal if needed, then delete HALT.
3. **STALE_MARKET_DATA** — no tick for 300s during regular hours. Usually the
   Gateway lost its data farm connection. Restart Gateway, confirm quotes,
   delete HALT.
4. **DAILY_LOSS** — down more than 10% today. With this strategy that is not
   a normal day; suspect a fat order or bad data before suspecting the market.
5. **DRAWDOWN** — more than 35% below the high-water mark. The preregistered
   risk budget is spent. The system stays down; the decision to re-fund is a
   human one, made away from the screen.
6. **CONSECUTIVE_REJECTIONS** — three rejections in a row. Read the raw IBKR
   error codes in the log (they are always logged). Common causes: fractional
   permission missing, cash account settlement, market closed.

## Quarter-end triggers, KST (you are in Seoul; the runner is in UTC)

Computed from the XNYS calendar. The wall-clock hour shifts when US DST
changes (November and March) — the runner uses zoneinfo, so only your
expectations need adjusting, never the code.

| Session (NY) | Trigger in KST |
|---|---|
| 2026-09-30 | 2026-10-01 04:45 |
| 2026-12-31 | 2027-01-01 05:45 |
| 2027-03-31 | 2027-04-01 04:45 |
| 2027-06-30 | 2027-07-01 04:45 |
| 2027-09-30 | 2027-10-01 04:45 |
| 2027-12-31 | 2028-01-01 05:45 |
| 2028-03-31 | 2028-04-01 04:45 |
| 2028-06-30 | 2028-07-01 04:45 |

Note the hour difference between summer (04:45) and winter (05:45) rows:
that is US DST, and it is why no UTC offset is ever hardcoded.

## If reconciliation breaks

Nothing automatic — by design. Read the journal
(`sqlite3 journal.sqlite "select * from orders order by updated_at"`), read
`ib.trades()` and `ib.positions()` from a Python shell against the Gateway,
work out which side is right, fix by hand in Client Portal, then and only
then delete HALT.
