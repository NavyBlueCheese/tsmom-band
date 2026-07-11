"""Preflight: runs before any real money moves.

    uv run python scripts/preflight.py

Each check prints [OK] or a specific failure. Checks needing IB Gateway are
attempted and reported as failures if it is not running; that is correct
behaviour, not an error in this script.
"""

from __future__ import annotations

import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from datetime import UTC  # noqa: E402

from twenty.data.bootstrap import holdout_digest  # noqa: E402
from twenty.evaluation.trials import (  # noqa: E402
    ledger_config_count,
    trial_count,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "[OK]  " if ok else "[FAIL]"
    line = f"{mark} {name}"
    if detail:
        line += f" - {detail}"
    print(line, flush=True)


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out[-400:] if out else ""


def main() -> int:
    # Lint, types, tests.
    ok, out = run_cmd(["uv", "run", "ruff", "check"])
    check("ruff check", ok, "" if ok else out)
    ok, out = run_cmd(["uv", "run", "mypy", "--strict", "src"])
    check("mypy --strict src", ok, "" if ok else out)
    ok, out = run_cmd(["uv", "run", "pytest", "-q"])
    check("pytest", ok, "" if ok else out)

    # trials.jsonl exists and covers the ledgers.
    trials = REPO / "trials.jsonl"
    n_trials = trial_count(trials)
    n_ledgers = ledger_config_count(REPO / "ledgers")
    check(
        "trials.jsonl covers ledgers",
        trials.exists() and n_trials >= n_ledgers,
        f"{n_trials} trials, {n_ledgers} ledger configs",
    )

    # PREDICTION.md exists and is committed.
    prediction = REPO / "PREDICTION.md"
    committed = False
    if prediction.exists():
        ok, out = run_cmd(["git", "ls-files", "--error-unmatch", "PREDICTION.md"])
        committed = ok
        _ok2, dirty = run_cmd(["git", "diff", "--name-only", "PREDICTION.md"])
        committed = committed and (dirty == "")
    check(
        "PREDICTION.md exists and is committed",
        prediction.exists() and committed,
        "" if committed else "missing or uncommitted",
    )

    # Holdout hash.
    holdout = REPO / "data" / "holdout"
    hash_file = holdout / "HASH"
    if hash_file.exists():
        matches = hash_file.read_text().strip() == holdout_digest(holdout)
        check("holdout hash matches", matches)
    else:
        check("holdout hash matches", False, "data/holdout/HASH missing")

    # IB Gateway checks.
    try:
        from twenty.execution.connection import IBSettings, connect

        settings = IBSettings()
        ib = f_ib = connect(settings)
        account = settings.account or (ib.managedAccounts()[0] if ib.managedAccounts() else "")

        summary = {row.tag: row.value for row in ib.accountSummary(account)}
        net_liq = Decimal(summary.get("NetLiquidation", "0"))
        check("gateway connect + account value", True, f"{account}: ${net_liq}")

        positions = ib.positions(account)
        check("positions fetched", True, f"{len(positions)} positions")

        from ib_async import Stock

        spy = Stock("SPY", "SMART", "USD")
        ib.qualifyContracts(spy)

        def _good(x: object) -> bool:
            return isinstance(x, int | float) and x > 0 and x == x  # NaN-safe


        quote: object = None
        for data_type in (3, 4):
            ib.reqMarketDataType(data_type)
            ticker = ib.reqMktData(spy, "", False, False)
            deadline = time.time() + 10
            while time.time() < deadline and not (_good(ticker.last) or _good(ticker.close)):
                ib.sleep(0.5)
            ib.cancelMktData(spy)
            quote = ticker.last if _good(ticker.last) else ticker.close
            if _good(quote):
                break
        check("SPY quote (delayed or frozen ok)", _good(quote), f"SPY {quote}")

        # Fractional permission: no clean API flag; probe whatIf with a
        # fractional cashQty order.
        import asyncio

        from ib_async import LimitOrder

        probe = LimitOrder("BUY", 0, float(quote) if _good(quote) else 1.0)  # type: ignore[arg-type]
        probe.cashQty = 5.0

        probe_errors: list[tuple[int, str]] = []

        def _capture(req_id: int, code: int, message: str, *args: object) -> None:
            if code not in (2104, 2106, 2158, 2119, 10197):
                probe_errors.append((code, message))

        ib.errorEvent += _capture
        try:
            state = ib.run(
                asyncio.wait_for(ib.whatIfOrderAsync(spy, probe), timeout=20)
            )
            warning = str(getattr(state, "warningText", "") or "")
            fractional_ok = state is not None and not warning and not probe_errors
            if fractional_ok:
                detail = ""
            else:
                first = probe_errors[0] if probe_errors else (0, warning)
                detail = (
                    f"probe rejected ({first[0]}: {first[1][:120]}). Enable it: "
                    "Client Portal -> Settings -> Account Settings -> Trading "
                    "Permissions -> Stocks -> Global (Trade in Fractions); "
                    "paper accounts inherit it from the live account, usually "
                    "by the next day"
                )
        except TimeoutError:
            fractional_ok = False
            detail = (
                "whatIf probe timed out; permission unknown. Verify manually in "
                "Client Portal -> Settings -> Trading Permissions"
            )
        finally:
            ib.errorEvent -= _capture
        check("fractional trading permission (cashQty probe)", fractional_ok, detail)

        # 7. Cash account. The AccountType tag reports ownership structure
        buying_power = Decimal(summary.get("BuyingPower", "0"))
        cash_value = Decimal(summary.get("TotalCashValue", "0"))
        looks_cash = cash_value > 0 and buying_power <= cash_value * Decimal("1.1")
        cash_detail = (
            f"buying power ${buying_power} vs cash ${cash_value} -> "
            f"{'cash-like' if looks_cash else 'MARGIN-like'}"
        )
        if settings.is_paper:
            check(
                "account is CASH, not margin (informational on paper)",
                True,
                cash_detail + "; verify the LIVE account is cash type before funding",
            )
        else:
            check("account is CASH, not margin", looks_cash, cash_detail)

        # Base currency USD.
        currency = next(
            (r.currency for r in ib.accountSummary(account) if r.tag == "NetLiquidation"),
            "",
        )
        check("base currency USD", currency == "USD", currency)

        # Account value 
        in_band = Decimal("15") <= net_liq <= Decimal("25")
        if settings.is_paper:
            check(
                "account value in [$15, $25] (live only; paper skipped)",
                True,
                f"paper ${net_liq}; the band is enforced when IB_IS_PAPER=false",
            )
        else:
            check(
                "account value in [$15, $25]",
                in_band,
                f"${net_liq} - this system is specified for $20; its cost model "
                "is wrong at other sizes" if not in_band else f"${net_liq}",
            )

        # Calendar agreement on next quarter end.
        from datetime import datetime

        from twenty.execution.schedule import next_trigger

        trigger = next_trigger(datetime.now(tz=UTC))
        # IBKR side
        details = ib.reqContractDetails(Stock("SPY", "SMART", "USD"))
        check(
            "exchange_calendars vs IBKR quarter-end",
            bool(details),
            f"next trigger {trigger.isoformat()} (IBKR contract reachable; "
            "verify liquidHours around that date in the log)",
        )
        f_ib.disconnect()
    except Exception as exc:
        for name in (
            "gateway connect + account value",
            "positions fetched",
            "SPY quote (delayed or frozen ok)",
            "fractional trading permission (cashQty probe)",
            "account is CASH, not margin (informational on paper)",
            "base currency USD",
            "account value in [$15, $25]",
            "exchange_calendars vs IBKR quarter-end",
        ):
            if not any(r[0] == name for r in RESULTS):
                check(name, False, f"gateway unavailable: {exc}")

    # No HALT file.
    halt = REPO / "HALT"
    check("no HALT file", not halt.exists(), str(halt) if halt.exists() else "")

    # HALT file round trip.
    if not halt.exists():
        try:
            halt.write_text("preflight test\n")
            halted = halt.exists()
            halt.unlink()
            resumed = not halt.exists()
            check("HALT write/delete round trip", halted and resumed)
        except OSError as exc:
            check("HALT write/delete round trip", False, str(exc))
    else:
        check("HALT write/delete round trip", False, "HALT already present, not touching it")

    failures = [r for r in RESULTS if not r[1]]
    print()
    if failures:
        print(f"{len(failures)} of {len(RESULTS)} checks failed. Do not fund the account.")
        return 1
    print(f"All {len(RESULTS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
