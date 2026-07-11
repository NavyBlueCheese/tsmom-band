"""Pre-commit hook: fail if the sealed holdout directory's hash has changed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty.data.bootstrap import holdout_digest


def main() -> int:
    holdout = Path(__file__).resolve().parents[1] / "data" / "holdout"
    hash_file = holdout / "HASH"
    if not holdout.exists() or not hash_file.exists():
        return 0
    recorded = hash_file.read_text().strip()
    current = holdout_digest(holdout)
    if recorded != current:
        print("FAIL: data/holdout contents no longer match data/holdout/HASH.")
        print(f"  recorded: {recorded}")
        print(f"  current:  {current}")
        print("The holdout is sealed. Do not modify it.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
