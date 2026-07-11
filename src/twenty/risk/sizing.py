"""Sizing guards. The vol-target scalar lives here, not in the strategy."""

from __future__ import annotations

from decimal import Decimal

import numpy as np

MAX_LEVERAGE = 1.0
VOL_TARGET = 0.15


def vol_target_scalar(w_norm: np.ndarray, cov_annualised: np.ndarray) -> float:
    """k = min(MAX_LEVERAGE, VOL_TARGET / portfolio sigma). The cap is
    load-bearing: a cash account cannot supply the ~150% gross a 15% vol
    target asks for in calm markets."""
    sigma_p = float(np.sqrt(w_norm @ cov_annualised @ w_norm))
    if sigma_p <= 0:
        return 0.0
    return min(MAX_LEVERAGE, VOL_TARGET / sigma_p)


def assert_no_leverage(weights: dict[str, Decimal]) -> None:
    """Raise if gross weights exceed 1. A cash account has no margin; a sum
    above 1 means the sizing pipeline is broken."""
    gross = sum(weights.values(), Decimal(0))
    if gross > Decimal("1.0") + Decimal("1e-9"):
        raise AssertionError(f"Gross exposure {gross} exceeds 1.0; sizing bug")
