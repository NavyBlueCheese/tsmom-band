"""Risk limits. Defaults are the live configuration."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_gross_exposure: float = 1.00
    max_position_weight: float = 0.60
    max_orders_per_day: int = 8
    max_notional_per_order: Decimal = Decimal("15.00")
    min_notional_per_order: Decimal = Decimal("2.00")
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.35
