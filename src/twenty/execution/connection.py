"""IB Gateway connection settings and safety properties."""

from __future__ import annotations

from typing import Any

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger(__name__)

GATEWAY_PAPER_PORT = 4002
GATEWAY_LIVE_PORT = 4001
TWS_PAPER_PORT = 7497
TWS_LIVE_PORT = 7496

PAPER_PORTS = frozenset({GATEWAY_PAPER_PORT, TWS_PAPER_PORT})
LIVE_PORTS = frozenset({GATEWAY_LIVE_PORT, TWS_LIVE_PORT})


class IBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IB_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = GATEWAY_PAPER_PORT
    client_id: int = 17
    account: str = ""
    is_paper: bool = True


class PortSafetyError(RuntimeError):
    pass


def assert_port_consistent(settings: IBSettings) -> None:
    """IB_PORT must agree with IB_IS_PAPER. This is a real safety property:
    the paper flag gates every other guard, and a live port behind a paper
    flag defeats all of them."""
    if settings.is_paper and settings.port not in PAPER_PORTS:
        raise PortSafetyError(
            f"IB_IS_PAPER is true but port {settings.port} is not a paper port "
            f"(paper: {sorted(PAPER_PORTS)}). Refusing to connect."
        )
    if not settings.is_paper and settings.port not in LIVE_PORTS:
        raise PortSafetyError(
            f"IB_IS_PAPER is false but port {settings.port} is not a live port "
            f"(live: {sorted(LIVE_PORTS)}). Refusing to connect."
        )


def connect(settings: IBSettings | None = None) -> Any:
    """Connect to IB Gateway, enforcing port consistency first.

    The daily Gateway logout is an expected event, not an exception: callers
    register ``on_disconnect`` handlers that reconnect, then reconcile, then
    resume — never resume before reconciling. That sequencing lives in
    runner.py; this function only builds the session.
    """
    from ib_async import IB

    settings = settings or IBSettings()
    assert_port_consistent(settings)
    ib = IB()
    ib.connect(
        settings.host, settings.port, clientId=settings.client_id, timeout=20
    )
    log.info(
        "Connected to IB",
        host=settings.host,
        port=settings.port,
        paper=settings.is_paper,
    )
    return ib
