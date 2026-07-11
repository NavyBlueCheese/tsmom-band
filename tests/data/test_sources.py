"""Rate limiter tests, clock controlled via a fake."""

from __future__ import annotations

from twenty.data.sources import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_identical_requests_spaced_two_seconds() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=60, window_s=600.0, min_identical_gap_s=2.0, clock=clock)
    assert bucket.acquire("SPY:2020") == 0.0
    slept = bucket.acquire("SPY:2020")
    assert slept == 2.0


def test_different_keys_not_spaced() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=60, window_s=600.0, min_identical_gap_s=2.0, clock=clock)
    assert bucket.acquire("SPY:2020") == 0.0
    assert bucket.acquire("EFA:2020") == 0.0


def test_sixty_requests_per_ten_minutes() -> None:
    clock = FakeClock()
    bucket = TokenBucket(capacity=60, window_s=600.0, min_identical_gap_s=2.0, clock=clock)
    for i in range(60):
        assert bucket.acquire(f"key-{i}") == 0.0
    slept = bucket.acquire("key-60")
    assert slept > 0.0
    # After sleeping, the oldest stamp has aged out of the window.
    assert clock.now - 1_000_000.0 >= 600.0
