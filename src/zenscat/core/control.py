"""Shared cancellation and progress contracts for long-running solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class CancellationToken(Protocol):
    def is_set(self) -> bool: ...


ProgressCallback = Callable[[int, int], None]


class SimulationCancelled(RuntimeError):
    """Raised at a deterministic solver boundary after cancellation is requested."""


def checkpoint(cancel_token: CancellationToken | None) -> None:
    if cancel_token is not None and cancel_token.is_set():
        raise SimulationCancelled("simulation cancelled")
