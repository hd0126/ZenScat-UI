"""Harmonic convergence workflow for RCWA requests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import workflows
from .core import CancellationToken, ProgressCallback, SimulationCancelled
from .core.control import checkpoint
from .workflows import AnalyticRCWARequest, ImportedRCWARequest, PhCRCWARequest, RCWARun

FloatArray = NDArray[np.float64]
ConvergenceRequest = AnalyticRCWARequest | ImportedRCWARequest | PhCRCWARequest


class ConvergenceCancelled(RuntimeError):
    """Raised when a harmonic convergence sweep is cancelled."""


@dataclass(frozen=True)
class ConvergenceSample:
    harmonic_count: int
    run: RCWARun
    t0: FloatArray
    r0: FloatArray
    energy: FloatArray
    energy_error: float
    delta_t0: float | None
    delta_r0: float | None
    delta_energy: float | None


@dataclass(frozen=True)
class ConvergenceRun:
    samples: tuple[ConvergenceSample, ...]
    recommended_harmonic_count: int | None
    converged: bool
    tolerance: float | None
    metadata: dict[str, object]

    @property
    def final_run(self) -> RCWARun:
        if not self.samples:
            raise ValueError("convergence run has no samples")
        return self.samples[-1].run


def run_harmonic_convergence(
    request: ConvergenceRequest,
    *,
    min_harmonic_count: int = 1,
    max_harmonic_count: int,
    tolerance: float | None = None,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> ConvergenceRun:
    """Run an RCWA request at increasing harmonic counts and score stability.

    ``progress`` reports completed harmonic counts over the requested harmonic
    count range. Solver-level progress is intentionally collapsed so GUI callers
    get one stable progress contract regardless of sweep size.
    """

    _validate_inputs(request, min_harmonic_count, max_harmonic_count, tolerance)
    counts = tuple(range(min_harmonic_count, max_harmonic_count + 1))
    samples: list[ConvergenceSample] = []
    for index, harmonic_count in enumerate(counts, start=1):
        try:
            checkpoint(cancel_token)
            run = _run_one(replace(request, harmonic_count=harmonic_count), cancel_token=cancel_token)
            samples.append(_sample_for_run(harmonic_count, run, samples[-1] if samples else None))
            checkpoint(cancel_token)
        except SimulationCancelled as exc:
            raise ConvergenceCancelled("convergence cancelled") from exc
        if progress is not None:
            progress(index, len(counts))

    recommended = None if tolerance is None else recommend_harmonic_count(samples, tolerance=tolerance)
    return ConvergenceRun(
        samples=tuple(samples),
        recommended_harmonic_count=recommended,
        converged=recommended is not None,
        tolerance=tolerance,
        metadata={
            "workflow": "harmonic_convergence",
            "request_type": type(request).__name__,
            "min_harmonic_count": min_harmonic_count,
            "max_harmonic_count": max_harmonic_count,
            "sample_count": len(samples),
        },
    )


def recommend_harmonic_count(samples: list[ConvergenceSample] | tuple[ConvergenceSample, ...], *, tolerance: float) -> int | None:
    """Return the first harmonic count whose zero-order metrics are stable."""

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive value")
    for sample in samples:
        if sample.delta_t0 is None or sample.delta_r0 is None or sample.delta_energy is None:
            continue
        if (
            sample.delta_t0 <= tolerance
            and sample.delta_r0 <= tolerance
            and sample.delta_energy <= tolerance
            and sample.energy_error <= tolerance
        ):
            return sample.harmonic_count
    return None


def _validate_inputs(
    request: ConvergenceRequest,
    min_harmonic_count: int,
    max_harmonic_count: int,
    tolerance: float | None,
) -> None:
    if not isinstance(request, (AnalyticRCWARequest, ImportedRCWARequest, PhCRCWARequest)):
        raise TypeError("request must be an AnalyticRCWARequest, ImportedRCWARequest, or PhCRCWARequest")
    if min_harmonic_count < 1:
        raise ValueError("min_harmonic_count must be at least one")
    if max_harmonic_count < 1:
        raise ValueError("max_harmonic_count must be at least one")
    if min_harmonic_count > max_harmonic_count:
        raise ValueError("min_harmonic_count must be less than or equal to max_harmonic_count")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be a finite positive value")


def _run_one(request: ConvergenceRequest, *, cancel_token: CancellationToken | None) -> RCWARun:
    if isinstance(request, AnalyticRCWARequest):
        return workflows.run_analytic_rcwa(request, cancel_token=cancel_token)
    if isinstance(request, ImportedRCWARequest):
        return workflows.run_imported_rcwa(request, cancel_token=cancel_token)
    return workflows.run_phc_rcwa(request, cancel_token=cancel_token)


def _sample_for_run(harmonic_count: int, run: RCWARun, previous: ConvergenceSample | None) -> ConvergenceSample:
    t0 = _zero_order(run.transmission.TRN0, run.transmission.sum, "TRN0")
    r0 = _zero_order(run.reflection.REF0, run.reflection.sum, "REF0")
    energy = _energy(run)
    delta_t0 = None if previous is None else _max_abs_delta(t0, previous.t0)
    delta_r0 = None if previous is None else _max_abs_delta(r0, previous.r0)
    delta_energy = None if previous is None else _max_abs_delta(energy, previous.energy)
    return ConvergenceSample(
        harmonic_count=harmonic_count,
        run=run,
        t0=t0,
        r0=r0,
        energy=energy,
        energy_error=float(np.max(np.abs(energy - 1.0))),
        delta_t0=delta_t0,
        delta_r0=delta_r0,
        delta_energy=delta_energy,
    )


def _zero_order(primary: Any, fallback_sum: Any, name: str) -> FloatArray:
    source = primary if primary is not None else fallback_sum
    if source is None:
        raise ValueError(f"{name} or sum must be available for convergence scoring")
    array = np.asarray(source, dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _energy(run: RCWARun) -> FloatArray:
    if run.transmission.sum is not None and run.reflection.sum is not None:
        transmission = np.asarray(run.transmission.sum, dtype=np.float64)
        reflection = np.asarray(run.reflection.sum, dtype=np.float64)
    else:
        transmission = _zero_order(run.transmission.TRN0, None, "TRN0")
        reflection = _zero_order(run.reflection.REF0, None, "REF0")
    energy = transmission + reflection
    if np.any(~np.isfinite(energy)):
        raise ValueError("energy must contain finite values")
    return energy.copy()


def _max_abs_delta(current: FloatArray, previous: FloatArray) -> float:
    if current.shape != previous.shape:
        raise ValueError("convergence samples must have matching sweep shapes")
    return float(np.max(np.abs(current - previous)))
