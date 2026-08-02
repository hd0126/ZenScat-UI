from __future__ import annotations

from threading import Event

import numpy as np
import pytest

import zenscat.convergence as convergence_module
from zenscat.convergence import (
    ConvergenceCancelled,
    ConvergenceSample,
    recommend_harmonic_count,
    run_harmonic_convergence,
)
from zenscat.core import DiffractionResult, SimulationCancelled
from zenscat.legacy_io import ImportedDevice
from zenscat.workflows import (
    AnalyticRCWARequest,
    ImportedRCWARequest,
    PhCRCWARequest,
    RCWARun,
)


def _fake_run(harmonic_count: int) -> RCWARun:
    value = 1.0 - 1.0 / (10.0 * harmonic_count)
    transmission = DiffractionResult(
        minus_1=np.zeros((1, 1)),
        plus_1=np.zeros((1, 1)),
        TRN0=np.array([[0.6 * value]]),
        sum=np.array([[0.6 * value]]),
    )
    reflection = DiffractionResult(
        minus_1=np.zeros((1, 1)),
        plus_1=np.zeros((1, 1)),
        REF0=np.array([[0.4 * value]]),
        sum=np.array([[0.4 * value]]),
    )
    return RCWARun(
        wavelengths_m=np.array([500e-9]),
        angles_rad=np.array([0.0]),
        transmission=transmission,
        reflection=reflection,
        grid=object(),  # type: ignore[arg-type]
        device=object(),  # type: ignore[arg-type]
        device_er=np.ones((1, 1), dtype=np.complex128),
        elapsed_s=0.01,
        metadata={"harmonic_count": harmonic_count},
    )


def test_convergence_runs_harmonic_sweep_and_recommends_count(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_analytic(request, *, cancel_token=None, progress=None):
        seen.append(request.harmonic_count)
        if progress is not None:
            progress(1, 1)
        return _fake_run(request.harmonic_count)

    monkeypatch.setattr(convergence_module.workflows, "run_analytic_rcwa", fake_analytic)
    progress_events: list[tuple[int, int]] = []
    request = AnalyticRCWARequest(
        params=[0.1, 0.2, 1.5, 1.6],
        layer_num=2,
        wavelengths_m=[500e-9],
        angles_rad=[0.0],
        harmonic_count=5,
    )

    result = run_harmonic_convergence(
        request,
        min_harmonic_count=1,
        max_harmonic_count=4,
        tolerance=0.04,
        progress=lambda done, total: progress_events.append((done, total)),
    )

    assert seen == [1, 2, 3, 4]
    assert [sample.harmonic_count for sample in result.samples] == [1, 2, 3, 4]
    assert result.recommended_harmonic_count == 3
    assert result.converged is True
    assert result.samples[0].delta_t0 is None
    assert result.samples[1].delta_t0 == pytest.approx(0.03)
    assert result.samples[-1].energy_error == pytest.approx(0.025)
    assert progress_events == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert result.metadata["workflow"] == "harmonic_convergence"
    assert result.metadata["request_type"] == "AnalyticRCWARequest"


def test_convergence_supports_phc_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_phc(request, *, cancel_token=None, progress=None):
        seen.append(request.harmonic_count)
        return _fake_run(request.harmonic_count)

    monkeypatch.setattr(convergence_module.workflows, "run_phc_rcwa", fake_phc)
    request = PhCRCWARequest(
        params=[0.32, 1.7, 1.2],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        layer_count=2,
        interface="PhC_rec_square",
        harmonic_count=2,
    )

    result = run_harmonic_convergence(request, max_harmonic_count=3)

    assert seen == [1, 2, 3]
    assert result.metadata["request_type"] == "PhCRCWARequest"


def test_convergence_supports_imported_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_imported(request, *, cancel_token=None, progress=None):
        seen.append(request.harmonic_count)
        return _fake_run(request.harmonic_count)

    monkeypatch.setattr(convergence_module.workflows, "run_imported_rcwa", fake_imported)
    request = ImportedRCWARequest(
        device=ImportedDevice(
            ER=np.ones((1, 8), dtype=np.complex128),
            sub_L_um=np.array([0.1]),
            x_um=np.linspace(0.0, 0.32, 8),
            Lx_um=0.32,
        ),
        wavelengths_m=[500e-9],
        angles_rad=[0.0],
        harmonic_count=2,
    )

    result = run_harmonic_convergence(request, max_harmonic_count=3)

    assert seen == [1, 2, 3]
    assert result.metadata["request_type"] == "ImportedRCWARequest"


def test_convergence_translates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_analytic(request, *, cancel_token=None, progress=None):
        raise SimulationCancelled("simulation cancelled")

    monkeypatch.setattr(convergence_module.workflows, "run_analytic_rcwa", fake_analytic)
    request = AnalyticRCWARequest(
        params=[0.1, 0.2, 1.5, 1.6],
        layer_num=2,
        wavelengths_m=[500e-9],
        angles_rad=[0.0],
    )

    with pytest.raises(ConvergenceCancelled, match="cancelled"):
        run_harmonic_convergence(request, max_harmonic_count=2)


def test_convergence_honors_preflight_cancellation() -> None:
    cancelled = Event()
    cancelled.set()
    request = AnalyticRCWARequest(
        params=[0.1, 0.2, 1.5, 1.6],
        layer_num=2,
        wavelengths_m=[500e-9],
        angles_rad=[0.0],
    )

    with pytest.raises(ConvergenceCancelled, match="cancelled"):
        run_harmonic_convergence(request, max_harmonic_count=2, cancel_token=cancelled)


def test_convergence_validates_bounds_and_tolerance() -> None:
    request = AnalyticRCWARequest(
        params=[0.1, 0.2, 1.5, 1.6],
        layer_num=2,
        wavelengths_m=[500e-9],
        angles_rad=[0.0],
    )

    with pytest.raises(ValueError, match="max_harmonic_count"):
        run_harmonic_convergence(request, max_harmonic_count=0)
    with pytest.raises(ValueError, match="min_harmonic_count"):
        run_harmonic_convergence(request, min_harmonic_count=4, max_harmonic_count=2)
    with pytest.raises(ValueError, match="tolerance"):
        run_harmonic_convergence(request, max_harmonic_count=2, tolerance=0.0)


def test_recommend_harmonic_count_uses_delta_and_energy_error() -> None:
    samples = [
        ConvergenceSample(
            harmonic_count=1,
            run=_fake_run(1),
            t0=np.array([[0.5]]),
            r0=np.array([[0.4]]),
            energy=np.array([[0.9]]),
            energy_error=0.1,
            delta_t0=None,
            delta_r0=None,
            delta_energy=None,
        ),
        ConvergenceSample(
            harmonic_count=2,
            run=_fake_run(2),
            t0=np.array([[0.51]]),
            r0=np.array([[0.405]]),
            energy=np.array([[0.915]]),
            energy_error=0.085,
            delta_t0=0.01,
            delta_r0=0.005,
            delta_energy=0.015,
        ),
    ]

    assert recommend_harmonic_count(samples, tolerance=0.1) == 2
    assert recommend_harmonic_count(samples, tolerance=0.001) is None
