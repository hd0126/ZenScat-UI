from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import zenscat.advanced_workflows as advanced_module
from zenscat.advanced_workflows import (
    PhCFDFDRequest,
    run_dual_fdfd,
    run_dual_phc_fdfd,
    run_phc_fdfd,
)
from zenscat.core import FDFDDevice, FDFDGrid, FDFDResult, PhCInterfaceParams
from zenscat.workflows import FDFDRequest, FDFDRun


def _fdfd_run(polarization: str, *, elapsed_s: float = 0.01) -> FDFDRun:
    result = FDFDResult(
        TRN={"sum": 0.0, "TRN0": np.zeros((1, 1))},
        REF={"sum": 0.0, "REF0": np.zeros((1, 1))},
        f=np.full((2, 2), 1 if polarization == "E" else 2, dtype=np.complex128),
    )
    return FDFDRun(
        wavelengths_um=np.array([0.52]),
        angles_rad=np.array([0.0]),
        result=result,
        grid=cast(FDFDGrid, SimpleNamespace()),
        device=cast(FDFDDevice, SimpleNamespace(ER2=np.full((2, 2), 3.0))),
        elapsed_s=elapsed_s,
        metadata={"workflow": "fdfd_fields", "polarization": polarization},
    )


def test_dual_fdfd_runs_e_then_h_and_remaps_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_fdfd(request, *, cancel_token=None, progress=None):
        calls.append(request.polarization)
        if progress is not None:
            progress(1, 3)
            progress(3, 3)
        return _fdfd_run(request.polarization, elapsed_s=0.25 if request.polarization == "E" else 0.5)

    monkeypatch.setattr(advanced_module, "run_fdfd", fake_run_fdfd)
    progress_events: list[tuple[int, int]] = []
    request = FDFDRequest(
        params=[0.1, 1.5],
        layer_num=1,
        wavelengths_um=[0.52],
        angles_rad=[0.0],
        polarization="H",
    )

    run = run_dual_fdfd(request, progress=lambda done, total: progress_events.append((done, total)))

    assert calls == ["E", "H"]
    assert progress_events == [(1, 6), (3, 6), (4, 6), (6, 6)]
    assert run.electric.metadata["polarization"] == "E"
    assert run.magnetic.metadata["polarization"] == "H"
    assert run.run_for("E") is run.electric
    assert run.run_for("H") is run.magnetic
    assert run.elapsed_s >= 0
    assert run.metadata["workflow"] == "dual_fdfd_fields"
    assert run.metadata["polarizations"] == ("E", "H")
    assert run.metadata["child_elapsed_s"] == {"E": 0.25, "H": 0.5}


def test_dual_fdfd_exposes_export_bundles_for_each_polarization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        advanced_module,
        "run_fdfd",
        lambda request, **_kwargs: _fdfd_run(request.polarization),
    )

    run = run_dual_fdfd(
        FDFDRequest(
            params=[0.1, 1.5],
            layer_num=1,
            wavelengths_um=[0.52],
            angles_rad=[0.0],
        )
    )

    bundles = run.fdfd_bundles(params=[0.1, 1.5])

    assert set(bundles) == {"E", "H"}
    np.testing.assert_array_equal(bundles["E"].result.f, np.ones((2, 2)))
    np.testing.assert_array_equal(bundles["H"].result.f, np.full((2, 2), 2.0))
    np.testing.assert_array_equal(run.fdfd_bundle("E").er2, np.full((2, 2), 3.0))


def test_phc_fdfd_rasterizes_corrected_device_and_runs_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_fdfd(grid, device, mode, *, cancel_token=None, progress=None):
        calls["grid"] = grid
        calls["device"] = device
        calls["mode"] = mode
        if progress is not None:
            progress(1, 1)
        return FDFDResult(
            TRN={"sum": 0.4, "TRN0": np.array([[0.4]])},
            REF={"sum": 0.2, "REF0": np.array([[0.2]])},
            f=np.ones((grid.Nx, grid.Ny), dtype=np.complex128),
        )

    monkeypatch.setattr(advanced_module, "fdfd_2d", fake_fdfd)
    progress_events: list[tuple[int, int]] = []

    run = run_phc_fdfd(
        PhCFDFDRequest(
            params=[0.32, 1.5, 2.0],
            wavelengths_um=[0.52],
            angles_rad=[0.0],
            polarization="H",
            interface="PhC_honeycomb",
            interface_params=PhCInterfaceParams(radius_star_ellipse=0.24),
            h_um=0.16,
            nres=6,
            spacer_um=[0.08, 0.08],
            npml=[1, 1],
        ),
        progress=lambda done, total: progress_events.append((done, total)),
    )

    device = cast(FDFDDevice, calls["device"])
    assert calls["mode"] == "H"
    assert progress_events == [(1, 1)]
    assert device.ER2.shape == (run.grid.Nx2, run.grid.Ny2)
    assert device.UR2.shape == device.ER2.shape
    assert run.metadata["workflow"] == "phc_fdfd_fields"
    assert run.metadata["polarization"] == "H"
    assert run.metadata["phc"]["interface"] == "PhC_honeycomb"
    assert run.metadata["phc"]["compatibility"] == "corrected"
    np.testing.assert_array_equal(run.fdfd_bundle().er2, device.ER2)


def test_phc_fdfd_rejects_legacy_exact_compatibility() -> None:
    request = PhCFDFDRequest(
        params=[0.32, 1.5, 2.0],
        wavelengths_um=[0.52],
        angles_rad=[0.0],
        compatibility="legacy_exact",
    )

    with pytest.raises(NotImplementedError, match="legacy Device_FDFD_PhC.m is incomplete"):
        run_phc_fdfd(request)


def test_dual_phc_fdfd_runs_e_then_h_and_exposes_bundles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_phc_fdfd(request, *, cancel_token=None, progress=None):
        calls.append(request.polarization)
        if progress is not None:
            progress(1, 2)
            progress(2, 2)
        result = FDFDResult(
            TRN={"sum": 0.0, "TRN0": np.zeros((1, 1))},
            REF={"sum": 0.0, "REF0": np.zeros((1, 1))},
            f=np.full((2, 2), 5 if request.polarization == "E" else 7, dtype=np.complex128),
        )
        return advanced_module.PhCFDFDRun(
            wavelengths_um=np.array([0.52]),
            angles_rad=np.array([0.0]),
            result=result,
            grid=cast(FDFDGrid, SimpleNamespace()),
            device=cast(FDFDDevice, SimpleNamespace(ER2=np.full((2, 2), 9.0))),
            elapsed_s=0.1 if request.polarization == "E" else 0.2,
            metadata={"workflow": "phc_fdfd_fields", "polarization": request.polarization},
        )

    monkeypatch.setattr(advanced_module, "run_phc_fdfd", fake_run_phc_fdfd)
    progress_events: list[tuple[int, int]] = []

    run = run_dual_phc_fdfd(
        PhCFDFDRequest(
            params=[0.32, 1.5, 2.0],
            wavelengths_um=[0.52],
            angles_rad=[0.0],
            polarization="H",
        ),
        progress=lambda done, total: progress_events.append((done, total)),
    )

    bundles = run.fdfd_bundles(params=[0.32, 1.5, 2.0])
    assert calls == ["E", "H"]
    assert progress_events == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert run.metadata["workflow"] == "dual_phc_fdfd_fields"
    assert run.metadata["polarizations"] == ("E", "H")
    assert run.metadata["child_elapsed_s"] == {"E": 0.1, "H": 0.2}
    assert run.run_for("E") is run.electric
    assert run.run_for("H") is run.magnetic
    np.testing.assert_array_equal(bundles["E"].result.f, np.full((2, 2), 5.0))
    np.testing.assert_array_equal(bundles["H"].result.f, np.full((2, 2), 7.0))


def test_dual_phc_fdfd_preserves_legacy_exact_rejection() -> None:
    request = PhCFDFDRequest(
        params=[0.32, 1.5, 2.0],
        wavelengths_um=[0.52],
        angles_rad=[0.0],
        compatibility="legacy_exact",
    )

    with pytest.raises(NotImplementedError, match="legacy Device_FDFD_PhC.m is incomplete"):
        run_dual_phc_fdfd(request)
