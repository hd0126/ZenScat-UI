from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.io import loadmat

import zenscat.workflows as workflow_module
from zenscat.core import FDFDDevice, FDFDGrid, FDFDResult, InterfaceParams
from zenscat.legacy_io import load_imported_device
from zenscat.workflows import (
    AnalyticRCWARequest,
    FDFDRequest,
    ImportedRCWARequest,
    MatrixMethod,
    Polarization,
    run_analytic_rcwa,
    run_fdfd,
    run_imported_rcwa,
)

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"
REPO_ROOT = Path(__file__).parents[1]


def _fixtures() -> dict[str, Any]:
    payload = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return {str(fixture.name): fixture for fixture in np.atleast_1d(payload["fixtures"]).ravel()}


@pytest.mark.parametrize(
    ("fixture_name", "method", "mode"),
    (
        ("small_s_e_3x3", "S", "E"),
        ("small_s_h_3x3", "S", "H"),
        ("small_t_e_3x3", "T", "E"),
        ("small_t_h_3x3", "T", "H"),
    ),
)
def test_analytic_workflow_matches_matlab_end_to_end(fixture_name: str, method: str, mode: str) -> None:
    fixture = _fixtures()[fixture_name]
    config = fixture.config
    request = AnalyticRCWARequest(
        params=np.asarray(config.params, dtype=np.float64),
        layer_num=np.asarray(config.params).size // 2,
        wavelengths_m=np.asarray(fixture.Lam0, dtype=np.float64),
        angles_rad=np.asarray(fixture.Theta, dtype=np.float64),
        harmonic_count=int(config.NH),
        matrix_method=cast(MatrixMethod, method),
        polarization=cast(Polarization, mode),
        distribution=cast(Any, str(config.distribution)),
        interface=cast(Any, str(config.interface)),
        Lx_um=float(config.Lx_um),
        h_um=float(config.h_um),
        Nx=int(config.Nx),
        Nz=int(config.Nz),
        n_superstrate=float(config.n_sup),
        n_substrate=float(config.n_sub),
        flat_substrate=bool(config.flat_substrate),
    )

    run = run_analytic_rcwa(request)

    assert run.transmission.TRN0 is not None
    assert run.reflection.REF0 is not None
    np.testing.assert_allclose(run.transmission.TRN0, fixture.TRN.TRN0, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(run.reflection.REF0, fixture.REF.REF0, rtol=2e-10, atol=2e-12)
    np.testing.assert_array_equal(run.device_er.real, fixture.device.ER)
    np.testing.assert_array_equal(run.device.sub_L, fixture.device.sub_L)
    assert run.metadata["workflow"] == "casual_rcwa"
    assert run.metadata["matrix_method"] == method


def test_imported_workflow_runs_bundled_mat_without_manual_conversion() -> None:
    imported = load_imported_device(REPO_ROOT / "Lumerical Comparison" / "RCWA_DATA.mat")
    request = ImportedRCWARequest(
        device=imported,
        wavelengths_m=[650e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        matrix_method="S",
        polarization="E",
        n_substrate=1.47,
    )

    run = run_imported_rcwa(request)

    assert run.transmission.sum is not None
    assert run.reflection.sum is not None
    energy = float(run.transmission.sum.item() + run.reflection.sum.item())
    assert energy == pytest.approx(1.0, abs=1e-9)
    assert run.metadata["workflow"] == "custom_import_rcwa"
    assert isinstance(run.metadata["source"], str)
    assert run.metadata["source"].endswith("Lumerical Comparison/RCWA_DATA.mat")


def test_workflow_result_converts_to_legacy_export_contract() -> None:
    fixture = _fixtures()["small_s_e_3x3"]
    config = fixture.config
    request = AnalyticRCWARequest(
        params=config.params,
        layer_num=2,
        wavelengths_m=fixture.Lam0,
        angles_rad=fixture.Theta,
        harmonic_count=2,
        Nx=128,
        Nz=5,
        n_substrate=float(config.n_sub),
    )
    run = run_analytic_rcwa(request)
    bundle = run.legacy_bundle(params=config.params)

    assert bundle.transmission.TRN0 is not None
    assert bundle.metadata is not None
    np.testing.assert_array_equal(bundle.wavelengths_m, run.wavelengths_m)
    np.testing.assert_array_equal(bundle.transmission.TRN0, run.transmission.TRN0)
    assert bundle.metadata["legacy_compatibility"] is True


def test_rcwa_workflow_exposes_second_orders_when_harmonics_allow_them() -> None:
    request = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        Nx=64,
        Nz=3,
    )

    run = run_analytic_rcwa(request)

    assert run.transmission.minus_2 is not None
    assert run.transmission.plus_2 is not None
    assert run.reflection.minus_2 is not None
    assert run.reflection.plus_2 is not None
    assert run.transmission.minus_2.shape == (1, 1)


@pytest.mark.parametrize("fixture_index", (0, 1))
def test_fdfd_workflow_matches_matlab_end_to_end(fixture_index: int) -> None:
    payload = loadmat(
        Path(__file__).parent / "golden" / "zenscat_matlab_fdfd_golden.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    fixture = cast(Any, np.atleast_1d(payload["fixtures"]).ravel()[fixture_index])
    config = fixture.config
    request = FDFDRequest(
        params=config.params,
        layer_num=int(config.layer_num),
        wavelengths_um=np.asarray(config.lam0_um, dtype=np.float64),
        angles_rad=np.asarray(config.theta_deg, dtype=np.float64) * np.pi / 180,
        polarization=cast(Polarization, str(config.mode)),
        distribution=cast(Any, str(config.distribution)),
        interface=cast(Any, str(config.interface)),
        Lx_um=float(config.Lx_um),
        h_um=float(config.h_um),
        n_superstrate=float(config.n_sup),
        n_substrate=float(config.n_sub),
        nres=float(config.NRES),
        spacer_um=np.asarray(config.SPACER, dtype=np.float64),
        npml=np.asarray(config.NPML, dtype=np.int64),
    )

    run = run_fdfd(request)

    np.testing.assert_allclose(run.result.TRN["TRN0"], fixture.TRN.TRN0, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(run.result.REF["REF0"], fixture.REF.REF0, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(run.result.f, fixture.f, rtol=2e-10, atol=2e-10)
    assert run.metadata["legacy_final_sweep_sum"] is True


def test_fdfd_request_forwards_full_grid_and_interface_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    interface_params = InterfaceParams(trapz_w_top=0.11, trapz_w_bot=0.05)
    calls: dict[str, object] = {}

    def fake_build_grid(params, **kwargs):
        calls["params"] = np.asarray(params)
        calls["grid_kwargs"] = kwargs
        return SimpleNamespace(Lam0=np.array([0.51]), Theta=np.array([0.0]))

    def fake_build_device(params, grid, *, interface, interface_params=None):
        calls["device_params"] = np.asarray(params)
        calls["interface"] = interface
        calls["interface_params"] = interface_params
        return SimpleNamespace(ER2=np.ones((2, 2)))

    def fake_fdfd(grid, device, mode, *, cancel_token=None, progress=None):
        calls["mode"] = mode
        if progress is not None:
            progress(1, 1)
        return FDFDResult(
            TRN={"sum": 0.0, "TRN0": np.zeros((1, 1))},
            REF={"sum": 0.0, "REF0": np.zeros((1, 1))},
            f=np.zeros((1, 1), dtype=np.complex128),
        )

    monkeypatch.setattr(workflow_module, "build_fdfd_grid", fake_build_grid)
    monkeypatch.setattr(workflow_module, "build_fdfd_device", fake_build_device)
    monkeypatch.setattr(workflow_module, "fdfd_2d", fake_fdfd)
    progress_events: list[tuple[int, int]] = []
    request = FDFDRequest(
        params=[0.1, 0.2, 1.5, 1.7],
        layer_num=2,
        wavelengths_um=[0.51],
        angles_rad=[0.0],
        polarization="H",
        distribution="two",
        interface="DE1",
        interface_params=interface_params,
        Lx_um=0.32,
        h_um=0.154,
        n_superstrate=1.0,
        n_substrate=1.516,
        nres=12,
        spacer_um=[1.0, 1.2],
        npml=[3, 4],
        is_periodic=True,
        period_num=7,
        refractive_idx=True,
        dispersion_mode="corrected_um",
    )

    run = run_fdfd(request, progress=lambda done, total: progress_events.append((done, total)))

    kwargs = cast(dict[str, Any], calls["grid_kwargs"])
    assert kwargs["distribution"] == "two"
    assert kwargs["period_num"] == 7
    assert kwargs["refractive_idx"] is True
    assert kwargs["dispersion_mode"] == "corrected_um"
    assert calls["interface"] == "DE1"
    assert calls["interface_params"] is interface_params
    assert calls["mode"] == "H"
    assert progress_events == [(1, 1)]
    assert run.metadata["dispersion_mode"] == "corrected_um"
    assert run.metadata["refractive_idx"] is True


def test_fdfd_run_builds_export_bundle() -> None:
    result = FDFDResult(
        TRN={"sum": 0.0, "TRN0": np.zeros((1, 1))},
        REF={"sum": 0.0, "REF0": np.zeros((1, 1))},
        f=np.zeros((1, 1), dtype=np.complex128),
    )
    run = workflow_module.FDFDRun(
        wavelengths_um=np.array([0.51]),
        angles_rad=np.array([0.0]),
        result=result,
        grid=cast(FDFDGrid, SimpleNamespace()),
        device=cast(FDFDDevice, SimpleNamespace(ER2=np.ones((2, 2)))),
        elapsed_s=0.01,
        metadata={"workflow": "fdfd_fields"},
    )

    bundle = run.fdfd_bundle(params=[0.1, 1.5])

    assert bundle.metadata == {"workflow": "fdfd_fields"}
    np.testing.assert_array_equal(bundle.er2, np.ones((2, 2)))
    np.testing.assert_array_equal(bundle.params, np.array([0.1, 1.5]))
