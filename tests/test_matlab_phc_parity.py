from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from scipy.io import loadmat

from zenscat.core import (
    PhCInterfaceParams,
    SimulationCancelled,
    build_phc_device,
    build_phc_grid,
    launch_rcwa_s_phc,
)
from zenscat.workflows import PhCRCWARequest, run_phc_rcwa

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_phc_golden.mat"
RESULT_FIELDS = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}


def _as_list(value):
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    return [value]


def _mat_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.ravel().astype(str)).strip()
        if value.size == 1:
            return _mat_text(value.item())
    return str(value)


def _mat_number(value) -> float:
    array = np.asarray(value)
    assert array.size == 1
    return float(array.item())


def _mat_bool(value) -> bool:
    return bool(_mat_number(value))


def _mat_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).ravel()


def _sha256_numeric(value) -> str:
    array = np.asarray(value, order="F")
    if np.iscomplexobj(array):
        payload = np.concatenate(
            [
                np.asarray(array.real, dtype="<f8", order="F").ravel(order="F"),
                np.asarray(array.imag, dtype="<f8", order="F").ravel(order="F"),
            ]
        )
    else:
        payload = np.asarray(array, dtype="<f8", order="F").ravel(order="F")
    return hashlib.sha256(payload.astype("<f8", copy=False).tobytes()).hexdigest()


def _load_fixtures():
    data = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return data["manifest"], _as_list(data["fixtures"])


def _params(config) -> PhCInterfaceParams:
    return PhCInterfaceParams(
        rec_2D_wx=_mat_number(config.rec_2D_wx),
        rec_2D_wy=_mat_number(config.rec_2D_wy),
        rec_rot_angle=_mat_number(config.rec_rot_angle),
        ax=_mat_number(config.ax),
        ay=_mat_number(config.ay),
        ellipse_rot_angle=_mat_number(config.ellipse_rot_angle),
        radius_star_ellipse=_mat_number(config.radius_star_ellipse),
    )


def _build(config):
    grid = build_phc_grid(
        config.params,
        wavelengths_m=_mat_array(config.wavelengths_m),
        angles_rad=_mat_array(config.angles_rad),
        interface=_mat_text(config.interface),
        Nx=int(_mat_number(config.Nx)),
        Nz=int(_mat_number(config.Nz)),
        n_superstrate=_mat_number(config.n_sup),
        n_substrate=_mat_number(config.n_sub),
    )
    device = build_phc_device(
        int(_mat_number(config.NH)),
        grid,
        _mat_text(config.interface),
        _params(config),
    )
    return grid, device


def test_phc_golden_manifest_contract():
    manifest, fixtures = _load_fixtures()

    assert _mat_text(manifest.schema_version) == "1.0"
    assert _mat_text(manifest.generator) == "matlab_oracle/generate_phc_golden.m"
    assert _mat_text(manifest.matlab_release) == "R2024b"
    assert _mat_number(manifest.fixture_count) == len(fixtures) == 4
    assert {_mat_text(fixture.name) for fixture in fixtures} == {
        "rectangle_e_layer3_legacy_bug",
        "ellipse_h_layer2",
        "hex_e_layer3_legacy_bug",
        "hex_h_layer2",
    }


def test_phc_device_3_arrays_match_matlab_fixtures():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        grid, device = _build(fixture.config)
        matlab_grid = fixture.grid_solver
        matlab_device = fixture.device

        np.testing.assert_array_equal(grid.Lam0, matlab_grid.Lam0)
        np.testing.assert_array_equal(grid.Theta, matlab_grid.Theta)
        np.testing.assert_array_equal(grid.Length, matlab_grid.Length)
        np.testing.assert_array_equal(grid.erIdx, matlab_grid.erIdx)
        assert grid.Lx == _mat_number(matlab_grid.Lx)
        assert grid.Nx == _mat_number(matlab_grid.Nx)
        assert grid.Nz == _mat_number(matlab_grid.Nz)
        assert grid.erR == _mat_number(matlab_grid.erR)
        assert grid.erT == _mat_number(matlab_grid.erT)

        np.testing.assert_array_equal(device.ER, matlab_device.ER)
        np.testing.assert_allclose(device.ERC, matlab_device.ERC, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(device.sub_L, matlab_device.sub_L)
        assert device.is_top == _mat_bool(matlab_device.is_top)
        assert device.is_bot == _mat_bool(matlab_device.is_bot)
        assert _sha256_numeric(device.ER) == _mat_text(fixture.checksums.device_ER_sha256)
        assert _sha256_numeric(matlab_device.ERC) == _mat_text(fixture.checksums.device_ERC_sha256)
        assert _sha256_numeric(device.sub_L) == _mat_text(fixture.checksums.device_sub_L_sha256)

        if device.is_top:
            np.testing.assert_array_equal(device.ER_top, matlab_device.ER_top)
            np.testing.assert_allclose(device.ERC_top, matlab_device.ERC_top, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(device.sub_L_top, matlab_device.sub_L_top)
            np.testing.assert_array_equal(device.ER_bot, matlab_device.ER_bot)
            np.testing.assert_allclose(device.ERC_bot, matlab_device.ERC_bot, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(device.sub_L_bot, matlab_device.sub_L_bot)


def test_launch_rcwa_s_phc_matches_matlab_legacy_results():
    _, fixtures = _load_fixtures()

    max_delta = 0.0
    for fixture in fixtures:
        config = fixture.config
        grid, device = _build(config)
        trn, ref = launch_rcwa_s_phc(
            int(_mat_number(config.layer_count)),
            int(_mat_number(config.NH)),
            grid,
            device,
            _mat_text(config.mode),
            bool(_mat_number(config.calc_fresnel)),
            repeat_mode="legacy",
        )
        for result, matlab_result, fields in (
            (trn, fixture.TRN, RESULT_FIELDS["TRN"]),
            (ref, fixture.REF, RESULT_FIELDS["REF"]),
        ):
            for field in fields:
                actual = getattr(result, field)
                expected = getattr(matlab_result, field)
                max_delta = max(max_delta, float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-12)
    assert max_delta < 3e-12


def test_phc_workflow_matches_matlab_fixtures_and_reports_metadata():
    _, fixtures = _load_fixtures()
    fixture = next(item for item in fixtures if _mat_text(item.name) == "rectangle_e_layer3_legacy_bug")
    config = fixture.config

    updates: list[tuple[int, int]] = []
    run = run_phc_rcwa(
        PhCRCWARequest(
            params=_mat_array(config.params),
            wavelengths_m=_mat_array(config.wavelengths_m),
            angles_rad=_mat_array(config.angles_rad),
            layer_count=int(_mat_number(config.layer_count)),
            interface=_mat_text(config.interface),
            interface_params=_params(config),
            harmonic_count=int(_mat_number(config.NH)),
            polarization=_mat_text(config.mode),
            Nx=int(_mat_number(config.Nx)),
            Nz=int(_mat_number(config.Nz)),
            n_superstrate=_mat_number(config.n_sup),
            n_substrate=_mat_number(config.n_sub),
            repeat_mode="legacy",
        ),
        progress=lambda done, total: updates.append((done, total)),
    )

    np.testing.assert_allclose(run.transmission.TRN0, fixture.TRN.TRN0, rtol=0, atol=3e-12)
    np.testing.assert_allclose(run.reflection.REF0, fixture.REF.REF0, rtol=0, atol=3e-12)
    assert run.metadata["workflow"] == "casual_phc_rcwa"
    assert run.metadata["repeat_mode"] == "legacy"
    assert run.metadata["legacy_non_power_of_two_repeat_bug"] is True
    assert run.metadata["legacy_angle_units"] == "radians"
    assert updates[-1] == (4, 4)


def test_phc_non_power_of_two_repeat_bug_is_named_and_correctable():
    _, fixtures = _load_fixtures()
    fixture = next(item for item in fixtures if _mat_text(item.name) == "rectangle_e_layer3_legacy_bug")
    config = fixture.config
    grid, device = _build(config)

    legacy_trn, _ = launch_rcwa_s_phc(
        3,
        int(_mat_number(config.NH)),
        grid,
        device,
        _mat_text(config.mode),
        repeat_mode="legacy",
    )
    corrected_trn, _ = launch_rcwa_s_phc(
        3,
        int(_mat_number(config.NH)),
        grid,
        device,
        _mat_text(config.mode),
        repeat_mode="corrected",
    )

    np.testing.assert_allclose(legacy_trn.TRN0, fixture.TRN.TRN0, rtol=0, atol=3e-12)
    assert np.max(np.abs(legacy_trn.TRN0 - corrected_trn.TRN0)) > 1e-4


def test_phc_rejects_zero_rectangle_widths_from_live_app_bug():
    grid = build_phc_grid(
        [0.32, 1.781, 1.2],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_rec_square",
        Nx=16,
        Nz=6,
    )
    with pytest.raises(ValueError, match="rectangle widths"):
        build_phc_device(
            1,
            grid,
            "PhC_rec_square",
            PhCInterfaceParams(rec_2D_wx=0.0, rec_2D_wy=0.25),
        )


def test_phc_workflow_honors_cancellation_before_solver_work():
    cancelled = Event()
    cancelled.set()
    with pytest.raises(SimulationCancelled):
        run_phc_rcwa(
            PhCRCWARequest(
                params=[0.32, 1.781, 1.2],
                wavelengths_m=[520e-9],
                angles_rad=[0.0],
                layer_count=2,
                interface="PhC_rec_circ",
                harmonic_count=1,
                Nx=16,
                Nz=6,
            ),
            cancel_token=cancelled,
        )
