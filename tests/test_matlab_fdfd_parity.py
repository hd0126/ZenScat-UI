from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from zenscat.core import (
    DEFAULT_DISPERSION_COEFFICIENTS,
    build_fdfd_device,
    build_fdfd_grid,
    dispersion_refractive_indices,
    fdfd_2d,
)
from zenscat.core.device import InterfaceParams

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_fdfd_golden.mat"


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


def _mat_number(value):
    array = np.asarray(value)
    assert array.size == 1
    return array.item()


def _mat_bool(value) -> bool:
    return bool(_mat_number(value))


def _interface_params(config) -> InterfaceParams:
    return InterfaceParams(
        trapz_w_bot=float(_mat_number(config.trapz_w_bot)),
        trapz_w_top=float(_mat_number(config.trapz_w_top)),
        supergauss_sigma=float(_mat_number(config.supergauss_sigma)),
        supergauss_m=float(_mat_number(config.supergauss_m)),
        triangle_w1=float(_mat_number(config.triangle_w1)),
        triangle_w2=float(_mat_number(config.triangle_w2)),
        triangle_w3=float(_mat_number(config.triangle_w3)),
    )


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


def _build_grid_and_device(fixture):
    config = fixture.config
    grid = build_fdfd_grid(
        config.params,
        layer_num=int(_mat_number(config.layer_num)),
        distribution=_mat_text(config.distribution),
        Lam0=np.asarray(config.lam0_um, dtype=np.float64),
        Theta=np.asarray(config.theta_deg, dtype=np.float64) * np.pi / 180,
        Lx=float(_mat_number(config.Lx_um)),
        h=float(_mat_number(config.h_um)),
        n_sup=float(_mat_number(config.n_sup)),
        n_sub=float(_mat_number(config.n_sub)),
        nres=float(_mat_number(config.NRES)),
        spacer=np.asarray(config.SPACER, dtype=np.float64),
        npml=np.asarray(config.NPML, dtype=np.int64),
        is_periodic=_mat_bool(config.is_periodic),
        period_num=int(_mat_number(config.period_num)),
        refractive_idx=_mat_bool(config.refractive_idx),
        dispersion_mode=_mat_text(config.dispersion_mode)
        if _mat_bool(config.refractive_idx)
        else "legacy_fdfd",
    )
    device = build_fdfd_device(
        config.params,
        grid,
        interface=_mat_text(config.interface),
        interface_params=_interface_params(config),
    )
    return grid, device


def test_fdfd_manifest_documents_scope_and_environment():
    manifest, fixtures = _load_fixtures()

    assert _mat_text(manifest.schema_version) == "1.0"
    assert _mat_text(manifest.generator) == "matlab_oracle/generate_fdfd_golden.m"
    assert len(_mat_text(manifest.source_commit)) == 40
    assert _mat_text(manifest.matlab_release) == "R2024b"
    assert _mat_number(manifest.fixture_count) == len(fixtures) == 7
    assert "PhC" in _mat_text(manifest.exclusions)
    assert "corrected_um" in _mat_text(manifest.dispersion_legacy_mode)

    names = {_mat_text(fixture.name) for fixture in fixtures}
    assert names == {
        "fdfd_small_sin_e_2x2",
        "fdfd_small_sin_h_2x2",
        "fdfd_small_de1_e_2x2",
        "fdfd_small_de4_e_2x2",
        "fdfd_small_tri_e_2x2",
        "fdfd_small_sin_two_periodic_e_2x2",
        "fdfd_small_sin_material_palette_e_2x2",
    }


def test_dispersion_palette_matches_matlab_legacy_and_exposes_corrected_mode():
    _, fixtures = _load_fixtures()
    fixture = {
        _mat_text(candidate.name): candidate for candidate in fixtures
    }["fdfd_small_sin_material_palette_e_2x2"]
    config = fixture.config

    np.testing.assert_allclose(
        DEFAULT_DISPERSION_COEFFICIENTS,
        np.asarray(fixture.dispersion.coefficients, dtype=np.float64),
        rtol=0,
        atol=5e-9,
    )
    legacy_n = dispersion_refractive_indices(config.lam0_um, mode="legacy_fdfd")
    corrected_n = dispersion_refractive_indices(config.lam0_um, mode="corrected_um")
    np.testing.assert_allclose(legacy_n, fixture.dispersion.legacy_n, rtol=0, atol=5e-9)
    assert np.max(np.abs(legacy_n - corrected_n)) > 1e-3
    assert _mat_text(config.dispersion_mode) == "legacy_fdfd"


def test_python_fdfd_grid_and_device_match_matlab_oracle():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        grid, device = _build_grid_and_device(fixture)

        assert grid.Nx == _mat_number(fixture.grid.Nx)
        assert grid.Ny == _mat_number(fixture.grid.Ny)
        assert grid.Nx2 == _mat_number(fixture.grid.Nx2)
        assert grid.Ny2 == _mat_number(fixture.grid.Ny2)
        assert grid.dx == _mat_number(fixture.grid.dx)
        assert grid.dy == _mat_number(fixture.grid.dy)
        assert grid.distribution == _mat_text(fixture.grid.distribution)
        np.testing.assert_array_equal(grid.Lam0, fixture.grid.Lam0)
        np.testing.assert_allclose(grid.Theta, fixture.grid.Theta, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(grid.Length, fixture.grid.Length)
        np.testing.assert_array_equal(grid.erIdx, fixture.grid.erIdx)
        np.testing.assert_array_equal(grid.X, fixture.grid.X)
        np.testing.assert_array_equal(grid.Y, fixture.grid.Y)
        np.testing.assert_allclose(grid.xa2, fixture.grid.xa2, rtol=0, atol=1e-16)
        np.testing.assert_array_equal(grid.ya2, fixture.grid.ya2)

        np.testing.assert_array_equal(device.ER2, fixture.device.ER2)
        np.testing.assert_array_equal(device.UR2, fixture.device.UR2)
        assert _sha256_numeric(device.ER2) == _mat_text(fixture.checksums.ER2_sha256)
        assert _sha256_numeric(device.UR2) == _mat_text(fixture.checksums.UR2_sha256)


def test_python_fdfd_solver_matches_matlab_fields_and_schema():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        grid, device = _build_grid_and_device(fixture)
        result = fdfd_2d(grid, device, _mat_text(fixture.config.mode))

        for field in ("TRN0", "TRN_plus1", "TRN_minus1"):
            np.testing.assert_allclose(
                result.TRN[field],
                np.asarray(getattr(fixture.TRN, field), dtype=np.float64),
                rtol=2e-11,
                atol=2e-11,
                err_msg=f"{fixture.name} TRN.{field}",
            )
        assert abs(result.TRN["sum"] - float(fixture.TRN.sum)) < 2e-11

        for field in ("REF0", "REF_plus1", "TRN_minus1"):
            np.testing.assert_allclose(
                result.REF[field],
                np.asarray(getattr(fixture.REF, field), dtype=np.float64),
                rtol=2e-11,
                atol=2e-11,
                err_msg=f"{fixture.name} REF.{field}",
            )
        assert abs(result.REF["sum"] - float(fixture.REF.sum)) < 2e-11

        np.testing.assert_allclose(
            result.f,
            np.asarray(fixture.f, dtype=np.complex128),
            rtol=2e-10,
            atol=2e-10,
            err_msg=f"{fixture.name} raw field f",
        )
        assert _sha256_numeric(fixture.f) == _mat_text(fixture.checksums.f_sha256)
