from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from zenscat.core.device import InterfaceParams, build_legacy_device, build_legacy_grid

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"
VARIANT_GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_device_variant_goldens.mat"


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
    return _as_list(data["fixtures"])


def _load_variant_fixtures():
    data = loadmat(VARIANT_GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return _as_list(data["fixtures"])


def _mat_bool(value) -> bool:
    return bool(_mat_number(value))


def _layer_num(config) -> int:
    params_size = int(np.asarray(config.params).size)
    if _mat_text(config.distribution) == "all":
        return int(np.ceil(params_size / 2))
    return max(params_size - 2, 1)


def _interface_params(config) -> InterfaceParams:
    return InterfaceParams(
        smooth=_mat_bool(config.smooth),
        trapz_w_bot=float(_mat_number(config.trapz_w_bot)),
        trapz_w_top=float(_mat_number(config.trapz_w_top)),
        supergauss_sigma=float(_mat_number(config.supergauss_sigma)),
        supergauss_m=float(_mat_number(config.supergauss_m)),
        triangle_w1=float(_mat_number(config.triangle_w1)),
        triangle_w2=float(_mat_number(config.triangle_w2)),
        triangle_w3=float(_mat_number(config.triangle_w3)),
    )


def test_python_device_geometry_matches_matlab_golden_arrays_and_checksums():
    for fixture in _load_fixtures():
        config = fixture.config
        grid = build_legacy_grid(
            config.params,
            layer_num=_layer_num(config),
            distribution=_mat_text(config.distribution),
            interface=_mat_text(config.interface),
            Lx=float(_mat_number(config.Lx_um)),
            h=float(_mat_number(config.h_um)),
            Nx=int(_mat_number(config.Nx)),
            Nz=int(_mat_number(config.Nz)),
            Lam0=fixture.Lam0,
            Theta=fixture.Theta,
            n_sup=float(_mat_number(config.n_sup)),
            n_sub=float(_mat_number(config.n_sub)),
        )
        device = build_legacy_device(
            int(_mat_number(config.NH)),
            grid,
            interface=_mat_text(config.interface),
            flat_substrate=bool(_mat_number(config.flat_substrate)),
        )

        assert device.ER.shape == np.asarray(fixture.device.ER).shape
        assert device.ERC.shape == np.asarray(fixture.device.ERC).shape
        assert device.sub_L.shape == np.asarray(fixture.device.sub_L).shape
        np.testing.assert_array_equal(device.ER, fixture.device.ER)
        np.testing.assert_allclose(device.ERC, fixture.device.ERC, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(device.sub_L, fixture.device.sub_L)
        assert _sha256_numeric(device.ER) == _mat_text(fixture.device.ER_checksum_sha256)
        assert _sha256_numeric(fixture.device.ERC) == _mat_text(fixture.device.ERC_checksum_sha256)
        assert _sha256_numeric(device.sub_L) == _mat_text(fixture.device.sub_L_checksum_sha256)


def test_python_grid_medium_fields_match_matlab_solver_contract():
    for fixture in _load_fixtures():
        config = fixture.config
        grid = build_legacy_grid(
            config.params,
            layer_num=_layer_num(config),
            distribution=_mat_text(config.distribution),
            interface=_mat_text(config.interface),
            Lx=float(_mat_number(config.Lx_um)),
            h=float(_mat_number(config.h_um)),
            Nx=int(_mat_number(config.Nx)),
            Nz=int(_mat_number(config.Nz)),
            Lam0=fixture.Lam0,
            Theta=fixture.Theta,
            n_sup=float(_mat_number(config.n_sup)),
            n_sub=float(_mat_number(config.n_sub)),
        )

        np.testing.assert_array_equal(grid.Lam0, fixture.grid_solver.Lam0)
        np.testing.assert_array_equal(grid.Theta, fixture.grid_solver.Theta)
        assert grid.Lx == _mat_number(fixture.grid_solver.Lx)
        assert grid.layer_num == _mat_number(fixture.grid_solver.layer_num)
        assert grid.erR == _mat_number(fixture.grid_solver.erR)
        assert grid.urR == _mat_number(fixture.grid_solver.urR)
        assert grid.erT == _mat_number(fixture.grid_solver.erT)
        assert grid.urT == _mat_number(fixture.grid_solver.urT)
        assert grid.erSub == _mat_number(fixture.grid_solver.erSub)


def test_python_device_geometry_matches_matlab_variant_fixtures():
    expected_names = {
        "de1_all_ui",
        "de1_two_periodic_ui",
        "de4_all_ui_smooth",
        "de4_two_periodic_ui_smooth",
        "tri_all_ui",
        "tri_two_periodic_ui",
    }
    fixtures = _load_variant_fixtures()
    assert {_mat_text(fixture.name) for fixture in fixtures} == expected_names

    for fixture in fixtures:
        config = fixture.config
        grid = build_legacy_grid(
            config.params,
            layer_num=_layer_num(config),
            distribution=_mat_text(config.distribution),
            interface=_mat_text(config.interface),
            Lx=float(_mat_number(config.Lx_um)),
            h=float(_mat_number(config.h_um)),
            Nx=int(_mat_number(config.Nx)),
            Nz=int(_mat_number(config.Nz)),
            Lam0=fixture.grid_solver.Lam0,
            Theta=fixture.grid_solver.Theta,
            n_sup=float(_mat_number(config.n_sup)),
            n_sub=float(_mat_number(config.n_sub)),
            is_periodic=_mat_bool(config.is_periodic),
            period_num=int(_mat_number(config.period_num)),
        )
        device = build_legacy_device(
            int(_mat_number(config.NH)),
            grid,
            interface=_mat_text(config.interface),
            interface_params=_interface_params(config),
            flat_substrate=_mat_bool(config.flat_substrate),
        )

        assert device.ER.shape == np.asarray(fixture.device.ER).shape
        assert device.ERC.shape == np.asarray(fixture.device.ERC).shape
        assert device.sub_L.shape == np.asarray(fixture.device.sub_L).shape
        np.testing.assert_array_equal(device.ER, fixture.device.ER)
        np.testing.assert_allclose(device.ERC, fixture.device.ERC, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(device.sub_L, fixture.device.sub_L)
        assert _sha256_numeric(device.ER) == _mat_text(fixture.device.ER_checksum_sha256)
        assert _sha256_numeric(fixture.device.ERC) == _mat_text(fixture.device.ERC_checksum_sha256)
        assert _sha256_numeric(device.sub_L) == _mat_text(fixture.device.sub_L_checksum_sha256)


def test_python_grid_geometry_matches_matlab_variant_fixtures():
    for fixture in _load_variant_fixtures():
        config = fixture.config
        grid = build_legacy_grid(
            config.params,
            layer_num=_layer_num(config),
            distribution=_mat_text(config.distribution),
            interface=_mat_text(config.interface),
            Lx=float(_mat_number(config.Lx_um)),
            h=float(_mat_number(config.h_um)),
            Nx=int(_mat_number(config.Nx)),
            Nz=int(_mat_number(config.Nz)),
            Lam0=fixture.grid_solver.Lam0,
            Theta=fixture.grid_solver.Theta,
            n_sup=float(_mat_number(config.n_sup)),
            n_sub=float(_mat_number(config.n_sub)),
            is_periodic=_mat_bool(config.is_periodic),
            period_num=int(_mat_number(config.period_num)),
        )
        geometry = fixture.grid_geometry

        assert _mat_text(geometry.distribution) == grid.distribution
        assert grid.h == _mat_number(geometry.h)
        assert grid.Lx == _mat_number(geometry.Lx)
        assert grid.Lz == _mat_number(geometry.Lz)
        assert grid.Nx == _mat_number(geometry.Nx)
        assert grid.Nz == _mat_number(geometry.Nz)
        assert grid.dz == _mat_number(geometry.dz)
        assert grid.qx == _mat_number(geometry.qx)
        assert grid.qz == _mat_number(geometry.qz)
        assert grid.delta == _mat_number(geometry.delta)
        assert grid.layer_num == _mat_number(geometry.layer_num)
        np.testing.assert_array_equal(grid.Length, geometry.Length)
        np.testing.assert_array_equal(grid.erIdx, geometry.erIdx)
        np.testing.assert_allclose(grid.x, geometry.x, rtol=0, atol=1e-16)
        assert _sha256_numeric(grid.Length) == _mat_text(fixture.checksums.grid_Length_sha256)
        assert _sha256_numeric(grid.erIdx) == _mat_text(fixture.checksums.grid_erIdx_sha256)
        assert _sha256_numeric(geometry.x) == _mat_text(fixture.checksums.grid_x_sha256)
