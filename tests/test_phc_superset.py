from __future__ import annotations

import numpy as np
import pytest

from zenscat.core import (
    PhCInterfaceParams,
    build_fdfd_grid,
    build_phc_device,
    build_phc_fdfd_device,
    build_phc_grid,
    launch_rcwa_s_phc,
)


def _grid():
    return build_phc_grid(
        [0.32, 1.5, 2.0],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_honeycomb",
        Nx=32,
        Nz=10,
    )


def test_phc_honeycomb_uses_three_corrected_unit_slices_and_metadata() -> None:
    grid = _grid()
    device = build_phc_device(1, grid, "PhC_honeycomb", PhCInterfaceParams(radius_star_ellipse=0.24))

    assert device.ER.shape == (3 * grid.Nz, grid.Nx)
    assert device.sub_L.shape == (3 * grid.Nz,)
    assert device.is_top is True
    assert device.is_bot is True
    assert device.ER_top is not None
    assert device.ER_bot is not None
    np.testing.assert_allclose(np.unique(device.ER), [1.5**2, 2.0**2])
    assert device.geometry is not None
    assert device.geometry["interface"] == "PhC_honeycomb"
    assert device.geometry["compatibility"] == "corrected"
    assert "legacy Device_3.m honeycomb branch" in str(device.geometry["semantics"])


def test_phc_rotated_hex_columns_changes_orientation_without_changing_range() -> None:
    normal_grid = build_phc_grid(
        [0.32, 1.5, 2.0],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_hex_columns",
        Nx=36,
        Nz=12,
    )
    rotated_grid = build_phc_grid(
        [0.32, 1.5, 2.0],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_hex_columns_rot",
        Nx=36,
        Nz=12,
    )

    params = PhCInterfaceParams(radius_star_ellipse=0.28)
    normal = build_phc_device(1, normal_grid, "PhC_hex_columns", params)
    rotated = build_phc_device(1, rotated_grid, "PhC_hex_columns_rot", params)

    assert normal.ER.shape == rotated.ER.shape
    np.testing.assert_allclose(np.unique(rotated.ER), [1.5**2, 2.0**2])
    assert not np.array_equal(normal.ER, rotated.ER)
    assert rotated.geometry is not None
    assert rotated.geometry["orientation"] == "rotated_hex_lattice"


def test_phc_generic_hex_polygon_rotation_is_explicit_and_symmetric() -> None:
    grid = build_phc_grid(
        [0.32, 1.5, 2.0],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_hex_polygon",
        Nx=41,
        Nz=41,
    )

    unrotated = build_phc_device(
        1,
        grid,
        "PhC_hex_polygon",
        PhCInterfaceParams(radius_star_ellipse=0.42, hex_rot_angle=0.0),
    )
    rotated = build_phc_device(
        1,
        grid,
        "PhC_hex_polygon",
        PhCInterfaceParams(radius_star_ellipse=0.42, hex_rot_angle=30.0),
    )

    assert unrotated.ER.shape == (grid.Nz, grid.Nx)
    np.testing.assert_array_equal(unrotated.ER, np.flipud(unrotated.ER))
    np.testing.assert_array_equal(unrotated.ER, np.fliplr(unrotated.ER))
    assert not np.array_equal(unrotated.ER, rotated.ER)
    assert rotated.geometry is not None
    assert rotated.geometry["semantics"] == "corrected regular hexagonal polygon inclusion"


def test_phc_fdfd_device_builds_corrected_er2_with_metadata_and_rotation_effect() -> None:
    grid = build_fdfd_grid(
        [0.08, 0.08, 1.5, 2.0],
        layer_num=2,
        Lam0=[0.52],
        Theta=[0.0],
        Lx=0.32,
        h=0.16,
        nres=8,
        spacer=[0.08, 0.08],
        npml=[2, 2],
    )
    params = [0.32, 1.5, 2.0]
    common = PhCInterfaceParams(radius_star_ellipse=0.32)

    normal = build_phc_fdfd_device(params, grid, "PhC_hex_columns", common)
    rotated = build_phc_fdfd_device(params, grid, "PhC_hex_columns_rot", common)

    assert normal.ER2.shape == (grid.Nx2, grid.Ny2)
    assert normal.UR2.shape == normal.ER2.shape
    assert normal.metadata["compatibility"] == "corrected"
    assert normal.metadata["legacy_exact"] is False
    assert normal.metadata["interface"] == "PhC_hex_columns"
    assert np.min(normal.ER2) >= min(grid.erSup, grid.erSub, 1.5**2)
    assert np.max(normal.ER2) <= max(grid.erSup, grid.erSub, 2.0**2)
    assert not np.array_equal(normal.ER2, rotated.ER2)


def test_phc_fdfd_rejects_legacy_exact_claim() -> None:
    grid = build_fdfd_grid(
        [0.08, 0.08, 1.5, 2.0],
        layer_num=2,
        Lam0=[0.52],
        Theta=[0.0],
        Lx=0.32,
        h=0.16,
        nres=8,
        spacer=[0.08, 0.08],
        npml=[2, 2],
    )
    with pytest.raises(NotImplementedError, match="legacy Device_FDFD_PhC.m is incomplete"):
        build_phc_fdfd_device(
            [0.32, 1.5, 2.0],
            grid,
            "PhC_rec_square",
            compatibility="legacy_exact",
        )


def test_phc_honeycomb_corrected_rcwa_path_runs_finite_lossless_point() -> None:
    grid = build_phc_grid(
        [0.32, 1.5, 2.0],
        wavelengths_m=[520e-9],
        angles_rad=[0.0],
        interface="PhC_honeycomb",
        Nx=16,
        Nz=6,
    )
    device = build_phc_device(
        1,
        grid,
        "PhC_honeycomb",
        PhCInterfaceParams(radius_star_ellipse=0.2),
    )

    trn, ref = launch_rcwa_s_phc(1, 1, grid, device, "E", repeat_mode="corrected")

    assert trn.TRN0.shape == (1, 1)
    assert ref.REF0.shape == (1, 1)
    assert np.isfinite(trn.TRN0).all()
    assert np.isfinite(ref.REF0).all()
    np.testing.assert_allclose(trn.TRN0 + ref.REF0, [[1.0]], atol=1e-10)
