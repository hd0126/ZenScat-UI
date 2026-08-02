import numpy as np
import pytest

from zenscat.core import (
    Device1D,
    Grid1D,
    RayleighCutoffError,
    ScatteringMatrix,
    calc_free_space,
    calc_k,
    calc_layer,
    convmat1d,
    launch_rcwa_s,
    redheffer_star,
)


def test_convmat1d_constant_profile_matches_legacy_toeplitz_shape() -> None:
    conv = convmat1d(np.full((2, 16), 2.25), harmonic_count=1)

    expected = np.zeros((3, 3, 2), dtype=np.complex128)
    expected[:, :, 0] = np.eye(3) * 2.25
    expected[:, :, 1] = np.eye(3) * 2.25
    np.testing.assert_allclose(conv, expected, atol=1e-12)


def test_calc_k_uses_legacy_harmonic_ordering_and_outer_signs() -> None:
    grid = Grid1D(Lam0=[1.55e-6], Theta=[0.0], Lx=1.55, erR=1.0, erT=1.0)

    kx, kz_t, kz_r = calc_k(grid, lam0=1.55e-6, theta=0.0, harmonic_count=1)

    zero_guard = 1e-6 / (2 * np.pi / 1.55e-6)
    np.testing.assert_allclose(np.diag(kx), [-1.0, zero_guard, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.diag(kz_t), [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.diag(kz_r), [0.0, -1.0, 0.0], atol=1e-6)


def test_redheffer_star_identity_passthrough_is_neutral() -> None:
    size = 3
    identity = ScatteringMatrix.identity_passthrough(size)
    sample = ScatteringMatrix(
        S11=np.eye(size, dtype=np.complex128) * 0.1,
        S12=np.eye(size, dtype=np.complex128) * 0.8,
        S21=np.eye(size, dtype=np.complex128) * 0.7,
        S22=np.eye(size, dtype=np.complex128) * -0.2,
    )

    left = redheffer_star(identity, sample)
    right = redheffer_star(sample, identity)

    for field in ("S11", "S12", "S21", "S22"):
        np.testing.assert_allclose(getattr(left, field), getattr(sample, field))
        np.testing.assert_allclose(getattr(right, field), getattr(sample, field))


def test_free_space_returns_legacy_passthrough_smatrix() -> None:
    grid = Grid1D(Lam0=[1.55e-6], Theta=[0.0], Lx=3.1)
    kx, _, _ = calc_k(grid, lam0=1.55e-6, theta=0.0, harmonic_count=1)

    sg, w0, v0 = calc_free_space(kx)

    np.testing.assert_allclose(sg.S11, np.zeros((3, 3)))
    np.testing.assert_allclose(sg.S12, np.eye(3))
    np.testing.assert_allclose(sg.S21, np.eye(3))
    np.testing.assert_allclose(sg.S22, np.zeros((3, 3)))
    np.testing.assert_allclose(w0, np.eye(3))
    assert v0.shape == (3, 3)


def test_public_free_space_rejects_non_diagonal_kx() -> None:
    kx = np.array(
        [
            [0.2, 0.01, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.2],
        ],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="kx must be diagonal"):
        calc_free_space(kx)


def test_launch_rcwa_s_homogeneous_air_conserves_normal_incidence_power() -> None:
    harmonic_count = 1
    grid = Grid1D(Lam0=[1.55e-6], Theta=[0.0], Lx=0.7, erR=1.0, urR=1.0, erT=1.0, urT=1.0)
    erc = np.eye(2 * harmonic_count + 1, dtype=np.complex128)[:, :, np.newaxis]
    device = Device1D(ERC=erc, sub_L=[0.0])

    trn, ref = launch_rcwa_s(harmonic_count, grid, device, modes=("E", "H"))

    np.testing.assert_allclose(ref.REF0, [[0.0]], atol=1e-10)
    np.testing.assert_allclose(trn.TRN0, [[1.0]], atol=1e-10)
    np.testing.assert_allclose(ref.sum, [[0.0]], atol=1e-10)
    np.testing.assert_allclose(trn.sum, [[1.0]], atol=1e-10)
    np.testing.assert_allclose(trn.minus_1, [[0.0]], atol=1e-10)
    np.testing.assert_allclose(trn.plus_1, [[0.0]], atol=1e-10)


def test_launch_rcwa_s_boundary_only_matches_normal_incidence_fresnel_power() -> None:
    harmonic_count = 1
    grid = Grid1D(
        Lam0=[1.55e-6],
        Theta=[0.0],
        Lx=0.7,
        erR=1.0,
        urR=1.0,
        erT=4.0,
        urT=1.0,
        layer_num=0,
    )
    erc = np.eye(2 * harmonic_count + 1, dtype=np.complex128)[:, :, np.newaxis]
    device = Device1D(ERC=erc, sub_L=[0.0])

    trn_e, ref_e = launch_rcwa_s(harmonic_count, grid, device, modes=("E",))
    trn_h, ref_h = launch_rcwa_s(harmonic_count, grid, device, modes=("H",))

    for trn, ref in ((trn_e, ref_e), (trn_h, ref_h)):
        np.testing.assert_allclose(ref.REF0, [[1.0 / 9.0]], atol=1e-10)
        np.testing.assert_allclose(trn.TRN0, [[8.0 / 9.0]], atol=1e-10)
        np.testing.assert_allclose(ref.sum + trn.sum, [[1.0]], atol=1e-10)


def test_launch_rcwa_s_raises_domain_error_at_exact_rayleigh_cutoff() -> None:
    harmonic_count = 1
    grid = Grid1D(Lam0=[1.55e-6], Theta=[0.0], Lx=1.55, erR=1.0, erT=1.0)
    erc = np.eye(2 * harmonic_count + 1, dtype=np.complex128)[:, :, np.newaxis]
    device = Device1D(ERC=erc, sub_L=[0.0])

    with pytest.raises(RayleighCutoffError, match="Rayleigh cutoff"):
        launch_rcwa_s(harmonic_count, grid, device, modes=("E",))


def test_calc_layer_nonzero_thickness_non_diagonal_erc_is_finite() -> None:
    harmonic_count = 1
    grid = Grid1D(Lam0=[1.55e-6], Theta=[0.0], Lx=0.7)
    kx, _, _ = calc_k(grid, lam0=1.55e-6, theta=0.0, harmonic_count=harmonic_count)
    _, w0, v0 = calc_free_space(kx)
    erc = np.array(
        [
            [2.25, 0.05, 0.0],
            [0.05, 2.25, 0.05],
            [0.0, 0.05, 2.25],
        ],
        dtype=np.complex128,
    )
    device = Device1D(ERC=erc[:, :, np.newaxis], sub_L=[0.2e-6])

    layer = calc_layer(kx, lam0=1.55e-6, w0=w0, v0=v0, device=device, sub_layer=0, mode="E")

    for block in (layer.S11, layer.S12, layer.S21, layer.S22):
        assert block.shape == (3, 3)
        assert np.all(np.isfinite(block))
    np.testing.assert_allclose(layer.S12, layer.S21)
