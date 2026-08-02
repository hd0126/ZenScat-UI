"""MATLAB-compatible 1D S-matrix RCWA primitives.

The equations in this module intentionally follow the legacy ZenScat
``ZENSCAT_MAIN`` MATLAB helpers: ``convmat1D``, ``calcK``, boundary
builders, ``calcLayer``, ``star``, and ``Launch_RCWA_S``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from .control import CancellationToken, ProgressCallback, checkpoint

Mode = Literal["E", "H"]
ComplexMatrix = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Grid1D:
    """Ambient and sweep parameters for the legacy 1D RCWA solver."""

    Lam0: FloatArray
    Theta: FloatArray
    Lx: float
    erR: complex = 1.0
    urR: complex = 1.0
    erT: complex = 1.0
    urT: complex = 1.0
    erSub: complex = 1.0
    layer_num: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "Lam0", _as_1d_float(self.Lam0, "Lam0"))
        object.__setattr__(self, "Theta", _as_1d_float(self.Theta, "Theta"))
        if self.Lx <= 0:
            raise ValueError("Lx must be positive")
        if self.layer_num < 0:
            raise ValueError("layer_num must be non-negative")


@dataclass(frozen=True)
class Device1D:
    """Layer stack expressed as convolution matrices and layer thicknesses."""

    ERC: ComplexMatrix
    sub_L: FloatArray

    def __post_init__(self) -> None:
        erc = np.asarray(self.ERC, dtype=np.complex128)
        if erc.ndim == 2:
            erc = erc[:, :, np.newaxis]
        if erc.ndim != 3 or erc.shape[0] != erc.shape[1]:
            raise ValueError("ERC must have shape (N, N, layer_count)")
        sub_l = _as_1d_float(self.sub_L, "sub_L")
        if erc.shape[2] != sub_l.size:
            raise ValueError("ERC third dimension must match sub_L length")
        if np.any(sub_l < 0):
            raise ValueError("sub_L values must be non-negative")
        object.__setattr__(self, "ERC", erc)
        object.__setattr__(self, "sub_L", sub_l)


@dataclass(frozen=True)
class ScatteringMatrix:
    """Four-block S-matrix used by the Redheffer star product."""

    S11: ComplexMatrix
    S12: ComplexMatrix
    S21: ComplexMatrix
    S22: ComplexMatrix

    @classmethod
    def identity_passthrough(cls, size: int) -> ScatteringMatrix:
        zeros = np.zeros((size, size), dtype=np.complex128)
        eye = np.eye(size, dtype=np.complex128)
        return cls(zeros.copy(), eye.copy(), eye.copy(), zeros.copy())


@dataclass(frozen=True)
class DiffractionResult:
    """Legacy diffraction result field names from ``Launch_RCWA_S``."""

    minus_1: FloatArray
    plus_1: FloatArray
    TRN0: FloatArray | None = None
    REF0: FloatArray | None = None
    sum: FloatArray | None = None
    minus_2: FloatArray | None = None
    plus_2: FloatArray | None = None


class RayleighCutoffError(ValueError):
    """Raised when a launch point lands exactly on a Rayleigh cutoff."""


def convmat1d(values: ArrayLike, harmonic_count: int) -> ComplexMatrix:
    """Build the 1D Fourier convolution matrix stack.

    Parameters match MATLAB ``convmat1D(A, NH)``. Input is shaped as
    ``(z_samples, x_samples)``; a one-dimensional input is treated as one
    sampled layer. The returned array has shape ``(N, N, z_samples)`` where
    ``N = 2 * harmonic_count + 1``.
    """

    if harmonic_count < 0:
        raise ValueError("harmonic_count must be non-negative")
    sampled = np.asarray(values, dtype=np.complex128)
    if sampled.ndim == 1:
        sampled = sampled[np.newaxis, :]
    if sampled.ndim != 2:
        raise ValueError("values must be one- or two-dimensional")

    nz, nx = sampled.shape
    order_count = 2 * harmonic_count + 1
    if order_count >= 2 * nx:
        raise ValueError("total harmonic number must be less than 2 * sample count")

    coefficients = np.fft.fftshift(np.fft.fft(sampled, axis=1) / nx, axes=1)
    p_orders = np.arange(-harmonic_count, harmonic_count + 1)
    center = nx // 2
    conv = np.empty((order_count, order_count, nz), dtype=np.complex128)
    for layer in range(nz):
        for row, p_row in enumerate(p_orders):
            for col, p_col in enumerate(p_orders):
                conv[row, col, layer] = coefficients[layer, center + p_row - p_col]
    return conv


def calc_k(
    grid: Grid1D,
    lam0: float,
    theta: float,
    harmonic_count: int,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    """Calculate normalized diagonal ``Kx``, ``KzT``, and ``KzR`` matrices."""

    if lam0 <= 0:
        raise ValueError("lam0 must be positive")
    if harmonic_count < 0:
        raise ValueError("harmonic_count must be non-negative")

    n_inc = np.sqrt(grid.erR * grid.urR + 0j)
    k0 = 2 * np.pi / lam0
    kinc_x = k0 * n_inc * np.sin(theta)
    matlab_indices = np.arange(1, 2 * harmonic_count + 2)
    harmonic_orders = -(matlab_indices - harmonic_count - 1)
    k_x = kinc_x - 2 * np.pi * harmonic_orders / (grid.Lx * 1e-6)
    k_x = np.where(k_x == 0, 1e-6, k_x)
    kx = np.diag((k_x / k0).astype(np.complex128))
    eye = np.eye(kx.shape[0], dtype=np.complex128)
    kz_t = np.conj(_matrix_sqrt(grid.erT * grid.urT * eye - kx @ kx))
    kz_r = -np.conj(_matrix_sqrt(grid.erR * grid.urR * eye - kx @ kx))
    return kx, kz_t, kz_r


def calc_free_space(kx: ComplexMatrix) -> tuple[ScatteringMatrix, ComplexMatrix, ComplexMatrix]:
    """Return free-space S-matrix and modal matrices ``W0``/``V0``."""

    size = _require_diagonal_square_matrix(kx, "kx")
    eye = np.eye(size, dtype=np.complex128)
    kx2 = _diagonal_matrix_square(kx)
    kz = np.conj(_diagonal_matrix_sqrt(eye - kx2))
    omega2 = kx2 - eye
    w0 = eye
    lam = 1j * kz
    v0 = _right_solve(omega2, lam)
    return ScatteringMatrix.identity_passthrough(size), w0, v0


def calc_reflection_side(
    kx: ComplexMatrix,
    kz_r: ComplexMatrix,
    w0: ComplexMatrix,
    v0: ComplexMatrix,
    grid: Grid1D,
    mode: Mode,
) -> tuple[ComplexMatrix, ScatteringMatrix]:
    """Build the incident/reflection-side S-matrix."""

    size = _require_diagonal_square_matrix(kx, "kx")
    eye = np.eye(size, dtype=np.complex128)
    w_ref = eye
    lam = -1j * kz_r
    if mode == "E":
        q = _diagonal_matrix_square(kx) - eye * grid.erR * grid.urR
        v_ref = _right_solve(q / grid.urR, lam)
    elif mode == "H":
        q = _diagonal_matrix_square(kx) - eye * grid.urR * grid.erR
        v_ref = _right_solve(q / grid.erR, lam)
    else:
        raise ValueError("mode must be 'E' or 'H'")
    a = np.linalg.solve(w0, w_ref) + np.linalg.solve(v0, v_ref)
    b = np.linalg.solve(w0, w_ref) - np.linalg.solve(v0, v_ref)
    s_ref = ScatteringMatrix(
        S11=-np.linalg.solve(a, b),
        S12=2 * np.linalg.inv(a),
        S21=0.5 * (a - _right_solve(b, a) @ b),
        S22=_right_solve(b, a),
    )
    return w_ref, s_ref


def calc_transmission_side(
    kx: ComplexMatrix,
    kz_t: ComplexMatrix,
    w0: ComplexMatrix,
    v0: ComplexMatrix,
    grid: Grid1D,
    mode: Mode,
) -> tuple[ComplexMatrix, ScatteringMatrix]:
    """Build the transmission-side S-matrix."""

    size = _require_diagonal_square_matrix(kx, "kx")
    eye = np.eye(size, dtype=np.complex128)
    w_trn = eye
    lam = 1j * kz_t
    if mode == "E":
        q = _diagonal_matrix_square(kx) / grid.urT - eye * grid.erT
        v_trn = _right_solve(q, lam)
    elif mode == "H":
        v_trn = (w_trn @ lam) / grid.erT
    else:
        raise ValueError("mode must be 'E' or 'H'")
    a = np.linalg.solve(w0, w_trn) + np.linalg.solve(v0, v_trn)
    b = np.linalg.solve(w0, w_trn) - np.linalg.solve(v0, v_trn)
    s_trn = ScatteringMatrix(
        S11=_right_solve(b, a),
        S12=0.5 * (a - _right_solve(b, a) @ b),
        S21=2 * np.linalg.inv(a),
        S22=-np.linalg.solve(a, b),
    )
    return w_trn, s_trn


def calc_layer(
    kx: ComplexMatrix,
    lam0: float,
    w0: ComplexMatrix,
    v0: ComplexMatrix,
    device: Device1D,
    sub_layer: int,
    mode: Mode,
) -> ScatteringMatrix:
    """Calculate one device sub-layer S-matrix."""

    if lam0 <= 0:
        raise ValueError("lam0 must be positive")
    size = _square_size(kx, "kx")
    if not 0 <= sub_layer < device.sub_L.size:
        raise IndexError("sub_layer out of range")
    erc = device.ERC[:, :, sub_layer]
    if erc.shape != (size, size):
        raise ValueError("device ERC shape must match kx")

    eye = np.eye(size, dtype=np.complex128)
    if mode == "E":
        omega2 = kx @ kx - erc
        eigvals, w = np.linalg.eig(omega2)
        lam = np.diag(np.sqrt(eigvals + 0j))
        v = w @ lam
    elif mode == "H":
        omega2 = erc @ (kx @ np.linalg.solve(erc, kx) - eye)
        eigvals, w = np.linalg.eig(omega2)
        lam = np.diag(np.sqrt(eigvals + 0j))
        v = np.linalg.solve(erc, w @ lam)
    else:
        raise ValueError("mode must be 'E' or 'H'")

    x_prop = expm(-lam * (2 * np.pi / lam0) * device.sub_L[sub_layer])
    a = np.linalg.solve(w, w0) + np.linalg.solve(v, v0)
    b = np.linalg.solve(w, w0) - np.linalg.solve(v, v0)
    f = a - x_prop @ _right_solve(b, a) @ x_prop @ b
    s11 = np.linalg.solve(f, x_prop @ _right_solve(b, a) @ x_prop @ a - b)
    s12 = np.linalg.solve(f, x_prop @ (a - _right_solve(b, a) @ b))
    return ScatteringMatrix(S11=s11, S12=s12, S21=s12.copy(), S22=s11.copy())


def redheffer_star(sa: ScatteringMatrix, sb: ScatteringMatrix) -> ScatteringMatrix:
    """Redheffer star product for two S-matrices."""

    size = _square_size(sa.S12, "sa.S12")
    eye = np.eye(size, dtype=np.complex128)
    d = _right_solve(sa.S12, eye - sb.S11 @ sa.S22)
    f = _right_solve(sb.S21, eye - sa.S22 @ sb.S11)
    return ScatteringMatrix(
        S11=sa.S11 + d @ sb.S11 @ sa.S21,
        S12=d @ sb.S12,
        S21=f @ sa.S21,
        S22=sb.S22 + f @ sa.S22 @ sb.S12,
    )


def launch_rcwa_s(
    harmonic_count: int,
    grid: Grid1D,
    device: Device1D,
    modes: Iterable[Mode] | str = ("E",),
    calc_fresnel: bool = False,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Run the legacy S-matrix RCWA sweep.

    Returns ``(TRN, REF)`` with the same field names used by MATLAB
    ``Launch_RCWA_S``: ``minus_1``, ``plus_1``, ``TRN0``/``REF0``, and
    ``sum``. If both modes are requested, later modes overwrite earlier
    modes, matching the legacy loop behavior.
    """

    if harmonic_count < 1:
        raise ValueError("harmonic_count must be at least 1 for +/-1 fields")
    size = 2 * harmonic_count + 1
    if device.ERC.shape[:2] != (size, size):
        raise ValueError("device ERC shape must match harmonic_count")

    sweep_shape = (grid.Lam0.size, grid.Theta.size)
    trn_minus_1 = np.zeros(sweep_shape, dtype=np.float64)
    trn_plus_1 = np.zeros(sweep_shape, dtype=np.float64)
    trn_minus_2 = np.zeros(sweep_shape, dtype=np.float64) if harmonic_count >= 2 else None
    trn_plus_2 = np.zeros(sweep_shape, dtype=np.float64) if harmonic_count >= 2 else None
    trn0 = np.zeros(sweep_shape, dtype=np.float64)
    trn_sum = np.zeros(sweep_shape, dtype=np.float64)
    ref_minus_1 = np.zeros(sweep_shape, dtype=np.float64)
    ref_plus_1 = np.zeros(sweep_shape, dtype=np.float64)
    ref_minus_2 = np.zeros(sweep_shape, dtype=np.float64) if harmonic_count >= 2 else None
    ref_plus_2 = np.zeros(sweep_shape, dtype=np.float64) if harmonic_count >= 2 else None
    ref0 = np.zeros(sweep_shape, dtype=np.float64)
    ref_sum = np.zeros(sweep_shape, dtype=np.float64)

    mode_list = _normalize_modes(modes)
    total_points = len(mode_list) * grid.Lam0.size * grid.Theta.size
    completed_points = 0
    src = np.zeros((size, 1), dtype=np.complex128)
    src[size // 2, 0] = 1.0

    for mode in mode_list:
        if mode not in {"E", "H"}:
            raise ValueError("modes may only contain 'E' and 'H'")
        for lam_index, lam0 in enumerate(grid.Lam0):
            for theta_index, theta in enumerate(grid.Theta):
                checkpoint(cancel_token)
                kx, kz_t, kz_r = calc_k(grid, float(lam0), float(theta), harmonic_count)
                _raise_on_rayleigh_cutoff(kx, kz_t, kz_r, float(lam0), float(theta))
                sg, w0, v0 = calc_free_space(kx)
                w_ref, s_ref = calc_reflection_side(kx, kz_r, w0, v0, grid, mode)
                w_trn, s_trn = calc_transmission_side(kx, kz_t, w0, v0, grid, mode)

                for _ in range(grid.layer_num):
                    for sub_layer in range(device.sub_L.size):
                        si = calc_layer(kx, float(lam0), w0, v0, device, sub_layer, mode)
                        sg = redheffer_star(sg, si)
                sg = redheffer_star(s_ref, sg)
                sg = redheffer_star(sg, s_trn)

                csrc = np.linalg.solve(w_ref, src)
                ry = w_ref @ sg.S11 @ csrc
                r = np.abs(ry[:, 0]) ** 2
                kz_inc = np.cos(theta) * np.sqrt(grid.erR * grid.urR + 0j)
                r = np.real(np.diag(-kz_r) / kz_inc) * r

                ty = w_trn @ sg.S21 @ csrc
                t = np.abs(ty[:, 0]) ** 2
                if mode == "E":
                    t = np.real((grid.urR / grid.urT) * np.diag(kz_t) / kz_inc) * t
                else:
                    t = np.real(np.diag(kz_t) * np.sqrt(grid.erR + 0j) / (grid.erT * np.cos(theta))) * t
                if calc_fresnel:
                    t = _apply_legacy_fresnel_correction(t, grid, theta, mode, size)

                ref_minus_1[lam_index, theta_index] = r[size // 2 - 1]
                ref_plus_1[lam_index, theta_index] = r[size // 2 + 1]
                if ref_minus_2 is not None and ref_plus_2 is not None:
                    ref_minus_2[lam_index, theta_index] = r[size // 2 - 2]
                    ref_plus_2[lam_index, theta_index] = r[size // 2 + 2]
                ref0[lam_index, theta_index] = r[size // 2]
                ref_sum[lam_index, theta_index] = abs(np.sum(r))
                trn_minus_1[lam_index, theta_index] = t[size // 2 - 1]
                trn_plus_1[lam_index, theta_index] = t[size // 2 + 1]
                if trn_minus_2 is not None and trn_plus_2 is not None:
                    trn_minus_2[lam_index, theta_index] = t[size // 2 - 2]
                    trn_plus_2[lam_index, theta_index] = t[size // 2 + 2]
                trn0[lam_index, theta_index] = t[size // 2]
                trn_sum[lam_index, theta_index] = abs(np.sum(t))
                completed_points += 1
                if progress is not None:
                    progress(completed_points, total_points)

    trn = DiffractionResult(
        minus_1=trn_minus_1,
        plus_1=trn_plus_1,
        TRN0=trn0,
        sum=trn_sum,
        minus_2=trn_minus_2,
        plus_2=trn_plus_2,
    )
    ref = DiffractionResult(
        minus_1=ref_minus_1,
        plus_1=ref_plus_1,
        REF0=ref0,
        sum=ref_sum,
        minus_2=ref_minus_2,
        plus_2=ref_plus_2,
    )
    return trn, ref


def _apply_legacy_fresnel_correction(
    transmission: FloatArray,
    grid: Grid1D,
    theta: float,
    mode: Mode,
    size: int,
) -> FloatArray:
    corrected = transmission.copy()
    theta_i = np.arcsin((1 / grid.erSub) ** 2 * np.sin(theta))
    if mode == "E":
        root = np.sqrt(grid.erSub + 0j)
        r = root * np.cos(theta_i) - np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j)
        r = r / (root * np.cos(theta_i) + np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j))
    else:
        root = np.sqrt(grid.erSub + 0j)
        r = root * np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j) - np.cos(theta_i)
        r = r / (root * np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j) + np.cos(theta_i))
    corrected[size // 2] *= np.sqrt(1 - abs(r) ** 2)
    return corrected


def _normalize_modes(modes: Iterable[Mode] | str) -> tuple[Mode, ...]:
    mode_list = tuple(modes)
    for mode in mode_list:
        if mode not in {"E", "H"}:
            raise ValueError("modes may only contain 'E' and 'H'")
    return cast(tuple[Mode, ...], mode_list)


def _matrix_sqrt(matrix: ComplexMatrix) -> ComplexMatrix:
    if np.allclose(matrix, np.diag(np.diag(matrix))):
        return _diagonal_matrix_sqrt(matrix)
    eigvals, eigvecs = np.linalg.eig(matrix)
    return eigvecs @ np.diag(np.sqrt(eigvals + 0j)) @ np.linalg.inv(eigvecs)


def _raise_on_rayleigh_cutoff(
    kx: ComplexMatrix,
    kz_t: ComplexMatrix,
    kz_r: ComplexMatrix,
    lam0: float,
    theta: float,
) -> None:
    cutoff_masks = {
        "free-space": np.diag(np.eye(kx.shape[0], dtype=np.complex128) - _diagonal_matrix_square(kx)) == 0,
        "transmission": np.diag(kz_t) == 0,
        "reflection": np.diag(kz_r) == 0,
    }
    for region, mask in cutoff_masks.items():
        if np.any(mask):
            orders = np.flatnonzero(mask) - (kx.shape[0] // 2)
            order_text = ", ".join(str(int(order)) for order in orders)
            raise RayleighCutoffError(
                f"Rayleigh cutoff in {region} region at lam0={lam0}, "
                f"theta={theta}, harmonic_order(s)={order_text}"
            )


def _diagonal_matrix_square(matrix: ComplexMatrix) -> ComplexMatrix:
    return np.diag(np.diag(matrix) ** 2)


def _diagonal_matrix_sqrt(matrix: ComplexMatrix) -> ComplexMatrix:
    return np.diag(np.sqrt(np.diag(matrix) + 0j))


def _right_solve(numerator: ComplexMatrix, denominator: ComplexMatrix) -> ComplexMatrix:
    """MATLAB-style ``numerator / denominator``."""

    return np.linalg.solve(denominator.T, numerator.T).T


def _square_size(matrix: ComplexMatrix, name: str) -> int:
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return int(arr.shape[0])


def _require_diagonal_square_matrix(matrix: ComplexMatrix, name: str) -> int:
    size = _square_size(matrix, name)
    if not np.array_equal(matrix, np.diag(np.diag(matrix))):
        raise ValueError(f"{name} must be diagonal")
    return size


def _as_1d_float(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr
