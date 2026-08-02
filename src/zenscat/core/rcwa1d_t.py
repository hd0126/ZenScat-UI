"""MATLAB-compatible legacy 1D T-matrix RCWA launcher."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .control import CancellationToken, ProgressCallback, checkpoint
from .rcwa1d import Device1D, DiffractionResult, Grid1D, Mode, _right_solve

ComplexMatrix = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def launch_rcwa_t(
    harmonic_count: int,
    grid: Grid1D,
    device: Device1D,
    mode: Mode = "E",
    calc_fresnel: bool = False,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Run the legacy ``Launch_RCWA_T`` sweep.

    Returns ``(TRN, REF)`` with the MATLAB public fields
    ``minus_1``, ``plus_1``, ``TRN0``/``REF0``, and ``sum``.
    """

    if harmonic_count < 1:
        raise ValueError("harmonic_count must be at least 1 for +/-1 fields")
    if mode not in {"E", "H"}:
        raise ValueError("mode must be 'E' or 'H'")
    size = 2 * harmonic_count + 1
    if device.ERC.shape[:2] != (size, size):
        raise ValueError("device ERC shape must match harmonic_count")
    layer_count = device.sub_L.size
    if layer_count < 2:
        raise ValueError("T-matrix launch requires at least two sub-layers")

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

    eye = np.eye(size, dtype=np.complex128)
    n_inc = np.sqrt(grid.erR * grid.urR + 0j)
    matlab_orders = -(np.arange(1, size + 1) - harmonic_count - 1).astype(np.float64)
    total_points = grid.Lam0.size * grid.Theta.size
    completed_points = 0

    for lam_index, lam0 in enumerate(grid.Lam0):
        k0 = 2 * np.pi / float(lam0)
        for theta_index, theta in enumerate(grid.Theta):
            checkpoint(cancel_token)
            kx = k0 * n_inc * np.sin(theta) - 2 * np.pi * matlab_orders / (grid.Lx * 1e-6)
            kx = np.where(kx == 0, 1e-12, kx)
            kx_norm = (kx / k0).astype(np.complex128)
            kx_matrix = np.diag(kx_norm)
            kz_r = np.conj(k0 * np.sqrt(n_inc**2 - kx_norm**2 + 0j))
            kz_t = np.conj(k0 * np.sqrt(grid.erT * grid.urT - kx_norm**2 + 0j))

            if mode == "E":
                kz_r_matrix = np.diag((kz_r / k0).astype(np.complex128))
                kz_t_matrix = np.diag((kz_t / k0).astype(np.complex128))
            else:
                kz_r_matrix = np.diag((kz_r / k0 / (n_inc**2)).astype(np.complex128))
                kz_t_matrix = np.diag((kz_t / k0 / grid.erT).astype(np.complex128))

            w, v, x_prop = _layer_modes(kx_matrix, float(lam0), device, mode)
            a = np.zeros((size, size, layer_count + 2), dtype=np.complex128)
            b = np.zeros_like(a)
            cp = np.zeros((size, layer_count + 2), dtype=np.complex128)
            cm = np.zeros_like(cp)

            fg = np.vstack((eye, 1j * kz_t_matrix))
            for sub_layer in range(layer_count - 2, 0, -1):
                left = np.hstack(
                    (
                        np.vstack((-w[:, :, sub_layer], v[:, :, sub_layer])),
                        fg,
                    )
                )
                right = np.vstack(
                    (
                        w[:, :, sub_layer] @ x_prop[:, :, sub_layer],
                        v[:, :, sub_layer] @ x_prop[:, :, sub_layer],
                    )
                )
                ab = np.linalg.solve(left, right)
                a[:, :, sub_layer] = ab[:size, :]
                b[:, :, sub_layer] = ab[size:, :]
                fg = np.vstack(
                    (
                        w[:, :, sub_layer] @ (eye + x_prop[:, :, sub_layer] @ a[:, :, sub_layer]),
                        v[:, :, sub_layer] @ (eye - x_prop[:, :, sub_layer] @ a[:, :, sub_layer]),
                    )
                )

            mr = np.hstack((np.vstack((-eye, 1j * kz_r_matrix)), fg))
            source = np.zeros((2 * size, 1), dtype=np.complex128)
            source[harmonic_count, 0] = 1.0
            if mode == "E":
                source[size + harmonic_count, 0] = 1j * n_inc * np.cos(theta)
            else:
                source[size + harmonic_count, 0] = 1j / n_inc * np.cos(theta)

            rt2 = np.linalg.solve(mr, source)
            reflected = rt2[:size, 0]
            cp[:, 1] = rt2[size:, 0]
            cm[:, 1] = a[:, :, 1] @ cp[:, 1]
            for sub_layer in range(2, layer_count):
                cp[:, sub_layer] = b[:, :, sub_layer - 1] @ cp[:, sub_layer - 1]
                cm[:, sub_layer] = a[:, :, sub_layer] @ cp[:, sub_layer]
            transmitted = b[:, :, layer_count - 2] @ cp[:, layer_count - 2]

            kz_inc = np.cos(theta) * np.sqrt(grid.erR * grid.urR + 0j)
            r = np.abs(reflected) ** 2
            r = np.real(np.diag(kz_r_matrix) / kz_inc) * r

            t = np.abs(transmitted) ** 2
            t = _scale_transmission(t, kz_t_matrix, kz_inc, grid, float(theta), mode, calc_fresnel, size)

            center = size // 2
            ref_minus_1[lam_index, theta_index] = r[center - 1]
            ref_plus_1[lam_index, theta_index] = r[center + 1]
            if ref_minus_2 is not None and ref_plus_2 is not None:
                ref_minus_2[lam_index, theta_index] = r[center - 2]
                ref_plus_2[lam_index, theta_index] = r[center + 2]
            ref0[lam_index, theta_index] = r[center]
            ref_sum[lam_index, theta_index] = abs(np.sum(r))
            trn_minus_1[lam_index, theta_index] = t[center - 1]
            trn_plus_1[lam_index, theta_index] = t[center + 1]
            if trn_minus_2 is not None and trn_plus_2 is not None:
                trn_minus_2[lam_index, theta_index] = t[center - 2]
                trn_plus_2[lam_index, theta_index] = t[center + 2]
            trn0[lam_index, theta_index] = t[center]
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


def _layer_modes(
    kx: ComplexMatrix,
    lam0: float,
    device: Device1D,
    mode: Mode,
) -> tuple[ComplexMatrix, ComplexMatrix, ComplexMatrix]:
    size = kx.shape[0]
    layer_count = device.sub_L.size
    eye = np.eye(size, dtype=np.complex128)
    w = np.zeros((size, size, layer_count), dtype=np.complex128)
    v = np.zeros_like(w)
    x_prop = np.zeros_like(w)

    for sub_layer in range(layer_count):
        erc = device.ERC[:, :, sub_layer]
        if mode == "E":
            omega2 = kx @ kx - erc
            eigvals, eigvecs = np.linalg.eig(omega2)
            lam = np.diag(np.sqrt(eigvals + 0j))
            w[:, :, sub_layer] = eigvecs
            v[:, :, sub_layer] = eigvecs @ lam
        else:
            omega2 = erc @ (_right_solve(kx, erc) @ kx - eye)
            eigvals, eigvecs = np.linalg.eig(omega2)
            lam = np.diag(np.sqrt(eigvals + 0j))
            w[:, :, sub_layer] = eigvecs
            v[:, :, sub_layer] = np.linalg.solve(erc, eigvecs) @ lam
        x_prop[:, :, sub_layer] = expm(-lam * (2 * np.pi / lam0) * device.sub_L[sub_layer])
    return w, v, x_prop


def _scale_transmission(
    transmission: FloatArray,
    kz_t: ComplexMatrix,
    kz_inc: complex,
    grid: Grid1D,
    theta: float,
    mode: Mode,
    calc_fresnel: bool,
    size: int,
) -> FloatArray:
    if calc_fresnel:
        if mode == "E":
            scaled = np.real((grid.urR / grid.urT) * np.diag(kz_t) / kz_inc) * transmission
            return _apply_t_fresnel_correction(scaled, grid, theta, mode, size)
        scaled = np.real(np.diag(kz_t) * np.sqrt(grid.erR + 0j) / (grid.erT * np.cos(theta))) * transmission
        return _apply_t_fresnel_correction(scaled, grid, theta, mode, size)
    if mode == "E":
        return np.real((grid.urR / grid.urT) * np.diag(kz_t) / kz_inc) * transmission
    return np.real(np.diag(kz_t) * np.sqrt(grid.erR * grid.urR + 0j) / np.cos(theta)) * transmission


def _apply_t_fresnel_correction(
    transmission: FloatArray,
    grid: Grid1D,
    theta: float,
    mode: Mode,
    size: int,
) -> FloatArray:
    corrected = transmission.copy()
    theta_i = np.arcsin((1 / grid.erSub) ** 2 * np.sin(theta))
    root = np.sqrt(grid.erSub + 0j)
    if mode == "E":
        r = root * np.cos(theta_i) - np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j)
        r = r / (root * np.cos(theta_i) + np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j))
    else:
        r = root * np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j) - np.cos(theta_i)
        r = r / (root * np.sqrt(1 - grid.erSub * np.sin(theta_i) ** 2 + 0j) + np.cos(theta_i))
    corrected[size // 2] *= np.sqrt(1 - abs(r) ** 2)
    return corrected


__all__ = ["launch_rcwa_t"]
