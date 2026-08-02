"""Legacy Casual 2D/PhC RCWA compatibility helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .control import CancellationToken, ProgressCallback, checkpoint
from .rcwa1d import (
    Device1D,
    DiffractionResult,
    Grid1D,
    ScatteringMatrix,
    calc_free_space,
    calc_k,
    calc_layer,
    calc_reflection_side,
    calc_transmission_side,
    convmat1d,
    redheffer_star,
)

PhCInterface = Literal["PhC_rec_circ", "PhC_rec_square", "PhC_hex_columns"]
Mode = Literal["E", "H"]
RepeatMode = Literal["legacy", "corrected"]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PhCInterfaceParams:
    """Subset of ``Int_Params.m`` used by ``Device_3.m``."""

    rec_2D_wx: float = 0.2
    rec_2D_wy: float = 0.25
    rec_rot_angle: float = 15.0
    ax: float = 1.0
    ay: float = 0.5
    ellipse_rot_angle: float = 15.0
    radius_star_ellipse: float = 0.45


@dataclass(frozen=True)
class PhCGrid:
    """Subset of ``Grid.m`` used by ``Device_3.m`` and ``Launch_RCWA_S_PhC.m``."""

    Lam0: FloatArray
    Theta: FloatArray
    Length: FloatArray
    Lx: float
    Nx: int
    Nz: int
    erIdx: FloatArray
    erR: complex
    urR: complex
    erT: complex
    urT: complex
    erSub: complex
    layer_num: int = 1

    def solver_grid(self) -> Grid1D:
        return Grid1D(
            Lam0=self.Lam0,
            Theta=self.Theta,
            Lx=self.Length[0],
            erR=self.erR,
            urR=self.urR,
            erT=self.erT,
            urT=self.urT,
            erSub=self.erSub,
            layer_num=1,
        )


@dataclass(frozen=True)
class PhCDevice:
    """Device arrays produced by legacy ``Device_3.m``."""

    ER: FloatArray
    ERC: ComplexArray
    sub_L: FloatArray
    is_top: bool = False
    is_bot: bool = False
    ER_top: FloatArray | None = None
    ER_bot: FloatArray | None = None
    ERC_top: ComplexArray | None = None
    ERC_bot: ComplexArray | None = None
    sub_L_top: FloatArray | None = None
    sub_L_bot: FloatArray | None = None

    def unit_device(self) -> Device1D:
        return Device1D(self.ERC, self.sub_L)


def build_phc_grid(
    params: ArrayLike,
    *,
    wavelengths_m: ArrayLike,
    angles_rad: ArrayLike,
    interface: PhCInterface,
    Nx: int = 96,
    Nz: int = 12,
    n_superstrate: complex = 1.0,
    n_substrate: complex = 1.516,
) -> PhCGrid:
    """Build the legacy PhC grid contract.

    ``params`` follows the Casual PhC convention: first value is lattice
    pitch in micrometers, followed by at least two refractive indices.
    """

    values = np.asarray(params, dtype=np.float64).ravel()
    if values.size < 3:
        raise ValueError("PhC params must be [pitch_um, n_background, n_inclusion]")
    if interface not in {"PhC_rec_circ", "PhC_rec_square", "PhC_hex_columns"}:
        raise NotImplementedError("only rectangle, ellipse, and hex-column PhC interfaces are implemented")
    if values[0] <= 0:
        raise ValueError("PhC pitch must be positive")
    if Nx <= 0 or Nz <= 1:
        raise ValueError("Nx must be positive and Nz must be greater than one")
    lam0 = np.asarray(wavelengths_m, dtype=np.float64).ravel()
    theta = np.asarray(angles_rad, dtype=np.float64).ravel()
    if lam0.size == 0 or theta.size == 0:
        raise ValueError("wavelengths and angles must be non-empty")
    if np.any(lam0 <= 0) or not np.isfinite(lam0).all() or not np.isfinite(theta).all():
        raise ValueError("wavelengths must be positive and angles finite")
    return PhCGrid(
        Lam0=lam0,
        Theta=theta,
        Length=np.array([values[0]], dtype=np.float64),
        Lx=float(values[0]),
        Nx=int(Nx),
        Nz=int(Nz),
        erIdx=np.asarray(values[1:] ** 2, dtype=np.float64),
        erR=n_superstrate**2,
        urR=1.0,
        erT=n_substrate**2,
        urT=1.0,
        erSub=n_substrate**2,
        layer_num=1,
    )


def build_phc_device(
    harmonic_count: int,
    grid: PhCGrid,
    interface: PhCInterface,
    interface_params: PhCInterfaceParams,
) -> PhCDevice:
    """Direct Python port of legacy ``Device_3.m`` for mainline PhC cases."""

    if harmonic_count < 1:
        raise ValueError("harmonic_count must be at least one")
    if grid.erIdx.size < 2:
        raise ValueError("PhC device requires two material indices")
    a = float(grid.Length[0])
    radius = float(interface_params.radius_star_ellipse) * a
    if radius <= 0:
        raise ValueError("PhC inclusion radius must be positive")

    is_top = False
    is_bot = False
    er_top = er_bot = None
    sub_l_top = sub_l_bot = None

    if interface == "PhC_rec_circ":
        if interface_params.ax == 0 or interface_params.ay == 0:
            raise ValueError("ellipse axes must be non-zero")
        x = np.linspace(-a / 2, a / 2, grid.Nx)
        y = np.linspace(-a / 2, a / 2, grid.Nz)
        X, Y = np.meshgrid(x, y)
        TH, R = np.arctan2(Y, X), np.hypot(X, Y)
        X, Y = R * np.cos(TH + interface_params.ellipse_rot_angle), R * np.sin(
            TH + interface_params.ellipse_rot_angle
        )
        er_mask = (X / interface_params.ax) ** 2 + (Y / interface_params.ay) ** 2 <= radius**2
        er = _materialize(er_mask, grid)
        step_size = a / grid.Nz
    elif interface == "PhC_rec_square":
        wx = interface_params.rec_2D_wx * a
        wy = interface_params.rec_2D_wy * a
        if wx == 0 or wy == 0:
            raise ValueError("rectangle widths must be non-zero")
        x = np.linspace(-grid.Lx / 2, grid.Lx / 2, grid.Nx)
        y = np.linspace(-a / 2, a / 2, grid.Nz)
        X, Y = np.meshgrid(x, y)
        TH, R = np.arctan2(Y, X), np.hypot(X, Y)
        X, Y = R * np.cos(TH + interface_params.rec_rot_angle), R * np.sin(
            TH + interface_params.rec_rot_angle
        )
        er_mask = (np.abs(X / wx / 2) <= 0.5) * (np.abs(Y / wy / 2) <= 0.5)
        er = _materialize(er_mask, grid)
        step_size = a / grid.Nz
    elif interface == "PhC_hex_columns":
        x = np.linspace(-a / 2, a / 2, grid.Nx)
        y = np.sqrt(3) * np.linspace(-a / 2, a / 2, grid.Nz)
        b1 = np.max(x)
        b2 = np.max(y)
        X, Y = np.meshgrid(x, y)
        er_mask = X**2 + Y**2 <= radius**2
        er_mask = er_mask | ((X - b1) ** 2 + (Y + b2) ** 2 <= radius**2)
        er_mask = er_mask | ((X + b1) ** 2 + (Y - b2) ** 2 <= radius**2)
        er_mask = er_mask | ((X + b1) ** 2 + (Y + b2) ** 2 <= radius**2)
        er_mask = er_mask | ((X - b1) ** 2 + (Y - b2) ** 2 <= radius**2)
        y_top = np.linspace(0, a / 2, int(np.floor(grid.Nz / 2)))
        X_top, Y_top = np.meshgrid(x, y_top)
        er_top_mask = (X_top - a / 2) ** 2 + (Y_top - a / 2) ** 2 <= radius**2
        er_top_mask = er_top_mask | ((X_top + a / 2) ** 2 + (Y_top - a / 2) ** 2 <= radius**2)
        er_bot_mask = (X_top - a / 2) ** 2 + Y_top**2 <= radius**2
        er_bot_mask = er_bot_mask | ((X_top + a / 2) ** 2 + Y_top**2 <= radius**2)
        er = _materialize(er_mask, grid)
        er_top = _materialize(er_top_mask, grid)
        er_bot = _materialize(er_bot_mask, grid)
        top_step = y_top[1] - y_top[0]
        sub_l_top = np.ones(y_top.size, dtype=np.float64) * top_step * 1e-6
        sub_l_bot = sub_l_top.copy()
        step_size = a * np.sqrt(3) / grid.Nz
        is_top = True
        is_bot = True
    else:
        raise NotImplementedError(f"unsupported PhC interface: {interface}")

    sub_l = np.ones(grid.Nz, dtype=np.float64) * step_size * 1e-6
    erc = convmat1d(er, harmonic_count)
    erc_top = convmat1d(er_top, harmonic_count) if er_top is not None else None
    erc_bot = convmat1d(er_bot, harmonic_count) if er_bot is not None else None
    return PhCDevice(
        ER=np.asarray(er, dtype=np.float64),
        ERC=erc,
        sub_L=sub_l,
        is_top=is_top,
        is_bot=is_bot,
        ER_top=er_top,
        ER_bot=er_bot,
        ERC_top=erc_top,
        ERC_bot=erc_bot,
        sub_L_top=sub_l_top,
        sub_L_bot=sub_l_bot,
    )


def launch_rcwa_s_phc(
    layer_count: int,
    harmonic_count: int,
    grid: PhCGrid,
    device: PhCDevice,
    modes: Iterable[Mode] | str = ("E",),
    calc_fresnel: bool = False,
    *,
    repeat_mode: RepeatMode = "legacy",
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[DiffractionResult, DiffractionResult]:
    """Port of ``Launch_RCWA_S_PhC.m``.

    ``repeat_mode='legacy'`` preserves the MATLAB non-power-of-two bug where
    the unit cell is repeated ``layer_count + 1`` times. ``'corrected'``
    repeats the unit cell exactly ``layer_count`` times.
    """

    if layer_count < 1:
        raise ValueError("layer_count must be at least one")
    if repeat_mode not in {"legacy", "corrected"}:
        raise ValueError("repeat_mode must be 'legacy' or 'corrected'")
    solver_grid = grid.solver_grid()
    unit_device = device.unit_device()
    size = 2 * harmonic_count + 1
    if unit_device.ERC.shape[:2] != (size, size):
        raise ValueError("device ERC shape must match harmonic_count")
    sweep_shape = (solver_grid.Lam0.size, solver_grid.Theta.size)
    trn_minus_1 = np.zeros(sweep_shape, dtype=np.float64)
    trn_plus_1 = np.zeros(sweep_shape, dtype=np.float64)
    trn0 = np.zeros(sweep_shape, dtype=np.float64)
    trn_sum = np.zeros(sweep_shape, dtype=np.float64)
    ref_minus_1 = np.zeros(sweep_shape, dtype=np.float64)
    ref_plus_1 = np.zeros(sweep_shape, dtype=np.float64)
    ref0 = np.zeros(sweep_shape, dtype=np.float64)
    ref_sum = np.zeros(sweep_shape, dtype=np.float64)
    mode_list = _normalize_modes(modes)
    total_points = len(mode_list) * solver_grid.Lam0.size * solver_grid.Theta.size
    completed_points = 0
    src = np.zeros((size, 1), dtype=np.complex128)
    src[size // 2, 0] = 1.0

    for mode in mode_list:
        if mode not in {"E", "H"}:
            raise ValueError("modes may only contain 'E' and 'H'")
        for lam_index, lam0 in enumerate(solver_grid.Lam0):
            for theta_index, theta in enumerate(solver_grid.Theta):
                checkpoint(cancel_token)
                kx, kz_t, kz_r = calc_k(solver_grid, float(lam0), float(theta), harmonic_count)
                sg, w0, v0 = calc_free_space(kx)
                w_ref, s_ref = calc_reflection_side(kx, kz_r, w0, v0, solver_grid, mode)
                w_trn, s_trn = calc_transmission_side(kx, kz_t, w0, v0, solver_grid, mode)

                for sub_layer in range(unit_device.sub_L.size):
                    sg = redheffer_star(sg, calc_layer(kx, float(lam0), w0, v0, unit_device, sub_layer, mode))
                suc = sg
                sg = _repeat_unit_cell(sg, suc, layer_count, repeat_mode)

                if device.is_top:
                    if device.ERC_top is None or device.sub_L_top is None:
                        raise ValueError("top PhC fields are missing")
                    sg_top, _, _ = calc_free_space(kx)
                    top_device = Device1D(device.ERC_top, device.sub_L_top)
                    for sub_layer in range(top_device.sub_L.size):
                        sg_top = redheffer_star(
                            sg_top,
                            calc_layer(kx, float(lam0), w0, v0, top_device, sub_layer, mode),
                        )
                    sg = redheffer_star(sg_top, sg)
                if device.is_bot:
                    if device.ERC_bot is None or device.sub_L_bot is None:
                        raise ValueError("bottom PhC fields are missing")
                    sg_bot, _, _ = calc_free_space(kx)
                    bot_device = Device1D(device.ERC_bot, device.sub_L_bot)
                    for sub_layer in range(bot_device.sub_L.size):
                        sg_bot = redheffer_star(
                            sg_bot,
                            calc_layer(kx, float(lam0), w0, v0, bot_device, sub_layer, mode),
                        )
                    sg = redheffer_star(sg, sg_bot)

                sg = redheffer_star(s_ref, sg)
                sg = redheffer_star(sg, s_trn)
                csrc = np.linalg.solve(w_ref, src)
                ry = w_ref @ sg.S11 @ csrc
                r = np.abs(ry[:, 0]) ** 2
                kz_inc = np.cos(theta) * np.sqrt(solver_grid.erR * solver_grid.urR + 0j)
                r = np.real(np.diag(-kz_r) / kz_inc) * r
                ty = w_trn @ sg.S21 @ csrc
                t = np.abs(ty[:, 0]) ** 2
                if mode == "E":
                    t = np.real((solver_grid.urR / solver_grid.urT) * np.diag(kz_t) / kz_inc) * t
                else:
                    t = np.real(
                        np.diag(kz_t) * np.sqrt(solver_grid.erR + 0j) / (solver_grid.erT * np.cos(theta))
                    ) * t
                if calc_fresnel:
                    t = _apply_legacy_fresnel_correction(t, solver_grid, float(theta), mode, size)

                ref_minus_1[lam_index, theta_index] = r[size // 2 - 1]
                ref_plus_1[lam_index, theta_index] = r[size // 2 + 1]
                ref0[lam_index, theta_index] = r[size // 2]
                ref_sum[lam_index, theta_index] = abs(np.sum(r))
                trn_minus_1[lam_index, theta_index] = t[size // 2 - 1]
                trn_plus_1[lam_index, theta_index] = t[size // 2 + 1]
                trn0[lam_index, theta_index] = t[size // 2]
                trn_sum[lam_index, theta_index] = abs(np.sum(t))
                completed_points += 1
                if progress is not None:
                    progress(completed_points, total_points)

    return (
        DiffractionResult(trn_minus_1, trn_plus_1, TRN0=trn0, sum=trn_sum),
        DiffractionResult(ref_minus_1, ref_plus_1, REF0=ref0, sum=ref_sum),
    )


def _repeat_unit_cell(
    sg: ScatteringMatrix,
    suc: ScatteringMatrix,
    layer_count: int,
    repeat_mode: RepeatMode,
) -> ScatteringMatrix:
    if repeat_mode == "corrected":
        for _ in range(layer_count - 1):
            sg = redheffer_star(sg, suc)
        return sg
    loop_num = np.log2(layer_count)
    if np.floor(loop_num) == loop_num:
        for _ in range(int(loop_num)):
            sg = redheffer_star(sg, sg)
    else:
        for _ in range(layer_count - 1):
            sg = redheffer_star(sg, suc)
        sg = redheffer_star(sg, suc)
    return sg


def _materialize(mask: NDArray[np.bool_], grid: PhCGrid) -> FloatArray:
    return (grid.erIdx[1] - grid.erIdx[0]) * mask.astype(np.float64) + grid.erIdx[0]


def _apply_legacy_fresnel_correction(
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


def _normalize_modes(modes: Iterable[Mode] | str) -> tuple[Mode, ...]:
    mode_list = tuple(modes)
    for mode in mode_list:
        if mode not in {"E", "H"}:
            raise ValueError("modes may only contain 'E' and 'H'")
    return cast(tuple[Mode, ...], mode_list)
