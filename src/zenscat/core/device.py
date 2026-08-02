"""MATLAB-compatible ZenScat grid and 1D device geometry builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rcwa1d import convmat1d

Distribution = Literal["all", "two"]
Interface = Literal["sin", "DE1", "DE4", "tri"]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class InterfaceParams:
    """Subset of ``Int_Params.m`` used by the 1D device builder."""

    smooth: bool = False
    trapz_w_bot: float = 0.195
    trapz_w_top: float = 0.375
    supergauss_sigma: float = 0.30615 / 2.355
    supergauss_m: float = 2.0
    triangle_w1: float = 0.25
    triangle_w2: float = 0.3
    triangle_w3: float = 0.1


@dataclass(frozen=True)
class LegacyGrid:
    """Geometry and medium fields produced by legacy ``Grid.m``."""

    Lam0: FloatArray
    Theta: FloatArray
    distribution: str
    urR: float
    erR: float
    urT: float
    erT: float
    erSub: float
    erIdx: FloatArray
    h: float
    Lx: float
    Lz: float
    Nx: int
    Nz: int
    dz: float
    qx: float
    qz: float
    x: FloatArray
    Length: FloatArray
    delta: int
    layer_num: int

    def solver_kwargs(self) -> dict[str, object]:
        """Return fields accepted by ``rcwa1d.Grid1D``."""

        return {
            "Lam0": self.Lam0,
            "Theta": self.Theta,
            "Lx": self.Lx,
            "erR": self.erR,
            "urR": self.urR,
            "erT": self.erT,
            "urT": self.urT,
            "erSub": self.erSub,
            "layer_num": self.layer_num,
        }


@dataclass(frozen=True)
class LegacyDevice:
    """Device arrays produced by legacy ``Device.m``."""

    ER: FloatArray
    ERC: ComplexArray
    sub_L: FloatArray


def build_legacy_grid(
    params: ArrayLike,
    *,
    layer_num: int | None = None,
    distribution: Distribution = "all",
    interface: Interface = "sin",
    Lx: float = 0.32,
    h: float = 0.154,
    Nx: int = 1028,
    Nz: int = 15,
    Lam0: ArrayLike | None = None,
    Theta: ArrayLike | None = None,
    n_sup: float = 1.0,
    n_sub: float = 1.516,
    is_periodic: bool = False,
    period_num: int = 33,
) -> LegacyGrid:
    """Build the subset of ``Grid.m`` needed by 1D RCWA.

    Lengths are stored in micrometers, matching the MATLAB geometry code.
    ``Lam0`` is stored in meters and ``Theta`` in radians.
    """

    values = np.asarray(params, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError("params must contain at least one length and one refractive index")
    if layer_num is None:
        layer_num = int(np.ceil(values.size / 2))
    if layer_num <= 0:
        raise ValueError("layer_num must be positive")
    if values.size < layer_num + 1:
        raise ValueError("params must contain layer lengths followed by refractive indices")
    if interface not in {"sin", "DE1", "DE4", "tri"}:
        raise NotImplementedError("only analytic 1D Device.m interfaces are implemented")
    if distribution not in {"all", "two"}:
        raise ValueError("distribution must be 'all' or 'two'")
    if Nx <= 0 or Nz <= 0:
        raise ValueError("Nx and Nz must be positive")
    if Lx <= 0 or h <= 0:
        raise ValueError("Lx and h must be positive")

    lam0 = _as_1d_float(1e-9 * np.linspace(480, 540, 201) if Lam0 is None else Lam0)
    theta = _as_1d_float(np.array([0.0]) if Theta is None else Theta)
    refractive_indices = values[layer_num:].copy()

    if distribution == "two":
        if refractive_indices.size == 0:
            raise ValueError("two-material distribution requires at least one index")
        if refractive_indices.size == 1:
            refractive_indices = np.array([refractive_indices[0], refractive_indices[0]])
        expanded = np.empty(layer_num, dtype=np.float64)
        for i in range(layer_num):
            expanded[i] = refractive_indices[0 if (i + 1) % 2 == 1 else 1]
        refractive_indices = expanded

    if is_periodic:
        if values.size < 2:
            raise ValueError("periodic grids require two thickness values")
        er_idx = np.empty(period_num, dtype=np.float64)
        periodic_indices = refractive_indices
        if periodic_indices.size == 1:
            periodic_indices = np.array([periodic_indices[0], periodic_indices[0]])
        for i in range(period_num):
            er_idx[i] = periodic_indices[1 if (i + 1) % 2 == 1 else 0] ** 2
        layer_thickness = values[:2]
        length = np.empty(period_num, dtype=np.float64)
        for i in range(period_num):
            length[i] = layer_thickness[1 if (i + 1) % 2 == 1 else 0]
        delta = (period_num + 2) * Nz
    else:
        er_idx = refractive_indices**2
        length = values[:layer_num].copy()
        delta = (layer_num + 2) * Nz

    Lz = float(h)
    dz = Lz / Nz
    qx = 2 * np.pi / Lx
    qz = 2 * np.pi / Lz
    x = np.linspace(-Lx / 2, Lx / 2, Nx)

    return LegacyGrid(
        Lam0=lam0,
        Theta=theta,
        distribution=distribution,
        urR=1.0,
        erR=float(n_sup**2),
        urT=1.0,
        erT=float(n_sub**2),
        erSub=float(n_sub**2),
        erIdx=er_idx,
        h=float(h),
        Lx=float(Lx),
        Lz=Lz,
        Nx=int(Nx),
        Nz=int(Nz),
        dz=float(dz),
        qx=float(qx),
        qz=float(qz),
        x=x,
        Length=length,
        delta=int(delta),
        layer_num=1,
    )


def build_legacy_device(
    harmonic_count: int,
    grid: LegacyGrid,
    *,
    interface: Interface = "sin",
    interface_params: InterfaceParams | None = None,
    flat_substrate: bool = False,
) -> LegacyDevice:
    """Build ``ER``, ``ERC``, and ``sub_L`` following legacy ``Device.m``."""

    if harmonic_count < 0:
        raise ValueError("harmonic_count must be non-negative")
    if interface not in {"sin", "DE1", "DE4", "tri"}:
        raise NotImplementedError("only analytic 1D Device.m interfaces are implemented")
    int_params = InterfaceParams() if interface_params is None else interface_params

    length = grid.Length.copy()
    z = _interface_profile(grid, interface, int_params)
    d = np.zeros((1 + length.size, z.size), dtype=np.float64)
    d[0, :] = z
    for i in range(length.size):
        if length[i] > grid.h:
            length[i] = grid.h / grid.Nz + grid.h
        d[i + 1, :] = z - np.sum(length[: i + 1])

    area = grid.erR * np.ones((grid.Nx, grid.Nz + grid.delta), dtype=np.float64)
    idx = 0
    for i in range(length.size + 1):
        if i == length.size:
            idx = length.size - 1
        for nx in range(grid.Nx):
            nz = _matlab_round((d[i, nx] + grid.Lz / 2) / grid.dz)
            area = _assign_matlab_prefix(area, nx, nz + grid.delta, grid.erIdx[idx])
        idx += 1

    if flat_substrate:
        d_0 = np.zeros_like(z) - np.sum(length) + grid.h / 2
    else:
        d_0 = d[-1, :]
    for nx in range(grid.Nx):
        nz = _matlab_round((d_0[nx] + grid.Lz / 2) / grid.dz)
        area = _assign_matlab_prefix(area, nx, nz + grid.delta, grid.erSub)

    er = np.rot90(area)
    er[er <= 0] = 1

    ind_substr = np.mean(er, axis=1) - grid.erSub
    idx_substr = np.abs(ind_substr) < 1e-6
    int_substr = np.diff(idx_substr.astype(np.float64))
    substr_index = np.flatnonzero(int_substr == 1)
    if substr_index.size:
        er = np.delete(er, np.s_[substr_index[0] + 4 :], axis=0)

    indices = np.zeros((length.size, er.shape[0]), dtype=np.float64)
    for j in range(length.size):
        indices[j, :] = np.mean(er, axis=1) - grid.erIdx[j]
    idx_same = np.abs(indices) < 1e-6
    sub_l = _geometry_sub_lengths(grid, idx_same) * 1e-6

    if int_params.smooth:
        er = _smooth_layers_like_matlab(er, area, grid)

    return LegacyDevice(
        ER=np.asarray(er, dtype=np.float64),
        ERC=convmat1d(er, harmonic_count),
        sub_L=np.asarray(sub_l, dtype=np.float64),
    )


def _interface_profile(grid: LegacyGrid, interface: Interface, params: InterfaceParams) -> FloatArray:
    if interface == "sin":
        return grid.h / 2 * np.sin(grid.qx * grid.x - np.pi / 2)
    if interface == "DE1":
        z = _trapezium(grid.x, grid, params) - grid.h / 2
        start = int(np.floor(grid.Nx / 2 + 1)) - 1
        right_half = z[start:]
        return np.concatenate([right_half, np.flip(right_half)])
    if interface == "DE4":
        return _trapezium_softened(grid.x, grid, params) - grid.h / 2
    if interface == "tri":
        return _triangle(grid.x, grid, params)
    raise NotImplementedError(f"unsupported analytic interface: {interface}")


def _trapezium(x: FloatArray, grid: LegacyGrid, params: InterfaceParams) -> FloatArray:
    Lx = grid.Lx
    h = grid.h
    w_top = params.trapz_w_top
    wx1 = (Lx - w_top) / 2
    w_bot = params.trapz_w_bot
    wx2 = (w_top - w_bot) / 2
    alpha = h / wx2
    b = alpha * (wx1 + wx2 - Lx / 2)
    z = np.zeros(x.size, dtype=np.float64)
    for i, x_i in enumerate(x):
        if x_i <= -Lx / 2 + wx1:
            z[i] = h
        elif x_i >= -Lx / 2 + wx1:
            z[i] = -alpha * x_i + b
        if x_i > -Lx / 2 + wx1 + wx2:
            z[i] = 1e-6
        if x_i > -Lx / 2 + wx1 + wx2 + w_bot:
            z[i] = alpha * x_i + b
        if x_i > -Lx / 2 + wx1 + 2 * wx2 + w_bot:
            z[i] = h
    return z


def _trapezium_softened(x: FloatArray, grid: LegacyGrid, params: InterfaceParams) -> FloatArray:
    sigma = params.supergauss_sigma
    n = 2 * params.supergauss_m
    return grid.h * np.exp(-0.5 * ((x / sigma) ** n))


def _triangle(x: FloatArray, grid: LegacyGrid, params: InterfaceParams) -> FloatArray:
    h = grid.h
    w1 = params.triangle_w1 * grid.Lx
    w2 = params.triangle_w2 * grid.Lx
    w3 = params.triangle_w3 * grid.Lx
    min_x = np.min(x)
    gamma = min_x + w1
    delta = min_x + w1 + w2
    sigma = delta + w3
    a1 = h / (delta - gamma)
    b1 = -gamma * a1
    a2 = h / (sigma - delta)
    b2 = sigma * a2
    y = np.zeros(x.size, dtype=np.float64)
    for i, x_i in enumerate(x):
        # MATLAB evaluates chained comparisons left-to-right; keep that
        # quirk for byte-level parity with Triangle.m.
        if float(min_x <= x_i) <= min_x + w1:
            y[i] = 0
        if (min_x + w1 <= x_i) and (x_i <= min_x + w1 + w2):  # noqa: PLR1716
            y[i] = a1 * x_i + b1
        if min_x + w1 + w2 <= x_i:
            y[i] = -a2 * x_i + b2
        if x_i >= min_x + w1 + w2 + w3:
            y[i] = 0
    return y


def _geometry_sub_lengths(grid: LegacyGrid, idx: NDArray[np.bool_]) -> FloatArray:
    """Port of ``GeometrySub_Lengths.m``."""

    c = grid.Length - grid.h
    d = c[c > 0]
    sub_l = np.full(idx.shape[1], grid.dz, dtype=np.float64)
    if grid.distribution == "two" and c.size > 1:
        layer_idx = idx[0, :].astype(np.float64) + idx[1, :].astype(np.float64)
    else:
        layer_idx = np.sum(idx, axis=0)
    col = np.flatnonzero(layer_idx >= 1)

    if col.size:
        if col.size == d.size:
            sub_l[col] = d
        else:
            sub_l[:] = grid.dz
    else:
        sub_l[:] = grid.dz
        if grid.Length.size == 1:
            sub_l[:] = grid.Length[0] / grid.Nz
        else:
            nz = grid.Nz
            dz = grid.Length / grid.Nz
            for i in range(grid.Length.size):
                sub_l[i * nz : (i + 1) * nz] = dz[i]

    if np.sum(sub_l) == 0:
        sub_l[:] = grid.dz
    return sub_l


def _smooth_layers_like_matlab(er: FloatArray, area: FloatArray, grid: LegacyGrid) -> FloatArray:
    smoothed = er.astype(np.complex128)
    b = np.exp(-((grid.x - np.mean(grid.x)) / grid.Lx * 100) ** 8)
    start = area.shape[1] - 1
    for i in range(start, er.shape[0]):
        er0 = np.fft.fft(er[i, :]) * np.fft.fft(b) / np.sum(b)
        smoothed[i, :] = np.fft.ifftshift(np.fft.ifft(er0))
    if np.max(np.abs(smoothed.imag)) < 1e-12:
        return smoothed.real.astype(np.float64)
    raise ValueError("smoothed ER unexpectedly has non-negligible imaginary values")


def _assign_matlab_prefix(area: FloatArray, row: int, count: int, value: float) -> FloatArray:
    if count < 1:
        return area
    if count > area.shape[1]:
        extra = np.zeros((area.shape[0], count - area.shape[1]), dtype=area.dtype)
        area = np.concatenate([area, extra], axis=1)
    area[row, :count] = value
    return area


def _as_1d_float(value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("array values must not be empty")
    return array


def _matlab_round(value: float) -> int:
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))
