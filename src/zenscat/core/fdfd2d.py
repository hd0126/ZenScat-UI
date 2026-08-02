"""MATLAB-compatible 2D FDFD path for the mainline sinusoidal ZenScat flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .control import CancellationToken, ProgressCallback, checkpoint
from .device import InterfaceParams

Mode = Literal["E", "H"]
Distribution = Literal["all", "two"]
Interface = Literal["sin", "DE1", "DE4", "tri"]
DispersionMode = Literal["legacy_fdfd", "corrected_um"]
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DEFAULT_DISPERSION_COEFFICIENTS: FloatArray = np.array(
    [
        [1.0470360360304749, 0.10062044805139791, 1.7626520390806792, 0.025857685491254445, 7.9442504603011166e-06, 0.01488250447597661],
        [0.7670067391554749, 3.124244511665399, 0.0022550641286480655, 0.089628459528276805, 0.021107109216242037, 29.169364749130747],
        [1.1716240487257874, 0.0062782600289601476, 0.10664365404621634, 0.010968127162218622, 41.500665274658573, 42.426637697757954],
        [2.1818007073561585, 1.0908911851183873, 7.0513498675595088e-06, 8.7187438499114478e-05, 0.066201722499762639, 10.902777516279798],
        [1.0939998767535126, 2.8404176966034811e-05, 3.1709038984157232, 0.10939622075874555, 22.313330197829295, 0.007646813524629259],
        [1.2875450692335999, 0.068446427340640881, 0.11622261612341589, 0.0088687914709417726, 8.2797100287373464, 8.49380347063763],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FDFDGrid:
    """Fields produced by legacy ``Grid_FDFD.m`` for the mainline path."""

    distribution: str
    erIdx: FloatArray
    Length: FloatArray
    h: float
    Lam0: FloatArray
    Theta: FloatArray
    X: FloatArray
    Y: FloatArray
    dx: float
    dy: float
    xa: FloatArray
    ya: FloatArray
    Lx: float
    Ly: float
    Sx: float
    Sy: float
    Nx: int
    Ny: int
    xa2: FloatArray
    ya2: FloatArray
    Nx2: int
    Ny2: int
    dx2: float
    dy2: float
    NRES: float
    SPACER: FloatArray
    NPML: NDArray[np.int64]
    nmax: float
    erSup: float
    erSub: float


@dataclass(frozen=True)
class FDFDDevice:
    """Device arrays produced by legacy ``Device_FDFD.m``."""

    ER2: FloatArray
    UR2: FloatArray


@dataclass(frozen=True)
class FDFDResult:
    """Legacy FDFD public result schema and raw final field."""

    TRN: dict[str, FloatArray | float]
    REF: dict[str, FloatArray | float]
    f: ComplexArray


def build_fdfd_grid(
    params: ArrayLike,
    *,
    layer_num: int | None = None,
    distribution: Distribution = "all",
    Lam0: ArrayLike = (1.0,),
    Theta: ArrayLike = (0.0,),
    Lx: float = 0.625,
    h: float = 0.2,
    n_sup: float = 1.0,
    n_sub: float = 1.47,
    nres: float = 20,
    spacer: ArrayLike | None = None,
    npml: ArrayLike = (25, 25),
    is_periodic: bool = False,
    period_num: int = 33,
    refractive_idx: bool = False,
    dispersion_mode: DispersionMode = "legacy_fdfd",
    dispersion_coefficients: ArrayLike | None = None,
) -> FDFDGrid:
    """Port of ``Grid_FDFD.m`` for explicit or material-palette indices.

    ``dispersion_mode='legacy_fdfd'`` preserves ``Dispersion.m`` exactly,
    including its ``1e6 * Lam0(mid)`` wavelength scaling. Use
    ``'corrected_um'`` only when FDFD wavelengths are already in micrometers.
    """

    values = np.asarray(params, dtype=np.float64).ravel()
    if layer_num is None:
        layer_num = int(np.ceil(values.size / 2))
    if values.size < layer_num + 1:
        raise ValueError("params must contain layer lengths followed by refractive indices")
    if distribution not in {"all", "two"}:
        raise ValueError("distribution must be 'all' or 'two'")

    lam0 = _as_1d_float(Lam0, "Lam0")
    theta = _as_1d_float(Theta, "Theta")
    raw_idx = values[layer_num:].copy()
    if refractive_idx:
        palette = dispersion_refractive_indices(
            lam0,
            coefficients=dispersion_coefficients,
            mode=dispersion_mode,
        )
        n_idx = np.empty(raw_idx.size, dtype=np.float64)
        for i, material_index in enumerate(raw_idx):
            index = int(material_index)
            if index != material_index or index < 1 or index > palette.size:
                raise ValueError("material palette indices must be integer values from 1 to 6")
            n_idx[i] = palette[index - 1]
    else:
        n_idx = raw_idx
    if is_periodic:
        er_idx = np.empty(period_num, dtype=np.float64)
        periodic_idx = n_idx
        if periodic_idx.size == 1:
            periodic_idx = np.array([periodic_idx[0], periodic_idx[0]], dtype=np.float64)
        for i in range(period_num):
            er_idx[i] = periodic_idx[1 if (i + 1) % 2 == 1 else 0] ** 2
        length = np.empty(period_num, dtype=np.float64)
        layer_thickness = values[:2]
        for i in range(period_num):
            length[i] = layer_thickness[1 if (i + 1) % 2 == 1 else 0]
    else:
        er_idx = n_idx**2
        length = values[:layer_num].copy()

    spacer_values = np.asarray(max(lam0) * np.array([2.5, 2.0]) if spacer is None else spacer, dtype=np.float64)
    npml_values = np.asarray(npml, dtype=np.int64).ravel()
    if npml_values.size != 2:
        raise ValueError("npml must contain [low, high] y-PML cell counts")
    if nres <= 0 or Lx <= 0 or h <= 0:
        raise ValueError("nres, Lx, and h must be positive")

    ly = float(h)
    nmax = float(np.max(np.abs(n_idx)))
    dx = float(np.min(lam0) / abs(nmax) / nres)
    dy = float(np.min(lam0) / abs(nmax) / nres)
    nx_pre = int(np.ceil(Lx / dx))
    dx = float(Lx / nx_pre)
    ny_pre = int(np.ceil(ly / dy))
    dy = float(ly / ny_pre)

    nx = int(np.ceil(Lx / dx))
    sx = float(nx * dx)
    sy_requested = float(spacer_values[0] + h + spacer_values[1])
    ny = int(npml_values[0] + np.ceil(sy_requested / dy) + npml_values[0])
    sy = float(ny * dy)

    nx2 = 2 * nx
    ny2 = 2 * ny
    dx2 = dx / 2
    dy2 = dy / 2

    xa = np.arange(1, nx + 1, dtype=np.float64) * dx
    ya = np.arange(1, ny + 1, dtype=np.float64) * dy
    y_mesh, x_mesh = np.meshgrid(ya, xa)
    xa2 = np.arange(1, nx2 + 1, dtype=np.float64) * dx2
    xa2 = xa2 - np.mean(xa2)
    ya2 = np.arange(1, ny2 + 1, dtype=np.float64) * dy2

    return FDFDGrid(
        distribution=distribution,
        erIdx=er_idx,
        Length=length,
        h=float(h),
        Lam0=lam0,
        Theta=theta,
        X=x_mesh,
        Y=y_mesh,
        dx=dx,
        dy=dy,
        xa=xa,
        ya=ya,
        Lx=float(Lx),
        Ly=ly,
        Sx=sx,
        Sy=sy,
        Nx=nx,
        Ny=ny,
        xa2=xa2,
        ya2=ya2,
        Nx2=nx2,
        Ny2=ny2,
        dx2=dx2,
        dy2=dy2,
        NRES=float(nres),
        SPACER=spacer_values,
        NPML=npml_values,
        nmax=nmax,
        erSup=float(n_sup**2),
        erSub=float(n_sub**2),
    )


def dispersion_refractive_indices(
    lam0: ArrayLike,
    *,
    coefficients: ArrayLike | None = None,
    mode: DispersionMode = "legacy_fdfd",
) -> FloatArray:
    """Evaluate legacy ``Dispersion.m`` coefficient rows.

    ``legacy_fdfd`` is the compatibility default and multiplies the middle
    FDFD wavelength by ``1e6`` exactly as MATLAB does. ``corrected_um``
    evaluates the same formula treating FDFD wavelengths as micrometers.
    """

    wavelengths = _as_1d_float(lam0, "Lam0")
    center_wavelength = wavelengths[wavelengths.size // 2]
    if mode == "legacy_fdfd":
        wavelength = 1e6 * center_wavelength
    elif mode == "corrected_um":
        wavelength = center_wavelength
    else:
        raise ValueError("dispersion mode must be 'legacy_fdfd' or 'corrected_um'")

    coeffs = np.abs(
        np.asarray(DEFAULT_DISPERSION_COEFFICIENTS if coefficients is None else coefficients, dtype=np.float64)
    )
    if coeffs.ndim != 2 or coeffs.shape[1] % 2 != 0:
        raise ValueError("dispersion coefficients must have shape (material_count, 2 * term_count)")
    split = coeffs.shape[1] // 2
    a_coeffs = coeffs[:, :split]
    b_coeffs = coeffs[:, split:]
    wavelength2 = wavelength**2
    terms = a_coeffs * wavelength2 / (wavelength2 - b_coeffs)
    return np.sqrt(1 + np.sum(terms, axis=1))


def build_fdfd_device(
    params: ArrayLike,
    grid: FDFDGrid,
    *,
    interface: Interface = "sin",
    interface_params: InterfaceParams | None = None,
) -> FDFDDevice:
    """Port of ``Device_FDFD.m`` for working mainline analytic interfaces."""

    if interface not in {"sin", "DE1", "DE4", "tri"}:
        raise NotImplementedError("only mainline Device_FDFD.m analytic interfaces are implemented")
    int_params = InterfaceParams() if interface_params is None else interface_params
    _ = np.asarray(params, dtype=np.float64).ravel()
    layer_num = grid.Length.size
    thickness = grid.Length
    d = np.zeros((layer_num + 1, grid.Nx2), dtype=np.float64)
    z = _interface_profile(grid, interface, int_params)
    d[0, :] = z
    for i in range(layer_num):
        d[i + 1, :] = z - np.sum(thickness[: i + 1])

    er2 = grid.erSup * np.ones((grid.Nx2, grid.Ny2), dtype=np.float64)
    for i in range(layer_num):
        d_0 = d[i, :]
        for nx in range(grid.Nx2):
            ny = _matlab_round((d_0[nx] + grid.h + grid.SPACER[0]) / grid.dy2)
            if ny > 0:
                er2[nx, :ny] = grid.erIdx[i]

    d_0 = d[-1, :]
    for nx in range(grid.Nx2):
        ny = _matlab_round((d_0[nx] + grid.h + grid.SPACER[0]) / grid.dy2)
        if ny > 0:
            er2[nx, :ny] = grid.erSub

    er2 = np.rot90(er2, 2)
    ur2 = np.ones_like(er2)
    return FDFDDevice(ER2=er2, UR2=ur2)


def _interface_profile(grid: FDFDGrid, interface: Interface, params: InterfaceParams) -> FloatArray:
    if interface == "sin":
        return grid.h / 2 * np.sin(2 * np.pi / grid.Lx * grid.xa2 - np.pi / 2)
    if interface == "DE1":
        z = _trapezium(grid.xa2, grid, params) - grid.h / 2
        start = int(np.floor(grid.Nx2 / 2 + 1)) - 1
        right_half = z[start:]
        return np.concatenate([right_half, np.flip(right_half)])
    if interface == "DE4":
        return _trapezium_softened(grid.xa2, grid, params) - grid.h / 2
    if interface == "tri":
        return _triangle(grid.xa2, grid, params)
    raise NotImplementedError(f"unsupported FDFD interface: {interface}")


def _trapezium(x: FloatArray, grid: FDFDGrid, params: InterfaceParams) -> FloatArray:
    lx = grid.Lx
    h = grid.h
    w_top = params.trapz_w_top
    wx1 = (lx - w_top) / 2
    w_bot = params.trapz_w_bot
    wx2 = (w_top - w_bot) / 2
    alpha = h / wx2
    b = alpha * (wx1 + wx2 - lx / 2)
    z = np.zeros(x.size, dtype=np.float64)
    for idx, x_i in enumerate(x):
        if x_i <= -lx / 2 + wx1:
            z[idx] = h
        elif x_i >= -lx / 2 + wx1:
            z[idx] = -alpha * x_i + b
        if x_i > -lx / 2 + wx1 + wx2:
            z[idx] = 1e-6
        if x_i > -lx / 2 + wx1 + wx2 + w_bot:
            z[idx] = alpha * x_i + b
        if x_i > -lx / 2 + wx1 + 2 * wx2 + w_bot:
            z[idx] = h
    return z


def _trapezium_softened(x: FloatArray, grid: FDFDGrid, params: InterfaceParams) -> FloatArray:
    sigma = params.supergauss_sigma
    n = 2 * params.supergauss_m
    return grid.h * np.exp(-0.5 * ((x / sigma) ** n))


def _triangle(x: FloatArray, grid: FDFDGrid, params: InterfaceParams) -> FloatArray:
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
    for idx, x_i in enumerate(x):
        if float(min_x <= x_i) <= min_x + w1:
            y[idx] = 0
        if (min_x + w1 <= x_i) and (x_i <= min_x + w1 + w2):  # noqa: PLR1716
            y[idx] = a1 * x_i + b1
        if min_x + w1 + w2 <= x_i:
            y[idx] = -a2 * x_i + b2
        if x_i >= min_x + w1 + w2 + w3:
            y[idx] = 0
    return y


def fdfd_2d(
    grid: FDFDGrid,
    device: FDFDDevice,
    mode: Mode,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> FDFDResult:
    """Port of ``FDFD_2D.m`` including legacy result field names."""

    if mode not in {"E", "H"}:
        raise ValueError("mode must be 'E' or 'H'")
    nx = grid.Nx
    ny = grid.Ny
    ns = (nx, ny)
    res = np.array([grid.dx, grid.dy], dtype=np.float64)
    bc = np.array([1, 0], dtype=np.int64)

    trn0 = np.zeros((grid.Lam0.size, grid.Theta.size), dtype=np.float64)
    ref0 = np.zeros_like(trn0)
    ref_minus1 = np.zeros_like(trn0)
    ref_plus1 = np.zeros_like(trn0)
    trn_plus1 = np.zeros_like(trn0)
    trn_minus1 = np.zeros_like(trn0)
    final_f = np.zeros((nx, ny), dtype=np.complex128)
    final_tde = np.array([], dtype=np.float64)
    final_rde = np.array([], dtype=np.float64)
    total_points = grid.Lam0.size * grid.Theta.size
    completed_points = 0

    for i, lam0 in enumerate(grid.Lam0):
        for j, theta in enumerate(grid.Theta):
            checkpoint(cancel_token)
            erxx, eryy, erzz, urxx, uryy, urzz = addupml2d(
                device.ER2, device.UR2, np.array([0, 0, grid.NPML[0], grid.NPML[1]], dtype=np.int64)
            )
            erxx_d = _sparse_diag(erxx)
            eryy_d = _sparse_diag(eryy)
            erzz_d = _sparse_diag(erzz)
            urxx_d = _sparse_diag(urxx)
            uryy_d = _sparse_diag(uryy)
            urzz_d = _sparse_diag(urzz)

            k0 = 2 * np.pi / lam0
            nsrc = np.sqrt(device.ER2[0, 0] * device.UR2[0, 0])
            kx_inc = nsrc * k0 * np.sin(theta)
            ky_inc = nsrc * k0 * np.cos(theta)
            kinc = np.array([kx_inc, ky_inc], dtype=np.float64)
            dex, dey, dhx, dhy = yeeder2d(ns, k0 * res, bc, kinc / k0)

            if mode == "E":
                a_mat = dhx @ _right_diag_inverse(uryy_d) @ dex + dhy @ _right_diag_inverse(urxx_d) @ dey + erzz_d
            else:
                a_mat = dex @ _right_diag_inverse(eryy_d) @ dhx + dey @ _right_diag_inverse(erxx_d) @ dhy + urzz_d

            fsrc = np.exp(-1j * (kx_inc * grid.X + ky_inc * grid.Y))
            q_mask = np.zeros((nx, ny), dtype=np.float64)
            q_mask[:, : grid.NPML[0] + 2] = 1
            q_diag = sparse.diags(q_mask.ravel(order="F"), format="csr")
            source = (q_diag @ a_mat - a_mat @ q_diag) @ fsrc.ravel(order="F")
            solved = spsolve(a_mat.tocsr(), source)
            final_f = np.asarray(solved, dtype=np.complex128).reshape((nx, ny), order="F")

            urref = device.UR2[0, 0]
            urtrn = device.UR2[0, grid.Ny2 - 1]
            erref = device.ER2[0, 0]
            ertrn = device.ER2[0, grid.Ny2 - 1]
            nref = np.sqrt(urref * erref)
            ntrn = np.sqrt(urtrn * ertrn)

            m_orders = np.arange(-np.floor(nx / 2), np.floor((nx - 1) / 2) + 1, dtype=np.float64)
            kx = kx_inc - m_orders * 2 * np.pi / grid.Sx
            kyref = np.sqrt((k0 * nref) ** 2 - kx**2 + 0j)
            kytrn = np.sqrt((k0 * ntrn) ** 2 - kx**2 + 0j)

            fref = final_f[:, grid.NPML[0]] / fsrc[:, 0]
            ftrn = final_f[:, ny - grid.NPML[1] - 1] / fsrc[:, 0]
            aref = np.fft.fftshift(np.fft.fft(fref)) / nx
            atrn = np.fft.fftshift(np.fft.fft(ftrn)) / nx
            rde = np.abs(aref) ** 2 * np.real(kyref / ky_inc)
            if mode == "E":
                tde = np.abs(atrn) ** 2 * np.real(urref / urtrn * kytrn / ky_inc)
            else:
                tde = np.abs(atrn) ** 2 * np.real(erref / ertrn * kytrn / ky_inc)

            idx0 = int(np.flatnonzero(m_orders == 0)[0])
            idx_minus1 = int(np.flatnonzero(m_orders == -1)[0])
            idx_plus1 = int(np.flatnonzero(m_orders == 1)[0])
            trn0[i, j] = tde[idx0]
            ref0[i, j] = rde[idx0]
            ref_minus1[i, j] = rde[idx_minus1]
            ref_plus1[i, j] = rde[idx_plus1]
            trn_plus1[i, j] = tde[idx_plus1]
            trn_minus1[i, j] = tde[idx_minus1]
            final_tde = tde
            final_rde = rde
            completed_points += 1
            if progress is not None:
                progress(completed_points, total_points)

    trn = {
        "sum": float(np.sum(final_tde)),
        "TRN0": trn0,
        "TRN_plus1": trn_plus1,
        "TRN_minus1": trn_minus1,
    }
    ref = {
        "sum": float(np.sum(final_rde)),
        "REF0": ref0,
        "REF_plus1": ref_plus1,
        "TRN_minus1": trn_minus1,
    }
    return FDFDResult(TRN=trn, REF=ref, f=final_f)


def yeeder2d(
    ns: tuple[int, int] | ArrayLike,
    res: ArrayLike,
    bc: ArrayLike,
    kinc: ArrayLike = (0.0, 0.0),
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Port of ``yeeder2d.m`` derivative matrices on a 2D Yee grid."""

    nx, ny = [int(v) for v in np.asarray(ns).ravel()]
    dx, dy = [float(v) for v in np.asarray(res).ravel()]
    bc_values = np.asarray(bc, dtype=np.int64).ravel()
    kinc_values = np.asarray(kinc, dtype=np.complex128).ravel()
    m_size = nx * ny

    if nx == 1:
        dex = sparse.csr_matrix((-1j * kinc_values[0]) * sparse.eye(m_size, format="csr"))
    else:
        dex = sparse.lil_matrix((m_size, m_size), dtype=np.complex128)
        for row in range(m_size):
            dex[row, row] = -1 / dx
            if (row + 1) % nx != 0 and row + 1 < m_size:
                dex[row, row + 1] = 1 / dx
        if bc_values[0] == 1:
            phase = np.exp(-1j * kinc_values[0] * nx * dx) / dx
            for row in range(nx - 1, m_size, nx):
                dex[row, row - nx + 1] = phase
        dex = sparse.csr_matrix(dex)

    if ny == 1:
        dey = sparse.csr_matrix((-1j * kinc_values[1]) * sparse.eye(m_size, format="csr"))
    else:
        dey = sparse.lil_matrix((m_size, m_size), dtype=np.complex128)
        for row in range(m_size):
            dey[row, row] = -1 / dy
            if row + nx < m_size:
                dey[row, row + nx] = 1 / dy
        if bc_values[1] == 1:
            phase = np.exp(-1j * kinc_values[1] * ny * dy) / dy
            for row in range(m_size - nx, m_size):
                dey[row, row - (m_size - nx)] = phase
        dey = sparse.csr_matrix(dey)

    dhx = sparse.csr_matrix(-dex.conjugate().transpose())
    dhy = sparse.csr_matrix(-dey.conjugate().transpose())
    return dex, dey, dhx, dhy


def addupml2d(
    er2: ArrayLike,
    ur2: ArrayLike,
    npml: ArrayLike,
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    """Port of ``addupml2d.m`` for 2D UPML material tensors."""

    er2_array = np.asarray(er2, dtype=np.complex128)
    ur2_array = np.asarray(ur2, dtype=np.complex128)
    nx2, ny2 = er2_array.shape
    nxlo, nxhi, nylo, nyhi = [int(v) * 2 for v in np.asarray(npml, dtype=np.int64).ravel()]
    amax = 4
    cmax = 1
    p_power = 3

    sx = np.ones((nx2, ny2), dtype=np.complex128)
    sy = np.ones((nx2, ny2), dtype=np.complex128)

    for nx in range(1, nxlo + 1):
        ax = 1 + (amax - 1) * (nx / nxlo) ** p_power
        cx = cmax * np.sin(0.5 * np.pi * nx / nxlo) ** 2
        sx[nxlo - nx, :] = ax * (1 - 1j * 60 * cx)
    for nx in range(1, nxhi + 1):
        ax = 1 + (amax - 1) * (nx / nxhi) ** p_power
        cx = cmax * np.sin(0.5 * np.pi * nx / nxhi) ** 2
        sx[nx2 - nxhi + nx - 1, :] = ax * (1 - 1j * 60 * cx)
    for ny in range(1, nylo + 1):
        ay = 1 + (amax - 1) * (ny / nylo) ** p_power
        cy = cmax * np.sin(0.5 * np.pi * ny / nylo) ** 2
        sy[:, nylo - ny] = ay * (1 - 1j * 60 * cy)
    for ny in range(1, nyhi + 1):
        ay = 1 + (amax - 1) * (ny / nyhi) ** p_power
        cy = cmax * np.sin(0.5 * np.pi * ny / nyhi) ** 2
        sy[:, ny2 - nyhi + ny - 1] = ay * (1 - 1j * 60 * cy)

    erxx = er2_array / sx * sy
    eryy = er2_array * sx / sy
    erzz = er2_array * sx * sy
    urxx = ur2_array / sx * sy
    uryy = ur2_array * sx / sy
    urzz = ur2_array * sx * sy

    return (
        erxx[1::2, 0::2],
        eryy[0::2, 1::2],
        erzz[0::2, 0::2],
        urxx[0::2, 1::2],
        uryy[1::2, 0::2],
        urzz[1::2, 1::2],
    )


def _sparse_diag(values: ArrayLike) -> sparse.csr_matrix:
    return sparse.csr_matrix(sparse.diags(np.asarray(values).ravel(order="F"), format="csr"))


def _right_diag_inverse(diagonal: sparse.csr_matrix) -> sparse.csr_matrix:
    return sparse.csr_matrix(sparse.diags(1 / diagonal.diagonal(), format="csr"))


def _as_1d_float(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _matlab_round(value: float) -> int:
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))
