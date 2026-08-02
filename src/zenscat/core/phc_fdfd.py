"""Corrected PhC-to-FDFD material rasterization.

The legacy ``Device_FDFD_PhC.m`` branch references undefined variables and
never returns a complete device.  This module intentionally exposes a
corrected Python contract instead of claiming MATLAB-exact parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fdfd2d import FDFDGrid
from .phc import PhCInterface, PhCInterfaceParams

CompatibilityMode = Literal["corrected", "legacy_exact"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PhCFDFDDevice:
    """FDFD material tensors for a corrected PhC geometry branch."""

    ER2: FloatArray
    UR2: FloatArray
    metadata: dict[str, Any]


def build_phc_fdfd_device(
    params: ArrayLike,
    grid: FDFDGrid,
    interface: PhCInterface,
    interface_params: PhCInterfaceParams | None = None,
    *,
    compatibility: CompatibilityMode = "corrected",
) -> PhCFDFDDevice:
    """Rasterize a PhC unit cell into the FDFD ``ER2``/``UR2`` grid.

    ``compatibility='legacy_exact'`` is deliberately rejected because the
    corresponding MATLAB file is incomplete.  The default corrected mode uses
    explicit geometry semantics and returns metadata that makes this boundary
    auditable by the GUI/export layer.
    """

    if compatibility == "legacy_exact":
        raise NotImplementedError(
            "legacy Device_FDFD_PhC.m is incomplete; use compatibility='corrected'"
        )
    if compatibility != "corrected":
        raise ValueError("compatibility must be 'corrected' or 'legacy_exact'")

    values = np.asarray(params, dtype=np.float64).ravel()
    if values.size < 3:
        raise ValueError("PhC FDFD params must be [pitch_um, n_background, n_inclusion]")
    pitch = float(values[0])
    if pitch <= 0:
        raise ValueError("PhC pitch must be positive")
    n_background = float(values[1])
    n_inclusion = float(values[2])
    if n_background <= 0 or n_inclusion <= 0:
        raise ValueError("PhC refractive indices must be positive")

    int_params = PhCInterfaceParams() if interface_params is None else interface_params
    radius = int_params.radius_star_ellipse * pitch
    if radius <= 0:
        raise ValueError("PhC inclusion radius must be positive")

    x = grid.xa2
    y_rel = grid.ya2 - (grid.SPACER[0] + grid.h / 2)
    X, Y = np.meshgrid(x, y_rel, indexing="ij")
    stack = np.abs(Y) <= grid.h / 2

    er2 = np.full((grid.Nx2, grid.Ny2), grid.erSup, dtype=np.float64)
    er2[:, grid.ya2 > grid.SPACER[0] + grid.h] = grid.erSub
    er2[stack] = n_background**2
    inclusion = _phc_mask(interface, X, Y, pitch, radius, int_params)
    er2[stack & inclusion] = n_inclusion**2
    ur2 = np.ones_like(er2)

    return PhCFDFDDevice(
        ER2=er2,
        UR2=ur2,
        metadata={
            "workflow": "phc_fdfd",
            "interface": interface,
            "compatibility": "corrected",
            "legacy_exact": False,
            "semantics": _semantics(interface),
            "pitch_um": pitch,
            "radius_um": radius,
            "background_n": n_background,
            "inclusion_n": n_inclusion,
        },
    )


def _phc_mask(
    interface: PhCInterface,
    x: FloatArray,
    y: FloatArray,
    pitch: float,
    radius: float,
    params: PhCInterfaceParams,
) -> NDArray[np.bool_]:
    if interface == "PhC_rec_circ":
        if params.ax == 0 or params.ay == 0:
            raise ValueError("ellipse axes must be non-zero")
        xr, yr = _rotate(x, y, params.ellipse_rot_angle)
        return (xr / params.ax) ** 2 + (yr / params.ay) ** 2 <= radius**2
    if interface == "PhC_rec_square":
        wx = params.rec_2D_wx * pitch
        wy = params.rec_2D_wy * pitch
        if wx == 0 or wy == 0:
            raise ValueError("rectangle widths must be non-zero")
        xr, yr = _rotate(x, y, params.rec_rot_angle)
        return (np.abs(xr) <= wx / 2) & (np.abs(yr) <= wy / 2)
    if interface == "PhC_hex_columns":
        return _hex_columns(x, y, pitch, radius, rotated=False)
    if interface == "PhC_hex_columns_rot":
        return _hex_columns(x, y, pitch, radius, rotated=True)
    if interface == "PhC_honeycomb":
        return _honeycomb(x, y, pitch, radius)
    if interface == "PhC_hex_polygon":
        xr, yr = _rotate(x, y, params.hex_rot_angle)
        qx = np.abs(xr)
        qy = np.abs(yr)
        apothem = radius * np.sqrt(3) / 2
        return (qx <= radius) & (np.sqrt(3) * qx + qy <= 2 * apothem)
    raise NotImplementedError(f"unsupported PhC FDFD interface: {interface}")


def _hex_columns(
    x: FloatArray,
    y: FloatArray,
    pitch: float,
    radius: float,
    *,
    rotated: bool,
) -> NDArray[np.bool_]:
    x_scale = np.sqrt(3) if rotated else 1.0
    y_scale = 1.0 if rotated else np.sqrt(3)
    xr = x * x_scale
    yr = y * y_scale
    b1 = pitch / 2
    b2 = np.sqrt(3) * pitch / 2
    return (
        xr**2 + yr**2 <= radius**2
    ) | (
        (xr - b1) ** 2 + (yr + b2) ** 2 <= radius**2
    ) | (
        (xr + b1) ** 2 + (yr - b2) ** 2 <= radius**2
    ) | (
        (xr + b1) ** 2 + (yr + b2) ** 2 <= radius**2
    ) | (
        (xr - b1) ** 2 + (yr - b2) ** 2 <= radius**2
    )


def _honeycomb(x: FloatArray, y: FloatArray, pitch: float, radius: float) -> NDArray[np.bool_]:
    cell_height = np.sqrt(3) * pitch
    period = 3 * cell_height
    y_fold = ((y + 1.5 * cell_height) % period) - 1.5 * cell_height
    b1 = pitch / 2
    b2 = cell_height / 2
    return (
        (x - b1) ** 2 + (y_fold + b2) ** 2 <= radius**2
    ) | (
        (x + b1) ** 2 + (y_fold - b2) ** 2 <= radius**2
    ) | (
        x**2 + y_fold**2 <= radius**2
    )


def _rotate(x: FloatArray, y: FloatArray, angle_deg: float) -> tuple[FloatArray, FloatArray]:
    theta = np.deg2rad(angle_deg)
    return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)


def _semantics(interface: PhCInterface) -> str:
    return {
        "PhC_rec_circ": "corrected rotated ellipse inclusion",
        "PhC_rec_square": "corrected rotated rectangle inclusion",
        "PhC_hex_columns": "corrected hexagonal circular column lattice",
        "PhC_hex_columns_rot": "corrected rotated hexagonal circular column lattice",
        "PhC_honeycomb": "corrected three-sublattice honeycomb circular column lattice",
        "PhC_hex_polygon": "corrected regular hexagonal polygon inclusion",
    }[interface]
