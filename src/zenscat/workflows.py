"""Headless workflow API shared by the ZenScat desktop UI and automation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    CancellationToken,
    Device1D,
    DiffractionResult,
    FDFDDevice,
    FDFDGrid,
    FDFDResult,
    Grid1D,
    InterfaceParams,
    PhCInterfaceParams,
    ProgressCallback,
    build_fdfd_device,
    build_fdfd_grid,
    build_legacy_device,
    build_legacy_grid,
    build_phc_device,
    build_phc_grid,
    fdfd_2d,
    launch_rcwa_s,
    launch_rcwa_s_phc,
    launch_rcwa_t,
)
from .legacy_io import FDFDResultBundle, ImportedDevice, LegacyResultBundle

MatrixMethod = Literal["S", "T"]
Polarization = Literal["E", "H"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AnalyticRCWARequest:
    params: ArrayLike
    layer_num: int
    wavelengths_m: ArrayLike
    angles_rad: ArrayLike
    harmonic_count: int = 5
    matrix_method: MatrixMethod = "S"
    polarization: Polarization = "E"
    distribution: Literal["all", "two"] = "all"
    interface: Literal["sin", "DE1", "DE4", "tri"] = "sin"
    interface_params: InterfaceParams = field(default_factory=InterfaceParams)
    Lx_um: float = 0.32
    h_um: float = 0.154
    Nx: int = 1028
    Nz: int = 15
    n_superstrate: float = 1.0
    n_substrate: float = 1.516
    is_periodic: bool = False
    period_num: int = 33
    flat_substrate: bool = False
    calc_fresnel: bool = False


@dataclass(frozen=True)
class ImportedRCWARequest:
    device: ImportedDevice
    wavelengths_m: ArrayLike
    angles_rad: ArrayLike
    harmonic_count: int = 5
    matrix_method: MatrixMethod = "S"
    polarization: Polarization = "E"
    n_superstrate: complex = 1.0
    n_substrate: complex = 1.516
    calc_fresnel: bool = False


@dataclass(frozen=True)
class FDFDRequest:
    params: ArrayLike
    layer_num: int
    wavelengths_um: ArrayLike
    angles_rad: ArrayLike
    polarization: Polarization = "E"
    distribution: Literal["all", "two"] = "all"
    interface: Literal["sin", "DE1", "DE4", "tri"] = "sin"
    interface_params: InterfaceParams = field(default_factory=InterfaceParams)
    Lx_um: float = 0.625
    h_um: float = 0.2
    n_superstrate: float = 1.0
    n_substrate: float = 1.47
    nres: float = 20
    spacer_um: ArrayLike | None = None
    npml: ArrayLike = (25, 25)
    is_periodic: bool = False
    period_num: int = 33
    refractive_idx: bool = False
    dispersion_mode: Literal["legacy_fdfd", "corrected_um"] = "legacy_fdfd"


@dataclass(frozen=True)
class PhCRCWARequest:
    params: ArrayLike
    wavelengths_m: ArrayLike
    angles_rad: ArrayLike
    layer_count: int
    interface: Literal["PhC_rec_circ", "PhC_rec_square", "PhC_hex_columns"]
    interface_params: PhCInterfaceParams = field(default_factory=PhCInterfaceParams)
    harmonic_count: int = 2
    polarization: Polarization = "E"
    Nx: int = 96
    Nz: int = 12
    n_superstrate: complex = 1.0
    n_substrate: complex = 1.516
    calc_fresnel: bool = False
    repeat_mode: Literal["legacy", "corrected"] = "legacy"


@dataclass(frozen=True)
class RCWARun:
    wavelengths_m: FloatArray
    angles_rad: FloatArray
    transmission: DiffractionResult
    reflection: DiffractionResult
    grid: Grid1D
    device: Device1D
    device_er: NDArray[np.complex128]
    elapsed_s: float
    metadata: dict[str, object]

    def legacy_bundle(
        self,
        params: ArrayLike | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LegacyResultBundle:
        return LegacyResultBundle(
            wavelengths_m=self.wavelengths_m,
            angles_rad=self.angles_rad,
            transmission=self.transmission,
            reflection=self.reflection,
            params=None if params is None else np.asarray(params, dtype=np.float64).ravel(),
            metadata=self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True)
class FDFDRun:
    wavelengths_um: FloatArray
    angles_rad: FloatArray
    result: FDFDResult
    grid: FDFDGrid
    device: FDFDDevice
    elapsed_s: float
    metadata: dict[str, object]

    def fdfd_bundle(self, params: ArrayLike | None = None) -> FDFDResultBundle:
        return FDFDResultBundle(
            wavelengths_um=self.wavelengths_um,
            angles_rad=self.angles_rad,
            result=self.result,
            er2=self.device.ER2,
            params=None if params is None else np.asarray(params, dtype=np.float64).ravel(),
            metadata=self.metadata,
        )


def run_analytic_rcwa(
    request: AnalyticRCWARequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> RCWARun:
    """Build a legacy analytic device and run the requested matrix method."""

    _validate_common_request(request)
    legacy_grid = build_legacy_grid(
        request.params,
        layer_num=request.layer_num,
        distribution=request.distribution,
        interface=request.interface,
        Lx=request.Lx_um,
        h=request.h_um,
        Nx=request.Nx,
        Nz=request.Nz,
        Lam0=request.wavelengths_m,
        Theta=request.angles_rad,
        n_sup=request.n_superstrate,
        n_sub=request.n_substrate,
        is_periodic=request.is_periodic,
        period_num=request.period_num,
    )
    legacy_device = build_legacy_device(
        request.harmonic_count,
        legacy_grid,
        interface=request.interface,
        interface_params=request.interface_params,
        flat_substrate=request.flat_substrate,
    )
    grid = Grid1D(
        Lam0=legacy_grid.Lam0,
        Theta=legacy_grid.Theta,
        Lx=legacy_grid.Lx,
        erR=legacy_grid.erR,
        urR=legacy_grid.urR,
        erT=legacy_grid.erT,
        urT=legacy_grid.urT,
        erSub=legacy_grid.erSub,
        layer_num=legacy_grid.layer_num,
    )
    device = Device1D(legacy_device.ERC, legacy_device.sub_L)
    metadata = {
        "workflow": "casual_rcwa",
        "matrix_method": request.matrix_method,
        "polarization": request.polarization,
        "interface": request.interface,
        "distribution": request.distribution,
        "harmonic_count": request.harmonic_count,
        "legacy_compatibility": True,
    }
    return _run_rcwa(
        grid,
        device,
        np.asarray(legacy_device.ER, dtype=np.complex128),
        request.matrix_method,
        request.polarization,
        request.harmonic_count,
        request.calc_fresnel,
        metadata,
        cancel_token,
        progress,
    )


def run_imported_rcwa(
    request: ImportedRCWARequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> RCWARun:
    """Run a legacy MAT/NumPy imported device without altering its arrays."""

    _validate_common_request(request)
    grid, device = request.device.solver_inputs(
        request.harmonic_count,
        wavelengths_m=request.wavelengths_m,
        angles_rad=request.angles_rad,
        n_superstrate=request.n_superstrate,
        n_substrate=request.n_substrate,
    )
    metadata = {
        "workflow": "custom_import_rcwa",
        "matrix_method": request.matrix_method,
        "polarization": request.polarization,
        "harmonic_count": request.harmonic_count,
        "source": None if request.device.source is None else str(request.device.source),
        "legacy_compatibility": True,
    }
    return _run_rcwa(
        grid,
        device,
        request.device.ER,
        request.matrix_method,
        request.polarization,
        request.harmonic_count,
        request.calc_fresnel,
        metadata,
        cancel_token,
        progress,
    )


def run_phc_rcwa(
    request: PhCRCWARequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> RCWARun:
    """Run the legacy Casual 2D/PhC S-matrix workflow."""

    wavelengths = np.asarray(request.wavelengths_m, dtype=np.float64).ravel()
    angles = np.asarray(request.angles_rad, dtype=np.float64).ravel()
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_m must contain finite positive values")
    if angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("angles_rad must contain finite values")
    if request.harmonic_count < 1:
        raise ValueError("harmonic_count must be at least one")
    if request.layer_count < 1:
        raise ValueError("layer_count must be at least one")
    if request.polarization not in {"E", "H"}:
        raise ValueError("polarization must be 'E' or 'H'")
    if request.repeat_mode not in {"legacy", "corrected"}:
        raise ValueError("repeat_mode must be 'legacy' or 'corrected'")

    phc_grid = build_phc_grid(
        request.params,
        wavelengths_m=wavelengths,
        angles_rad=angles,
        interface=request.interface,
        Nx=request.Nx,
        Nz=request.Nz,
        n_superstrate=request.n_superstrate,
        n_substrate=request.n_substrate,
    )
    phc_device = build_phc_device(
        request.harmonic_count,
        phc_grid,
        request.interface,
        request.interface_params,
    )
    started = perf_counter()
    transmission, reflection = launch_rcwa_s_phc(
        request.layer_count,
        request.harmonic_count,
        phc_grid,
        phc_device,
        request.polarization,
        request.calc_fresnel,
        repeat_mode=request.repeat_mode,
        cancel_token=cancel_token,
        progress=progress,
    )
    return RCWARun(
        wavelengths_m=phc_grid.Lam0.copy(),
        angles_rad=phc_grid.Theta.copy(),
        transmission=transmission,
        reflection=reflection,
        grid=phc_grid.solver_grid(),
        device=phc_device.unit_device(),
        device_er=np.asarray(phc_device.ER, dtype=np.complex128).copy(),
        elapsed_s=perf_counter() - started,
        metadata={
            "workflow": "casual_phc_rcwa",
            "matrix_method": "S",
            "polarization": request.polarization,
            "interface": request.interface,
            "harmonic_count": request.harmonic_count,
            "layer_count": request.layer_count,
            "repeat_mode": request.repeat_mode,
            "legacy_non_power_of_two_repeat_bug": request.repeat_mode == "legacy",
            "legacy_angle_units": "radians",
            "legacy_compatibility": True,
        },
    )


def run_fdfd(
    request: FDFDRequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> FDFDRun:
    """Run the mainline legacy FDFD field workflow, including raw ``f``."""

    wavelengths = np.asarray(request.wavelengths_um, dtype=np.float64).ravel()
    angles = np.asarray(request.angles_rad, dtype=np.float64).ravel()
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_um must contain finite positive values")
    if angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("angles_rad must contain finite values")
    if request.polarization not in {"E", "H"}:
        raise ValueError("polarization must be 'E' or 'H'")
    if request.interface not in {"sin", "DE1", "DE4", "tri"}:
        raise ValueError("interface must be one of sin, DE1, DE4, tri")
    if request.dispersion_mode not in {"legacy_fdfd", "corrected_um"}:
        raise ValueError("dispersion_mode must be 'legacy_fdfd' or 'corrected_um'")
    grid = build_fdfd_grid(
        request.params,
        layer_num=request.layer_num,
        distribution=request.distribution,
        Lam0=wavelengths,
        Theta=angles,
        Lx=request.Lx_um,
        h=request.h_um,
        n_sup=request.n_superstrate,
        n_sub=request.n_substrate,
        nres=request.nres,
        spacer=request.spacer_um,
        npml=request.npml,
        is_periodic=request.is_periodic,
        period_num=request.period_num,
        refractive_idx=request.refractive_idx,
        dispersion_mode=request.dispersion_mode,
    )
    device = build_fdfd_device(
        request.params,
        grid,
        interface=request.interface,
        interface_params=request.interface_params,
    )
    started = perf_counter()
    result = fdfd_2d(
        grid,
        device,
        request.polarization,
        cancel_token=cancel_token,
        progress=progress,
    )
    return FDFDRun(
        wavelengths_um=grid.Lam0.copy(),
        angles_rad=grid.Theta.copy(),
        result=result,
        grid=grid,
        device=device,
        elapsed_s=perf_counter() - started,
        metadata={
            "workflow": "fdfd_fields",
            "polarization": request.polarization,
            "interface": request.interface,
            "distribution": request.distribution,
            "period_num": request.period_num,
            "refractive_idx": request.refractive_idx,
            "dispersion_mode": request.dispersion_mode,
            "legacy_final_sweep_sum": True,
            "legacy_ref_trn_minus1_alias": True,
        },
    )
def _run_rcwa(
    grid: Grid1D,
    device: Device1D,
    device_er: NDArray[np.complex128],
    matrix_method: MatrixMethod,
    polarization: Polarization,
    harmonic_count: int,
    calc_fresnel: bool,
    metadata: dict[str, object],
    cancel_token: CancellationToken | None,
    progress: ProgressCallback | None,
) -> RCWARun:
    started = perf_counter()
    if matrix_method == "S":
        transmission, reflection = launch_rcwa_s(
            harmonic_count,
            grid,
            device,
            polarization,
            calc_fresnel,
            cancel_token=cancel_token,
            progress=progress,
        )
    elif matrix_method == "T":
        transmission, reflection = launch_rcwa_t(
            harmonic_count,
            grid,
            device,
            polarization,
            calc_fresnel,
            cancel_token=cancel_token,
            progress=progress,
        )
    else:
        raise ValueError("matrix_method must be 'S' or 'T'")
    return RCWARun(
        wavelengths_m=grid.Lam0.copy(),
        angles_rad=grid.Theta.copy(),
        transmission=transmission,
        reflection=reflection,
        grid=grid,
        device=device,
        device_er=np.asarray(device_er, dtype=np.complex128).copy(),
        elapsed_s=perf_counter() - started,
        metadata=metadata,
    )


def _validate_common_request(request: AnalyticRCWARequest | ImportedRCWARequest) -> None:
    wavelengths = np.asarray(request.wavelengths_m, dtype=np.float64).ravel()
    angles = np.asarray(request.angles_rad, dtype=np.float64).ravel()
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_m must contain finite positive values")
    if angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("angles_rad must contain finite values")
    if request.harmonic_count < 1:
        raise ValueError("harmonic_count must be at least one")
    if request.matrix_method not in {"S", "T"}:
        raise ValueError("matrix_method must be 'S' or 'T'")
    if request.polarization not in {"E", "H"}:
        raise ValueError("polarization must be 'E' or 'H'")
