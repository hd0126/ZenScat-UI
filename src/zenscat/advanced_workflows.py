"""Higher-level corrected workflows built from the legacy solver contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import (
    CancellationToken,
    FDFDDevice,
    FDFDGrid,
    FDFDResult,
    PhCInterfaceParams,
    ProgressCallback,
    build_fdfd_grid,
    build_phc_fdfd_device,
    fdfd_2d,
)
from .core.phc import PhCInterface
from .core.phc_fdfd import CompatibilityMode
from .legacy_io import FDFDResultBundle
from .workflows import FDFDRequest, FDFDRun, Polarization, run_fdfd

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DualFDFDRun:
    """Paired E/H FDFD field solve with export helpers for each polarization."""

    electric: FDFDRun
    magnetic: FDFDRun
    elapsed_s: float
    metadata: dict[str, object]

    def run_for(self, polarization: Polarization) -> FDFDRun:
        if polarization == "E":
            return self.electric
        if polarization == "H":
            return self.magnetic
        raise ValueError("polarization must be 'E' or 'H'")

    def fdfd_bundle(
        self,
        polarization: Polarization,
        params: ArrayLike | None = None,
    ) -> FDFDResultBundle:
        return self.run_for(polarization).fdfd_bundle(params=params)

    def fdfd_bundles(self, params: ArrayLike | None = None) -> dict[Polarization, FDFDResultBundle]:
        return {
            "E": self.electric.fdfd_bundle(params=params),
            "H": self.magnetic.fdfd_bundle(params=params),
        }


@dataclass(frozen=True)
class PhCFDFDRequest:
    """Corrected PhC geometry FDFD request.

    ``params`` follows the PhC convention: ``[pitch_um, n_background,
    n_inclusion]``.  ``legacy_exact`` is intentionally rejected because the
    original MATLAB ``Device_FDFD_PhC.m`` branch is incomplete.
    """

    params: ArrayLike
    wavelengths_um: ArrayLike
    angles_rad: ArrayLike
    polarization: Polarization = "E"
    interface: PhCInterface = "PhC_rec_circ"
    interface_params: PhCInterfaceParams = field(default_factory=PhCInterfaceParams)
    Lx_um: float | None = None
    h_um: float | None = None
    n_superstrate: float = 1.0
    n_substrate: float = 1.516
    nres: float = 20
    spacer_um: ArrayLike | None = None
    npml: ArrayLike = (25, 25)
    compatibility: CompatibilityMode = "corrected"


@dataclass(frozen=True)
class PhCFDFDRun:
    """FDFD run produced from a corrected PhC material rasterization."""

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


@dataclass(frozen=True)
class DualPhCFDFDRun:
    """Paired E/H corrected PhC-FDFD solve with per-polarization export helpers."""

    electric: PhCFDFDRun
    magnetic: PhCFDFDRun
    elapsed_s: float
    metadata: dict[str, object]

    def run_for(self, polarization: Polarization) -> PhCFDFDRun:
        if polarization == "E":
            return self.electric
        if polarization == "H":
            return self.magnetic
        raise ValueError("polarization must be 'E' or 'H'")

    def fdfd_bundle(
        self,
        polarization: Polarization,
        params: ArrayLike | None = None,
    ) -> FDFDResultBundle:
        return self.run_for(polarization).fdfd_bundle(params=params)

    def fdfd_bundles(self, params: ArrayLike | None = None) -> dict[Polarization, FDFDResultBundle]:
        return {
            "E": self.electric.fdfd_bundle(params=params),
            "H": self.magnetic.fdfd_bundle(params=params),
        }


def run_dual_fdfd(
    request: FDFDRequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> DualFDFDRun:
    """Run E and H field solves sequentially with a single aggregate progress stream."""

    started = perf_counter()
    progress_state: dict[Polarization, tuple[int, int]] = {}

    def child_progress(polarization: Polarization) -> ProgressCallback:
        def _remap(done: int, total: int) -> None:
            progress_state[polarization] = (done, total)
            if progress is not None:
                offset = 0
                if polarization == "H":
                    offset = progress_state.get("E", (0, total))[1]
                progress(offset + done, total * 2)

        return _remap

    electric = run_fdfd(
        replace(request, polarization="E"),
        cancel_token=cancel_token,
        progress=child_progress("E"),
    )
    magnetic = run_fdfd(
        replace(request, polarization="H"),
        cancel_token=cancel_token,
        progress=child_progress("H"),
    )
    return DualFDFDRun(
        electric=electric,
        magnetic=magnetic,
        elapsed_s=perf_counter() - started,
        metadata={
            "workflow": "dual_fdfd_fields",
            "polarizations": ("E", "H"),
            "child_elapsed_s": {"E": electric.elapsed_s, "H": magnetic.elapsed_s},
            "child_workflows": {
                "E": _metadata_dict(electric.metadata),
                "H": _metadata_dict(magnetic.metadata),
            },
        },
    )


def run_dual_phc_fdfd(
    request: PhCFDFDRequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> DualPhCFDFDRun:
    """Run corrected PhC-FDFD E and H solves with one aggregate progress stream."""

    started = perf_counter()
    progress_state: dict[Polarization, tuple[int, int]] = {}

    def child_progress(polarization: Polarization) -> ProgressCallback:
        def _remap(done: int, total: int) -> None:
            progress_state[polarization] = (done, total)
            if progress is not None:
                offset = 0
                if polarization == "H":
                    offset = progress_state.get("E", (0, total))[1]
                progress(offset + done, total * 2)

        return _remap

    electric = run_phc_fdfd(
        replace(request, polarization="E"),
        cancel_token=cancel_token,
        progress=child_progress("E"),
    )
    magnetic = run_phc_fdfd(
        replace(request, polarization="H"),
        cancel_token=cancel_token,
        progress=child_progress("H"),
    )
    return DualPhCFDFDRun(
        electric=electric,
        magnetic=magnetic,
        elapsed_s=perf_counter() - started,
        metadata={
            "workflow": "dual_phc_fdfd_fields",
            "polarizations": ("E", "H"),
            "child_elapsed_s": {"E": electric.elapsed_s, "H": magnetic.elapsed_s},
            "child_workflows": {
                "E": _metadata_dict(electric.metadata),
                "H": _metadata_dict(magnetic.metadata),
            },
        },
    )


def run_phc_fdfd(
    request: PhCFDFDRequest,
    *,
    cancel_token: CancellationToken | None = None,
    progress: ProgressCallback | None = None,
) -> PhCFDFDRun:
    """Rasterize a corrected PhC interface and run the existing 2D FDFD solver."""

    values = np.asarray(request.params, dtype=np.float64).ravel()
    if values.size < 3:
        raise ValueError("PhC FDFD params must be [pitch_um, n_background, n_inclusion]")
    if request.polarization not in {"E", "H"}:
        raise ValueError("polarization must be 'E' or 'H'")
    pitch_um = float(values[0])
    background_n = float(values[1])
    if pitch_um <= 0:
        raise ValueError("PhC pitch must be positive")
    if background_n <= 0:
        raise ValueError("PhC background refractive index must be positive")

    height_um = pitch_um if request.h_um is None else float(request.h_um)
    if height_um <= 0:
        raise ValueError("h_um must be positive")

    grid = build_fdfd_grid(
        [height_um, background_n],
        layer_num=1,
        Lam0=request.wavelengths_um,
        Theta=request.angles_rad,
        Lx=pitch_um if request.Lx_um is None else request.Lx_um,
        h=height_um,
        n_sup=request.n_superstrate,
        n_sub=request.n_substrate,
        nres=request.nres,
        spacer=request.spacer_um,
        npml=request.npml,
        dispersion_mode="corrected_um",
    )
    phc_device = build_phc_fdfd_device(
        values,
        grid,
        request.interface,
        request.interface_params,
        compatibility=request.compatibility,
    )
    device = FDFDDevice(ER2=phc_device.ER2, UR2=phc_device.UR2)
    started = perf_counter()
    result = fdfd_2d(
        grid,
        device,
        request.polarization,
        cancel_token=cancel_token,
        progress=progress,
    )
    return PhCFDFDRun(
        wavelengths_um=grid.Lam0.copy(),
        angles_rad=grid.Theta.copy(),
        result=result,
        grid=grid,
        device=device,
        elapsed_s=perf_counter() - started,
        metadata={
            "workflow": "phc_fdfd_fields",
            "polarization": request.polarization,
            "interface": request.interface,
            "compatibility": request.compatibility,
            "dispersion_mode": "corrected_um",
            "phc": _metadata_dict(phc_device.metadata),
            "legacy_exact": False,
        },
    )


def _metadata_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return dict(metadata)
