"""Compatibility I/O for the files produced and consumed by MATLAB ZenScat."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.io import loadmat, savemat

from .core import Device1D, DiffractionResult, FDFDResult, Grid1D, convmat1d

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
RESULT_FIELDS = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}


@dataclass(frozen=True)
class ImportedDevice:
    """Legacy ``DIF``/``RCWA_DATA`` payload, with geometry lengths in micrometers."""

    ER: ComplexArray
    sub_L_um: FloatArray
    x_um: FloatArray
    Lx_um: float
    source: Path | None = None

    def __post_init__(self) -> None:
        er = np.asarray(self.ER, dtype=np.complex128)
        if er.ndim == 1:
            er = er[np.newaxis, :]
        sub_l = np.asarray(self.sub_L_um, dtype=np.float64).ravel()
        x = np.asarray(self.x_um, dtype=np.float64).ravel()
        if er.ndim != 2 or er.shape[0] == 0 or er.shape[1] == 0:
            raise ValueError("ER must be a non-empty two-dimensional array")
        if sub_l.size != er.shape[0]:
            raise ValueError("sub_L length must match the number of ER layers")
        if x.size != er.shape[1]:
            raise ValueError("x length must match the number of ER samples")
        if not np.isfinite(er.real).all() or not np.isfinite(er.imag).all():
            raise ValueError("ER must contain only finite values")
        if not np.isfinite(sub_l).all() or np.any(sub_l < 0):
            raise ValueError("sub_L must contain finite, non-negative values")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not np.isfinite(self.Lx_um) or self.Lx_um <= 0:
            raise ValueError("Lx must be finite and positive")
        object.__setattr__(self, "ER", er)
        object.__setattr__(self, "sub_L_um", sub_l)
        object.__setattr__(self, "x_um", x)
        object.__setattr__(self, "Lx_um", float(self.Lx_um))

    def solver_inputs(
        self,
        harmonic_count: int,
        *,
        wavelengths_m: ArrayLike,
        angles_rad: ArrayLike,
        n_superstrate: complex = 1.0,
        n_substrate: complex = 1.0,
    ) -> tuple[Grid1D, Device1D]:
        """Lower this imported device exactly as legacy ``Device_IMPORT.m`` does."""

        grid = Grid1D(
            Lam0=np.asarray(wavelengths_m, dtype=np.float64).ravel(),
            Theta=np.asarray(angles_rad, dtype=np.float64).ravel(),
            Lx=self.Lx_um,
            erR=n_superstrate**2,
            urR=1.0,
            erT=n_substrate**2,
            urT=1.0,
            erSub=n_substrate**2,
            layer_num=1,
        )
        device = Device1D(
            ERC=convmat1d(self.ER, harmonic_count),
            sub_L=self.sub_L_um * 1e-6,
        )
        return grid, device


@dataclass(frozen=True)
class LegacyResultBundle:
    """Public result contract shared by legacy Casual and Import workflows."""

    wavelengths_m: FloatArray
    angles_rad: FloatArray
    transmission: DiffractionResult
    reflection: DiffractionResult
    params: FloatArray | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FDFDResultBundle:
    """Python FDFD export contract with explicit micrometre wavelength units."""

    wavelengths_um: FloatArray
    angles_rad: FloatArray
    result: FDFDResult
    er2: FloatArray
    params: FloatArray | None = None
    metadata: Mapping[str, Any] | None = None


def load_imported_device(path: str | Path, *, trusted_pickle: bool = False) -> ImportedDevice:
    """Load a legacy MAT device or the bundled pickled NumPy dictionary.

    NumPy ``.npy`` dictionaries require ``trusted_pickle=True`` because loading
    arbitrary pickle data is unsafe. MATLAB files never require that opt-in.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffixes = source.suffixes
    if source.suffix.lower() == ".mat":
        payload: Any = loadmat(source, squeeze_me=True, struct_as_record=False)
    elif source.suffix.lower() == ".npy" or suffixes[-2:] == [".py", ".npy"]:
        if not trusted_pickle:
            raise ValueError("loading legacy NumPy dictionaries requires trusted_pickle=True")
        loaded = np.load(source, allow_pickle=True)
        payload = loaded.item() if isinstance(loaded, np.ndarray) and loaded.shape == () else loaded
    else:
        raise ValueError(f"unsupported legacy device format: {source.suffix or '<none>'}")

    data = _unwrap_device_payload(payload)
    return ImportedDevice(
        ER=_field(data, "ER"),
        sub_L_um=_field(data, "sub_L"),
        x_um=_field(data, "x"),
        Lx_um=float(np.asarray(_field(data, "Lx")).item()),
        source=source,
    )


def save_legacy_result_bundle(
    directory: str | Path,
    bundle: LegacyResultBundle,
    *,
    overwrite: bool = False,
) -> Path:
    """Write MATLAB-compatible result files plus a reproducibility manifest."""

    destination = Path(directory).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"result directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    wavelengths = np.asarray(bundle.wavelengths_m, dtype=np.float64).ravel()
    angles = np.asarray(bundle.angles_rad, dtype=np.float64).ravel()
    expected_shape = (wavelengths.size, angles.size)
    trn = _result_dict(bundle.transmission, "TRN", expected_shape)
    ref = _result_dict(bundle.reflection, "REF", expected_shape)

    savemat(destination / "TRN.mat", {"TRN": trn}, do_compression=True)
    savemat(destination / "REF.mat", {"REF": ref}, do_compression=True)
    savemat(destination / "Lam.mat", {"Lam0": wavelengths}, do_compression=True)
    savemat(destination / "Theta.mat", {"Theta": angles}, do_compression=True)

    files = ["TRN.mat", "REF.mat", "Lam.mat", "Theta.mat"]
    params = None
    if bundle.params is not None:
        params = np.asarray(bundle.params, dtype=np.float64).ravel()
        savemat(destination / "Output.mat", {"Output": params}, do_compression=True)
        savemat(destination / "Params.mat", {"Params": params}, do_compression=True)
        files.extend(("Output.mat", "Params.mat"))

    manifest = {
        "schema": "zenscat.result-bundle",
        "schema_version": "1.0",
        "units": {"wavelength": "m", "angle": "rad"},
        "shape": list(expected_shape),
        "files": files,
        "checksums": {
            "Lam0": _sha256_numeric(wavelengths),
            "Theta": _sha256_numeric(angles),
            **{f"TRN.{name}": _sha256_numeric(value) for name, value in trn.items()},
            **{f"REF.{name}": _sha256_numeric(value) for name, value in ref.items()},
        },
        "metadata": _json_safe(dict(bundle.metadata or {})),
    }
    if params is not None:
        manifest["checksums"]["Params"] = _sha256_numeric(params)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def save_fdfd_result_bundle(
    directory: str | Path,
    bundle: FDFDResultBundle,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic Python FDFD result bundle.

    This is intentionally not labeled as MATLAB legacy output: the original
    FDFD plot path did not define a durable file schema. Wavelengths are saved
    in micrometres and the raw final field is preserved as ``Field.mat``.
    """

    destination = Path(directory).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"result directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    wavelengths = np.asarray(bundle.wavelengths_um, dtype=np.float64).ravel()
    angles = np.asarray(bundle.angles_rad, dtype=np.float64).ravel()
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("wavelengths_um must contain finite positive values")
    if angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("angles_rad must contain finite values")
    expected_shape = (wavelengths.size, angles.size)
    trn = _fdfd_result_dict(bundle.result.TRN, "TRN", expected_shape)
    ref = _fdfd_result_dict(bundle.result.REF, "REF", expected_shape)
    field = np.asarray(bundle.result.f, dtype=np.complex128)
    er2 = np.asarray(bundle.er2, dtype=np.float64)
    if field.ndim != 2 or field.size == 0:
        raise ValueError("FDFD field must be a non-empty two-dimensional array")
    if er2.ndim != 2 or er2.size == 0 or not np.isfinite(er2).all():
        raise ValueError("ER2 must be a finite non-empty two-dimensional array")

    savemat(destination / "TRN.mat", {"TRN": trn}, do_compression=True)
    savemat(destination / "REF.mat", {"REF": ref}, do_compression=True)
    savemat(destination / "Lam.mat", {"Lam0": wavelengths}, do_compression=True)
    savemat(destination / "Theta.mat", {"Theta": angles}, do_compression=True)
    savemat(destination / "Field.mat", {"f": field}, do_compression=True)
    savemat(destination / "ER2.mat", {"ER2": er2}, do_compression=True)

    files = ["TRN.mat", "REF.mat", "Lam.mat", "Theta.mat", "Field.mat", "ER2.mat"]
    params = None
    if bundle.params is not None:
        params = np.asarray(bundle.params, dtype=np.float64).ravel()
        savemat(destination / "Params.mat", {"Params": params}, do_compression=True)
        files.append("Params.mat")

    manifest = {
        "schema": "zenscat.fdfd-result-bundle",
        "schema_version": "1.0",
        "units": {"wavelength": "um", "angle": "rad", "field": "raw_solver_units"},
        "shape": list(expected_shape),
        "field_shape": list(field.shape),
        "er2_shape": list(er2.shape),
        "files": files,
        "checksums": {
            "Lam0": _sha256_numeric(wavelengths),
            "Theta": _sha256_numeric(angles),
            "Field.f": _sha256_numeric(field),
            "ER2": _sha256_numeric(er2),
            **{f"TRN.{name}": _sha256_numeric(value) for name, value in trn.items()},
            **{f"REF.{name}": _sha256_numeric(value) for name, value in ref.items()},
        },
        "metadata": _json_safe(dict(bundle.metadata or {})),
    }
    if params is not None:
        manifest["checksums"]["Params"] = _sha256_numeric(params)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_legacy_result_bundle(directory: str | Path) -> LegacyResultBundle:
    """Read a directory written by MATLAB ZenScat or ``save_legacy_result_bundle``."""

    source = Path(directory).expanduser().resolve()
    wavelengths = _load_named_mat(source / "Lam.mat", "Lam0")
    angles = _load_named_mat(source / "Theta.mat", "Theta")
    expected_shape = (np.asarray(wavelengths).size, np.asarray(angles).size)
    trn = _load_result(source / "TRN.mat", "TRN", expected_shape)
    ref = _load_result(source / "REF.mat", "REF", expected_shape)
    params = None
    for filename, variable in (("Output.mat", "Output"), ("Params.mat", "Params")):
        candidate = source / filename
        if candidate.is_file():
            params = np.asarray(_load_named_mat(candidate, variable), dtype=np.float64).ravel()
            break
    metadata = None
    manifest_path = source / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest.get("metadata")
    return LegacyResultBundle(
        wavelengths_m=np.asarray(wavelengths, dtype=np.float64).ravel(),
        angles_rad=np.asarray(angles, dtype=np.float64).ravel(),
        transmission=trn,
        reflection=ref,
        params=params,
        metadata=metadata,
    )


def _unwrap_device_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        public = {key: value for key, value in payload.items() if not str(key).startswith("__")}
        if all(name in public for name in ("ER", "sub_L", "x", "Lx")):
            return public
        for wrapper in ("DIF", "DEVICE", "device"):
            if wrapper in public:
                return public[wrapper]
    if all(hasattr(payload, name) for name in ("ER", "sub_L", "x", "Lx")):
        return payload
    raise ValueError("device payload must provide ER, sub_L, x, and Lx")


def _field(payload: Any, name: str) -> Any:
    if isinstance(payload, Mapping):
        if name not in payload:
            raise ValueError(f"legacy device is missing {name}")
        return payload[name]
    if hasattr(payload, name):
        return getattr(payload, name)
    raise ValueError(f"legacy device is missing {name}")


def _result_dict(result: DiffractionResult, kind: str, expected_shape: tuple[int, int]) -> dict[str, FloatArray]:
    fields: dict[str, FloatArray] = {}
    for name in RESULT_FIELDS[kind]:
        value = getattr(result, name)
        if value is None:
            raise ValueError(f"{kind}.{name} is required for legacy export")
        array = np.asarray(value, dtype=np.float64)
        if array.shape != expected_shape:
            raise ValueError(f"{kind}.{name} must have shape {expected_shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{kind}.{name} contains non-finite values")
        fields[name] = array
    return fields


def _fdfd_result_dict(
    result: Mapping[str, ArrayLike | float],
    kind: str,
    expected_shape: tuple[int, int],
) -> dict[str, FloatArray | float]:
    fields: dict[str, FloatArray | float] = {}
    for name, value in result.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape == ():
            scalar = float(array)
            if not np.isfinite(scalar):
                raise ValueError(f"{kind}.{name} contains a non-finite scalar")
            fields[name] = scalar
            continue
        if array.shape != expected_shape:
            raise ValueError(f"{kind}.{name} must have shape {expected_shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{kind}.{name} contains non-finite values")
        fields[name] = array
    return fields


def _load_named_mat(path: Path, variable: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    if variable not in payload:
        raise ValueError(f"{path.name} is missing variable {variable}")
    return payload[variable]


def _load_result(path: Path, kind: str, expected_shape: tuple[int, int]) -> DiffractionResult:
    payload = _load_named_mat(path, kind)
    values: dict[str, FloatArray] = {}
    for name in RESULT_FIELDS[kind]:
        value = np.asarray(_field(payload, name), dtype=np.float64)
        if value.size != expected_shape[0] * expected_shape[1]:
            raise ValueError(
                f"{path.name} field {kind}.{name} has {value.size} values; "
                f"expected {expected_shape[0] * expected_shape[1]}"
            )
        values[name] = value.reshape(expected_shape)
    if kind == "TRN":
        return DiffractionResult(**values)
    return DiffractionResult(**values)


def _sha256_numeric(value: ArrayLike) -> str:
    array = np.asarray(value, order="F")
    if np.iscomplexobj(array):
        complex_array = np.asarray(value, dtype=np.complex128, order="F")
        payload = np.concatenate(
            (
                np.asarray(complex_array.real, dtype="<f8", order="F").ravel(order="F"),
                np.asarray(complex_array.imag, dtype="<f8", order="F").ravel(order="F"),
            )
        )
    else:
        payload = np.asarray(array, dtype="<f8", order="F").ravel(order="F")
    return sha256(payload.astype("<f8", copy=False).tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"metadata value is not JSON serializable: {type(value).__name__}")
