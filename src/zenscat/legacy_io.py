"""Compatibility I/O for the files produced and consumed by MATLAB ZenScat."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

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
CompatibilityMode = Literal["legacy_exact", "corrected", "modern"]
COMPATIBILITY_MODES = {"legacy_exact", "corrected", "modern"}
DEVICE_FIELD_ALIASES = {
    "ER": ("ER", "er", "EPS", "eps", "epsilon", "permittivity"),
    "sub_L": ("sub_L", "subL", "sub_l", "thickness", "thickness_um", "layer_thickness"),
    "x": ("x", "X", "x_um", "xgrid", "x_grid"),
    "Lx": ("Lx", "lx", "period", "Period", "period_um", "Lx_um"),
}
RESULT_FIELD_ALIASES = {
    "TRN": {
        "minus_2": ("minus_2", "TRN_minus2", "TRN_minus_2", "minus2", "m2"),
        "minus_1": ("minus_1", "TRN_minus1", "TRN_minus_1", "minus1", "m1"),
        "plus_1": ("plus_1", "TRN_plus1", "TRN_plus_1", "plus1", "p1"),
        "plus_2": ("plus_2", "TRN_plus2", "TRN_plus_2", "plus2", "p2"),
        "TRN0": ("TRN0", "zero", "zeroth", "TRN_0", "T0"),
        "sum": ("sum", "TRN_sum", "total", "Total", "Tsum"),
    },
    "REF": {
        "minus_2": ("minus_2", "REF_minus2", "REF_minus_2", "minus2", "m2"),
        "minus_1": ("minus_1", "REF_minus1", "REF_minus_1", "minus1", "m1"),
        "plus_1": ("plus_1", "REF_plus1", "REF_plus_1", "plus1", "p1"),
        "plus_2": ("plus_2", "REF_plus2", "REF_plus_2", "plus2", "p2"),
        "REF0": ("REF0", "zero", "zeroth", "REF_0", "R0"),
        "sum": ("sum", "REF_sum", "total", "Total", "Rsum"),
    },
}
AXIS_ALIASES = {
    "Lam0": ("Lam0", "Lam", "lambda", "Lambda", "wavelengths", "wavelength_m"),
    "Theta": ("Theta", "theta", "angles", "angle", "Theta_rad"),
}
PARAM_ALIASES = {
    "Output": ("Output", "output", "Params", "params", "parameters"),
    "Params": ("Params", "params", "Output", "output", "parameters"),
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
    compatibility_mode: CompatibilityMode | None = None,
) -> Path:
    """Write MATLAB-compatible result files plus a reproducibility manifest."""

    destination = Path(directory).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"result directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    _validate_compatibility_mode(compatibility_mode)

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
    if compatibility_mode is not None:
        files.extend(_write_rcwa_compatibility_artifacts(destination, wavelengths, angles, trn, ref))

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
    if compatibility_mode is not None:
        manifest["compatibility_mode"] = compatibility_mode
        manifest["artifact_contract"] = {
            "Data.txt": "flat legacy-compatible table with wavelength_nm, theta_deg, selected orders, sums, energy_error",
            "RCWA_plot_data.csv": "CSV table suitable for 1D line plots or 2D heatmaps",
            "RCWA_axes.csv": "axis vectors in legacy display units",
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
    compatibility_mode: CompatibilityMode | None = None,
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
    _validate_compatibility_mode(compatibility_mode)

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
    if compatibility_mode is not None:
        files.extend(_write_fdfd_compatibility_artifacts(destination, wavelengths, angles, trn, ref, field, er2))

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
    if compatibility_mode is not None:
        manifest["compatibility_mode"] = compatibility_mode
        manifest["artifact_contract"] = {
            "FDFD_metrics.csv": "flat wavelength/angle metrics table with TRN, REF, and energy_error columns",
            "Field_real.csv": "raw final field real component",
            "Field_imag.csv": "raw final field imaginary component",
            "Field_abs.csv": "raw final field magnitude",
            "ER2.csv": "permittivity map used for contour overlays",
        }
    if params is not None:
        manifest["checksums"]["Params"] = _sha256_numeric(params)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_legacy_result_bundle(directory: str | Path, *, strict: bool = True) -> LegacyResultBundle:
    """Read a directory written by MATLAB ZenScat or ``save_legacy_result_bundle``."""

    source = Path(directory).expanduser().resolve()
    wavelengths = _load_named_mat(source / "Lam.mat", AXIS_ALIASES["Lam0"])
    angles = _load_named_mat(source / "Theta.mat", AXIS_ALIASES["Theta"])
    expected_shape = (np.asarray(wavelengths).size, np.asarray(angles).size)
    trn = _load_result(source / "TRN.mat", "TRN", expected_shape, strict=strict)
    ref = _load_result(source / "REF.mat", "REF", expected_shape, strict=strict)
    params = None
    for filename, variable in (("Output.mat", "Output"), ("Params.mat", "Params")):
        candidate = source / filename
        if candidate.is_file():
            params = np.asarray(_load_named_mat(candidate, PARAM_ALIASES[variable]), dtype=np.float64).ravel()
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
        if all(_has_field(public, name) for name in ("ER", "sub_L", "x", "Lx")):
            return public
        for wrapper in ("DIF", "DEVICE", "device"):
            if wrapper in public:
                return public[wrapper]
    if all(_has_field(payload, name) for name in ("ER", "sub_L", "x", "Lx")):
        return payload
    raise ValueError("device payload must provide ER, sub_L, x, and Lx")


def _field(payload: Any, name: str) -> Any:
    aliases = DEVICE_FIELD_ALIASES.get(name, (name,))
    return _field_with_aliases(payload, aliases, f"legacy device is missing {name}")


def _has_field(payload: Any, name: str) -> bool:
    aliases = DEVICE_FIELD_ALIASES.get(name, (name,))
    if isinstance(payload, Mapping):
        return any(alias in payload for alias in aliases)
    return any(hasattr(payload, alias) for alias in aliases)


def _field_with_aliases(payload: Any, aliases: tuple[str, ...], missing_message: str) -> Any:
    if isinstance(payload, Mapping):
        for alias in aliases:
            if alias in payload:
                return payload[alias]
    else:
        for alias in aliases:
            if hasattr(payload, alias):
                return getattr(payload, alias)
    raise ValueError(missing_message)


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
    for name in ("minus_2", "plus_2"):
        value = getattr(result, name, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape != expected_shape:
            raise ValueError(
                f"{kind}.{name} must have shape {expected_shape}, got {array.shape}"
            )
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


def _validate_compatibility_mode(mode: CompatibilityMode | None) -> None:
    if mode is not None and mode not in COMPATIBILITY_MODES:
        raise ValueError(f"unsupported compatibility_mode: {mode}")


def _write_rcwa_compatibility_artifacts(
    destination: Path,
    wavelengths_m: FloatArray,
    angles_rad: FloatArray,
    trn: Mapping[str, FloatArray],
    ref: Mapping[str, FloatArray],
) -> list[str]:
    fields = {
        "TRN_minus_1": trn["minus_1"],
        "TRN_0": trn["TRN0"],
        "TRN_plus_1": trn["plus_1"],
        "TRN_sum": trn["sum"],
        "REF_minus_1": ref["minus_1"],
        "REF_0": ref["REF0"],
        "REF_plus_1": ref["plus_1"],
        "REF_sum": ref["sum"],
        "energy_error": 1.0 - (trn["sum"] + ref["sum"]),
    }
    if "minus_2" in trn and "plus_2" in trn:
        fields["TRN_minus_2"] = trn["minus_2"]
        fields["TRN_plus_2"] = trn["plus_2"]
    if "minus_2" in ref and "plus_2" in ref:
        fields["REF_minus_2"] = ref["minus_2"]
        fields["REF_plus_2"] = ref["plus_2"]
    rows = _grid_rows(
        wavelengths_m * 1e9,
        np.rad2deg(angles_rad),
        "wavelength_nm",
        fields,
    )
    header = list(rows)
    table = np.column_stack([rows[name] for name in header])
    _write_csv(destination / "Data.txt", header, table)
    _write_csv(destination / "RCWA_plot_data.csv", header, table)
    axis_len = max(wavelengths_m.size, angles_rad.size)
    axes = np.full((axis_len, 2), np.nan, dtype=np.float64)
    axes[: wavelengths_m.size, 0] = wavelengths_m * 1e9
    axes[: angles_rad.size, 1] = np.rad2deg(angles_rad)
    _write_csv(destination / "RCWA_axes.csv", ["wavelength_nm", "theta_deg"], axes)
    return ["Data.txt", "RCWA_plot_data.csv", "RCWA_axes.csv"]


def _write_fdfd_compatibility_artifacts(
    destination: Path,
    wavelengths_um: FloatArray,
    angles_rad: FloatArray,
    trn: Mapping[str, FloatArray | float],
    ref: Mapping[str, FloatArray | float],
    field: ComplexArray,
    er2: FloatArray,
) -> list[str]:
    expected_shape = (wavelengths_um.size, angles_rad.size)
    metric_arrays: dict[str, FloatArray] = {}
    for prefix, fields in (("TRN", trn), ("REF", ref)):
        for name, value in fields.items():
            metric_arrays[f"{prefix}_{name}"] = _broadcast_metric(value, expected_shape)
    trn_energy_key = "TRN_sum_grid" if "TRN_sum_grid" in metric_arrays else "TRN_sum"
    ref_energy_key = "REF_sum_grid" if "REF_sum_grid" in metric_arrays else "REF_sum"
    if trn_energy_key in metric_arrays and ref_energy_key in metric_arrays:
        metric_arrays["energy_error"] = 1.0 - (
            metric_arrays[trn_energy_key] + metric_arrays[ref_energy_key]
        )
    rows = _grid_rows(wavelengths_um, np.rad2deg(angles_rad), "wavelength_um", metric_arrays)
    header = list(rows)
    table = np.column_stack([rows[name] for name in header])
    _write_csv(destination / "FDFD_metrics.csv", header, table)
    np.savetxt(destination / "Field_real.csv", field.real, delimiter=",")
    np.savetxt(destination / "Field_imag.csv", field.imag, delimiter=",")
    np.savetxt(destination / "Field_abs.csv", np.abs(field), delimiter=",")
    np.savetxt(destination / "ER2.csv", er2, delimiter=",")
    return ["FDFD_metrics.csv", "Field_real.csv", "Field_imag.csv", "Field_abs.csv", "ER2.csv"]


def _grid_rows(
    wavelengths_display: FloatArray,
    angles_display: FloatArray,
    wavelength_column: str,
    fields: Mapping[str, ArrayLike],
) -> dict[str, FloatArray]:
    wave_grid, angle_grid = np.meshgrid(wavelengths_display, angles_display, indexing="ij")
    rows: dict[str, FloatArray] = {
        wavelength_column: wave_grid.ravel(),
        "theta_deg": angle_grid.ravel(),
    }
    for name, value in fields.items():
        rows[name] = np.asarray(value, dtype=np.float64).ravel()
    return rows


def _broadcast_metric(value: ArrayLike | float, expected_shape: tuple[int, int]) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return np.full(expected_shape, float(array), dtype=np.float64)
    return array.reshape(expected_shape)


def _write_csv(path: Path, header: list[str], table: FloatArray) -> None:
    np.savetxt(path, table, delimiter=",", header=",".join(header), comments="")


def _load_named_mat(path: Path, variable: str | tuple[str, ...]) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    variables = (variable,) if isinstance(variable, str) else variable
    for name in variables:
        if name in payload:
            return payload[name]
    raise ValueError(f"{path.name} is missing variable {'/'.join(variables)}")


def _load_result(path: Path, kind: str, expected_shape: tuple[int, int], *, strict: bool) -> DiffractionResult:
    payload = _load_named_mat(path, kind)
    values: dict[str, FloatArray] = {}
    for name in RESULT_FIELDS[kind]:
        aliases = RESULT_FIELD_ALIASES[kind][name]
        try:
            value = np.asarray(
                _field_with_aliases(payload, aliases, f"{path.name} is missing {kind}.{name}"),
                dtype=np.float64,
            )
        except ValueError:
            if strict:
                raise
            value = _compat_missing_result_field(values, name, expected_shape)
        if value.size != expected_shape[0] * expected_shape[1]:
            raise ValueError(
                f"{path.name} field {kind}.{name} has {value.size} values; "
                f"expected {expected_shape[0] * expected_shape[1]}"
            )
        values[name] = value.reshape(expected_shape)
    for name in ("minus_2", "plus_2"):
        aliases = RESULT_FIELD_ALIASES[kind][name]
        try:
            value = np.asarray(
                _field_with_aliases(payload, aliases, "optional field missing"),
                dtype=np.float64,
            )
        except ValueError:
            continue
        if value.size != expected_shape[0] * expected_shape[1]:
            raise ValueError(
                f"{path.name} field {kind}.{name} has {value.size} values; "
                f"expected {expected_shape[0] * expected_shape[1]}"
            )
        values[name] = value.reshape(expected_shape)
    if kind == "TRN":
        return DiffractionResult(**values)
    return DiffractionResult(**values)


def _compat_missing_result_field(
    values: Mapping[str, FloatArray],
    name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    if name in ("minus_1", "plus_1"):
        return np.zeros(expected_shape, dtype=np.float64)
    if name in ("TRN0", "REF0"):
        if "sum" in values:
            return values["sum"]
        return np.zeros(expected_shape, dtype=np.float64)
    if name == "sum":
        for zero_field in ("TRN0", "REF0"):
            if zero_field in values:
                return values[zero_field]
        return np.zeros(expected_shape, dtype=np.float64)
    return np.zeros(expected_shape, dtype=np.float64)


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
