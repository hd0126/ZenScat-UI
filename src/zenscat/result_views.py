"""Presentation-ready result views for RCWA and FDFD outputs.

The functions in this module are intentionally GUI-agnostic.  They turn
legacy-compatible solver objects into validated matrices, slice descriptors,
and flat table columns that desktop widgets and exporters can share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

ResultFamily = Literal["Transmission", "Reflection", "Energy", "Energy error"]
ResultOrder = Literal["-2", "-1", "0", "+1", "+2", "sum"]
FieldTransform = Literal["abs", "real", "imag", "phase", "log10abs"]
WavelengthUnit = Literal["m", "um", "nm"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_POSITIONS: tuple[str, str, str] = ("start", "mid", "end")
_ORDERS: tuple[ResultOrder, ...] = ("-1", "0", "+1", "sum")


@dataclass(frozen=True)
class AxisDescriptor:
    """Numeric axis plus a plot/table label."""

    label: str
    values: FloatArray
    unit: str


@dataclass(frozen=True)
class ResultSlice:
    """One-dimensional slice through a two-dimensional result matrix."""

    position: str
    index: int
    fixed_label: str
    fixed_value: float
    fixed_unit: str
    x: AxisDescriptor
    values: FloatArray
    label: str


@dataclass(frozen=True)
class LegacyWavelengthSlice:
    """Legacy MATLAB-style TRN0/REF0 slice at one wavelength."""

    position: str
    index: int
    wavelength_nm: float
    x: AxisDescriptor
    series: Mapping[str, FloatArray]
    series_labels: tuple[str, ...]


@dataclass(frozen=True)
class ResultView:
    """Validated matrix and common plot descriptors for a result selection."""

    kind: Literal["series", "heatmap"]
    family: str
    order: str
    label: str
    values: FloatArray
    x: AxisDescriptor
    y: AxisDescriptor
    wavelength_slices: tuple[ResultSlice, ...]
    angle_slices: tuple[ResultSlice, ...]


@dataclass(frozen=True)
class ResultTable:
    """CSV-ready flattened result columns at wavelength-angle grain."""

    columns: Mapping[str, FloatArray]
    column_names: tuple[str, ...]
    row_count: int


@dataclass(frozen=True)
class FieldView:
    """Transformed FDFD field with optional ER2 contour helpers."""

    transform: str
    values: FloatArray
    label: str
    contour_mask: BoolArray | None = None
    edge_mask: BoolArray | None = None


def build_result_view(
    transmission: object,
    reflection: object,
    *,
    wavelengths: ArrayLike,
    angles: ArrayLike,
    family: ResultFamily,
    order: ResultOrder,
    wavelength_unit: WavelengthUnit,
) -> ResultView:
    """Build a validated 2D/1D descriptor for one result family and order."""

    wavelength_nm = _wavelengths_nm(wavelengths, wavelength_unit)
    theta_deg = _as_1d_float(angles, "angles")
    theta_deg = np.rad2deg(theta_deg)
    values = result_matrix(transmission, reflection, family=family, order=order)
    _validate_matrix_axes(values, wavelength_nm, theta_deg)

    x = AxisDescriptor("angle", theta_deg, "deg")
    y = AxisDescriptor("wavelength", wavelength_nm, "nm")
    kind: Literal["series", "heatmap"] = "heatmap" if values.shape[0] > 1 and values.shape[1] > 1 else "series"
    return ResultView(
        kind=kind,
        family=family,
        order=order,
        label=f"{family} {order}",
        values=values,
        x=x,
        y=y,
        wavelength_slices=_wavelength_slices(values, x, wavelength_nm, "nm", f"{family} {order}"),
        angle_slices=_angle_slices(values, y, theta_deg, "deg", f"{family} {order}"),
    )


def result_matrix(
    transmission: object,
    reflection: object,
    *,
    family: ResultFamily,
    order: ResultOrder,
) -> FloatArray:
    """Return selected Transmission/Reflection/Energy/Energy error matrix."""

    trn = _result_array(transmission, "TRN", order)
    ref = _result_array(reflection, "REF", order)
    if trn.shape != ref.shape:
        raise ValueError(f"Transmission and reflection shapes differ: {trn.shape} != {ref.shape}")
    if family == "Transmission":
        return trn
    if family == "Reflection":
        return ref
    energy = trn + ref
    if family == "Energy":
        return energy
    if family == "Energy error":
        return 1.0 - energy
    raise ValueError(f"unknown result family: {family}")


def result_table_columns(
    transmission: object,
    reflection: object,
    wavelengths: ArrayLike,
    angles: ArrayLike,
    *,
    wavelength_unit: WavelengthUnit,
    orders: Sequence[ResultOrder] = _ORDERS,
) -> ResultTable:
    """Flatten all selected orders into deterministic CSV-ready columns."""

    wavelength_nm = _wavelengths_nm(wavelengths, wavelength_unit)
    theta_deg = np.rad2deg(_as_1d_float(angles, "angles"))
    if not orders:
        raise ValueError("orders must contain at least one order")

    wave_grid = np.repeat(wavelength_nm, theta_deg.size)
    angle_grid = np.tile(theta_deg, wavelength_nm.size)
    columns: dict[str, FloatArray] = {
        "wavelength_nm": wave_grid.astype(np.float64, copy=False),
        "theta_deg": angle_grid.astype(np.float64, copy=False),
    }
    column_names = ["wavelength_nm", "theta_deg"]

    for order in orders:
        trn = _result_array(transmission, "TRN", order)
        ref = _result_array(reflection, "REF", order)
        _validate_axis_shape(trn, wavelength_nm, theta_deg, f"TRN {order}")
        _validate_axis_shape(ref, wavelength_nm, theta_deg, f"REF {order}")
        trn_name = _column_name("TRN", order)
        ref_name = _column_name("REF", order)
        energy_name = _column_name("energy", order)
        error_name = _column_name("energy_error", order)
        energy = trn + ref
        columns[trn_name] = trn.ravel()
        columns[ref_name] = ref.ravel()
        columns[energy_name] = energy.ravel()
        columns[error_name] = (1.0 - energy).ravel()
        column_names.extend([trn_name, ref_name, energy_name, error_name])

    return ResultTable(columns=columns, column_names=tuple(column_names), row_count=wave_grid.size)


def legacy_wavelength_slices(
    transmission: object,
    reflection: object,
    wavelengths: ArrayLike,
    angles: ArrayLike,
    *,
    wavelength_unit: WavelengthUnit,
) -> tuple[LegacyWavelengthSlice, ...]:
    """Return start/mid/end wavelength slices for legacy TRN0/REF0 plots."""

    wavelength_nm = _wavelengths_nm(wavelengths, wavelength_unit)
    theta_deg = np.rad2deg(_as_1d_float(angles, "angles"))
    trn0 = _result_array(transmission, "TRN", "0")
    ref0 = _result_array(reflection, "REF", "0")
    _validate_matrix_axes(trn0, wavelength_nm, theta_deg)
    _validate_expected_shape(ref0, trn0.shape, "REF0")
    x = AxisDescriptor("angle", theta_deg, "deg")

    slices: list[LegacyWavelengthSlice] = []
    for position, idx in zip(_POSITIONS, _slice_indices(wavelength_nm.size), strict=True):
        slices.append(
            LegacyWavelengthSlice(
                position=position,
                index=idx,
                wavelength_nm=float(wavelength_nm[idx]),
                x=x,
                series={"TRN0": trn0[idx, :], "REF0": ref0[idx, :]},
                series_labels=("TRN0", "REF0"),
            )
        )
    return tuple(slices)


def field_transform(
    field: ArrayLike,
    transform: FieldTransform,
    *,
    er2: ArrayLike | None = None,
    log_floor: float = 1e-30,
) -> FieldView:
    """Transform a complex FDFD field and optionally attach ER2 contour masks."""

    field_array = np.asarray(field)
    if field_array.ndim != 2:
        raise ValueError("field must be a two-dimensional array")
    if transform == "abs":
        values = np.abs(field_array)
        label = "abs(f)"
    elif transform == "real":
        values = np.real(field_array)
        label = "real(f)"
    elif transform == "imag":
        values = np.imag(field_array)
        label = "imag(f)"
    elif transform == "phase":
        values = np.angle(field_array)
        label = "phase(f)"
    elif transform == "log10abs":
        if log_floor <= 0:
            raise ValueError("log_floor must be positive")
        values = np.log10(np.maximum(np.abs(field_array), log_floor))
        label = "log10(abs(f))"
    else:
        raise ValueError(f"unknown field transform: {transform}")

    contour_mask: BoolArray | None = None
    edge_mask: BoolArray | None = None
    if er2 is not None:
        contour_mask = er2_contour_mask(er2)
        edge_mask = er2_edge_mask(er2)
    return FieldView(
        transform=transform,
        values=np.asarray(values, dtype=np.float64),
        label=label,
        contour_mask=contour_mask,
        edge_mask=edge_mask,
    )


def er2_contour_mask(er2: ArrayLike) -> BoolArray:
    """Return a finite-material mask suitable for contour overlays."""

    array = np.asarray(er2, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("er2 must be a two-dimensional array")
    return np.isfinite(array)


def er2_edge_mask(er2: ArrayLike) -> BoolArray:
    """Return pixels bordering a material-permittivity transition."""

    array = np.asarray(er2, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("er2 must be a two-dimensional array")
    finite = np.isfinite(array)
    edge = np.zeros(array.shape, dtype=np.bool_)
    horizontal = finite[:, 1:] & finite[:, :-1] & (array[:, 1:] != array[:, :-1])
    vertical = finite[1:, :] & finite[:-1, :] & (array[1:, :] != array[:-1, :])
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def _result_array(source: object, prefix: Literal["TRN", "REF"], order: ResultOrder) -> FloatArray:
    if isinstance(source, Mapping):
        candidates = _mapping_keys(prefix, order)
        for key in candidates:
            if key in source:
                return _as_result_matrix(source[key], f"{prefix} {order}")
        raise ValueError(f"{prefix} result missing order {order}")

    for attr in _attribute_keys(prefix, order):
        if hasattr(source, attr):
            value = getattr(source, attr)
            if value is not None:
                return _as_result_matrix(value, f"{prefix} {order}")
    raise ValueError(f"{prefix} result missing order {order}")


def _mapping_keys(prefix: Literal["TRN", "REF"], order: ResultOrder) -> tuple[str, ...]:
    if prefix == "TRN":
        return {
            "-2": ("minus_2", "TRN_minus2"),
            "-1": ("minus_1", "TRN_minus1"),
            "0": ("TRN0", "zero"),
            "+1": ("plus_1", "TRN_plus1"),
            "+2": ("plus_2", "TRN_plus2"),
            "sum": ("sum_grid", "sum", "TRN_sum"),
        }[order]
    return {
        "-2": ("minus_2", "REF_minus2"),
        "-1": ("minus_1", "REF_minus1"),
        "0": ("REF0", "zero"),
        "+1": ("plus_1", "REF_plus1"),
        "+2": ("plus_2", "REF_plus2"),
        "sum": ("sum_grid", "sum", "REF_sum"),
    }[order]


def _attribute_keys(prefix: Literal["TRN", "REF"], order: ResultOrder) -> tuple[str, ...]:
    if prefix == "TRN":
        return {
            "-2": ("minus_2",),
            "-1": ("minus_1",),
            "0": ("TRN0",),
            "+1": ("plus_1",),
            "+2": ("plus_2",),
            "sum": ("sum",),
        }[order]
    return {
        "-2": ("minus_2",),
        "-1": ("minus_1",),
        "0": ("REF0",),
        "+1": ("plus_1",),
        "+2": ("plus_2",),
        "sum": ("sum",),
    }[order]


def _column_name(prefix: str, order: ResultOrder) -> str:
    suffix = {
        "-2": "minus2",
        "-1": "minus1",
        "0": "0",
        "+1": "plus1",
        "+2": "plus2",
        "sum": "sum",
    }[order]
    if prefix in {"TRN", "REF"}:
        return f"{prefix}_{suffix}" if order != "0" else f"{prefix}0"
    return f"{prefix}_{suffix}"


def _wavelengths_nm(values: ArrayLike, unit: WavelengthUnit) -> FloatArray:
    wavelengths = _as_1d_float(values, "wavelengths")
    if unit == "m":
        return wavelengths * 1e9
    if unit == "um":
        return wavelengths * 1e3
    if unit == "nm":
        return wavelengths
    raise ValueError(f"unknown wavelength unit: {unit}")


def _as_1d_float(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_result_matrix(values: object, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(array.size, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} shape must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _validate_matrix_axes(values: FloatArray, wavelengths_nm: FloatArray, theta_deg: FloatArray) -> None:
    _validate_axis_shape(values, wavelengths_nm, theta_deg, "result")


def _validate_axis_shape(values: FloatArray, wavelengths_nm: FloatArray, theta_deg: FloatArray, name: str) -> None:
    expected_shape = (wavelengths_nm.size, theta_deg.size)
    if values.shape != expected_shape:
        raise ValueError(
            f"{name} shape {values.shape} does not match "
            f"wavelength count {wavelengths_nm.size} and angle count {theta_deg.size}"
        )


def _validate_expected_shape(values: FloatArray, shape: tuple[int, int], name: str) -> None:
    if values.shape != shape:
        raise ValueError(f"{name} shape {values.shape} does not match expected shape {shape}")


def _slice_indices(size: int) -> tuple[int, int, int]:
    if size <= 0:
        raise ValueError("slice axis must not be empty")
    return (0, size // 2, size - 1)


def _wavelength_slices(
    values: FloatArray,
    x: AxisDescriptor,
    wavelengths_nm: FloatArray,
    unit: str,
    label: str,
) -> tuple[ResultSlice, ...]:
    slices: list[ResultSlice] = []
    for position, idx in zip(_POSITIONS, _slice_indices(values.shape[0]), strict=True):
        slices.append(
            ResultSlice(
                position=position,
                index=idx,
                fixed_label="wavelength",
                fixed_value=float(wavelengths_nm[idx]),
                fixed_unit=unit,
                x=x,
                values=values[idx, :],
                label=f"{label} at {wavelengths_nm[idx]:.6g} {unit}",
            )
        )
    return tuple(slices)


def _angle_slices(
    values: FloatArray,
    x: AxisDescriptor,
    theta_deg: FloatArray,
    unit: str,
    label: str,
) -> tuple[ResultSlice, ...]:
    slices: list[ResultSlice] = []
    for position, idx in zip(_POSITIONS, _slice_indices(values.shape[1]), strict=True):
        slices.append(
            ResultSlice(
                position=position,
                index=idx,
                fixed_label="angle",
                fixed_value=float(theta_deg[idx]),
                fixed_unit=unit,
                x=x,
                values=values[:, idx],
                label=f"{label} at {theta_deg[idx]:.6g} {unit}",
            )
        )
    return tuple(slices)
