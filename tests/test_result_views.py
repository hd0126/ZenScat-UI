from __future__ import annotations

import numpy as np
import pytest

from zenscat.core import DiffractionResult, FDFDResult
from zenscat.result_views import (
    build_result_view,
    field_transform,
    legacy_wavelength_slices,
    result_matrix,
    result_table_columns,
)


def _rcwa_results() -> tuple[DiffractionResult, DiffractionResult]:
    base = np.array([[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]])
    trn = DiffractionResult(
        minus_1=base + 0.001,
        plus_1=base + 0.002,
        TRN0=base + 0.003,
        sum=base + 0.004,
    )
    ref = DiffractionResult(
        minus_1=base + 0.010,
        plus_1=base + 0.020,
        REF0=base + 0.030,
        sum=base + 0.040,
    )
    return trn, ref


def test_rcwa_result_view_selects_energy_error_and_heatmap_axes() -> None:
    wavelengths_m = np.array([500e-9, 510e-9])
    angles_rad = np.array([0.0, 0.1, 0.2])
    trn, ref = _rcwa_results()

    view = build_result_view(
        trn,
        ref,
        wavelengths=wavelengths_m,
        angles=angles_rad,
        family="Energy error",
        order="sum",
        wavelength_unit="m",
    )

    assert view.kind == "heatmap"
    assert view.label == "Energy error sum"
    np.testing.assert_allclose(view.x.values, np.rad2deg(angles_rad))
    np.testing.assert_allclose(view.y.values, [500.0, 510.0])
    np.testing.assert_allclose(view.values, 1.0 - (trn.sum + ref.sum))


def test_result_view_builds_start_mid_end_slices_for_both_axes() -> None:
    wavelengths_um = np.array([0.50, 0.51, 0.52])
    angles_rad = np.array([0.0, 0.1, 0.2])
    trn = {
        "TRN_minus1": np.arange(9, dtype=np.float64).reshape(3, 3),
        "TRN_plus1": np.ones((3, 3)),
        "TRN0": np.full((3, 3), 0.5),
        "sum": np.full((3, 3), 0.7),
    }
    ref = {
        "REF_minus1": np.full((3, 3), 0.01),
        "REF_plus1": np.full((3, 3), 0.02),
        "REF0": np.full((3, 3), 0.03),
        "sum": np.full((3, 3), 0.08),
    }

    view = build_result_view(
        trn,
        ref,
        wavelengths=wavelengths_um,
        angles=angles_rad,
        family="Transmission",
        order="-1",
        wavelength_unit="um",
    )

    assert [slice_.position for slice_ in view.wavelength_slices] == ["start", "mid", "end"]
    assert [slice_.index for slice_ in view.wavelength_slices] == [0, 1, 2]
    assert view.wavelength_slices[1].fixed_label == "wavelength"
    assert view.wavelength_slices[1].fixed_value == pytest.approx(510.0)
    np.testing.assert_array_equal(view.wavelength_slices[1].values, [3.0, 4.0, 5.0])
    assert [slice_.index for slice_ in view.angle_slices] == [0, 1, 2]
    assert view.angle_slices[2].fixed_label == "angle"
    np.testing.assert_array_equal(view.angle_slices[2].values, [2.0, 5.0, 8.0])


def test_result_table_columns_are_csv_ready_for_all_orders() -> None:
    wavelengths_m = np.array([500e-9, 510e-9])
    angles_rad = np.array([0.0, 0.1, 0.2])
    trn, ref = _rcwa_results()

    table = result_table_columns(trn, ref, wavelengths_m, angles_rad, wavelength_unit="m")

    assert table.row_count == 6
    assert table.column_names[:2] == ("wavelength_nm", "theta_deg")
    assert "TRN_minus1" in table.columns
    assert "REF_plus1" in table.columns
    assert "energy_sum" in table.columns
    assert "energy_error_sum" in table.columns
    np.testing.assert_allclose(table.columns["wavelength_nm"], [500, 500, 500, 510, 510, 510])
    np.testing.assert_allclose(table.columns["TRN0"], trn.TRN0.ravel())
    np.testing.assert_allclose(table.columns["energy_sum"], (trn.sum + ref.sum).ravel())


def test_legacy_wavelength_slices_match_start_mid_end_matlab_plot_contract() -> None:
    wavelengths_m = np.array([500e-9, 510e-9, 520e-9])
    angles_rad = np.array([0.0, 0.2])
    trn = DiffractionResult(
        minus_1=np.zeros((3, 2)),
        plus_1=np.zeros((3, 2)),
        TRN0=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        sum=np.zeros((3, 2)),
    )
    ref = DiffractionResult(
        minus_1=np.zeros((3, 2)),
        plus_1=np.zeros((3, 2)),
        REF0=np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        sum=np.zeros((3, 2)),
    )

    slices = legacy_wavelength_slices(trn, ref, wavelengths_m, angles_rad, wavelength_unit="m")

    assert [slice_.position for slice_ in slices] == ["start", "mid", "end"]
    assert slices[0].series_labels == ("TRN0", "REF0")
    np.testing.assert_allclose(slices[1].x.values, np.rad2deg(angles_rad))
    np.testing.assert_allclose(slices[1].series["TRN0"], [0.3, 0.4])
    np.testing.assert_allclose(slices[2].series["REF0"], [1.1, 1.2])


def test_field_transform_supports_common_views_and_er2_edges() -> None:
    field = np.array([[1 + 1j, -1 + 0j], [0 - 2j, 0 + 0j]])
    er2 = np.array([[1.0, 1.0, 2.0], [1.0, 3.0, 3.0], [1.0, 3.0, 4.0]])

    abs_view = field_transform(field, "abs")
    real_view = field_transform(field, "real")
    log_view = field_transform(field, "log10abs")
    er_view = field_transform(field, "abs", er2=er2)

    np.testing.assert_allclose(abs_view.values, np.abs(field))
    np.testing.assert_allclose(real_view.values, field.real)
    assert np.isfinite(log_view.values).all()
    assert er_view.contour_mask is not None
    assert er_view.edge_mask is not None
    assert er_view.edge_mask.dtype == np.bool_
    assert np.count_nonzero(er_view.edge_mask) > 0


def test_result_views_reject_shape_mismatch() -> None:
    trn, ref = _rcwa_results()
    bad_ref = DiffractionResult(
        minus_1=np.zeros((2, 2)),
        plus_1=np.zeros((2, 2)),
        REF0=np.zeros((2, 2)),
        sum=np.zeros((2, 2)),
    )

    with pytest.raises(ValueError, match="shape"):
        build_result_view(
            trn,
            bad_ref,
            wavelengths=np.array([500e-9, 510e-9]),
            angles=np.array([0.0, 0.1, 0.2]),
            family="Energy",
            order="sum",
            wavelength_unit="m",
        )

    with pytest.raises(ValueError, match="wavelength"):
        result_table_columns(trn, ref, np.array([500e-9]), np.array([0.0, 0.1, 0.2]), wavelength_unit="m")


def test_fdfd_result_object_can_feed_table_columns() -> None:
    wavelengths_um = np.array([0.50, 0.51])
    angles_rad = np.array([0.0])
    result = FDFDResult(
        TRN={
            "TRN_minus1": np.array([[0.01], [0.02]]),
            "TRN_plus1": np.array([[0.03], [0.04]]),
            "TRN0": np.array([[0.50], [0.60]]),
            "sum": np.array([[0.54], [0.66]]),
        },
        REF={
            "REF_minus1": np.array([[0.05], [0.06]]),
            "REF_plus1": np.array([[0.07], [0.08]]),
            "REF0": np.array([[0.10], [0.20]]),
            "sum": np.array([[0.22], [0.34]]),
        },
        f=np.ones((2, 2), dtype=np.complex128),
    )

    table = result_table_columns(result.TRN, result.REF, wavelengths_um, angles_rad, wavelength_unit="um")

    assert table.row_count == 2
    np.testing.assert_allclose(table.columns["wavelength_nm"], [500.0, 510.0])
    np.testing.assert_allclose(table.columns["energy_error_sum"], [0.24, 0.0])


def test_reflection_view_rejects_transmission_prefixed_negative_order() -> None:
    transmission = {"TRN_minus1": np.array([[0.1]])}
    malformed_reflection = {"TRN_minus1": np.array([[0.9]])}

    with pytest.raises(ValueError, match="REF result missing order -1"):
        result_matrix(
            transmission,
            malformed_reflection,
            family="Reflection",
            order="-1",
        )
