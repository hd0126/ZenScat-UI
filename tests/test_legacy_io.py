from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from zenscat.core import DiffractionResult, FDFDResult
from zenscat.legacy_io import (
    FDFDResultBundle,
    LegacyResultBundle,
    load_imported_device,
    load_legacy_result_bundle,
    save_fdfd_result_bundle,
    save_legacy_result_bundle,
)

REPO_ROOT = Path(__file__).parents[1]
EXAMPLE_DEVICES = {
    "Spatial Filter/RCWA_DATA.mat": 2,
    "DFB files/RCWA_DATA.mat": 3,
    "Lumerical Comparison/RCWA_DATA.mat": 2,
    "LIDT coupler/RCWA_DATA.mat": 16,
}


@pytest.mark.parametrize(("relative_path", "layer_count"), EXAMPLE_DEVICES.items())
def test_bundled_matlab_devices_load_without_source_edits(relative_path: str, layer_count: int) -> None:
    imported = load_imported_device(REPO_ROOT / relative_path)

    assert imported.ER.shape == (layer_count, 1028)
    assert imported.sub_L_um.shape == (layer_count,)
    assert imported.x_um.shape == (1028,)
    assert imported.Lx_um > 0

    grid, device = imported.solver_inputs(
        2,
        wavelengths_m=[500e-9, 510e-9],
        angles_rad=[0.0, 0.1],
        n_superstrate=1.0,
        n_substrate=1.516,
    )
    assert grid.Lx == imported.Lx_um
    assert grid.erT == pytest.approx(1.516**2)
    assert device.ERC.shape == (5, 5, layer_count)
    np.testing.assert_array_equal(device.sub_L, imported.sub_L_um * 1e-6)


def test_pickled_numpy_device_requires_explicit_trust() -> None:
    path = REPO_ROOT / "Spatial Filter" / "RCWA_DATA.py.npy"
    with pytest.raises(ValueError, match="trusted_pickle"):
        load_imported_device(path)

    imported = load_imported_device(path, trusted_pickle=True)
    assert imported.ER.shape == (2, 1028)


def test_result_bundle_round_trips_through_matlab_schema(tmp_path: Path) -> None:
    wavelengths = np.array([500e-9, 510e-9])
    angles = np.array([0.0, 0.1, 0.2])
    shape = (wavelengths.size, angles.size)
    trn = DiffractionResult(
        minus_1=np.full(shape, 0.01),
        plus_1=np.full(shape, 0.02),
        TRN0=np.full(shape, 0.7),
        sum=np.full(shape, 0.73),
    )
    ref = DiffractionResult(
        minus_1=np.full(shape, 0.03),
        plus_1=np.full(shape, 0.04),
        REF0=np.full(shape, 0.2),
        sum=np.full(shape, 0.27),
    )
    bundle = LegacyResultBundle(
        wavelengths_m=wavelengths,
        angles_rad=angles,
        transmission=trn,
        reflection=ref,
        params=np.array([0.182, 0.120, 1.781, 1.650]),
        metadata={"solver": "S", "polarization": "E"},
    )

    destination = save_legacy_result_bundle(tmp_path / "result", bundle)
    loaded = load_legacy_result_bundle(destination)

    np.testing.assert_array_equal(loaded.wavelengths_m, wavelengths)
    np.testing.assert_array_equal(loaded.angles_rad, angles)
    np.testing.assert_array_equal(loaded.transmission.TRN0, trn.TRN0)
    np.testing.assert_array_equal(loaded.reflection.REF0, ref.REF0)
    np.testing.assert_array_equal(loaded.params, bundle.params)

    matlab_trn = loadmat(destination / "TRN.mat", squeeze_me=True, struct_as_record=False)["TRN"]
    np.testing.assert_array_equal(matlab_trn.minus_1, trn.minus_1)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "zenscat.result-bundle"
    assert manifest["shape"] == [2, 3]
    assert manifest["metadata"] == {"polarization": "E", "solver": "S"}
    assert loaded.metadata == {"polarization": "E", "solver": "S"}


def test_result_export_refuses_silent_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "result"
    destination.mkdir()
    (destination / "user-note.txt").write_text("keep", encoding="utf-8")
    empty = np.zeros((1, 1))
    bundle = LegacyResultBundle(
        wavelengths_m=np.array([500e-9]),
        angles_rad=np.array([0.0]),
        transmission=DiffractionResult(empty, empty, TRN0=empty, sum=empty),
        reflection=DiffractionResult(empty, empty, REF0=empty, sum=empty),
    )

    with pytest.raises(FileExistsError):
        save_legacy_result_bundle(destination, bundle)
    assert (destination / "user-note.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("shape", ((1, 3), (3, 1), (1, 1)))
def test_result_round_trip_preserves_single_axis_orientation(tmp_path: Path, shape: tuple[int, int]) -> None:
    wavelengths = np.linspace(500e-9, 520e-9, shape[0])
    angles = np.linspace(0.0, 0.2, shape[1])
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10
    bundle = LegacyResultBundle(
        wavelengths_m=wavelengths,
        angles_rad=angles,
        transmission=DiffractionResult(values, values, TRN0=values, sum=values),
        reflection=DiffractionResult(values, values, REF0=values, sum=values),
    )

    destination = save_legacy_result_bundle(tmp_path / f"result-{shape[0]}-{shape[1]}", bundle)
    loaded = load_legacy_result_bundle(destination)

    assert loaded.transmission.TRN0 is not None
    assert loaded.reflection.REF0 is not None
    assert loaded.transmission.TRN0.shape == shape
    assert loaded.reflection.REF0.shape == shape
    np.testing.assert_array_equal(loaded.transmission.TRN0, values)


def test_fdfd_result_export_writes_python_schema_with_raw_field(tmp_path: Path) -> None:
    wavelengths = np.array([0.51, 0.52])
    angles = np.array([0.0])
    field = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128)
    er2 = np.array([[1.0, 2.25], [2.25, 1.0]])
    result = FDFDResult(
        TRN={
            "sum": 0.6,
            "TRN0": np.array([[0.2], [0.3]]),
            "TRN_plus1": np.array([[0.01], [0.02]]),
            "TRN_minus1": np.array([[0.03], [0.04]]),
        },
        REF={
            "sum": 0.4,
            "REF0": np.array([[0.1], [0.2]]),
            "REF_plus1": np.array([[0.05], [0.06]]),
            "TRN_minus1": np.array([[0.03], [0.04]]),
        },
        f=field,
    )
    bundle = FDFDResultBundle(
        wavelengths_um=wavelengths,
        angles_rad=angles,
        result=result,
        er2=er2,
        params=np.array([0.182, 0.120, 1.781, 1.650]),
        metadata={"workflow": "fdfd_fields"},
    )

    destination = save_fdfd_result_bundle(tmp_path / "fdfd", bundle)

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "zenscat.fdfd-result-bundle"
    assert manifest["units"]["wavelength"] == "um"
    assert manifest["field_shape"] == [2, 2]
    assert "Field.f" in manifest["checksums"]
    np.testing.assert_array_equal(loadmat(destination / "Field.mat")["f"], field)
    np.testing.assert_array_equal(loadmat(destination / "ER2.mat")["ER2"], er2)


def test_fdfd_export_refuses_silent_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "fdfd"
    destination.mkdir()
    (destination / "notes.txt").write_text("keep", encoding="utf-8")
    result = FDFDResult(
        TRN={"sum": 0.0, "TRN0": np.zeros((1, 1))},
        REF={"sum": 0.0, "REF0": np.zeros((1, 1))},
        f=np.zeros((1, 1), dtype=np.complex128),
    )
    bundle = FDFDResultBundle(
        wavelengths_um=np.array([0.51]),
        angles_rad=np.array([0.0]),
        result=result,
        er2=np.ones((1, 1)),
    )

    with pytest.raises(FileExistsError):
        save_fdfd_result_bundle(destination, bundle)
    assert (destination / "notes.txt").read_text(encoding="utf-8") == "keep"
