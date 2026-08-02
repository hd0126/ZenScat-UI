from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from zenscat.core import DiffractionResult, InterfaceParams
from zenscat.legacy_io import ImportedDevice
from zenscat.optimization import (
    evaluate_analytic_parameters,
    evaluate_imported_parameters,
    legacy_objective_value,
)
from zenscat.workflows import AnalyticRCWARequest, ImportedRCWARequest

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_optimization_golden.mat"
NUMERIC_FIELDS = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}


def _as_list(value):
    if isinstance(value, np.ndarray):
        return list(value.ravel())
    return [value]


def _mat_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.ravel().astype(str)).strip()
        if value.size == 1:
            return _mat_text(value.item())
    return str(value)


def _mat_number(value) -> float:
    array = np.asarray(value)
    assert array.size == 1
    return float(array.item())


def _mat_array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).ravel()


def _sha256_numeric(value) -> str:
    array = np.asarray(value, order="F")
    if np.iscomplexobj(array):
        payload = np.concatenate(
            [
                np.asarray(array.real, dtype="<f8", order="F").ravel(order="F"),
                np.asarray(array.imag, dtype="<f8", order="F").ravel(order="F"),
            ]
        )
    else:
        payload = np.asarray(array, dtype="<f8", order="F").ravel(order="F")
    return hashlib.sha256(payload.astype("<f8", copy=False).tobytes()).hexdigest()


def _load_fixtures():
    data = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return data["manifest"], _as_list(data["fixtures"])


def _result_pair(fixture) -> tuple[DiffractionResult, DiffractionResult]:
    trn = fixture.TRN
    ref = fixture.REF
    return (
        DiffractionResult(
            minus_1=np.asarray(trn.minus_1, dtype=np.float64),
            plus_1=np.asarray(trn.plus_1, dtype=np.float64),
            TRN0=np.asarray(trn.TRN0, dtype=np.float64),
            sum=np.asarray(trn.sum, dtype=np.float64),
        ),
        DiffractionResult(
            minus_1=np.asarray(ref.minus_1, dtype=np.float64),
            plus_1=np.asarray(ref.plus_1, dtype=np.float64),
            REF0=np.asarray(ref.REF0, dtype=np.float64),
            sum=np.asarray(ref.sum, dtype=np.float64),
        ),
    )


def test_optimization_golden_manifest_contract():
    manifest, fixtures = _load_fixtures()

    assert _mat_text(manifest.schema_version) == "1.0"
    assert _mat_text(manifest.generator) == "matlab_oracle/generate_optimization_golden.m"
    assert _mat_text(manifest.matlab_release) == "R2024b"
    assert _mat_number(manifest.fixture_count) == len(fixtures) == 4
    assert {_mat_text(fixture.name) for fixture in fixtures} == {
        "analytic_e_t0",
        "analytic_h_absorption",
        "import_e_gain",
        "import_h_r0",
    }


def test_legacy_objective_value_matches_matlab_merit_fixtures():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        trn, ref = _result_pair(fixture)
        actual = legacy_objective_value(
            trn,
            ref,
            _mat_text(fixture.objective),
            flavor=_mat_text(fixture.flavor),
        )

        matlab_fitness = _mat_number(fixture.fitness)
        matlab_expected = _mat_number(fixture.python_expected)
        np.testing.assert_allclose([actual], [matlab_fitness], rtol=0, atol=2e-14)
        np.testing.assert_allclose([actual], [matlab_expected], rtol=0, atol=2e-14)


def test_workflow_objective_evaluators_match_matlab_merit_fixtures():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        config = fixture.config
        objective = _mat_text(fixture.objective)
        if _mat_text(fixture.kind) == "analytic":
            template = AnalyticRCWARequest(
                params=_mat_array(config.params),
                layer_num=int(np.ceil(np.asarray(config.params).size / 2)),
                wavelengths_m=1e-9 * _mat_array(config.wavelengths_nm),
                angles_rad=(np.pi / 180) * _mat_array(config.angles_deg),
                harmonic_count=int(_mat_number(config.NH)),
                polarization=_mat_text(config.mode),
                distribution=_mat_text(config.distribution),
                interface=_mat_text(config.interface),
                interface_params=InterfaceParams(smooth=True),
                Lx_um=_mat_number(config.Lx_um),
                h_um=_mat_number(config.h_um),
                Nx=int(_mat_number(config.Nx)),
                Nz=int(_mat_number(config.Nz)),
                n_superstrate=_mat_number(config.n_sup),
                n_substrate=_mat_number(config.n_sub),
            )
            actual = evaluate_analytic_parameters(template, _mat_array(config.X), objective)
        else:
            ER = np.asarray(config.initial_ER, dtype=np.complex128)
            sub_L_um = _mat_array(config.initial_sub_L_m) * 1e6
            imported = ImportedDevice(
                ER=ER,
                sub_L_um=sub_L_um,
                x_um=np.linspace(-_mat_number(config.Lx_um) / 2, _mat_number(config.Lx_um) / 2, ER.shape[1]),
                Lx_um=_mat_number(config.Lx_um),
            )
            template = ImportedRCWARequest(
                device=imported,
                wavelengths_m=_mat_array(config.Lam0),
                angles_rad=_mat_array(config.Theta),
                harmonic_count=int(_mat_number(config.NH)),
                polarization=_mat_text(config.mode),
                n_superstrate=_mat_number(config.n_sup),
                n_substrate=_mat_number(config.n_sub),
            )
            actual = evaluate_imported_parameters(template, _mat_array(config.X), objective)

        np.testing.assert_allclose([actual], [_mat_number(fixture.fitness)], rtol=0, atol=2e-12)


def test_optimization_fixture_field_checksums_are_stable():
    _, fixtures = _load_fixtures()

    for fixture in fixtures:
        for struct_name, field_names in NUMERIC_FIELDS.items():
            struct = getattr(fixture, struct_name)
            for field_name in field_names:
                value = getattr(struct, field_name)
                assert np.asarray(value).shape == (2,)
                assert np.all(np.isfinite(value))
                checksum_name = f"{struct_name}_{field_name}_sha256"
                assert _sha256_numeric(value) == _mat_text(getattr(fixture.checksums, checksum_name))
