from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

GOLDEN_PATH = Path(__file__).parent / "golden" / "zenscat_matlab_golden.mat"
EXPECTED_FIXTURES = {
    "casual_default_s_e_41x21": {
        "shape": (41, 21),
        "method": "S",
        "nh": 5,
        "polarization": "E",
        "nx": 1028,
        "nz": 11,
    },
    "small_s_e_3x3": {
        "shape": (3, 3),
        "method": "S",
        "nh": 2,
        "polarization": "E",
        "nx": 128,
        "nz": 5,
    },
    "small_s_h_3x3": {
        "shape": (3, 3),
        "method": "S",
        "nh": 2,
        "polarization": "H",
        "nx": 128,
        "nz": 5,
    },
    "small_t_e_3x3": {
        "shape": (3, 3),
        "method": "T",
        "nh": 2,
        "polarization": "E",
        "nx": 128,
        "nz": 5,
    },
    "small_t_h_3x3": {
        "shape": (3, 3),
        "method": "T",
        "nh": 2,
        "polarization": "H",
        "nx": 128,
        "nz": 5,
    },
}
NUMERIC_FIELDS = {
    "TRN": ("minus_1", "plus_1", "TRN0", "sum"),
    "REF": ("minus_1", "plus_1", "REF0", "sum"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _mat_number(value):
    array = np.asarray(value)
    assert array.size == 1
    return array.item()


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


def _load_golden():
    assert GOLDEN_PATH.exists(), f"Missing golden fixture: {GOLDEN_PATH}"
    data = loadmat(GOLDEN_PATH, squeeze_me=True, struct_as_record=False)
    return data["manifest"], _as_list(data["fixtures"])


def test_manifest_records_source_and_environment():
    manifest, fixtures = _load_golden()

    assert _mat_text(manifest.schema_version) == "1.0"
    assert _mat_text(manifest.generator) == "matlab_oracle/generate_golden_fixtures.m"
    assert len(_mat_text(manifest.source_commit)) == 40
    assert _mat_text(manifest.matlab_release) == "R2024b"
    assert _mat_number(manifest.fixture_count) == len(fixtures) == 5

    names = {_mat_text(fixture.name) for fixture in fixtures}
    assert names == set(EXPECTED_FIXTURES)


def test_fixture_shapes_checksums_and_config_contract():
    _, fixtures = _load_golden()

    for fixture in fixtures:
        name = _mat_text(fixture.name)
        expected = EXPECTED_FIXTURES[name]
        config = fixture.config

        assert _mat_text(config.method) == expected["method"]
        assert _mat_text(config.interface) == "sin"
        assert _mat_text(config.distribution) == "all"
        assert _mat_text(config.polarization) == expected["polarization"]
        assert _mat_number(config.NH) == expected["nh"]
        assert _mat_number(config.Nx) == expected["nx"]
        assert _mat_number(config.Nz) == expected["nz"]
        np.testing.assert_allclose(np.asarray(config.params), [0.182, 0.120, 1.781, 1.650])
        assert _mat_number(config.Lx_um) == 0.32
        assert _mat_number(config.h_um) == 0.154

        assert np.asarray(fixture.Lam0).shape == (expected["shape"][0],)
        assert np.asarray(fixture.Theta).shape == (expected["shape"][1],)
        np.testing.assert_allclose(np.asarray(fixture.Output), [0.182, 0.120, 1.781, 1.650])
        np.testing.assert_allclose(fixture.grid_solver.Lam0, fixture.Lam0)
        np.testing.assert_allclose(fixture.grid_solver.Theta, fixture.Theta)
        assert _mat_number(fixture.grid_solver.Lx) == 0.32
        assert _mat_number(fixture.grid_solver.layer_num) == 1
        assert _mat_number(fixture.grid_solver.erR) == 1.0
        assert _mat_number(fixture.grid_solver.urR) == 1.0
        assert _mat_number(fixture.grid_solver.urT) == 1.0
        assert _mat_number(fixture.grid_solver.erT) == _mat_number(fixture.grid_solver.erSub)
        assert _mat_number(fixture.grid_solver.erSub) == 1.516**2

        checksums = fixture.checksums
        assert _mat_text(checksums.Lam0_sha256) == _sha256_numeric(fixture.Lam0)
        assert _mat_text(checksums.Theta_sha256) == _sha256_numeric(fixture.Theta)
        assert _mat_text(checksums.Output_sha256) == _sha256_numeric(fixture.Output)

        for struct_name, field_names in NUMERIC_FIELDS.items():
            struct = getattr(fixture, struct_name)
            for field_name in field_names:
                value = getattr(struct, field_name)
                assert np.asarray(value).shape == expected["shape"]
                assert np.all(np.isfinite(value))
                checksum_name = f"{struct_name}_{field_name}_sha256"
                assert _mat_text(getattr(checksums, checksum_name)) == _sha256_numeric(value)

        assert tuple(np.asarray(fixture.device.ER_shape, dtype=int).ravel()) == np.asarray(fixture.device.ER).shape
        assert tuple(np.asarray(fixture.device.ERC_shape, dtype=int).ravel()) == np.asarray(fixture.device.ERC).shape
        assert np.prod(np.asarray(fixture.device.sub_L_shape, dtype=int)) == np.asarray(fixture.device.sub_L).size
        assert np.all(np.isfinite(fixture.device.ER))
        assert np.all(np.isfinite(fixture.device.ERC))
        assert np.all(np.isfinite(fixture.device.sub_L))
        assert _mat_number(fixture.device.sub_L_sum_m) > 0
        assert abs(_mat_number(fixture.device.sub_L_sum_m) - np.sum(fixture.device.sub_L)) < 1e-18
        assert _mat_text(checksums.device_ER_sha256) == _sha256_numeric(fixture.device.ER)
        assert _mat_text(checksums.device_ERC_sha256) == _sha256_numeric(fixture.device.ERC)
        assert _mat_text(checksums.device_sub_L_sha256) == _sha256_numeric(fixture.device.sub_L)
        assert _mat_text(fixture.device.ER_checksum_sha256) == _mat_text(checksums.device_ER_sha256)
        assert _mat_text(fixture.device.ERC_checksum_sha256) == _mat_text(checksums.device_ERC_sha256)
        assert SHA256_RE.match(_mat_text(fixture.device.sub_L_checksum_sha256))


def test_energy_is_conserved_for_all_golden_fields():
    _, fixtures = _load_golden()

    for fixture in fixtures:
        energy = np.asarray(fixture.TRN.sum) + np.asarray(fixture.REF.sum)
        assert np.max(np.abs(energy - 1.0)) < 1e-9
        assert abs(_mat_number(fixture.energy.sum_min) - np.min(energy)) < 1e-12
        assert abs(_mat_number(fixture.energy.sum_max) - np.max(energy)) < 1e-12
        assert abs(_mat_number(fixture.energy.max_abs_error) - np.max(np.abs(energy - 1.0))) < 1e-12
