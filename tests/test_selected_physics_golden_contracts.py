from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "tests" / "golden" / "selected_physics_manifest.json"
EXPECTED_CASES = {
    "asr_gate_b": "reference_golden",
    "nvf_curved_metal_2d": "reference_golden",
    "w18_meep_lossy_2d": "blocked_boundary",
    "conical_1d_sp": "reference_golden",
}


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _case(case_id: str) -> dict[str, Any]:
    for case in _manifest()["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"missing selected physics case: {case_id}")


def _fixture(case_id: str) -> dict[str, Any]:
    case = _case(case_id)
    path = ROOT / case["fixture"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == case["sha256"]
    fixture = json.loads(payload)
    assert fixture["case_id"] == case_id
    assert fixture["source_commit"] == _manifest()["reference_engine"]["commit"]
    return fixture


def test_selected_physics_manifest_is_explicit_about_support_boundaries() -> None:
    manifest = _manifest()

    assert manifest["schema"] == "zenscat.selected-physics-goldens"
    assert manifest["schema_version"] == "1.0"
    assert len(manifest["reference_engine"]["commit"]) == 40
    assert manifest["reference_engine"]["repository"].startswith("https://github.com/")
    assert {case["id"]: case["status"] for case in manifest["cases"]} == EXPECTED_CASES
    assert all(case["local_support"] == "not_implemented" for case in manifest["cases"])
    assert all(not case["status"].startswith("xfail") for case in manifest["cases"])


def test_asr_gate_b_reference_keeps_safe_refine_physically_invariant() -> None:
    fixture = _fixture("asr_gate_b")
    samples = fixture["samples"]

    assert {(row["material"], row["polarization"], row["orders"]) for row in samples} == {
        (material, polarization, orders)
        for material in ("TiO2", "Au")
        for polarization in ("TE", "TM")
        for orders in (21, 41)
    }
    deltas = np.asarray([abs(row["asr_T"] - row["base_T"]) for row in samples])
    assert np.all(np.isfinite(deltas))
    assert float(np.max(deltas)) < _case("asr_gate_b")["tolerance"]["abs_delta_T_max"]


def test_nvf_curved_metal_2d_reference_suppresses_high_order_oscillation() -> None:
    fixture = _fixture("nvf_curved_metal_2d")
    orders = np.asarray([row["orders"] for row in fixture["samples"]], dtype=np.int64)
    transmission = np.asarray([row["T"] for row in fixture["samples"]], dtype=np.float64)

    np.testing.assert_array_equal(orders, [11, 13, 15])
    assert fixture["parameters"]["factorization"] == "normal_vector_field"
    assert fixture["parameters"]["use_matched_coordinates"] is False
    assert np.all((0.0 <= transmission) & (transmission <= 1.0))
    spread = float(np.ptp(transmission))
    assert spread < _case("nvf_curved_metal_2d")["tolerance"]["spread_T_max"]


def test_w18_meep_lossy_2d_remains_blocked_until_volume_absorption_is_wired() -> None:
    fixture = _fixture("w18_meep_lossy_2d")

    assert fixture["status"] == "blocked"
    assert fixture["block_code"] == "W18"
    assert fixture["scope"] == "lossy_patterned_2d"
    assert set(fixture["validated_subscopes"]) == {"lossy_uniform", "lossy_patterned_1d"}
    assert fixture["required_gate"]["observable"] == "A_volume"
    assert fixture["required_gate"]["must_compare_against"] == "1-R-T"
    assert fixture["required_gate"]["volume_absorption_delta_max"] <= 0.03


def test_conical_1d_sp_reference_locks_basis_mapping_and_azimuth_continuity() -> None:
    fixture = _fixture("conical_1d_sp")
    samples = fixture["samples"]
    indexed = {(row["phi_deg"], row["polarization"]): row for row in samples}

    assert fixture["convention"]["selector_mapping_with_sp_basis"] == {"TE": "s", "TM": "p"}
    assert fixture["convention"]["conical_requires_cartesian_basis_rotation"] is True
    assert set(indexed) == {(phi, pol) for phi in (0.0, 0.05, 30.0) for pol in ("s", "p")}
    for row in samples:
        assert row["T"] >= 0.0
        assert row["R"] >= 0.0
        assert abs(row["T"] + row["R"] + row["A"] - 1.0) < 1e-12
        assert abs(row["A"]) < _case("conical_1d_sp")["tolerance"]["energy_residual_max"]
    for polarization in ("s", "p"):
        assert abs(indexed[(0.05, polarization)]["T"] - indexed[(0.0, polarization)]["T"]) < 5e-3
    assert indexed[(30.0, "s")]["T"] == pytest.approx(0.7610947824, abs=2e-9)
    assert indexed[(30.0, "p")]["T"] == pytest.approx(0.8748089655, abs=2e-9)
