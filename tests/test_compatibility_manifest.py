from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).parents[1] / "compatibility" / "manifest-v1.json"


def test_compatibility_manifest_enumerates_the_legacy_product_surface() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema"] == "zenscat.compatibility-manifest"
    assert manifest["schema_version"] == "1.0"
    assert manifest["production_runtime"].startswith("Python")
    assert manifest["public_result_fields"]["rcwa"] == {
        "TRN": ["minus_1", "plus_1", "TRN0", "sum"],
        "REF": ["minus_1", "plus_1", "REF0", "sum"],
    }
    assert manifest["public_result_fields"]["fdfd"] == {
        "TRN": ["TRN_minus1", "TRN_plus1", "TRN0", "sum"],
        "REF": ["TRN_minus1", "REF_plus1", "REF0", "sum"],
        "raw_field": "f",
        "device_field": "ER2",
    }
    workflows = {workflow["id"]: workflow for workflow in manifest["workflows"]}
    assert set(workflows) == {
        "casual_rcwa",
        "custom_import_rcwa",
        "casual_phc_rcwa",
        "optimization",
        "fdfd_fields",
    }
    for workflow in workflows.values():
        assert ".m" in workflow["legacy_entry"]
        assert workflow["inputs"]
        assert workflow["outputs"]
        assert workflow["status"] == "supported"
        assert workflow["gui_support"] == "implemented"
        assert "pending" not in workflow

    assert "T matrix" in workflows["casual_rcwa"]["supported"]
    assert "DE1 trapezium interface" in workflows["casual_rcwa"]["supported"]
    assert "legacy RCWA_DATA.mat files" in workflows["custom_import_rcwa"]["supported"]
    assert "PhC_hex_columns hex columns" in workflows["casual_phc_rcwa"]["supported"]
    assert "R(+1)" in workflows["optimization"]["supported"]
    assert "MATLAB proprietary GA trajectory compatibility" in workflows["optimization"]["not_claimed"]
    assert "tri triangle interface" in workflows["fdfd_fields"]["supported"]
    assert "material palette dispersion corrected_um mode" in workflows["fdfd_fields"]["supported"]


def test_every_declared_oracle_has_a_real_fixture_and_test() -> None:
    root = MANIFEST_PATH.parents[1]
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    fixture_ids = set()
    expected_families = {
        "casual_rcwa_s_t_e_h",
        "analytic_device_variants",
        "optimization_objective_parity",
        "fdfd_mainline_fields",
        "casual_phc_rcwa",
    }
    for oracle in manifest["oracle_fixture_families"]:
        assert oracle["id"] not in fixture_ids
        fixture_ids.add(oracle["id"])
        assert (root / oracle["fixture"]).is_file()
        assert (root / oracle["generator"]).is_file()
        assert oracle["fixtures"]
        for python_test in oracle["python_tests"]:
            assert (root / python_test).is_file()
        assert oracle["tolerance"]["rtol"] >= 0
        assert oracle["tolerance"]["atol"] >= 0
        assert oracle["units"]
    assert fixture_ids == expected_families


def test_known_legacy_quirks_have_explicit_policies() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    quirks = {item["id"]: item for item in manifest["legacy_quirks"]}

    assert {
        "fdfd-final-sweep-sum",
        "fdfd-ref-trn-minus1-alias",
        "fdfd-dispersion-unit-behavior",
        "phc-non-power-of-two-repeat",
        "phc-rectangle-zero-width-live-wiring",
        "optimizer-plus-minus-two-dropdown-options",
        "matlab-ga-trajectory",
        "tm-normalization-differs-between-s-and-t",
        "matlab-rounding-indexing-orientation",
        "broken-device-fdfd-phc-branch",
    } <= set(quirks)
    assert all(item["policy"] for item in quirks.values())
    assert all(item["status"] != "unverified" for item in quirks.values())


def test_units_and_export_schemas_prevent_rcwa_fdfd_confusion() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["length_units"]["rcwa_solver_wavelength"] == "m"
    assert manifest["length_units"]["fdfd_wavelength"] == "um"

    schemas = {schema["schema"]: schema for schema in manifest["export_schemas"]}
    assert schemas["zenscat.result-bundle"]["units"]["wavelength"] == "m"
    assert schemas["zenscat.fdfd-result-bundle"]["units"]["wavelength"] == "um"
    assert "Field.mat" in schemas["zenscat.fdfd-result-bundle"]["files"]
    assert "ER2.mat" in schemas["zenscat.fdfd-result-bundle"]["files"]


def test_release_claims_do_not_overstate_distribution_status() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    claims = manifest["release_claims"]
    assert claims["matlab_required_at_runtime"] is False
    assert claims["matlab_coder_required"] is False
    assert claims["windows_mex_required"] is False
    assert claims["notarization"] == "not claimed"
