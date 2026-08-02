from __future__ import annotations

import zenscat


def test_superset_workflows_are_available_from_package_root() -> None:
    expected = {
        "configured_external_backend",
        "decode_external_run",
        "run_dual_fdfd",
        "run_dual_phc_fdfd",
        "run_harmonic_convergence",
        "run_optimization",
        "run_phc_fdfd",
        "build_result_view",
        "field_transform",
        "result_table_columns",
    }

    assert expected <= set(zenscat.__all__)
    assert all(callable(getattr(zenscat, name)) for name in expected)


def test_superset_result_types_are_available_from_package_root() -> None:
    expected = {
        "DualFDFDRun",
        "DualPhCFDFDRun",
        "ExternalFDFDRun",
        "ExternalRCWARun",
        "OptimizationCheckpoint",
        "OptimizationMeritProgress",
        "PhCFDFDRequest",
        "ResultView",
    }

    assert expected <= set(zenscat.__all__)
    assert all(getattr(zenscat, name) is not None for name in expected)
