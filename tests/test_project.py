from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from zenscat.project import PROJECT_SCHEMA, ProjectDocument


def test_project_round_trip_preserves_numeric_and_complex_configuration(tmp_path: Path) -> None:
    project = ProjectDocument(
        name="GMR study",
        workflow="casual_rcwa",
        configuration={
            "params": np.array([0.182, 0.120, 1.781, 1.650]),
            "complex_er": np.array([[1 + 0.1j, 2 - 0.2j]]),
            "source": Path("devices/RCWA_DATA.mat"),
        },
        result_bundle="results/run-001",
    )

    path = project.save(tmp_path / "study")
    loaded = ProjectDocument.load(path)

    assert path.suffix == ".zenscat"
    assert loaded.name == project.name
    assert loaded.workflow == project.workflow
    assert loaded.configuration is not None
    assert project.configuration is not None
    np.testing.assert_array_equal(loaded.configuration["params"], project.configuration["params"])
    np.testing.assert_array_equal(loaded.configuration["complex_er"], project.configuration["complex_er"])
    assert loaded.configuration["source"] == Path("devices/RCWA_DATA.mat")
    assert loaded.result_bundle == "results/run-001"
    assert not (tmp_path / ".study.zenscat.tmp").exists()


def test_project_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.zenscat"
    path.write_text(
        json.dumps(
            {
                "schema": PROJECT_SCHEMA,
                "schema_version": "99.0",
                "name": "future",
                "workflow": "casual_rcwa",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported project schema version"):
        ProjectDocument.load(path)


def test_project_rejects_unknown_workflow() -> None:
    with pytest.raises(ValueError, match="unsupported workflow"):
        ProjectDocument(name="bad", workflow="unknown")


@pytest.mark.parametrize(
    "workflow",
    [
        "casual_rcwa",
        "custom_import_rcwa",
        "optimization",
        "fdfd_fields",
        "casual_phc_rcwa",
        "harmonic_convergence",
        "phc_fdfd_fields",
    ],
)
def test_project_accepts_supported_workflows(workflow: str) -> None:
    project = ProjectDocument(name="workflow", workflow=workflow)

    assert project.workflow == workflow
    assert workflow in ProjectDocument.WORKFLOWS
