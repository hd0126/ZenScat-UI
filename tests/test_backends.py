from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import numpy as np
import pytest

from zenscat.backends import (
    BackendUnavailableError,
    ExternalSolverBackend,
    ExternalSolverConfig,
    ExternalSolverError,
    ExternalSolverResponse,
    LocalPythonBackend,
    configured_external_backend,
    decode_external_run,
)


def _write_solver(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_local_backend_is_explicit_and_executes_callable() -> None:
    backend = LocalPythonBackend[int]()

    assert backend.availability().available is True
    assert backend.run(lambda left, right: left + right, 2, 3) == 5


def test_unconfigured_external_backend_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZENSCAT_EXTERNAL_SOLVER", raising=False)
    backend = configured_external_backend()

    assert backend.availability().available is False
    with pytest.raises(BackendUnavailableError, match="No external solver command"):
        backend.run({"workflow": "rcwa"})


def test_external_backend_uses_json_file_envelope_without_shell(tmp_path: Path) -> None:
    solver = _write_solver(
        tmp_path / "fake solver",
        """
request=""
result=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --zenscat-request) request="$2"; shift 2 ;;
    --zenscat-result) result="$2"; shift 2 ;;
    *) shift ;;
  esac
done
python3 - "$request" "$result" <<'PY'
import json, sys
request = json.load(open(sys.argv[1], encoding='utf-8'))
json.dump({'ok': True, 'workflow': request['workflow']}, open(sys.argv[2], 'w', encoding='utf-8'))
PY
""",
    )
    backend = ExternalSolverBackend(
        ExternalSolverConfig((str(solver),), timeout_s=2.0)
    )

    response = backend.run({"workflow": "rcwa", "values": [1, 2]})

    assert response.payload == {"ok": True, "workflow": "rcwa"}
    assert response.command[0] == str(solver)
    assert "--zenscat-request" in response.command
    assert "--zenscat-result" in response.command


def test_external_backend_reports_nonzero_exit(tmp_path: Path) -> None:
    solver = _write_solver(tmp_path / "fails", "echo solver-broke >&2\nexit 7\n")
    backend = ExternalSolverBackend(ExternalSolverConfig((str(solver),), timeout_s=2.0))

    with pytest.raises(ExternalSolverError, match="status 7: solver-broke"):
        backend.run({"workflow": "rcwa"})


def test_external_backend_honors_cancellation(tmp_path: Path) -> None:
    solver = _write_solver(tmp_path / "waits", "sleep 5\n")
    backend = ExternalSolverBackend(ExternalSolverConfig((str(solver),), timeout_s=10.0))
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(ExternalSolverError, match="cancelled"):
        backend.run({"workflow": "rcwa"}, cancel_event=cancellation)


def test_command_text_uses_platform_parser_not_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZENSCAT_EXTERNAL_SOLVER", '"/tmp/a solver" --profile exact')

    config = configured_external_backend().config

    assert config.command == ("/tmp/a solver", "--profile", "exact")
    assert os.environ["ZENSCAT_EXTERNAL_SOLVER"].startswith('"/tmp')


def test_external_inline_rcwa_result_decodes_to_exportable_run() -> None:
    matrix = [[0.1, 0.2]]
    response = ExternalSolverResponse(
        payload={
            "schema": "zenscat.external-result",
            "kind": "rcwa",
            "wavelengths_m": [500e-9],
            "angles_rad": [0.0, 0.1],
            "transmission": {
                "minus_1": matrix,
                "plus_1": matrix,
                "TRN0": matrix,
                "sum": matrix,
            },
            "reflection": {
                "minus_1": matrix,
                "plus_1": matrix,
                "REF0": matrix,
                "sum": matrix,
            },
        },
        stdout="",
        stderr="",
        elapsed_s=0.1,
        command=("fake-solver",),
    )

    run = decode_external_run(response)

    assert run.metadata["backend"] == "external"
    assert run.transmission.TRN0.shape == (1, 2)
    assert run.legacy_bundle().reflection.REF0.shape == (1, 2)


def test_external_relative_bundle_path_is_rejected_before_cwd_reinterpretation() -> None:
    response = ExternalSolverResponse(
        payload={
            "schema": "zenscat.external-result",
            "kind": "rcwa",
            "bundle_path": "deleted-temp-output",
        },
        stdout="",
        stderr="",
        elapsed_s=0.1,
        command=("fake-solver",),
    )

    with pytest.raises(ExternalSolverError, match="bundle_path must be absolute"):
        decode_external_run(response)


def test_external_inline_fdfd_result_decodes_to_exportable_run() -> None:
    response = ExternalSolverResponse(
        payload={
            "schema": "zenscat.external-result",
            "kind": "fdfd",
            "wavelengths_um": [0.5, 0.52],
            "angles_rad": [0.0],
            "transmission": {"TRN0": [[0.7], [0.6]], "TRN_sum": [[0.8], [0.7]]},
            "reflection": {"REF0": [[0.2], [0.3]], "REF_sum": [[0.2], [0.3]]},
            "field": {"real": [[1.0, 0.0], [0.5, -0.5]], "imag": [[0.0, 1.0], [0.5, 0.0]]},
            "er2": [[1.0, 1.2], [2.0, 2.1]],
        },
        stdout="",
        stderr="",
        elapsed_s=0.2,
        command=("fake-solver",),
    )

    run = decode_external_run(response)

    assert run.metadata["backend"] == "external"
    assert run.result.f.shape == (2, 2)
    assert np.iscomplexobj(run.result.f)
    assert run.fdfd_bundle().er2.shape == (2, 2)


def test_external_optimization_result_decodes_gui_history_and_checkpoint() -> None:
    matrix = [[0.1, 0.2]]
    checkpoint = {
        "lower_bounds": [0.6, 0.01, 0.2],
        "upper_bounds": [5.0, 0.5, 1.0],
        "seed": 3,
        "population_size": 5,
        "max_generations": 20,
        "integer_indices": [0],
        "sum_limit_count": 2,
        "max_sum": 4.0,
        "compatibility_profile": "scipy_de",
        "generations_completed": 2,
        "evaluations": 30,
        "best_fitness": -1.5,
        "best_parameters": [1.0, 0.2, 0.3],
        "history": [
            {
                "generation": 1,
                "evaluations": 15,
                "best_fitness": -1.0,
                "best_parameters": [0.8, 0.1, 0.2],
            },
            {
                "generation": 2,
                "evaluations": 30,
                "best_fitness": -1.5,
                "best_parameters": [1.0, 0.2, 0.3],
            },
        ],
    }
    response = ExternalSolverResponse(
        payload={
            "schema": "zenscat.external-result",
            "kind": "optimization",
            "source": "analytic",
            "objective": "R(+1)",
            "optimization": {
                "parameters": [1.0, 0.2, 0.3],
                "fitness": -1.5,
                "success": True,
                "message": "external complete",
                "generations": 2,
                "evaluations": 30,
                "history": checkpoint["history"],
                "checkpoint": checkpoint,
            },
            "final_parameters": [1.0, 0.2, 0.3],
            "final_fitness": -1.5,
            "final_run": {
                "wavelengths_m": [500e-9],
                "angles_rad": [0.0, 0.1],
                "transmission": {
                    "minus_1": matrix,
                    "plus_1": matrix,
                    "TRN0": matrix,
                    "sum": matrix,
                },
                "reflection": {
                    "minus_1": matrix,
                    "plus_1": matrix,
                    "REF0": matrix,
                    "sum": matrix,
                },
            },
        },
        stdout="",
        stderr="",
        elapsed_s=0.3,
        command=("fake-solver",),
    )

    run = decode_external_run(response)

    assert run.objective == "R(+1)"
    assert [item.best_fitness for item in run.optimization.history] == [-1.0, -1.5]
    assert run.optimization.checkpoint is not None
    assert run.optimization.checkpoint.to_dict()["generations_completed"] == 2
    assert run.final_run.reflection.REF0.shape == (1, 2)
    assert run.legacy_bundle().metadata["optimizer"] == "external"
