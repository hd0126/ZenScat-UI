from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import cast

import numpy as np
import pytest

import zenscat.optimization as optimization_module
from zenscat.core import DiffractionResult
from zenscat.core.control import SimulationCancelled
from zenscat.legacy_io import load_imported_device
from zenscat.optimization import (
    ObjectiveName,
    OptimizationCancelled,
    OptimizationProgress,
    OptimizationRequest,
    OptimizationResult,
    evaluate_analytic_parameters,
    evaluate_imported_parameters,
    legacy_objective_value,
    optimize_bounded,
    run_optimization,
)
from zenscat.workflows import (
    AnalyticRCWARequest,
    ImportedRCWARequest,
    run_analytic_rcwa,
    run_imported_rcwa,
)


def _results() -> tuple[DiffractionResult, DiffractionResult]:
    trn = DiffractionResult(
        minus_1=np.array([[0.1, 0.2]]),
        plus_1=np.array([[0.3, 0.4]]),
        TRN0=np.array([[0.5, 0.6]]),
        sum=np.array([[0.8, 0.9]]),
    )
    ref = DiffractionResult(
        minus_1=np.array([[0.01, 0.02]]),
        plus_1=np.array([[0.03, 0.04]]),
        REF0=np.array([[0.05, 0.06]]),
        sum=np.array([[0.1, 0.2]]),
    )
    return trn, ref


@pytest.mark.parametrize(
    ("objective", "expected"),
    (
        ("R(-1)", -0.03),
        ("R(0)", -0.11),
        ("R(+1)", -0.07),
        ("T(-1)", -0.3),
        ("T(0)", -1.1),
        ("T(+1)", -0.7),
        ("Absorption", 2.0),
        ("Gain", -2.0),
    ),
)
def test_analytic_objective_preserves_matlab_signs(objective: str, expected: float) -> None:
    trn, ref = _results()
    assert legacy_objective_value(trn, ref, cast(ObjectiveName, objective), flavor="analytic") == pytest.approx(
        expected
    )


def test_import_objective_preserves_absolute_value_semantics() -> None:
    trn, ref = _results()
    assert legacy_objective_value(trn, ref, "Absorption", flavor="imported") == pytest.approx(-0.2)
    assert legacy_objective_value(trn, ref, "Gain", flavor="imported") == pytest.approx(-2.0)


def test_deterministic_optimizer_honors_integer_and_sum_constraints() -> None:
    result = optimize_bounded(
        lambda x: float((x[0] - 0.25) ** 2 + (x[1] - 0.5) ** 2 + (x[2] - 2.0) ** 2),
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 3.0],
        max_generations=35,
        population_size=8,
        integer_indices=(2,),
        sum_limit_count=2,
        max_sum=0.8,
        seed=7,
    )

    assert result.parameters[0] + result.parameters[1] <= 0.8 + 1e-10
    assert result.parameters[2] == round(result.parameters[2])
    assert result.fitness < 1e-6
    assert result.evaluations > 0
    assert result.history


def test_optimizer_checks_cancellation_before_work() -> None:
    cancelled = Event()
    cancelled.set()
    with pytest.raises(OptimizationCancelled, match="cancelled"):
        optimize_bounded(lambda x: float(np.sum(x**2)), [-1.0], [1.0], cancel_event=cancelled)


def test_analytic_parameter_mapping_matches_direct_workflow() -> None:
    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        Nx=128,
        Nz=5,
    )
    direct = run_analytic_rcwa(template)

    value = evaluate_analytic_parameters(template, [0.32, 0.154, 0.182], "T(0)")

    assert value == pytest.approx(
        legacy_objective_value(direct.transmission, direct.reflection, "T(0)", flavor="analytic"),
        abs=1e-12,
    )


def test_imported_parameter_mapping_preserves_first_three_layer_quirk() -> None:
    imported = load_imported_device(Path(__file__).parents[1] / "DFB files" / "RCWA_DATA.mat")
    template = ImportedRCWARequest(
        device=imported,
        wavelengths_m=[650e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        n_substrate=1.47,
    )
    direct = run_imported_rcwa(template)
    parameters = np.concatenate(([imported.Lx_um], imported.sub_L_um[:3]))

    value = evaluate_imported_parameters(template, parameters, "R(0)")

    assert value == pytest.approx(
        legacy_objective_value(direct.transmission, direct.reflection, "R(0)", flavor="imported"),
        abs=1e-12,
    )


def test_optimization_service_returns_final_s_matrix_run(monkeypatch: pytest.MonkeyPatch) -> None:
    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        matrix_method="T",
        Nx=128,
        Nz=5,
    )
    progress_events: list[OptimizationProgress] = []

    def fake_optimize(objective, lower_bounds, upper_bounds, **kwargs):
        parameters = np.array([0.32, 0.154, 0.182])
        fitness = objective(parameters)
        snapshot = OptimizationProgress(1, 1, fitness, parameters)
        kwargs["progress"](snapshot)
        return OptimizationResult(
            parameters=parameters,
            fitness=fitness,
            success=True,
            message="fixed test optimizer",
            generations=1,
            evaluations=1,
            history=(snapshot,),
        )

    monkeypatch.setattr(optimization_module, "optimize_bounded", fake_optimize)
    request = OptimizationRequest(
        source="analytic",
        template=template,
        objective="T(0)",
        lower_bounds=[0.3, 0.1, 0.1],
        upper_bounds=[0.4, 0.2, 0.3],
    )

    run = run_optimization(request, progress=progress_events.append)

    assert run.metadata["workflow"] == "optimization"
    assert run.metadata["matrix_method_forced"] == "S"
    assert run.final_run.metadata["matrix_method"] == "S"
    assert progress_events and progress_events[0].generation == 1
    assert run.final_fitness == pytest.approx(
        legacy_objective_value(run.final_run.transmission, run.final_run.reflection, "T(0)", flavor="analytic"),
        abs=1e-12,
    )
    bundle = run.legacy_bundle()
    np.testing.assert_array_equal(bundle.params, np.array([0.32, 0.154, 0.182]))
    assert bundle.metadata is not None
    assert bundle.metadata["workflow"] == "optimization"
    assert bundle.metadata["matrix_method_forced"] == "S"
    assert bundle.metadata["final_objective"] == "T(0)"
    assert bundle.metadata["final_fitness"] == pytest.approx(run.final_fitness)
    assert bundle.metadata["optimization_evaluations"] == 1


def test_optimization_service_rejects_unimplemented_legacy_orders() -> None:
    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
    )
    request = OptimizationRequest(
        source="analytic",
        template=template,
        objective="R(+2)",  # type: ignore[arg-type]
        lower_bounds=[0.3, 0.1, 0.1],
        upper_bounds=[0.4, 0.2, 0.3],
    )

    with pytest.raises(ValueError, match="unsupported legacy objective"):
        run_optimization(request)


def test_optimization_service_honors_preflight_cancellation() -> None:
    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
    )
    request = OptimizationRequest(
        source="analytic",
        template=template,
        objective="R(0)",
        lower_bounds=[0.3, 0.1, 0.1],
        upper_bounds=[0.4, 0.2, 0.3],
    )
    cancelled = Event()
    cancelled.set()

    with pytest.raises(OptimizationCancelled, match="cancelled"):
        run_optimization(request, cancel_event=cancelled)


def test_analytic_evaluator_translates_solver_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    import zenscat.workflows as workflow_module

    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
    )

    def cancelled_run(*args, **kwargs):
        raise SimulationCancelled("simulation cancelled")

    monkeypatch.setattr(workflow_module, "run_analytic_rcwa", cancelled_run)

    with pytest.raises(OptimizationCancelled, match="cancelled") as raised:
        evaluate_analytic_parameters(template, [0.32, 0.154, 0.182], "R(0)")
    assert isinstance(raised.value.__cause__, SimulationCancelled)


def test_imported_evaluator_translates_solver_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    import zenscat.workflows as workflow_module

    imported = load_imported_device(Path(__file__).parents[1] / "DFB files" / "RCWA_DATA.mat")
    template = ImportedRCWARequest(
        device=imported,
        wavelengths_m=[650e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        n_substrate=1.47,
    )

    def cancelled_run(*args, **kwargs):
        raise SimulationCancelled("simulation cancelled")

    monkeypatch.setattr(workflow_module, "run_imported_rcwa", cancelled_run)

    with pytest.raises(OptimizationCancelled, match="cancelled") as raised:
        evaluate_imported_parameters(template, [imported.Lx_um, *imported.sub_L_um[:3]], "R(0)")
    assert isinstance(raised.value.__cause__, SimulationCancelled)


def test_optimization_service_translates_final_verification_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zenscat.workflows as workflow_module

    template = AnalyticRCWARequest(
        params=[0.182, 0.120, 1.781, 1.650],
        layer_num=2,
        wavelengths_m=[510e-9],
        angles_rad=[0.0],
        harmonic_count=2,
        Nx=128,
        Nz=5,
    )

    def fake_optimize(objective, lower_bounds, upper_bounds, **kwargs):
        return OptimizationResult(
            parameters=np.array([0.32, 0.154, 0.182]),
            fitness=-1.0,
            success=True,
            message="fixed test optimizer",
            generations=1,
            evaluations=1,
            history=(),
        )

    def cancelled_run(*args, **kwargs):
        raise SimulationCancelled("simulation cancelled")

    monkeypatch.setattr(optimization_module, "optimize_bounded", fake_optimize)
    monkeypatch.setattr(workflow_module, "run_analytic_rcwa", cancelled_run)
    request = OptimizationRequest(
        source="analytic",
        template=template,
        objective="T(0)",
        lower_bounds=[0.3, 0.1, 0.1],
        upper_bounds=[0.4, 0.2, 0.3],
    )

    with pytest.raises(OptimizationCancelled, match="cancelled") as raised:
        run_optimization(request)
    assert isinstance(raised.value.__cause__, SimulationCancelled)
