from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import zenscat.optimization as optimization_module
from zenscat.core import DiffractionResult
from zenscat.legacy_io import ImportedDevice
from zenscat.optimization import (
    OptimizationCheckpoint,
    OptimizationMeritProgress,
    OptimizationProgress,
    OptimizationRequest,
    evaluate_imported_layer_parameters,
    legacy_objective_value,
    optimize_bounded,
    restore_optimization_checkpoint,
)
from zenscat.workflows import ImportedRCWARequest


def _extended_results() -> tuple[DiffractionResult, DiffractionResult]:
    trn = DiffractionResult(
        minus_1=np.array([[0.1]]),
        plus_1=np.array([[0.3]]),
        TRN0=np.array([[0.5]]),
        sum=np.array([[0.9]]),
    )
    ref = DiffractionResult(
        minus_1=np.array([[0.01]]),
        plus_1=np.array([[0.03]]),
        REF0=np.array([[0.05]]),
        sum=np.array([[0.1]]),
    )
    object.__setattr__(trn, "minus_2", np.array([[0.07]]))
    object.__setattr__(trn, "plus_2", np.array([[0.11]]))
    object.__setattr__(ref, "minus_2", np.array([[0.13]]))
    object.__setattr__(ref, "plus_2", np.array([[0.17]]))
    return trn, ref


def test_plus_minus_two_objectives_are_mode_gated() -> None:
    trn, ref = _extended_results()

    with pytest.raises(ValueError, match="not implemented by legacy MATLAB merit"):
        legacy_objective_value(trn, ref, "R(+2)", compatibility_mode="legacy_exact")

    assert legacy_objective_value(trn, ref, "R(+2)", compatibility_mode="corrected") == pytest.approx(-0.17)
    assert legacy_objective_value(trn, ref, "R(-2)", compatibility_mode="modern") == pytest.approx(-0.13)
    assert legacy_objective_value(trn, ref, "T(+2)", compatibility_mode="corrected") == pytest.approx(-0.11)
    assert legacy_objective_value(trn, ref, "T(-2)", compatibility_mode="modern") == pytest.approx(-0.07)


def test_checkpoint_round_trip_and_resume_keeps_best_seeded() -> None:
    first = optimize_bounded(
        lambda x: float((x[0] - 0.2) ** 2 + (x[1] - 0.4) ** 2),
        [0.0, 0.0],
        [1.0, 1.0],
        max_generations=2,
        population_size=5,
        seed=3,
        polish=False,
    )
    assert first.checkpoint is not None

    restored = restore_optimization_checkpoint(first.checkpoint.to_dict())
    resumed = optimize_bounded(
        lambda x: float((x[0] - 0.2) ** 2 + (x[1] - 0.4) ** 2),
        [0.0, 0.0],
        [1.0, 1.0],
        max_generations=4,
        population_size=5,
        seed=3,
        polish=False,
        checkpoint=restored,
    )

    assert resumed.generations >= first.generations
    assert resumed.evaluations > first.evaluations
    assert resumed.fitness <= first.fitness + 1e-12
    assert any(item.generation == first.generations + 1 for item in resumed.history)


def test_resume_uses_remaining_generation_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_maxiter: list[int] = []

    def fake_de(func, *, maxiter, callback, **kwargs):
        seen_maxiter.append(maxiter)
        for _ in range(maxiter):
            candidate = np.array([0.25], dtype=np.float64)
            value = func(candidate)
            assert value == pytest.approx(0.0)
            callback(candidate, 0.0)
        return OptimizeResult(x=np.array([0.25]), fun=0.0, success=True, message="ok", nit=maxiter, nfev=0)

    monkeypatch.setattr(optimization_module, "differential_evolution", fake_de)
    checkpoint = OptimizationCheckpoint(
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        seed=11,
        population_size=4,
        max_generations=2,
        integer_indices=(),
        sum_limit_count=0,
        max_sum=None,
        compatibility_profile="scipy_de",
        generations_completed=2,
        evaluations=8,
        best_fitness=0.01,
        best_parameters=np.array([0.2]),
        history=(
            OptimizationProgress(1, 4, 0.02, np.array([0.1])),
            OptimizationProgress(2, 8, 0.01, np.array([0.2])),
        ),
    )

    result = optimize_bounded(
        lambda x: float((x[0] - 0.25) ** 2),
        [0.0],
        [1.0],
        max_generations=5,
        population_size=4,
        seed=11,
        checkpoint=checkpoint,
        polish=False,
    )

    assert seen_maxiter == [3]
    assert result.generations == 5
    assert result.checkpoint is not None
    assert result.checkpoint.generations_completed == 5
    assert [item.generation for item in result.history[-3:]] == [3, 4, 5]


def test_already_complete_checkpoint_returns_without_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_de(*args, **kwargs):
        raise AssertionError("completed checkpoint must not invoke the solver")

    monkeypatch.setattr(optimization_module, "differential_evolution", unexpected_de)
    checkpoint = OptimizationCheckpoint(
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        seed=2,
        population_size=3,
        max_generations=2,
        integer_indices=(),
        sum_limit_count=0,
        max_sum=None,
        compatibility_profile="scipy_de",
        generations_completed=2,
        evaluations=6,
        best_fitness=0.004,
        best_parameters=np.array([0.42]),
        history=(
            OptimizationProgress(1, 3, 0.02, np.array([0.3])),
            OptimizationProgress(2, 6, 0.004, np.array([0.42])),
        ),
    )

    result = optimize_bounded(
        lambda x: float((x[0] - 0.4) ** 2),
        [0.0],
        [1.0],
        max_generations=2,
        population_size=3,
        seed=2,
        checkpoint=checkpoint,
        polish=False,
    )

    assert result.success is True
    assert result.message == "checkpoint already completed requested generation budget"
    assert result.generations == 2
    assert result.evaluations == 6
    np.testing.assert_allclose(result.parameters, [0.42])


def test_checkpoint_ahead_of_requested_budget_is_rejected() -> None:
    checkpoint = OptimizationCheckpoint(
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        seed=2,
        population_size=3,
        max_generations=3,
        integer_indices=(),
        sum_limit_count=0,
        max_sum=None,
        compatibility_profile="scipy_de",
        generations_completed=3,
        evaluations=9,
        best_fitness=0.004,
        best_parameters=np.array([0.42]),
        history=(),
    )

    with pytest.raises(ValueError, match="checkpoint generations exceed"):
        optimize_bounded(
            lambda x: float((x[0] - 0.4) ** 2),
            [0.0],
            [1.0],
            max_generations=2,
            population_size=3,
            seed=2,
            checkpoint=checkpoint,
            polish=False,
        )


def test_merit_progress_reports_current_and_best_per_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_de(func, *, callback, **kwargs):
        first = np.array([0.9], dtype=np.float64)
        second = np.array([0.2], dtype=np.float64)
        third = np.array([0.6], dtype=np.float64)
        func(first)
        func(second)
        callback(second, 0.0)
        func(third)
        callback(third, 0.0)
        return OptimizeResult(x=second, fun=0.04, success=True, message="ok", nit=2, nfev=3)

    monkeypatch.setattr(optimization_module, "differential_evolution", fake_de)
    numeric_events: list[OptimizationProgress] = []
    merit_events: list[OptimizationMeritProgress] = []

    result = optimize_bounded(
        lambda x: float((x[0] - 0.0) ** 2),
        [0.0],
        [1.0],
        max_generations=2,
        population_size=3,
        seed=4,
        progress=numeric_events.append,
        merit_progress=merit_events.append,
        polish=False,
    )

    assert result.generations == 2
    assert [event.generation for event in numeric_events] == [1, 2]
    assert [event.generation for event in merit_events] == [1, 2]
    assert merit_events[0].current_fitness == pytest.approx(0.04)
    assert merit_events[0].best_fitness == pytest.approx(0.04)
    assert merit_events[1].current_fitness == pytest.approx(0.36)
    assert merit_events[1].best_fitness == pytest.approx(0.04)
    np.testing.assert_allclose(merit_events[1].current_parameters, [0.6])
    np.testing.assert_allclose(merit_events[1].best_parameters, [0.2])


def test_time_limit_returns_partial_result_after_first_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    clock_values = iter([0.0, 5.0])

    def fake_monotonic() -> float:
        return next(clock_values)

    def fake_de(func, **kwargs):
        func(np.array([0.3], dtype=np.float64))
        func(np.array([0.8], dtype=np.float64))
        raise AssertionError("second objective call should hit the time limit")

    monkeypatch.setattr(optimization_module, "monotonic", fake_monotonic)
    monkeypatch.setattr(optimization_module, "differential_evolution", fake_de)

    result = optimize_bounded(
        lambda x: float((x[0] - 0.25) ** 2),
        [0.0],
        [1.0],
        max_generations=5,
        population_size=3,
        seed=1,
        time_limit_s=1.0,
        polish=False,
    )

    assert result.success is False
    assert result.message == "optimization time limit reached; returning best candidate"
    assert result.generations == 0
    assert result.evaluations == 1
    assert result.fitness == pytest.approx(0.0025)
    np.testing.assert_allclose(result.parameters, [0.3])


def test_checkpoint_dict_validation_rejects_mismatched_resume_bounds() -> None:
    checkpoint = OptimizationCheckpoint(
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        seed=1,
        population_size=3,
        max_generations=1,
        integer_indices=(),
        sum_limit_count=0,
        max_sum=None,
        compatibility_profile="scipy_de",
        generations_completed=1,
        evaluations=3,
        best_fitness=0.1,
        best_parameters=np.array([0.5]),
        history=(),
    )

    with pytest.raises(ValueError, match="checkpoint bounds"):
        optimize_bounded(
            lambda x: float(np.sum(x**2)),
            [0.0, 0.0],
            [1.0, 1.0],
            checkpoint=checkpoint.to_dict(),
            polish=False,
        )


def test_imported_layer_parameter_contract_supports_arbitrary_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, np.ndarray] = {}

    def fake_run(request, **kwargs):
        captured["sub_L_um"] = request.device.sub_L_um.copy()
        return type(
            "Run",
            (),
            {
                "transmission": DiffractionResult(
                    minus_1=np.array([[0.0]]),
                    plus_1=np.array([[0.0]]),
                    TRN0=np.array([[0.2]]),
                    sum=np.array([[0.2]]),
                ),
                "reflection": DiffractionResult(
                    minus_1=np.array([[0.0]]),
                    plus_1=np.array([[0.0]]),
                    REF0=np.array([[0.3]]),
                    sum=np.array([[0.3]]),
                ),
            },
        )()

    monkeypatch.setattr("zenscat.workflows.run_imported_rcwa", fake_run)
    device = ImportedDevice(
        ER=np.ones((4, 8), dtype=np.complex128),
        sub_L_um=np.array([0.1, 0.2, 0.3, 0.4]),
        x_um=np.linspace(-0.5, 0.5, 8),
        Lx_um=1.0,
    )
    template = ImportedRCWARequest(device=device, wavelengths_m=[500e-9], angles_rad=[0.0])

    value = evaluate_imported_layer_parameters(
        template,
        [0.25, 0.45],
        "R(0)",
        layer_indices=[1, 3],
        compatibility_mode="modern",
    )

    np.testing.assert_allclose(captured["sub_L_um"], [0.1, 0.25, 0.3, 0.45])
    assert value == pytest.approx(-0.3)


def test_run_optimization_legacy_exact_rejects_plus_minus_two_before_solver() -> None:
    device = ImportedDevice(
        ER=np.ones((3, 8), dtype=np.complex128),
        sub_L_um=np.array([0.1, 0.2, 0.3]),
        x_um=np.linspace(-0.5, 0.5, 8),
        Lx_um=1.0,
    )
    request = OptimizationRequest(
        source="imported",
        template=ImportedRCWARequest(device=device, wavelengths_m=[500e-9], angles_rad=[0.0]),
        objective="T(-2)",
        lower_bounds=[1.0, 0.1, 0.2, 0.3],
        upper_bounds=[1.0, 0.1, 0.2, 0.3],
        compatibility_mode="legacy_exact",
    )

    with pytest.raises(ValueError, match="not implemented by legacy MATLAB merit"):
        optimization_module.run_optimization(request)
