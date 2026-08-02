"""Legacy objective semantics with a deterministic, cancellable Python optimizer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, differential_evolution

from .core import DiffractionResult, ProgressCallback, SimulationCancelled

if TYPE_CHECKING:
    from .workflows import AnalyticRCWARequest, ImportedRCWARequest, RCWARun


ObjectiveName = Literal[
    "R(-1)",
    "R(0)",
    "R(+1)",
    "T(-1)",
    "T(0)",
    "T(+1)",
    "Absorption",
    "Gain",
]
ObjectiveFlavor = Literal["analytic", "imported"]
FloatArray = NDArray[np.float64]
SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "R(-1)",
    "R(0)",
    "R(+1)",
    "T(-1)",
    "T(0)",
    "T(+1)",
    "Absorption",
    "Gain",
)


class OptimizationCancelled(RuntimeError):
    """Raised when a caller cancels an optimization job."""


@dataclass(frozen=True)
class OptimizationProgress:
    generation: int
    evaluations: int
    best_fitness: float
    best_parameters: FloatArray


@dataclass(frozen=True)
class OptimizationResult:
    parameters: FloatArray
    fitness: float
    success: bool
    message: str
    generations: int
    evaluations: int
    history: tuple[OptimizationProgress, ...]


@dataclass(frozen=True)
class OptimizationRequest:
    """Thin service request for legacy-compatible bounded optimization.

    ``matrix_method`` on the RCWA template is intentionally forced to ``S``
    during merit evaluation and final verification, matching ZenScat's
    ``Merit_Function3.m`` and ``Merit_Function_Import.m`` paths.
    """

    source: ObjectiveFlavor
    template: AnalyticRCWARequest | ImportedRCWARequest
    objective: ObjectiveName
    lower_bounds: ArrayLike
    upper_bounds: ArrayLike
    max_generations: int = 100
    population_size: int = 15
    seed: int = 0
    integer_indices: Sequence[int] = ()
    sum_limit_count: int = 0
    max_sum: float | None = None
    time_limit_s: float | None = None
    polish: bool = True


@dataclass(frozen=True)
class OptimizationRun:
    source: ObjectiveFlavor
    objective: ObjectiveName
    optimization: OptimizationResult
    final_run: RCWARun
    final_parameters: FloatArray
    final_fitness: float
    metadata: dict[str, object]

    def legacy_bundle(self):
        metadata = {
            **self.final_run.metadata,
            **self.metadata,
            "final_objective": self.objective,
            "final_fitness": self.final_fitness,
            "optimization_fitness": self.optimization.fitness,
            "optimization_success": self.optimization.success,
            "optimization_generations": self.optimization.generations,
            "optimization_evaluations": self.optimization.evaluations,
        }
        return self.final_run.legacy_bundle(params=self.final_parameters, metadata=metadata)


def legacy_objective_value(
    transmission: DiffractionResult,
    reflection: DiffractionResult,
    objective: ObjectiveName,
    *,
    flavor: ObjectiveFlavor = "analytic",
) -> float:
    """Return the exact scalar sign/absolute-value semantics of the MATLAB merits."""

    if flavor not in {"analytic", "imported"}:
        raise ValueError("flavor must be 'analytic' or 'imported'")
    values = {
        "R(-1)": reflection.minus_1,
        "R(0)": reflection.REF0,
        "R(+1)": reflection.plus_1,
        "T(-1)": transmission.minus_1,
        "T(0)": transmission.TRN0,
        "T(+1)": transmission.plus_1,
    }
    if objective in values:
        selected = values[objective]
    elif objective == "Absorption":
        if transmission.sum is None or reflection.sum is None:
            raise ValueError("TRN.sum and REF.sum are required for absorption")
        total = np.asarray(transmission.sum) + np.asarray(reflection.sum)
        selected = -total if flavor == "analytic" else 1.0 - total
    elif objective == "Gain":
        if transmission.sum is None or reflection.sum is None:
            raise ValueError("TRN.sum and REF.sum are required for gain")
        selected = np.asarray(transmission.sum) + np.asarray(reflection.sum)
    else:
        raise ValueError(f"unsupported legacy objective: {objective}")
    if selected is None:
        raise ValueError(f"required result field is missing for {objective}")

    selected_array = np.asarray(selected, dtype=np.float64)
    if not np.isfinite(selected_array).all():
        raise ValueError("objective inputs must be finite")
    if flavor == "imported":
        return -float(np.sum(np.abs(selected_array)))
    return -float(np.sum(selected_array))


def evaluate_analytic_parameters(
    template: AnalyticRCWARequest,
    parameters: ArrayLike,
    objective: ObjectiveName,
    *,
    cancel_event: Event | None = None,
) -> float:
    """Evaluate the exact ``Merit_Function3.m`` parameter mapping.

    MATLAB maps ``X`` to period, modulation depth, and only the first layer
    thickness, then always evaluates the S-matrix solver.
    """

    from .workflows import run_analytic_rcwa

    values = np.asarray(parameters, dtype=np.float64).ravel()
    if values.size < 3:
        raise ValueError("analytic legacy parameters require [Lx, h, first_thickness]")
    geometry_params = np.asarray(template.params, dtype=np.float64).ravel().copy()
    if geometry_params.size == 0:
        raise ValueError("template params must not be empty")
    geometry_params[0] = values[2]
    request = replace(
        template,
        params=geometry_params,
        Lx_um=float(values[0]),
        h_um=float(values[1]),
        matrix_method="S",
    )
    try:
        result = run_analytic_rcwa(request, cancel_token=cancel_event)
    except SimulationCancelled as exc:
        raise OptimizationCancelled("optimization cancelled") from exc
    return legacy_objective_value(result.transmission, result.reflection, objective, flavor="analytic")


def evaluate_imported_parameters(
    template: ImportedRCWARequest,
    parameters: ArrayLike,
    objective: ObjectiveName,
    *,
    cancel_event: Event | None = None,
) -> float:
    """Evaluate ``Merit_Function_Import.m``, including its first-three-layer quirk."""

    from .legacy_io import ImportedDevice
    from .workflows import run_imported_rcwa

    values = np.asarray(parameters, dtype=np.float64).ravel()
    if values.size < 4:
        raise ValueError("imported legacy parameters require [Lx, sub_L1, sub_L2, sub_L3]")
    if template.device.sub_L_um.size < 3:
        raise ValueError("legacy imported optimization requires at least three device layers")
    thicknesses = template.device.sub_L_um.copy()
    thicknesses[:3] = values[1:4]
    imported = ImportedDevice(
        ER=template.device.ER,
        sub_L_um=thicknesses,
        x_um=template.device.x_um,
        Lx_um=float(values[0]),
        source=template.device.source,
    )
    request = replace(template, device=imported, matrix_method="S")
    try:
        result = run_imported_rcwa(request, cancel_token=cancel_event)
    except SimulationCancelled as exc:
        raise OptimizationCancelled("optimization cancelled") from exc
    return legacy_objective_value(result.transmission, result.reflection, objective, flavor="imported")


def optimize_bounded(
    objective: Callable[[FloatArray], float],
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    *,
    max_generations: int = 100,
    population_size: int = 15,
    seed: int = 0,
    integer_indices: Sequence[int] = (),
    sum_limit_count: int = 0,
    max_sum: float | None = None,
    time_limit_s: float | None = None,
    cancel_event: Event | None = None,
    progress: Callable[[OptimizationProgress], None] | None = None,
    polish: bool = True,
) -> OptimizationResult:
    """Run a deterministic SciPy differential-evolution replacement for MATLAB GA.

    This intentionally preserves ZenScat's bounds, optional integer material
    variables, and the linear thickness-sum constraint. It does not claim to
    reproduce MATLAB's proprietary GA trajectory.
    """

    lower = np.asarray(lower_bounds, dtype=np.float64).ravel()
    upper = np.asarray(upper_bounds, dtype=np.float64).ravel()
    if lower.size == 0 or lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must be non-empty and have equal shape")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("bounds must be finite and lower <= upper")
    if max_generations < 1 or population_size < 1:
        raise ValueError("max_generations and population_size must be positive")
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if sum_limit_count < 0 or sum_limit_count > lower.size:
        raise ValueError("sum_limit_count is outside the parameter vector")
    if sum_limit_count and max_sum is None:
        raise ValueError("max_sum is required when sum_limit_count is non-zero")

    integrality = np.zeros(lower.size, dtype=bool)
    for index in integer_indices:
        if index < 0 or index >= lower.size:
            raise ValueError(f"integer index out of range: {index}")
        integrality[index] = True

    constraints: tuple[LinearConstraint, ...] = ()
    if sum_limit_count:
        assert max_sum is not None
        coefficients = np.zeros((1, lower.size), dtype=np.float64)
        coefficients[0, :sum_limit_count] = 1.0
        constraints = (LinearConstraint(coefficients, -np.inf, float(max_sum)),)

    started = monotonic()
    cancellation = cancel_event or Event()
    evaluations = 0
    best_value = np.inf
    best_parameters = lower.copy()
    history: list[OptimizationProgress] = []

    def checked_objective(parameters: FloatArray) -> float:
        nonlocal evaluations, best_value, best_parameters
        if cancellation.is_set():
            raise OptimizationCancelled("optimization cancelled")
        if time_limit_s is not None and monotonic() - started >= time_limit_s:
            raise OptimizationCancelled("optimization time limit reached")
        value = float(objective(np.asarray(parameters, dtype=np.float64)))
        if not np.isfinite(value):
            raise ValueError("objective returned a non-finite value")
        evaluations += 1
        if value < best_value:
            best_value = value
            best_parameters = np.asarray(parameters, dtype=np.float64).copy()
        return value

    generation = 0

    def on_generation(_parameters: FloatArray, _convergence: float) -> bool:
        nonlocal generation
        generation += 1
        snapshot = OptimizationProgress(generation, evaluations, best_value, best_parameters.copy())
        history.append(snapshot)
        if progress is not None:
            progress(snapshot)
        if cancellation.is_set():
            raise OptimizationCancelled("optimization cancelled")
        if time_limit_s is not None and monotonic() - started >= time_limit_s:
            raise OptimizationCancelled("optimization time limit reached")
        return False

    de_solver = cast(Any, differential_evolution)
    result = de_solver(
        checked_objective,
        bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)),
        constraints=constraints,
        integrality=integrality,
        maxiter=max_generations,
        popsize=population_size,
        seed=seed,
        callback=on_generation,
        polish=polish,
        workers=1,
        updating="immediate",
    )
    return OptimizationResult(
        parameters=np.asarray(result.x, dtype=np.float64),
        fitness=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        generations=int(result.nit),
        evaluations=int(result.nfev),
        history=tuple(history),
    )


def run_optimization(
    request: OptimizationRequest,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[OptimizationProgress], None] | None = None,
    solver_progress: ProgressCallback | None = None,
) -> OptimizationRun:
    """Run bounded optimization and return an export-ready final RCWA result."""

    _validate_optimization_request(request)
    from .workflows import AnalyticRCWARequest, ImportedRCWARequest

    cancellation = cancel_event or Event()
    if request.source == "analytic":
        if not isinstance(request.template, AnalyticRCWARequest):
            raise TypeError("analytic optimization requires AnalyticRCWARequest")
        analytic_template = request.template

        def objective(x: FloatArray) -> float:
            return evaluate_analytic_parameters(
                analytic_template,
                x,
                request.objective,
                cancel_event=cancellation,
            )
    elif request.source == "imported":
        if not isinstance(request.template, ImportedRCWARequest):
            raise TypeError("imported optimization requires ImportedRCWARequest")
        imported_template = request.template

        def objective(x: FloatArray) -> float:
            return evaluate_imported_parameters(
                imported_template,
                x,
                request.objective,
                cancel_event=cancellation,
            )
    else:
        raise ValueError("source must be 'analytic' or 'imported'")

    optimization = optimize_bounded(
        objective,
        request.lower_bounds,
        request.upper_bounds,
        max_generations=request.max_generations,
        population_size=request.population_size,
        seed=request.seed,
        integer_indices=request.integer_indices,
        sum_limit_count=request.sum_limit_count,
        max_sum=request.max_sum,
        time_limit_s=request.time_limit_s,
        cancel_event=cancellation,
        progress=progress,
        polish=request.polish,
    )
    try:
        final_run = _run_final_physical_rcwa(
            request,
            optimization.parameters,
            cancellation,
            solver_progress,
        )
    except SimulationCancelled as exc:
        raise OptimizationCancelled("optimization cancelled") from exc
    flavor: ObjectiveFlavor = "analytic" if request.source == "analytic" else "imported"
    final_fitness = legacy_objective_value(
        final_run.transmission,
        final_run.reflection,
        request.objective,
        flavor=flavor,
    )
    metadata = {
        "workflow": "optimization",
        "source": request.source,
        "objective": request.objective,
        "matrix_method_forced": "S",
        "supported_objectives": list(SUPPORTED_OBJECTIVES),
        "optimizer": "scipy.differential_evolution",
        "matlab_ga_trajectory_compatibility": False,
    }
    return OptimizationRun(
        source=request.source,
        objective=request.objective,
        optimization=optimization,
        final_run=final_run,
        final_parameters=np.asarray(optimization.parameters, dtype=np.float64).copy(),
        final_fitness=final_fitness,
        metadata=metadata,
    )


def _validate_optimization_request(request: OptimizationRequest) -> None:
    if request.source not in {"analytic", "imported"}:
        raise ValueError("source must be 'analytic' or 'imported'")
    if request.objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported legacy objective: {request.objective}")


def _run_final_physical_rcwa(
    request: OptimizationRequest,
    parameters: ArrayLike,
    cancel_event: Event,
    solver_progress: ProgressCallback | None,
) -> RCWARun:
    from .workflows import (
        AnalyticRCWARequest,
        ImportedRCWARequest,
        run_analytic_rcwa,
        run_imported_rcwa,
    )

    if request.source == "analytic":
        if not isinstance(request.template, AnalyticRCWARequest):
            raise TypeError("analytic optimization requires AnalyticRCWARequest")
        final_request = _analytic_request_for_parameters(request.template, parameters)
        return run_analytic_rcwa(final_request, cancel_token=cancel_event, progress=solver_progress)
    if not isinstance(request.template, ImportedRCWARequest):
        raise TypeError("imported optimization requires ImportedRCWARequest")
    final_request = _imported_request_for_parameters(request.template, parameters)
    return run_imported_rcwa(final_request, cancel_token=cancel_event, progress=solver_progress)


def _analytic_request_for_parameters(template: AnalyticRCWARequest, parameters: ArrayLike) -> AnalyticRCWARequest:
    values = np.asarray(parameters, dtype=np.float64).ravel()
    if values.size < 3:
        raise ValueError("analytic legacy parameters require [Lx, h, first_thickness]")
    geometry_params = np.asarray(template.params, dtype=np.float64).ravel().copy()
    if geometry_params.size == 0:
        raise ValueError("template params must not be empty")
    geometry_params[0] = values[2]
    return replace(
        template,
        params=geometry_params,
        Lx_um=float(values[0]),
        h_um=float(values[1]),
        matrix_method="S",
    )


def _imported_request_for_parameters(template: ImportedRCWARequest, parameters: ArrayLike) -> ImportedRCWARequest:
    from .legacy_io import ImportedDevice

    values = np.asarray(parameters, dtype=np.float64).ravel()
    if values.size < 4:
        raise ValueError("imported legacy parameters require [Lx, sub_L1, sub_L2, sub_L3]")
    if template.device.sub_L_um.size < 3:
        raise ValueError("legacy imported optimization requires at least three device layers")
    thicknesses = template.device.sub_L_um.copy()
    thicknesses[:3] = values[1:4]
    imported = ImportedDevice(
        ER=template.device.ER,
        sub_L_um=thicknesses,
        x_um=template.device.x_um,
        Lx_um=float(values[0]),
        source=template.device.source,
    )
    return replace(template, device=imported, matrix_method="S")
