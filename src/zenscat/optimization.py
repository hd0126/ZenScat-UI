"""Legacy objective semantics with a deterministic, cancellable Python optimizer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import LinearConstraint, OptimizeResult, differential_evolution

from .core import DiffractionResult, ProgressCallback, SimulationCancelled

if TYPE_CHECKING:
    from .workflows import AnalyticRCWARequest, ImportedRCWARequest, RCWARun


ObjectiveName = Literal[
    "R(-2)",
    "R(-1)",
    "R(0)",
    "R(+1)",
    "R(+2)",
    "T(-2)",
    "T(-1)",
    "T(0)",
    "T(+1)",
    "T(+2)",
    "Absorption",
    "Gain",
]
ObjectiveFlavor = Literal["analytic", "imported"]
OptimizationCompatibilityMode = Literal["legacy_exact", "corrected", "modern"]
OptimizationParameterContract = Literal["legacy", "imported_layers"]
OptimizerProfile = Literal["scipy_de", "ga_compat"]
FloatArray = NDArray[np.float64]
SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "R(-2)",
    "R(-1)",
    "R(0)",
    "R(+1)",
    "R(+2)",
    "T(-2)",
    "T(-1)",
    "T(0)",
    "T(+1)",
    "T(+2)",
    "Absorption",
    "Gain",
)
LEGACY_EXACT_OBJECTIVES: tuple[str, ...] = (
    "R(-1)",
    "R(0)",
    "R(+1)",
    "T(-1)",
    "T(0)",
    "T(+1)",
    "Absorption",
    "Gain",
)
PLUS_MINUS_TWO_OBJECTIVES = {"R(-2)", "R(+2)", "T(-2)", "T(+2)"}


class OptimizationCancelled(RuntimeError):
    """Raised when a caller cancels an optimization job."""


class OptimizationTimeLimitReached(RuntimeError):
    """Internal signal used to return the best candidate at the time limit."""


@dataclass(frozen=True)
class OptimizationProgress:
    generation: int
    evaluations: int
    best_fitness: float
    best_parameters: FloatArray


@dataclass(frozen=True)
class OptimizationMeritProgress:
    """Plain callback payload for UI-safe merit history rendering."""

    generation: int
    evaluations: int
    current_fitness: float
    current_parameters: FloatArray
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
    checkpoint: OptimizationCheckpoint | None = None


@dataclass(frozen=True)
class OptimizationCheckpoint:
    """Serializable state for deterministic best-seeded continuation.

    SciPy's differential-evolution implementation does not expose its private
    population RNG state as a stable public contract.  This checkpoint stores
    the bounded problem contract plus the best known candidate and history;
    resume seeds a fresh deterministic population around that best candidate.
    """

    lower_bounds: FloatArray
    upper_bounds: FloatArray
    seed: int
    population_size: int
    max_generations: int
    integer_indices: tuple[int, ...]
    sum_limit_count: int
    max_sum: float | None
    compatibility_profile: OptimizerProfile
    generations_completed: int
    evaluations: int
    best_fitness: float
    best_parameters: FloatArray
    history: tuple[OptimizationProgress, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "lower_bounds": self.lower_bounds.tolist(),
            "upper_bounds": self.upper_bounds.tolist(),
            "seed": self.seed,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "integer_indices": list(self.integer_indices),
            "sum_limit_count": self.sum_limit_count,
            "max_sum": self.max_sum,
            "compatibility_profile": self.compatibility_profile,
            "generations_completed": self.generations_completed,
            "evaluations": self.evaluations,
            "best_fitness": self.best_fitness,
            "best_parameters": self.best_parameters.tolist(),
            "history": [
                {
                    "generation": item.generation,
                    "evaluations": item.evaluations,
                    "best_fitness": item.best_fitness,
                    "best_parameters": item.best_parameters.tolist(),
                }
                for item in self.history
            ],
        }


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
    compatibility_mode: OptimizationCompatibilityMode = "legacy_exact"
    parameter_contract: OptimizationParameterContract = "legacy"
    imported_layer_indices: Sequence[int] | None = None
    optimizer_profile: OptimizerProfile = "scipy_de"
    checkpoint: OptimizationCheckpoint | Mapping[str, object] | None = None


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
    compatibility_mode: OptimizationCompatibilityMode = "legacy_exact",
) -> float:
    """Return the exact scalar sign/absolute-value semantics of the MATLAB merits."""

    if flavor not in {"analytic", "imported"}:
        raise ValueError("flavor must be 'analytic' or 'imported'")
    _validate_compatibility_mode(compatibility_mode)
    if compatibility_mode == "legacy_exact" and objective in PLUS_MINUS_TWO_OBJECTIVES:
        raise ValueError(f"unsupported legacy objective: {objective} is not implemented by legacy MATLAB merit functions")
    values = {
        "R(-2)": _order_value(reflection, -2),
        "R(-1)": reflection.minus_1,
        "R(0)": reflection.REF0,
        "R(+1)": reflection.plus_1,
        "R(+2)": _order_value(reflection, +2),
        "T(-2)": _order_value(transmission, -2),
        "T(-1)": transmission.minus_1,
        "T(0)": transmission.TRN0,
        "T(+1)": transmission.plus_1,
        "T(+2)": _order_value(transmission, +2),
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
    compatibility_mode: OptimizationCompatibilityMode = "legacy_exact",
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
    return legacy_objective_value(
        result.transmission,
        result.reflection,
        objective,
        flavor="analytic",
        compatibility_mode=compatibility_mode,
    )


def evaluate_imported_parameters(
    template: ImportedRCWARequest,
    parameters: ArrayLike,
    objective: ObjectiveName,
    *,
    cancel_event: Event | None = None,
    compatibility_mode: OptimizationCompatibilityMode = "legacy_exact",
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
    return legacy_objective_value(
        result.transmission,
        result.reflection,
        objective,
        flavor="imported",
        compatibility_mode=compatibility_mode,
    )


def evaluate_imported_layer_parameters(
    template: ImportedRCWARequest,
    parameters: ArrayLike,
    objective: ObjectiveName,
    *,
    layer_indices: Sequence[int] | None = None,
    cancel_event: Event | None = None,
    compatibility_mode: OptimizationCompatibilityMode = "modern",
) -> float:
    """Evaluate an imported-device optimization over arbitrary layer thicknesses."""

    from .legacy_io import ImportedDevice
    from .workflows import run_imported_rcwa

    values = np.asarray(parameters, dtype=np.float64).ravel()
    if layer_indices is None:
        indices = tuple(range(values.size))
    else:
        indices = tuple(int(index) for index in layer_indices)
    if len(indices) != values.size:
        raise ValueError("layer_indices length must match the parameter vector")
    thicknesses = template.device.sub_L_um.copy()
    for index, value in zip(indices, values, strict=True):
        if index < 0 or index >= thicknesses.size:
            raise ValueError(f"imported layer index out of range: {index}")
        thicknesses[index] = value
    imported = ImportedDevice(
        ER=template.device.ER,
        sub_L_um=thicknesses,
        x_um=template.device.x_um,
        Lx_um=template.device.Lx_um,
        source=template.device.source,
    )
    request = replace(template, device=imported, matrix_method="S")
    try:
        result = run_imported_rcwa(request, cancel_token=cancel_event)
    except SimulationCancelled as exc:
        raise OptimizationCancelled("optimization cancelled") from exc
    return legacy_objective_value(
        result.transmission,
        result.reflection,
        objective,
        flavor="imported",
        compatibility_mode=compatibility_mode,
    )


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
    merit_progress: Callable[[OptimizationMeritProgress], None] | None = None,
    polish: bool = True,
    optimizer_profile: OptimizerProfile = "scipy_de",
    checkpoint: OptimizationCheckpoint | Mapping[str, object] | None = None,
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
    if optimizer_profile not in {"scipy_de", "ga_compat"}:
        raise ValueError("optimizer_profile must be 'scipy_de' or 'ga_compat'")
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

    restored_checkpoint = restore_optimization_checkpoint(checkpoint) if checkpoint is not None else None
    if restored_checkpoint is not None:
        _validate_checkpoint_resume(
            restored_checkpoint,
            lower,
            upper,
            seed=seed,
            population_size=population_size,
            integer_indices=tuple(int(index) for index in integer_indices),
            sum_limit_count=sum_limit_count,
            max_sum=max_sum,
            optimizer_profile=optimizer_profile,
        )
        if restored_checkpoint.generations_completed > max_generations:
            raise ValueError("checkpoint generations exceed requested max_generations")

    started = monotonic()
    cancellation = cancel_event or Event()
    evaluations = 0 if restored_checkpoint is None else restored_checkpoint.evaluations
    starting_evaluations = evaluations
    best_value = np.inf if restored_checkpoint is None else restored_checkpoint.best_fitness
    best_parameters = lower.copy() if restored_checkpoint is None else restored_checkpoint.best_parameters.copy()
    history: list[OptimizationProgress] = [] if restored_checkpoint is None else list(restored_checkpoint.history)
    latest_value = best_value
    latest_parameters = best_parameters.copy()
    generation = 0 if restored_checkpoint is None else restored_checkpoint.generations_completed

    if restored_checkpoint is not None and restored_checkpoint.generations_completed == max_generations:
        completed_history = tuple(item for item in history if item.generation <= max_generations)
        result_checkpoint = OptimizationCheckpoint(
            lower_bounds=lower.copy(),
            upper_bounds=upper.copy(),
            seed=int(seed),
            population_size=int(population_size),
            max_generations=int(max_generations),
            integer_indices=tuple(int(index) for index in integer_indices),
            sum_limit_count=int(sum_limit_count),
            max_sum=None if max_sum is None else float(max_sum),
            compatibility_profile=optimizer_profile,
            generations_completed=int(max_generations),
            evaluations=int(evaluations),
            best_fitness=float(best_value),
            best_parameters=best_parameters.copy(),
            history=completed_history,
        )
        return OptimizationResult(
            parameters=best_parameters.copy(),
            fitness=float(best_value),
            success=True,
            message="checkpoint already completed requested generation budget",
            generations=int(max_generations),
            evaluations=int(evaluations),
            history=completed_history,
            checkpoint=result_checkpoint,
        )

    def checked_objective(parameters: FloatArray) -> float:
        nonlocal evaluations, best_value, best_parameters, latest_value, latest_parameters
        if cancellation.is_set():
            raise OptimizationCancelled("optimization cancelled")
        if time_limit_s is not None and evaluations > starting_evaluations and monotonic() - started >= time_limit_s:
            raise OptimizationTimeLimitReached
        value = float(objective(np.asarray(parameters, dtype=np.float64)))
        if not np.isfinite(value):
            raise ValueError("objective returned a non-finite value")
        evaluations += 1
        latest_value = value
        latest_parameters = np.asarray(parameters, dtype=np.float64).copy()
        if value < best_value:
            best_value = value
            best_parameters = latest_parameters.copy()
        return value

    def on_generation(_parameters: FloatArray, _convergence: float) -> bool:
        nonlocal generation
        generation += 1
        snapshot = OptimizationProgress(generation, evaluations, best_value, best_parameters.copy())
        history.append(snapshot)
        if progress is not None:
            progress(snapshot)
        if merit_progress is not None:
            merit_progress(
                OptimizationMeritProgress(
                    generation=generation,
                    evaluations=evaluations,
                    current_fitness=float(latest_value),
                    current_parameters=latest_parameters.copy(),
                    best_fitness=float(best_value),
                    best_parameters=best_parameters.copy(),
                )
            )
        if cancellation.is_set():
            raise OptimizationCancelled("optimization cancelled")
        return time_limit_s is not None and monotonic() - started >= time_limit_s

    init = None if restored_checkpoint is None else _resume_population(restored_checkpoint, lower, upper)
    completed_generations = 0 if restored_checkpoint is None else restored_checkpoint.generations_completed
    remaining_generations = max_generations - completed_generations
    de_solver = cast(Any, differential_evolution)
    try:
        result = de_solver(
            checked_objective,
            bounds=list(zip(lower.tolist(), upper.tolist(), strict=True)),
            constraints=constraints,
            integrality=integrality,
            maxiter=remaining_generations,
            popsize=population_size,
            seed=seed,
            callback=on_generation,
            polish=polish,
            workers=1,
            updating="immediate",
            init="latinhypercube" if init is None else init,
        )
    except OptimizationTimeLimitReached:
        result = OptimizeResult(
            x=best_parameters.copy(),
            fun=float(best_value),
            success=False,
            message="optimization time limit reached; returning best candidate",
            nit=generation - completed_generations,
            nfev=evaluations,
        )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    final_fitness = float(result.fun)
    if final_fitness <= best_value:
        best_value = final_fitness
        best_parameters = final_parameters.copy()
    completed_total = min(int(max_generations), completed_generations + int(result.nit))
    full_history = tuple(history)
    result_checkpoint = OptimizationCheckpoint(
        lower_bounds=lower.copy(),
        upper_bounds=upper.copy(),
        seed=int(seed),
        population_size=int(population_size),
        max_generations=int(max_generations),
        integer_indices=tuple(int(index) for index in integer_indices),
        sum_limit_count=int(sum_limit_count),
        max_sum=None if max_sum is None else float(max_sum),
        compatibility_profile=optimizer_profile,
        generations_completed=completed_total,
        evaluations=int(evaluations),
        best_fitness=float(best_value),
        best_parameters=best_parameters.copy(),
        history=full_history,
    )
    return OptimizationResult(
        parameters=final_parameters,
        fitness=final_fitness,
        success=bool(result.success),
        message=str(result.message),
        generations=completed_total,
        evaluations=int(evaluations),
        history=full_history,
        checkpoint=result_checkpoint,
    )


def run_optimization(
    request: OptimizationRequest,
    *,
    cancel_event: Event | None = None,
    progress: Callable[[OptimizationProgress], None] | None = None,
    merit_progress: Callable[[OptimizationMeritProgress], None] | None = None,
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
                compatibility_mode=request.compatibility_mode,
            )
    elif request.source == "imported":
        if not isinstance(request.template, ImportedRCWARequest):
            raise TypeError("imported optimization requires ImportedRCWARequest")
        imported_template = request.template

        if request.parameter_contract == "legacy":

            def objective(x: FloatArray) -> float:
                return evaluate_imported_parameters(
                    imported_template,
                    x,
                    request.objective,
                    cancel_event=cancellation,
                    compatibility_mode=request.compatibility_mode,
                )
        else:

            def objective(x: FloatArray) -> float:
                return evaluate_imported_layer_parameters(
                    imported_template,
                    x,
                    request.objective,
                    layer_indices=request.imported_layer_indices,
                    cancel_event=cancellation,
                    compatibility_mode=request.compatibility_mode,
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
        merit_progress=merit_progress,
        polish=request.polish,
        optimizer_profile=request.optimizer_profile,
        checkpoint=request.checkpoint,
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
        compatibility_mode=request.compatibility_mode,
    )
    metadata = {
        "workflow": "optimization",
        "source": request.source,
        "objective": request.objective,
        "matrix_method_forced": "S",
        "supported_objectives": list(SUPPORTED_OBJECTIVES),
        "legacy_exact_objectives": list(LEGACY_EXACT_OBJECTIVES),
        "compatibility_mode": request.compatibility_mode,
        "parameter_contract": request.parameter_contract,
        "optimizer": "scipy.differential_evolution",
        "optimizer_profile": request.optimizer_profile,
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
    _validate_compatibility_mode(request.compatibility_mode)
    if request.parameter_contract not in {"legacy", "imported_layers"}:
        raise ValueError("parameter_contract must be 'legacy' or 'imported_layers'")
    if request.parameter_contract == "imported_layers" and request.source != "imported":
        raise ValueError("imported_layers parameter contract requires imported source")
    if request.optimizer_profile not in {"scipy_de", "ga_compat"}:
        raise ValueError("optimizer_profile must be 'scipy_de' or 'ga_compat'")
    if request.objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"unsupported legacy objective: {request.objective}")
    if request.compatibility_mode == "legacy_exact" and request.objective in PLUS_MINUS_TWO_OBJECTIVES:
        raise ValueError(
            f"unsupported legacy objective: {request.objective} is not implemented by legacy MATLAB merit functions"
        )


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
    if request.parameter_contract == "imported_layers":
        final_request = _imported_layers_request_for_parameters(
            request.template,
            parameters,
            request.imported_layer_indices,
        )
        return run_imported_rcwa(final_request, cancel_token=cancel_event, progress=solver_progress)
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


def _imported_layers_request_for_parameters(
    template: ImportedRCWARequest,
    parameters: ArrayLike,
    layer_indices: Sequence[int] | None,
) -> ImportedRCWARequest:
    from .legacy_io import ImportedDevice

    values = np.asarray(parameters, dtype=np.float64).ravel()
    indices = tuple(range(values.size)) if layer_indices is None else tuple(int(index) for index in layer_indices)
    if len(indices) != values.size:
        raise ValueError("layer_indices length must match the parameter vector")
    thicknesses = template.device.sub_L_um.copy()
    for index, value in zip(indices, values, strict=True):
        if index < 0 or index >= thicknesses.size:
            raise ValueError(f"imported layer index out of range: {index}")
        thicknesses[index] = value
    imported = ImportedDevice(
        ER=template.device.ER,
        sub_L_um=thicknesses,
        x_um=template.device.x_um,
        Lx_um=template.device.Lx_um,
        source=template.device.source,
    )
    return replace(template, device=imported, matrix_method="S")


def restore_optimization_checkpoint(
    checkpoint: OptimizationCheckpoint | Mapping[str, object],
) -> OptimizationCheckpoint:
    if isinstance(checkpoint, OptimizationCheckpoint):
        return checkpoint
    history_items = []
    for raw_item in checkpoint.get("history", ()):
        if not isinstance(raw_item, Mapping):
            raise TypeError("checkpoint history entries must be mappings")
        history_items.append(
            OptimizationProgress(
                generation=int(raw_item["generation"]),
                evaluations=int(raw_item["evaluations"]),
                best_fitness=float(raw_item["best_fitness"]),
                best_parameters=np.asarray(raw_item["best_parameters"], dtype=np.float64).ravel(),
            )
        )
    profile = str(checkpoint["compatibility_profile"])
    if profile not in {"scipy_de", "ga_compat"}:
        raise ValueError("checkpoint optimizer profile is invalid")
    return OptimizationCheckpoint(
        lower_bounds=np.asarray(checkpoint["lower_bounds"], dtype=np.float64).ravel(),
        upper_bounds=np.asarray(checkpoint["upper_bounds"], dtype=np.float64).ravel(),
        seed=int(checkpoint["seed"]),
        population_size=int(checkpoint["population_size"]),
        max_generations=int(checkpoint["max_generations"]),
        integer_indices=tuple(int(index) for index in cast(Sequence[object], checkpoint["integer_indices"])),
        sum_limit_count=int(checkpoint["sum_limit_count"]),
        max_sum=None if checkpoint.get("max_sum") is None else float(cast(float, checkpoint["max_sum"])),
        compatibility_profile=cast(OptimizerProfile, profile),
        generations_completed=int(checkpoint["generations_completed"]),
        evaluations=int(checkpoint["evaluations"]),
        best_fitness=float(checkpoint["best_fitness"]),
        best_parameters=np.asarray(checkpoint["best_parameters"], dtype=np.float64).ravel(),
        history=tuple(history_items),
    )


def _validate_compatibility_mode(mode: str) -> None:
    if mode not in {"legacy_exact", "corrected", "modern"}:
        raise ValueError("compatibility_mode must be 'legacy_exact', 'corrected', or 'modern'")


def _validate_checkpoint_resume(
    checkpoint: OptimizationCheckpoint,
    lower: FloatArray,
    upper: FloatArray,
    *,
    seed: int,
    population_size: int,
    integer_indices: tuple[int, ...],
    sum_limit_count: int,
    max_sum: float | None,
    optimizer_profile: OptimizerProfile,
) -> None:
    if checkpoint.lower_bounds.shape != lower.shape or checkpoint.upper_bounds.shape != upper.shape:
        raise ValueError("checkpoint bounds shape does not match this optimization")
    if not np.array_equal(checkpoint.lower_bounds, lower) or not np.array_equal(checkpoint.upper_bounds, upper):
        raise ValueError("checkpoint bounds do not match this optimization")
    if checkpoint.best_parameters.shape != lower.shape:
        raise ValueError("checkpoint best parameter shape does not match bounds")
    if checkpoint.seed != seed:
        raise ValueError("checkpoint seed does not match this optimization")
    if checkpoint.population_size != population_size:
        raise ValueError("checkpoint population size does not match this optimization")
    if checkpoint.integer_indices != integer_indices:
        raise ValueError("checkpoint integer indices do not match this optimization")
    if checkpoint.sum_limit_count != sum_limit_count or checkpoint.max_sum != max_sum:
        raise ValueError("checkpoint sum constraint does not match this optimization")
    if checkpoint.compatibility_profile != optimizer_profile:
        raise ValueError("checkpoint optimizer profile does not match this optimization")


def _resume_population(checkpoint: OptimizationCheckpoint, lower: FloatArray, upper: FloatArray) -> FloatArray:
    rng = np.random.default_rng(checkpoint.seed + checkpoint.generations_completed + 1)
    rows = max(5, checkpoint.population_size * lower.size)
    population = rng.uniform(lower, upper, size=(rows, lower.size))
    population[0] = np.clip(checkpoint.best_parameters, lower, upper)
    for index in checkpoint.integer_indices:
        population[:, index] = np.rint(population[:, index])
    return population.astype(np.float64, copy=False)


def _order_value(result: DiffractionResult, order: int) -> FloatArray | None:
    if order == -2:
        return cast(FloatArray | None, getattr(result, "minus_2", None))
    if order == 2:
        return cast(FloatArray | None, getattr(result, "plus_2", None))
    raise ValueError("only second-order helper values are supported")
