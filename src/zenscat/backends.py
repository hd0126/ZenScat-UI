"""Stable execution boundaries for local and external ZenScat solvers.

The desktop application is always explicit about which backend executed a
request.  External commands use a small file-envelope protocol so MATLAB,
Octave, and native command-line bridges can be implemented without importing
their runtimes into the GUI process.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Any, cast

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a selected backend cannot execute on this machine."""


class ExternalSolverError(RuntimeError):
    """Raised when an external solver violates the execution contract."""


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    available: bool
    executable: str | None
    detail: str
    architecture: str


@dataclass(frozen=True)
class ExternalSolverConfig:
    """Configuration for the ZenScat external CLI file-envelope protocol."""

    command: tuple[str, ...]
    name: str = "External CLI"
    timeout_s: float = 3600.0

    @classmethod
    def from_command_text(
        cls,
        command: str,
        *,
        name: str = "External CLI",
        timeout_s: float = 3600.0,
    ) -> ExternalSolverConfig:
        values = tuple(shlex.split(command))
        return cls(values, name=name, timeout_s=timeout_s)


@dataclass(frozen=True)
class ExternalSolverResponse:
    payload: dict[str, Any]
    stdout: str
    stderr: str
    elapsed_s: float
    command: tuple[str, ...]


@dataclass(frozen=True)
class ExternalRCWARun:
    """RCWA result decoded from the external solver protocol."""

    wavelengths_m: np.ndarray
    angles_rad: np.ndarray
    transmission: Any
    reflection: Any
    elapsed_s: float
    metadata: dict[str, object]

    def legacy_bundle(
        self,
        params: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        from .legacy_io import LegacyResultBundle

        return LegacyResultBundle(
            wavelengths_m=self.wavelengths_m,
            angles_rad=self.angles_rad,
            transmission=self.transmission,
            reflection=self.reflection,
            params=None if params is None else np.asarray(params, dtype=np.float64),
            metadata=self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True)
class ExternalFDFDDevice:
    ER2: np.ndarray


@dataclass(frozen=True)
class ExternalFDFDRun:
    """FDFD result decoded from the external solver protocol."""

    wavelengths_um: np.ndarray
    angles_rad: np.ndarray
    result: Any
    device: ExternalFDFDDevice
    elapsed_s: float
    metadata: dict[str, object]

    def fdfd_bundle(self, params: Any | None = None) -> Any:
        from .legacy_io import FDFDResultBundle

        return FDFDResultBundle(
            wavelengths_um=self.wavelengths_um,
            angles_rad=self.angles_rad,
            result=self.result,
            er2=self.device.ER2,
            params=None if params is None else np.asarray(params, dtype=np.float64),
            metadata=self.metadata,
        )


ExternalDecodedRun = ExternalRCWARun | ExternalFDFDRun | Any


class LocalPythonBackend[T]:
    """Named wrapper used to keep local execution explicit in job metadata."""

    name = "Local Python"

    @staticmethod
    def availability() -> BackendAvailability:
        return BackendAvailability(
            name=LocalPythonBackend.name,
            available=True,
            executable=None,
            detail="In-process NumPy/SciPy solver is available.",
            architecture=_architecture(),
        )

    def run(self, operation: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        return operation(*args, **kwargs)


class ExternalSolverBackend:
    """Execute a registered solver without a shell or implicit fallback.

    The child process receives two appended arguments::

        --zenscat-request /path/request.json
        --zenscat-result /path/result.json

    It must write a JSON object to the result path and exit with status zero.
    This deliberately narrow contract is usable from Python, MATLAB batch
    launchers, Octave scripts, or native Apple Silicon executables.
    """

    def __init__(self, config: ExternalSolverConfig) -> None:
        self.config = config

    def availability(self) -> BackendAvailability:
        if not self.config.command:
            return BackendAvailability(
                name=self.config.name,
                available=False,
                executable=None,
                detail="No external solver command is configured.",
                architecture=_architecture(),
            )
        executable = _resolve_executable(self.config.command[0])
        if executable is None:
            return BackendAvailability(
                name=self.config.name,
                available=False,
                executable=None,
                detail=f"Executable was not found: {self.config.command[0]}",
                architecture=_architecture(),
            )
        if self.config.timeout_s <= 0:
            return BackendAvailability(
                name=self.config.name,
                available=False,
                executable=executable,
                detail="External solver timeout must be positive.",
                architecture=_architecture(),
            )
        return BackendAvailability(
            name=self.config.name,
            available=True,
            executable=executable,
            detail="External file-envelope solver is available.",
            architecture=_architecture(),
        )

    def run(
        self,
        request: Mapping[str, Any] | object,
        *,
        cancel_event: Event | None = None,
    ) -> ExternalSolverResponse:
        availability = self.availability()
        if not availability.available or availability.executable is None:
            raise BackendUnavailableError(availability.detail)
        cancellation = cancel_event or Event()
        started = monotonic()
        with tempfile.TemporaryDirectory(prefix="zenscat-external-") as temporary:
            work = Path(temporary)
            request_path = work / "request.json"
            result_path = work / "result.json"
            stdout_path = work / "stdout.log"
            stderr_path = work / "stderr.log"
            request_path.write_text(
                json.dumps(_json_safe(request), ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            command = (
                availability.executable,
                *self.config.command[1:],
                "--zenscat-request",
                str(request_path),
                "--zenscat-result",
                str(result_path),
            )
            with stdout_path.open("wb") as stdout_file, stderr_path.open(
                "wb"
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    shell=False,
                    cwd=work,
                )
                while process.poll() is None:
                    if cancellation.is_set():
                        process.terminate()
                        _wait_or_kill(process)
                        raise ExternalSolverError("external solver cancelled")
                    if monotonic() - started >= self.config.timeout_s:
                        process.terminate()
                        _wait_or_kill(process)
                        raise ExternalSolverError(
                            f"external solver timed out after {self.config.timeout_s:g} s"
                        )
                    sleep(0.02)
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            if process.returncode != 0:
                detail = stderr.strip() or stdout.strip() or "no diagnostic output"
                raise ExternalSolverError(
                    f"external solver exited with status {process.returncode}: {detail}"
                )
            if not result_path.is_file():
                raise ExternalSolverError("external solver did not write result.json")
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ExternalSolverError("external solver returned invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ExternalSolverError("external solver result must be a JSON object")
            return ExternalSolverResponse(
                payload=payload,
                stdout=stdout,
                stderr=stderr,
                elapsed_s=monotonic() - started,
                command=tuple(command),
            )


def configured_external_backend(
    command_text: str | None = None,
    *,
    timeout_s: float = 3600.0,
) -> ExternalSolverBackend:
    """Build the external backend from explicit text or the documented env var."""

    text = command_text if command_text is not None else os.environ.get(
        "ZENSCAT_EXTERNAL_SOLVER", ""
    )
    return ExternalSolverBackend(
        ExternalSolverConfig.from_command_text(text, timeout_s=timeout_s)
    )


def discover_solver_commands() -> tuple[BackendAvailability, ...]:
    """Report common external runtimes without claiming adapter readiness."""

    results = [LocalPythonBackend.availability()]
    configured = configured_external_backend().availability()
    results.append(configured)
    for name, executable in (("MATLAB", "matlab"), ("GNU Octave", "octave-cli")):
        resolved = shutil.which(executable)
        results.append(
            BackendAvailability(
                name=name,
                available=resolved is not None,
                executable=resolved,
                detail=(
                    "Runtime detected; configure a ZenScat file-envelope bridge."
                    if resolved
                    else f"{executable} is not on PATH."
                ),
                architecture=_architecture(),
            )
        )
    return tuple(results)


def decode_external_run(response: ExternalSolverResponse) -> ExternalDecodedRun:
    """Decode the documented inline or result-bundle external response."""

    payload = response.payload
    schema = payload.get("schema")
    if schema not in {None, "zenscat.external-result"}:
        raise ExternalSolverError(f"unsupported external result schema: {schema}")
    kind = payload.get("kind")
    metadata = dict(payload.get("metadata", {}))
    metadata.update(
        {
            "backend": "external",
            "external_command": list(response.command),
        }
    )
    if kind == "rcwa":
        return _decode_rcwa_payload(payload, response, metadata)
    if kind == "fdfd":
        return _decode_fdfd_payload(payload, response, metadata)
    if kind == "optimization":
        return _decode_optimization_payload(payload, response, metadata)
    raise ExternalSolverError("external result kind must be 'rcwa', 'fdfd', or 'optimization'")


def _decode_rcwa_payload(
    payload: Mapping[str, Any],
    response: ExternalSolverResponse,
    metadata: Mapping[str, object],
) -> ExternalRCWARun:
    bundle_path = payload.get("bundle_path")
    if bundle_path is not None:
        from .legacy_io import load_legacy_result_bundle

        path = Path(str(bundle_path)).expanduser()
        if not path.is_absolute():
            raise ExternalSolverError(
                "external bundle_path must be absolute; use inline JSON fields for temporary results"
            )
        bundle = load_legacy_result_bundle(path, strict=False)
        return ExternalRCWARun(
            wavelengths_m=np.asarray(bundle.wavelengths_m, dtype=np.float64),
            angles_rad=np.asarray(bundle.angles_rad, dtype=np.float64),
            transmission=bundle.transmission,
            reflection=bundle.reflection,
            elapsed_s=float(payload.get("elapsed_s", response.elapsed_s)),
            metadata={
                **dict(bundle.metadata or {}),
                **dict(metadata),
            },
        )
    transmission = _decode_diffraction(payload.get("transmission"), "TRN")
    reflection = _decode_diffraction(payload.get("reflection"), "REF")
    return ExternalRCWARun(
        wavelengths_m=_float_array(payload.get("wavelengths_m"), "wavelengths_m"),
        angles_rad=_float_array(payload.get("angles_rad"), "angles_rad"),
        transmission=transmission,
        reflection=reflection,
        elapsed_s=float(payload.get("elapsed_s", response.elapsed_s)),
        metadata=dict(metadata),
    )


def _decode_fdfd_payload(
    payload: Mapping[str, Any],
    response: ExternalSolverResponse,
    metadata: Mapping[str, object],
) -> ExternalFDFDRun:
    from .core import FDFDResult

    trn_payload = payload.get("transmission")
    ref_payload = payload.get("reflection")
    if not isinstance(trn_payload, Mapping) or not isinstance(ref_payload, Mapping):
        raise ExternalSolverError("FDFD external result requires transmission/reflection objects")
    result = FDFDResult(
        TRN={str(key): _decode_numeric(value) for key, value in trn_payload.items()},
        REF={str(key): _decode_numeric(value) for key, value in ref_payload.items()},
        f=np.asarray(_decode_numeric(payload.get("field")), dtype=np.complex128),
    )
    er2 = np.asarray(_decode_numeric(payload.get("er2")), dtype=np.float64)
    if result.f.ndim != 2 or er2.ndim != 2:
        raise ExternalSolverError("FDFD external field and ER2 must be two-dimensional")
    return ExternalFDFDRun(
        wavelengths_um=_float_array(payload.get("wavelengths_um"), "wavelengths_um"),
        angles_rad=_float_array(payload.get("angles_rad"), "angles_rad"),
        result=result,
        device=ExternalFDFDDevice(er2),
        elapsed_s=float(payload.get("elapsed_s", response.elapsed_s)),
        metadata=dict(metadata),
    )


def _decode_optimization_payload(
    payload: Mapping[str, Any],
    response: ExternalSolverResponse,
    metadata: Mapping[str, object],
) -> Any:
    from .optimization import (
        SUPPORTED_OBJECTIVES,
        ObjectiveFlavor,
        ObjectiveName,
        OptimizationResult,
        OptimizationRun,
        restore_optimization_checkpoint,
    )

    raw_optimization = payload.get("optimization")
    if not isinstance(raw_optimization, Mapping):
        raise ExternalSolverError("optimization external result requires an optimization object")
    raw_final_run = payload.get("final_run")
    if not isinstance(raw_final_run, Mapping):
        raise ExternalSolverError("optimization external result requires a final_run object")
    final_payload = dict(raw_final_run)
    final_payload.setdefault("kind", "rcwa")
    if final_payload.get("kind") != "rcwa":
        raise ExternalSolverError("optimization final_run must be an RCWA result")
    final_metadata = {
        **dict(final_payload.get("metadata", {})),
        **dict(metadata),
        "workflow": "optimization",
    }
    final_run = _decode_rcwa_payload(final_payload, response, final_metadata)
    history = tuple(_decode_optimization_history(raw_optimization.get("history", ())))
    raw_checkpoint = raw_optimization.get("checkpoint")
    checkpoint = (
        None
        if raw_checkpoint is None
        else restore_optimization_checkpoint(_require_mapping(raw_checkpoint, "optimization checkpoint"))
    )
    parameters = _float_array(raw_optimization.get("parameters"), "optimization.parameters")
    fitness = float(raw_optimization["fitness"])
    final_parameters = _float_array(
        payload.get("final_parameters", parameters),
        "final_parameters",
    )
    final_fitness = float(payload.get("final_fitness", fitness))
    run_metadata = {
        **dict(metadata),
        **dict(payload.get("metadata", {})),
        "workflow": "optimization",
        "optimizer": "external",
    }
    raw_source = str(payload.get("source", run_metadata.get("source", "analytic")))
    if raw_source not in {"analytic", "imported"}:
        raise ExternalSolverError("optimization source must be 'analytic' or 'imported'")
    raw_objective = str(payload["objective"])
    if raw_objective not in SUPPORTED_OBJECTIVES:
        raise ExternalSolverError(f"unsupported optimization objective: {raw_objective}")
    return OptimizationRun(
        source=cast(ObjectiveFlavor, raw_source),
        objective=cast(ObjectiveName, raw_objective),
        optimization=OptimizationResult(
            parameters=parameters,
            fitness=fitness,
            success=bool(raw_optimization.get("success", True)),
            message=str(raw_optimization.get("message", "external optimization completed")),
            generations=int(raw_optimization.get("generations", len(history))),
            evaluations=int(raw_optimization.get("evaluations", len(history))),
            history=history,
            checkpoint=checkpoint,
        ),
        final_run=final_run,
        final_parameters=final_parameters,
        final_fitness=final_fitness,
        metadata=run_metadata,
    )


def _decode_optimization_history(value: Any) -> list[Any]:
    from .optimization import OptimizationProgress

    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExternalSolverError("optimization history must be a sequence")
    history = []
    for item in value:
        raw = _require_mapping(item, "optimization history item")
        history.append(
            OptimizationProgress(
                generation=int(raw["generation"]),
                evaluations=int(raw["evaluations"]),
                best_fitness=float(raw["best_fitness"]),
                best_parameters=_float_array(raw["best_parameters"], "history.best_parameters"),
            )
        )
    return history


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalSolverError(f"{name} must be an object")
    return value


def _decode_diffraction(value: Any, prefix: str) -> Any:
    from .core import DiffractionResult

    if not isinstance(value, Mapping):
        raise ExternalSolverError(f"external {prefix} result must be an object")
    zero_name = "TRN0" if prefix == "TRN" else "REF0"
    try:
        minus = _decode_numeric(value["minus_1"])
        plus = _decode_numeric(value["plus_1"])
        zero = _decode_numeric(value[zero_name])
        total = _decode_numeric(value["sum"])
    except KeyError as exc:
        raise ExternalSolverError(f"external {prefix} result is missing {exc.args[0]}") from exc
    optional = {}
    for name in ("minus_2", "plus_2"):
        if name in value:
            optional[name] = np.asarray(_decode_numeric(value[name]), dtype=np.float64)
    return DiffractionResult(
        minus_1=np.asarray(minus, dtype=np.float64),
        plus_1=np.asarray(plus, dtype=np.float64),
        **{zero_name: np.asarray(zero, dtype=np.float64)},
        sum=np.asarray(total, dtype=np.float64),
        **optional,
    )


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(_decode_numeric(value), dtype=np.float64).ravel()
    if array.size == 0 or not np.isfinite(array).all():
        raise ExternalSolverError(f"external {name} must contain finite values")
    return array


def _decode_numeric(value: Any) -> Any:
    if isinstance(value, Mapping) and {"real", "imag"}.issubset(value):
        real = np.asarray(value["real"], dtype=np.float64)
        imag = np.asarray(value["imag"], dtype=np.float64)
        if real.shape != imag.shape:
            raise ExternalSolverError("complex external arrays require matching real/imag shapes")
        return real + 1j * imag
    return value


def _resolve_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
        return str(resolved) if resolved.is_file() and os.access(resolved, os.X_OK) else None
    return shutil.which(value)


def _wait_or_kill(process: subprocess.Popen[bytes]) -> None:
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)


def _architecture() -> str:
    return os.uname().machine if hasattr(os, "uname") else "unknown"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"external request value is not JSON serializable: {type(value).__name__}")


__all__ = [
    "BackendAvailability",
    "BackendUnavailableError",
    "ExternalFDFDRun",
    "ExternalRCWARun",
    "ExternalSolverBackend",
    "ExternalSolverConfig",
    "ExternalSolverError",
    "ExternalSolverResponse",
    "LocalPythonBackend",
    "configured_external_backend",
    "decode_external_run",
    "discover_solver_commands",
]
