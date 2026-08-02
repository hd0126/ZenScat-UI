"""Versioned, JSON-based ZenScat project persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

PROJECT_SCHEMA = "zenscat.project"
PROJECT_VERSION = "1.0"
WORKFLOWS = {
    "casual_rcwa",
    "custom_import_rcwa",
    "optimization",
    "fdfd_fields",
    "casual_phc_rcwa",
    "harmonic_convergence",
    "phc_fdfd_fields",
}


@dataclass
class ProjectDocument:
    WORKFLOWS: ClassVar[set[str]] = WORKFLOWS

    name: str
    workflow: str = "casual_rcwa"
    configuration: dict[str, Any] | None = None
    result_bundle: str | None = None
    created_at: str | None = None
    modified_at: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("project name must not be empty")
        if self.workflow not in self.WORKFLOWS:
            raise ValueError(f"unsupported workflow: {self.workflow}")
        if self.configuration is None:
            self.configuration = {}
        now = _utc_now()
        if self.created_at is None:
            self.created_at = now
        if self.modified_at is None:
            self.modified_at = self.created_at

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        if not destination.suffix:
            destination = destination.with_suffix(".zenscat")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.modified_at = _utc_now()
        payload = {
            "schema": PROJECT_SCHEMA,
            "schema_version": PROJECT_VERSION,
            "name": self.name,
            "workflow": self.workflow,
            "configuration": self.configuration,
            "result_bundle": self.result_bundle,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(payload, cls=_ProjectEncoder, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ProjectDocument:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"), object_hook=_project_decoder)
        if payload.get("schema") != PROJECT_SCHEMA:
            raise ValueError("not a ZenScat project document")
        if payload.get("schema_version") != PROJECT_VERSION:
            raise ValueError(
                f"unsupported project schema version: {payload.get('schema_version')!r}; "
                f"expected {PROJECT_VERSION}"
            )
        return cls(
            name=str(payload["name"]),
            workflow=str(payload["workflow"]),
            configuration=dict(payload.get("configuration", {})),
            result_bundle=payload.get("result_bundle"),
            created_at=payload.get("created_at"),
            modified_at=payload.get("modified_at"),
        )


class _ProjectEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        value = o
        if isinstance(value, np.ndarray):
            if np.iscomplexobj(value):
                return {
                    "__ndarray__": True,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "real": value.real.tolist(),
                    "imag": value.imag.tolist(),
                }
            return {
                "__ndarray__": True,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        if isinstance(value, complex):
            return {"__complex__": True, "real": value.real, "imag": value.imag}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return {"__path__": True, "value": str(value)}
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, Mapping):
            return dict(value)
        return super().default(value)


def _project_decoder(value: dict[str, Any]) -> Any:
    if value.get("__ndarray__"):
        if "real" in value:
            array = np.asarray(value["real"], dtype=np.float64) + 1j * np.asarray(value["imag"], dtype=np.float64)
            return array.astype(value["dtype"]).reshape(value["shape"])
        return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
    if value.get("__complex__"):
        return complex(value["real"], value["imag"])
    if value.get("__path__"):
        return Path(value["value"])
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
