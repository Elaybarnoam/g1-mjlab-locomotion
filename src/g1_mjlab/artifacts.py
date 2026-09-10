"""Crash-readable run lifecycle and structured metrics."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"completed", "budget_limited", "interrupted", "failed"}
_TRANSITIONS = {
    "created": {"starting", "failed"},
    "starting": {"running", "failed", "interrupted"},
    "running": TERMINAL_STATES,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass
class RunStore:
    root: Path
    monotonic_started: float

    @classmethod
    def create(cls, root: Path, manifest: dict[str, Any]) -> RunStore:
        root.mkdir(parents=True, exist_ok=False)
        for name in ("metrics", "traces", "checkpoints", "evaluation", "videos", "report"):
            (root / name).mkdir()
        now = utc_now()
        data = {"schema_version": 1, "status": "created", "created_at": now, **manifest}
        _atomic_json(root / "manifest.json", data)
        return cls(root=root, monotonic_started=time.monotonic())

    @classmethod
    def open(cls, root: Path) -> RunStore:
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"not a run directory: {root}")
        return cls(root=root, monotonic_started=time.monotonic())

    def manifest(self) -> dict[str, Any]:
        value = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("manifest root must be an object")
        return value

    def transition(self, status: str, **fields: Any) -> None:
        manifest = self.manifest()
        current = str(manifest["status"])
        if status not in _TRANSITIONS.get(current, set()):
            raise ValueError(f"invalid run transition {current!r} -> {status!r}")
        manifest.update(fields)
        manifest["status"] = status
        manifest["updated_at"] = utc_now()
        manifest["wall_seconds"] = time.monotonic() - self.monotonic_started
        _atomic_json(self.root / "manifest.json", manifest)

    def append_metric(self, record: dict[str, Any]) -> None:
        self.append_metrics([record])

    def append_metrics(self, records: Iterable[dict[str, Any]]) -> int:
        """Append a batch durably while paying the flush cost only once."""
        run_id = self.manifest()["run_id"]
        count = 0
        with (self.root / "metrics" / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            for record in records:
                complete = {"schema_version": 1, "run_id": run_id, **record}
                stream.write(json.dumps(complete, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        return count


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    records: list[dict[str, Any]] = []
    truncated = False
    if not path.exists():
        return records, truncated
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                truncated = True
                break
            raise
        if not isinstance(value, dict):
            raise ValueError("metric records must be objects")
        records.append(value)
    return records, truncated


def list_files(root: Path) -> Iterable[Path]:
    return (path for path in root.rglob("*") if path.is_file())


def snapshot_source(project: Path, destination: Path) -> dict[str, Any]:
    """Archive project-owned implementation, including untracked source files."""
    paths = [project / "pyproject.toml"]
    for directory in ("src/g1_mjlab", "tests", "configs/standing-v1"):
        paths.extend(
            path
            for path in (project / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    manifest = []
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if not path.exists():
                continue
            relative = path.relative_to(project).as_posix()
            archive.write(path, relative)
            manifest.append({"path": relative, "sha256": sha256_file(path)})
    return {"archive": destination.name, "sha256": sha256_file(destination), "files": manifest}
