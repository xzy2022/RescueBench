"""Durable image and metadata artifacts for stretcher goal capture."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.stretcher_capture_data import CaptureInputs
from benchmark.teleport_probe_artifacts import append_jsonl, utc_now, write_json_atomic
from benchmark.teleport_probe_data import ProbeError, TaskSelection, configured_pose


@dataclass(frozen=True)
class CaptureTargetPaths:
    """Identify the public image and sidecar for one Level/Point."""

    image_path: Path
    sidecar_path: Path


@dataclass(frozen=True)
class BatchArtifacts:
    """Own one batch's public target directory and diagnostic run directory."""

    stretcher_dir: Path
    run_dir: Path
    run_path: Path
    points_path: Path
    samples_path: Path
    diagnostics_dir: Path

    @classmethod
    def create(cls, inputs: CaptureInputs) -> BatchArtifacts:
        """Create the public and diagnostic directories for one batch."""

        stretcher_dir = inputs.topomap_dir / "stretcher"
        stretcher_dir.mkdir(parents=True, exist_ok=True)
        runs_dir = stretcher_dir / "_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = _create_unique_directory(runs_dir, f"stretcher-capture-{stamp}")
        diagnostics_dir = run_dir / "diagnostics"
        diagnostics_dir.mkdir()
        return cls(
            stretcher_dir=stretcher_dir,
            run_dir=run_dir,
            run_path=run_dir / "run.json",
            points_path=run_dir / "points.jsonl",
            samples_path=run_dir / "samples.jsonl",
            diagnostics_dir=diagnostics_dir,
        )

    def target_paths(self, selection: TaskSelection) -> CaptureTargetPaths:
        """Return deterministic public artifact paths for one task point."""

        stem = f"level_{selection.level}_{selection.point_id}"
        return CaptureTargetPaths(
            image_path=self.stretcher_dir / f"{stem}.png",
            sidecar_path=self.stretcher_dir / f"{stem}.json",
        )

    def append_point(self, record: dict[str, Any]) -> None:
        """Durably append one terminal point record."""

        append_jsonl(self.points_path, record)

    def append_sample(self, record: dict[str, Any]) -> None:
        """Durably append one runtime pose/image sample record."""

        append_jsonl(self.samples_path, record)


def _create_unique_directory(root: Path, base_name: str) -> Path:
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise ProbeError(f"could not create a unique run directory under: {root}")


def sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest for one payload."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_existing_capture(
    paths: CaptureTargetPaths,
    selection: TaskSelection,
) -> tuple[str, dict[str, Any] | None]:
    """Classify a public image/sidecar pair for resume handling."""

    image_exists = paths.image_path.is_file()
    sidecar_exists = paths.sidecar_path.is_file()
    if not image_exists and not sidecar_exists:
        return "missing", None
    if not image_exists or not sidecar_exists:
        return "conflict", None
    try:
        metadata = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "conflict", None
    if not isinstance(metadata, dict) or metadata.get("status") != "captured":
        return "conflict", metadata if isinstance(metadata, dict) else None
    if metadata.get("level") != selection.level:
        return "conflict", metadata
    if metadata.get("point_id") != selection.point_id:
        return "conflict", metadata
    if metadata.get("filename") != paths.image_path.name:
        return "conflict", metadata
    try:
        actual_hash = sha256_file(paths.image_path)
    except OSError:
        return "conflict", metadata
    if metadata.get("sha256") != actual_hash:
        return "conflict", metadata
    return "valid", metadata


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def save_capture(
    *,
    cv2: Any,
    paths: CaptureTargetPaths,
    image: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Encode one raw BGR frame and atomically publish it with its sidecar."""

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ProbeError(f"failed to encode PNG: {paths.image_path}")
    payload = encoded.tobytes()
    digest = sha256_bytes(payload)
    shape = [int(value) for value in image.shape]
    if len(shape) < 2:
        raise ProbeError("captured image has no valid shape")
    complete_metadata = {
        **metadata,
        "status": "captured",
        "filename": paths.image_path.name,
        "image_size": [shape[1], shape[0]],
        "image_shape": shape,
        "dtype": str(image.dtype),
        "sha256": digest,
        "captured_at_utc": utc_now(),
    }
    _write_bytes_atomic(paths.image_path, payload)
    write_json_atomic(paths.sidecar_path, complete_metadata)
    return complete_metadata


def save_diagnostic_frame(
    cv2: Any,
    artifacts: BatchArtifacts,
    selection: TaskSelection,
    image: Any,
    suffix: str,
) -> Path | None:
    """Save the final failed-attempt frame outside the public goal directory."""

    filename = f"L{selection.level}-P{selection.point_id}-{suffix}.png"
    path = artifacts.diagnostics_dir / filename
    success, encoded = cv2.imencode(".png", image)
    if not success:
        return None
    _write_bytes_atomic(path, encoded.tobytes())
    return path


def build_initial_run_metadata(inputs: CaptureInputs) -> dict[str, Any]:
    """Build durable batch-level metadata before UE startup."""

    return {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "topomap_dir": str(inputs.topomap_dir),
        "stretcher_dir": str(inputs.topomap_dir / "stretcher"),
        "requested_resolution": list(inputs.render.resolution),
        "offscreen": inputs.render.offscreen,
        "render_quality": inputs.render.quality,
        "resume": inputs.resume,
        "overwrite": inputs.overwrite,
        "tasks": [
            {
                "level": selection.level,
                "point_id": selection.point_id,
                "env_id": selection.task_context["env_id"],
            }
            for selection in inputs.selections
        ],
        "counts": {},
        "error": None,
    }


def configured_poses(selection: TaskSelection) -> dict[str, list[float]]:
    """Return the four configured actor poses recorded in a sidecar."""

    return {
        "agent": configured_pose(selection.raw_point, "agent_loc"),
        "injured": configured_pose(selection.raw_point, "injured_player_loc"),
        "stretcher": configured_pose(selection.raw_point, "stretcher_loc"),
        "ambulance": configured_pose(selection.raw_point, "ambulance_loc"),
    }
