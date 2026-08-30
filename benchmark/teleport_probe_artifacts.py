"""Durable run-level artifacts for the RescueBench teleport capture probe."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import gmtime, strftime
from typing import Any

from benchmark.teleport_probe_data import (
    ProbeInputs,
    configured_pose,
    create_run_directory,
)


def utc_now() -> str:
    """Return an ISO 8601 UTC timestamp on supported Python versions."""

    return strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document through a same-directory temporary file."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append and flush one JSONL record so prior samples survive interruption."""

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class ProbeArtifacts:
    """Identify the output paths owned by one probe run."""

    run_dir: Path
    samples_path: Path
    metadata_path: Path

    @classmethod
    def create(cls, inputs: ProbeInputs) -> ProbeArtifacts:
        """Create the unique run directory and its output paths."""

        run_dir = create_run_directory(
            inputs.output_root,
            inputs.selection.level,
            inputs.selection.point_id,
        )
        return cls(
            run_dir=run_dir,
            samples_path=run_dir / "samples.jsonl",
            metadata_path=run_dir / "run.json",
        )


def sample_filename(
    sequence_index: int,
    sample_index: int,
    pose_id: str,
    delay: float,
) -> str:
    """Build a filename unique even when two delays round to the same text."""

    return f"{sequence_index:03d}_{sample_index:03d}_{pose_id}_t{delay:.3f}s.png"


def build_run_metadata(inputs: ProbeInputs) -> dict[str, Any]:
    """Build the initial durable metadata for one probe run."""

    selection = inputs.selection
    return {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": utc_now(),
        "completed_at_utc": None,
        "error": None,
        "scene": {
            "index": selection.scene.index,
            "count": 7,
            "name": selection.scene.name,
            "env_id": selection.task_context["env_id"],
        },
        "selection": {
            "level": selection.level,
            "point_id": selection.point_id,
            "point_numbering": "zero_based",
            "source_path": str(selection.source_path),
            "source_line": selection.source_line,
        },
        "task_configuration": {
            "agent_pose": configured_pose(selection.raw_point, "agent_loc"),
            "injured_pose": configured_pose(
                selection.raw_point,
                "injured_player_loc",
            ),
            "stretcher_pose": configured_pose(selection.raw_point, "stretcher_loc"),
            "ambulance_pose": configured_pose(selection.raw_point, "ambulance_loc"),
            "injured_agent_id": selection.raw_point.get("injured_agent_id"),
            "reference_text": selection.task_context.get("reference_text"),
            "reference_image_path": selection.task_context.get("reference_image_path"),
            "timeout_s": selection.task_context["timeout"],
        },
        "probe_configuration": {
            "poses_file": str(inputs.poses_path),
            "poses": [
                {
                    "id": pose.pose_id,
                    "location_xyz": pose.location_xyz,
                    "rotation_rpy_deg": pose.rotation_rpy_deg,
                    "note": pose.note,
                }
                for pose in inputs.poses
            ],
            "requested_resolution": inputs.render.resolution,
            "sample_delays_s": list(inputs.sample_delays),
            "coordinate_frame": "unreal_world",
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "rotation_unit": "degree",
            "offscreen": inputs.render.offscreen,
            "render_quality": inputs.render.quality,
            "calls_env_step": False,
            "loads_navigation_model": False,
            "changes_agent_physics": False,
            "forces_standup": False,
        },
    }
