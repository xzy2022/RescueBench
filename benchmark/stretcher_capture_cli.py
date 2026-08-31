"""Command-line orchestration for stretcher semantic-goal capture."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.core.env_manager import EnvManager
from benchmark.stretcher_capture_artifacts import (
    BatchArtifacts,
    build_initial_run_metadata,
    configured_poses,
    inspect_existing_capture,
    save_capture,
    save_diagnostic_frame,
)
from benchmark.stretcher_capture_data import (
    DEFAULT_CAPTURE_DISTANCE_UU,
    DEFAULT_HEIGHT_OFFSET_UU,
    DEFAULT_HEIGHT_RETRY_STEP_UU,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_DELTA_Z_UU,
    DEFAULT_MIN_DELTA_Z_UU,
    DEFAULT_POSITION_EPSILON_UU,
    DEFAULT_RESOLUTION,
    DEFAULT_ROTATION_EPSILON_DEG,
    DEFAULT_SAMPLE_DELAYS,
    DEFAULT_STABLE_TAIL_SAMPLES,
    CaptureInputs,
    build_capture_inputs,
    policy_as_dict,
    validate_capture_policy,
)
from benchmark.stretcher_capture_runtime import (
    capture_point,
    outcome_as_dict,
)
from benchmark.teleport_probe_artifacts import utc_now, write_json_atomic
from benchmark.teleport_probe_capture import runtime_handles
from benchmark.teleport_probe_console import print_task_configuration
from benchmark.teleport_probe_data import ProbeError, TaskSelection


def _load_runtime_dependencies() -> Any:
    numpy = importlib.import_module("numpy")
    numpy.bool8 = numpy.bool_
    importlib.import_module("gym_rescue")
    return importlib.import_module("cv2")


def _resolve_inputs(args: argparse.Namespace) -> CaptureInputs:
    if "UnrealEnv" not in os.environ:
        raise ProbeError(
            "UnrealEnv is not set; run through the project rescue-run wrapper or "
            "set it to the UE asset directory"
        )
    policy = validate_capture_policy(
        capture_distance_uu=args.capture_distance_uu,
        initial_height_offset_uu=args.initial_height_offset_uu,
        height_retry_step_uu=args.height_retry_step_uu,
        min_delta_z_uu=args.min_delta_z_uu,
        max_delta_z_uu=args.max_delta_z_uu,
        max_attempts=args.max_attempts,
        sample_delays=args.sample_delays,
        stable_tail_samples=args.stable_tail_samples,
        position_epsilon_uu=args.position_epsilon_uu,
        rotation_epsilon_deg=args.rotation_epsilon_deg,
    )
    return build_capture_inputs(
        levels=args.levels,
        point_ids=args.point_ids,
        topomap_dir=Path(args.topomap_dir),
        resolution=args.resolution,
        offscreen=args.offscreen,
        render_quality=args.render_quality,
        policy=policy,
        resume=args.resume,
        overwrite=args.overwrite,
    )


def _selection_fields(selection: TaskSelection) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recorded_at_utc": utc_now(),
        "env_id": selection.task_context["env_id"],
        "scene": selection.scene.name,
        "scene_index": selection.scene.index,
        "level": selection.level,
        "point_id": selection.point_id,
        "point_numbering": "zero_based",
        "source_path": str(selection.source_path),
        "source_line": selection.source_line,
    }


def _capture_metadata(
    selection: TaskSelection,
    inputs: CaptureInputs,
    outcome_record: dict[str, Any],
) -> dict[str, Any]:
    actual_poses = outcome_record["actual_poses"]
    return {
        **_selection_fields(selection),
        "capture_state": "phase2_goal_view",
        "configured_poses": configured_poses(selection),
        "capture_geometry": outcome_record["capture_geometry"],
        "height_policy": policy_as_dict(inputs.policy),
        "attempts": outcome_record["attempts"],
        "requested_agent_pose": outcome_record["requested_agent_pose"],
        "actual_agent_pose": actual_poses["agent_actor"],
        "actual_injured_pose": actual_poses["injured"],
        "actual_stretcher_pose": actual_poses["stretcher"],
        "actual_ambulance_pose": actual_poses["ambulance"],
        "stretcher_loc": actual_poses["stretcher"],
        "ambulance_loc": actual_poses["ambulance"],
        "camera_pose": actual_poses["robot_camera"],
        "agent_minus_stretcher_z_uu": outcome_record["agent_minus_stretcher_z_uu"],
        "requested_resolution": list(inputs.render.resolution),
        "calls_env_step": False,
        "loads_navigation_model": False,
        "changes_agent_physics": False,
        "forces_standup": False,
    }


def _record_existing(
    artifacts: BatchArtifacts,
    selection: TaskSelection,
    metadata: dict[str, Any],
) -> str:
    record = {
        **_selection_fields(selection),
        "status": "already_valid",
        "reason": "resume_hash_verified",
        "filename": metadata.get("filename"),
        "sha256": metadata.get("sha256"),
    }
    artifacts.append_point(record)
    print(f"[Skip] L{selection.level} P{selection.point_id}: existing capture valid")
    return "already_valid"


def _record_conflict(
    artifacts: BatchArtifacts,
    selection: TaskSelection,
) -> str:
    record = {
        **_selection_fields(selection),
        "status": "skipped",
        "reason": "existing_output_conflict",
    }
    artifacts.append_point(record)
    print(
        f"[Skip] L{selection.level} P{selection.point_id}: "
        "existing image/sidecar conflict (use --overwrite to replace)"
    )
    return "skipped"


def _run_selection(
    *,
    manager: EnvManager,
    cv2: Any,
    inputs: CaptureInputs,
    artifacts: BatchArtifacts,
    selection: TaskSelection,
) -> str:
    paths = artifacts.target_paths(selection)
    existing_status, existing_metadata = inspect_existing_capture(paths, selection)
    if existing_status == "valid" and inputs.resume and not inputs.overwrite:
        if existing_metadata is None:
            raise ProbeError("valid existing capture has no metadata")
        return _record_existing(artifacts, selection, existing_metadata)
    if existing_status != "missing" and not inputs.overwrite:
        return _record_conflict(artifacts, selection)

    print_task_configuration(selection)
    print("\n[Stretcher capture]")
    manager.ensure_env(
        str(selection.task_context["env_id"]),
        selection.level,
    )
    manager.apply_task_context(selection.task_context)
    if manager.env is None:
        raise ProbeError("environment manager did not create an environment")
    manager.env.reset()
    runtime = runtime_handles(manager.env)
    outcome = capture_point(
        runtime=runtime,
        selection=selection,
        policy=inputs.policy,
        sample_sink=artifacts.append_sample,
    )
    outcome_record = outcome_as_dict(outcome)
    point_record = {
        **_selection_fields(selection),
        **outcome_record,
    }
    if outcome.status == "captured":
        metadata = save_capture(
            cv2=cv2,
            paths=paths,
            image=outcome.final_snapshot.image,
            metadata=_capture_metadata(selection, inputs, outcome_record),
        )
        point_record["filename"] = metadata["filename"]
        point_record["sha256"] = metadata["sha256"]
        point_record["sidecar"] = paths.sidecar_path.name
        print(
            f"[Captured] L{selection.level} P{selection.point_id}: "
            f"{paths.image_path} delta_z={outcome.delta_z_uu:.3f}"
        )
    else:
        diagnostic_path = save_diagnostic_frame(
            cv2,
            artifacts,
            selection,
            outcome.final_snapshot.image,
            outcome.reason,
        )
        point_record["diagnostic_frame"] = (
            str(diagnostic_path.relative_to(artifacts.run_dir))
            if diagnostic_path is not None
            else None
        )
        print(f"[Skip] L{selection.level} P{selection.point_id}: {outcome.reason}")
    artifacts.append_point(point_record)
    return outcome.status


def run_capture(args: argparse.Namespace) -> tuple[Path, int]:
    """Run a complete batch and return its run directory and process code."""

    inputs = _resolve_inputs(args)
    artifacts = BatchArtifacts.create(inputs)
    metadata = build_initial_run_metadata(inputs)
    metadata["policy"] = policy_as_dict(inputs.policy)
    write_json_atomic(artifacts.run_path, metadata)
    print(f"[Setup] Tasks: {len(inputs.selections)}")
    print(f"[Setup] Public stretcher directory: {artifacts.stretcher_dir}")
    print(f"[Setup] Diagnostic run directory: {artifacts.run_dir}")

    counts: Counter[str] = Counter()
    active_error = None
    manager = None
    try:
        cv2 = _load_runtime_dependencies()
        manager = EnvManager(
            resolution=inputs.render.resolution,
            render_quality=inputs.render.quality,
            offscreen=inputs.render.offscreen,
        )
        for selection in inputs.selections:
            try:
                status = _run_selection(
                    manager=manager,
                    cv2=cv2,
                    inputs=inputs,
                    artifacts=artifacts,
                    selection=selection,
                )
            except Exception as error:  # keep the remaining task points runnable
                status = "failed"
                artifacts.append_point(
                    {
                        **_selection_fields(selection),
                        "status": status,
                        "reason": "runtime_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(
                    f"[Failed] L{selection.level} P{selection.point_id}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )
            counts[status] += 1
    except BaseException as error:
        active_error = error
        raise
    finally:
        if manager is not None:
            manager.close_env()
        metadata["completed_at_utc"] = utc_now()
        metadata["counts"] = dict(counts)
        if active_error is None:
            metadata["status"] = "completed"
        elif isinstance(active_error, KeyboardInterrupt):
            metadata["status"] = "interrupted"
            metadata["error"] = "KeyboardInterrupt"
        else:
            metadata["status"] = "failed"
            metadata["error"] = f"{type(active_error).__name__}: {active_error}"
        write_json_atomic(artifacts.run_path, metadata)

    partial = counts.get("skipped", 0) > 0 or counts.get("failed", 0) > 0
    print(f"[Completed] counts={dict(counts)} run={artifacts.run_dir}")
    return artifacts.run_dir, 2 if partial else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the direct-script command-line contract."""

    parser = argparse.ArgumentParser(
        description=(
            "Capture per-Level/Point stretcher semantic goal images through "
            "validated actor teleport and physics settling."
        )
    )
    parser.add_argument("--levels", nargs="+", type=int, required=True)
    parser.add_argument("--point-ids", nargs="+", type=int, default=None)
    parser.add_argument("--topomap-dir", required=True)
    parser.add_argument(
        "--capture-distance-uu",
        type=float,
        default=DEFAULT_CAPTURE_DISTANCE_UU,
    )
    parser.add_argument(
        "--initial-height-offset-uu",
        type=float,
        default=DEFAULT_HEIGHT_OFFSET_UU,
    )
    parser.add_argument(
        "--height-retry-step-uu",
        type=float,
        default=DEFAULT_HEIGHT_RETRY_STEP_UU,
    )
    parser.add_argument(
        "--min-delta-z-uu",
        type=float,
        default=DEFAULT_MIN_DELTA_Z_UU,
    )
    parser.add_argument(
        "--max-delta-z-uu",
        type=float,
        default=DEFAULT_MAX_DELTA_Z_UU,
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--sample-delays",
        nargs="+",
        type=float,
        default=DEFAULT_SAMPLE_DELAYS,
    )
    parser.add_argument(
        "--stable-tail-samples",
        type=int,
        default=DEFAULT_STABLE_TAIL_SAMPLES,
    )
    parser.add_argument(
        "--position-epsilon-uu",
        type=float,
        default=DEFAULT_POSITION_EPSILON_UU,
    )
    parser.add_argument(
        "--rotation-epsilon-deg",
        type=float,
        default=DEFAULT_ROTATION_EPSILON_DEG,
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=DEFAULT_RESOLUTION,
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument("--render-quality", type=int, default=2)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--offscreen", dest="offscreen", action="store_true")
    display_group.add_argument("--onscreen", dest="offscreen", action="store_false")
    parser.set_defaults(offscreen=True)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stretcher goal-image capture command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _run_dir, exit_code = run_capture(args)
        return exit_code
    except KeyboardInterrupt:
        print("[Interrupted] stretcher capture stopped by user", file=sys.stderr)
        return 130
    except ProbeError as error:
        print(f"[Error] {error}", file=sys.stderr)
        return 1
