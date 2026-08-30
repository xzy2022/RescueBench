"""Command-line orchestration for the RescueBench teleport capture probe."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.core.env_manager import EnvManager
from benchmark.teleport_probe_artifacts import (
    ProbeArtifacts,
    build_run_metadata,
    utc_now,
    write_json_atomic,
)
from benchmark.teleport_probe_capture import (
    CaptureSession,
    CaptureWriter,
    SequenceRequest,
    runtime_handles,
)
from benchmark.teleport_probe_console import (
    print_completed,
    print_level_points,
    print_probe_inputs,
    print_runtime_objects,
    print_scene_catalog,
    print_teleport_request,
)
from benchmark.teleport_probe_data import (
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_DELAYS,
    TIME_LIMITS,
    ProbeError,
    ProbeInputs,
    RenderSettings,
    load_pose_manifest,
    load_task_selection,
    validate_resolution,
    validate_sample_delays,
)


def _resolve_inputs(args: argparse.Namespace) -> ProbeInputs:
    selection = load_task_selection(args.level, args.point_id)
    poses_path = Path(args.poses_file).expanduser().resolve()
    poses = load_pose_manifest(poses_path)
    resolution = validate_resolution(args.resolution)
    sample_delays = validate_sample_delays(args.sample_delays)
    if args.render_quality < 0:
        raise ProbeError("Render quality must be non-negative")
    if "UnrealEnv" not in os.environ:
        raise ProbeError(
            "UnrealEnv is not set; run through the project rescue-run wrapper or "
            "set it to the UE asset directory"
        )
    return ProbeInputs(
        selection=selection,
        poses_path=poses_path,
        poses=poses,
        output_root=Path(args.output_root),
        render=RenderSettings(
            resolution=resolution,
            offscreen=args.offscreen,
            quality=args.render_quality,
        ),
        sample_delays=sample_delays,
    )


def _load_runtime_dependencies() -> Any:
    numpy = importlib.import_module("numpy")
    numpy.bool8 = numpy.bool_
    importlib.import_module("gym_rescue")
    return importlib.import_module("cv2")


def _new_capture_session(
    manager: EnvManager,
    inputs: ProbeInputs,
    artifacts: ProbeArtifacts,
) -> CaptureSession:
    cv2 = _load_runtime_dependencies()
    manager.ensure_env(
        str(inputs.selection.task_context["env_id"]),
        inputs.selection.level,
    )
    manager.apply_task_context(inputs.selection.task_context)
    manager.env.reset()
    runtime = runtime_handles(manager.env)

    print_runtime_objects(runtime.agent_name, runtime.cam_id)

    writer = CaptureWriter(
        cv2=cv2,
        artifacts=artifacts,
        selection=inputs.selection,
        requested_resolution=inputs.render.resolution,
    )
    return CaptureSession(runtime=runtime, writer=writer)


def _execute_probe(
    manager: EnvManager,
    inputs: ProbeInputs,
    artifacts: ProbeArtifacts,
) -> dict[str, Any]:
    session = _new_capture_session(manager, inputs, artifacts)
    baseline_rows = session.capture_sequence(
        SequenceRequest(
            sequence_index=0,
            pose_id="baseline",
            delays=inputs.sample_delays,
            requested_pose=None,
            before_actor_pose=None,
        )
    )

    final_rows = baseline_rows
    for pose_index, pose in enumerate(inputs.poses, start=1):
        before = session.capture_state()
        before_actor_pose = tuple(before.actual["agent_actor"])
        print_teleport_request(
            pose_index,
            len(inputs.poses),
            pose,
            before_actor_pose,
            before.camera_pose,
        )

        session.runtime.env_unwrapped.unrealcv.set_obj_rotation(
            session.runtime.agent_name,
            list(pose.rotation_rpy_deg),
        )
        session.runtime.env_unwrapped.unrealcv.set_obj_location(
            session.runtime.agent_name,
            list(pose.location_xyz),
        )
        final_rows = session.capture_sequence(
            SequenceRequest(
                sequence_index=pose_index,
                pose_id=pose.pose_id,
                delays=inputs.sample_delays,
                requested_pose=tuple(pose.pose),
                before_actor_pose=before_actor_pose,
            )
        )
    return final_rows[-1]


def _record_failure(
    metadata: dict[str, Any],
    artifacts: ProbeArtifacts,
    error: BaseException,
) -> None:
    metadata["status"] = (
        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    )
    metadata["completed_at_utc"] = utc_now()
    metadata["error"] = f"{type(error).__name__}: {error}"
    write_json_atomic(artifacts.metadata_path, metadata)


def run_probe(args: argparse.Namespace) -> Path:
    """Validate inputs, run the probe, and persist terminal run status."""

    inputs = _resolve_inputs(args)
    artifacts = ProbeArtifacts.create(inputs)
    metadata = build_run_metadata(inputs)
    write_json_atomic(artifacts.metadata_path, metadata)
    print_probe_inputs(inputs, artifacts.run_dir)

    manager = EnvManager(
        resolution=inputs.render.resolution,
        render_quality=inputs.render.quality,
        offscreen=inputs.render.offscreen,
    )
    try:
        final_sample = _execute_probe(manager, inputs, artifacts)
        metadata["status"] = "completed"
        metadata["completed_at_utc"] = utc_now()
        metadata["final_sample"] = final_sample
        write_json_atomic(artifacts.metadata_path, metadata)
        print_completed(artifacts.run_dir)
        return artifacts.run_dir
    finally:
        active_error = sys.exc_info()[1]
        try:
            if active_error is not None:
                _record_failure(metadata, artifacts, active_error)
        finally:
            manager.close_env()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Load one RescueBench test point, teleport the protagonist actor to "
            "poses from an external JSON file, and save measured poses/raw PNGs."
        )
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="Print the seven-scene catalog and exit without launching UE.",
    )
    parser.add_argument(
        "--list-points",
        action="store_true",
        help="Print all zero-based test points for --level and exit without UE.",
    )
    parser.add_argument("--level", type=int, help="Benchmark level (0-4).")
    parser.add_argument(
        "--point-id",
        type=int,
        help="Zero-based point index in level_<LEVEL>.jsonl.",
    )
    parser.add_argument(
        "--poses-file",
        help="Path to the external JSON pose manifest.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory under which a unique timestamped run directory is created.",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_RESOLUTION,
        help="Requested UE/UnrealCV resolution (default: 640 640).",
    )
    parser.add_argument(
        "--sample-delays",
        nargs="+",
        type=float,
        default=DEFAULT_SAMPLE_DELAYS,
        metavar="SECONDS",
        help="Strictly increasing capture times after reset/teleport.",
    )
    parser.add_argument(
        "--render-quality",
        type=int,
        default=2,
        help="UE render quality passed to EnvManager (default: 2).",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--offscreen",
        dest="offscreen",
        action="store_true",
        help="Launch UE in offscreen mode (default).",
    )
    display_group.add_argument(
        "--onscreen",
        dest="offscreen",
        action="store_false",
        help="Launch UE with onscreen rendering.",
    )
    parser.set_defaults(offscreen=True)
    return parser


def _require_run_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    missing = [
        flag
        for flag, value in (
            ("--level", args.level),
            ("--point-id", args.point_id),
            ("--poses-file", args.poses_file),
            ("--output-root", args.output_root),
        )
        if value is None
    ]
    if missing:
        parser.error("normal probe mode requires: " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.list_scenes:
            print_scene_catalog()
            return 0
        if args.list_points:
            if args.level is None:
                parser.error("--list-points requires --level")
            if args.level not in TIME_LIMITS:
                raise ProbeError(f"Level must be one of {sorted(TIME_LIMITS)}")
            print_level_points(args.level)
            return 0

        _require_run_arguments(parser, args)
        run_probe(args)
        return 0
    except ProbeError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[Interrupted] Probe stopped by user.", file=sys.stderr)
        return 130
