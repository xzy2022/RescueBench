"""CLI: run one manually selected stage of a model-independent control experiment."""

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from benchmark.vint_waypoint_control import waypoint_to_move

DEFAULTS = {
    "level": 0,
    "point_id": 0,
    "resolution": [320, 320],
    "period_s": 0.2,
    "settle_s": 2.0,
    "observe_s": 3.0,
    "case_observe_s": 2.0,
    "diagnostic_observe_s": 2.0,
    "force_hold_s": 2.0,
    "release_observe_s": 1.0,
    "camera_mount_observe_s": 1.0,
    "stop_observe_s": 2.0,
    "waypoint_timeout_s": 15.0,
    "diagnostic_target_yaw_deg": 90.0,
    "rotation_axis_deg": 30.0,
    "frame": "start_local_cm",
    "waypoint_conversion": {
        "normalize": True,
        "max_v": 0.2,
        "rate_hz": 4.0,
    },
    "controller": {
        "turn_sign": 1,
        "turn_gain": 0.5,
        "forward_gain": 0.3,
        "max_turn": 15.0,
        "max_forward": 30.0,
        "turn_in_place_deg": 45.0,
        "arrival_radius_cm": 10.0,
    },
}


def read_plan(path):
    """Load explicit experiment inputs and the small set of controller defaults."""
    supplied = json.loads(path.read_text(encoding="utf-8-sig"))
    plan = {**DEFAULTS, **supplied}
    plan["waypoint_conversion"] = {
        **DEFAULTS["waypoint_conversion"],
        **supplied.get("waypoint_conversion", {}),
    }
    plan["controller"] = {**DEFAULTS["controller"], **supplied.get("controller", {})}
    if plan["frame"] not in ("start_local_cm", "world_cm"):
        raise ValueError("frame must be start_local_cm or world_cm")
    for name in (
        "period_s",
        "settle_s",
        "observe_s",
        "case_observe_s",
        "diagnostic_observe_s",
        "force_hold_s",
        "release_observe_s",
        "camera_mount_observe_s",
        "stop_observe_s",
        "waypoint_timeout_s",
    ):
        if not math.isfinite(plan[name]) or plan[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    controller = plan["controller"]
    if controller["turn_sign"] not in (-1, 1):
        raise ValueError("turn_sign must be -1 or 1")
    for name, value in controller.items():
        if not math.isfinite(value) or (name != "turn_sign" and value <= 0):
            raise ValueError(f"Invalid controller parameter: {name}")
    if controller["max_turn"] > 30 or controller["max_forward"] > 100:
        raise ValueError("Mixed action limits: max_turn <= 30, max_forward <= 100")
    return plan


def check_named_cases(cases, value_name, value_length=None):
    """Validate ordered calibration cases with unique IDs and finite values."""
    if not isinstance(cases, list) or not cases:
        raise ValueError("Calibration cases must be a non-empty JSON array")
    seen = set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Each calibration case requires a non-empty string id")
        if case_id in seen:
            raise ValueError(f"Duplicate calibration case id: {case_id}")
        seen.add(case_id)
        value = case.get(value_name)
        if value_length is not None:
            if not isinstance(value, list) or len(value) != value_length:
                raise ValueError(
                    f"{case_id}.{value_name} must contain {value_length} numbers"
                )
            values = value
        else:
            values = [value]
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in values
        ):
            raise ValueError(f"{case_id}.{value_name} must contain only numbers")
        if any(not math.isfinite(item) for item in values):
            raise ValueError(f"{case_id}.{value_name} must be finite")


def check_stage(plan, args):
    """Check the selected stage's required inputs before launching UE."""
    if args.stage == "position":
        check_named_cases(plan.get("position_cases"), "offset_world_xy_cm", 2)
    if args.stage == "yaw":
        check_named_cases(plan.get("yaw_cases"), "yaw_deg")
    if args.stage == "head":
        check_named_cases(plan.get("head_cases"), "head_index")
        if any(
            isinstance(case["head_index"], bool)
            or not isinstance(case["head_index"], int)
            or case["head_index"] < 0
            for case in plan["head_cases"]
        ):
            raise ValueError("head_index must be a non-negative integer")
    if args.stage == "yaw_diagnosis":
        arms = (
            "actor_only",
            "move_before",
            "stand_before",
            "full_step_before",
            "force_hold",
        )
        if args.case not in arms:
            raise ValueError(f"yaw_diagnosis --case must be one of: {', '.join(arms)}")
        if not math.isfinite(plan["diagnostic_target_yaw_deg"]):
            raise ValueError("diagnostic_target_yaw_deg must be finite")
    if args.stage == "rotation_axes":
        value = plan["rotation_axis_deg"]
        if not math.isfinite(value) or not 0 < abs(value) <= 180:
            raise ValueError("rotation_axis_deg must be finite and within 0-180")
    if args.stage == "camera_mount":
        check_named_cases(plan.get("camera_location_cases"), "relative_location", 3)
        check_named_cases(plan.get("camera_rotation_cases"), "head_rotation", 3)
    if args.stage == "actions":
        case = plan["actions"][args.case]
        waypoint = case.get("waypoint")
        if not isinstance(waypoint, list) or len(waypoint) not in (2, 4):
            raise ValueError("Action waypoint must contain 2 or 4 numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in waypoint
        ):
            raise ValueError("Action waypoint must contain only finite numbers")
        conversion = plan["waypoint_conversion"]
        if not isinstance(conversion.get("normalize"), bool):
            raise ValueError("waypoint_conversion.normalize must be boolean")
        for name in ("max_v", "rate_hz"):
            value = conversion.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"waypoint_conversion.{name} must be positive")
        turn, forward = waypoint_to_move(case["waypoint"], **conversion)
        if not (-30 <= turn <= 30 and -100 <= forward <= 100):
            raise ValueError("Action outside Mixed limits")
        if not math.isfinite(case["hold_s"]) or case["hold_s"] <= 0:
            raise ValueError("hold_s must be finite and positive")
    if args.stage == "follow":
        if not plan["waypoints"]:
            raise ValueError("waypoints must contain at least one [x, y] point")
        for x, y in plan["waypoints"]:
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("waypoints must be finite")


def write_json(path, value):
    """Write a readable run artifact."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_metadata(root, output):
    """Read Git provenance through a temporary protected config for runner users."""
    config_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output, delete=False
        ) as stream:
            stream.write(f"[safe]\n\tdirectory = {root.as_posix()}\n")
            config_path = Path(stream.name)
        env = os.environ.copy()
        env["GIT_CONFIG_GLOBAL"] = str(config_path)
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        command = ["git", "-C", str(root)]
        head = subprocess.run(
            [*command, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        status = subprocess.run(
            [*command, "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        error = "\n".join(
            part.strip() for part in (head.stderr, status.stderr) if part.strip()
        )
        return head.stdout.strip(), status.stdout, error
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


def run_stage(args):
    """Run exactly one stage; future stages require separate invocations."""
    plan = read_plan(args.plan)
    if args.level is not None:
        plan["level"] = args.level
    if args.point_id is not None:
        plan["point_id"] = args.point_id
    check_stage(plan, args)
    selected_case = args.case if args.stage in ("actions", "yaw_diagnosis") else None
    repeat_action = bool(args.repeat) if args.stage == "actions" else False
    if args.preview:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "case": selected_case,
                    "repeat": repeat_action,
                    "plan": plan,
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        raise ValueError("--output is required unless --preview is selected")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent.parent
    git_head, git_status, git_error = git_metadata(root, output)
    write_json(
        output / "run.json",
        {
            "stage": args.stage,
            "case": selected_case,
            "repeat": repeat_action,
            "plan_path": str(args.plan.resolve()),
            "effective_plan": plan,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "git_head": git_head,
            "git_status": git_status,
            "git_error": git_error,
            "time_basis": "wall_clock_perf_counter",
            "pose_order_assumption": ["x", "y", "z", "rotation0", "yaw", "rotation2"],
            "position_unit": "cm",
            "angle_unit": "degree",
            "controller": (
                "experiment copy of VINTAgent waypoint conversion"
                if args.stage == "actions"
                else "independent geometric controller; not NoMaD adapter"
            ),
        },
    )
    runtime = importlib.import_module("benchmark.waypoint_runtime")
    experiment = runtime.Experiment(plan, output)
    summary = {"status": "running"}
    try:
        prepare_action = args.stage not in (
            "yaw_diagnosis",
            "rotation_axes",
            "camera_mount",
        )
        origin, reset_comparison = experiment.start(prepare_action=prepare_action)
        if args.stage == "pose":
            result = experiment.run_pose(origin, reset_comparison)
        elif args.stage == "position":
            result = experiment.run_position(origin)
        elif args.stage == "yaw":
            result = experiment.run_yaw(origin)
        elif args.stage == "head":
            result = experiment.run_head(origin)
        elif args.stage == "yaw_diagnosis":
            diagnostics = importlib.import_module(
                "benchmark.waypoint_stage1_diagnostics"
            )
            result = diagnostics.run_yaw_diagnosis(experiment, origin, selected_case)
        elif args.stage == "rotation_axes":
            diagnostics = importlib.import_module(
                "benchmark.waypoint_stage1_diagnostics"
            )
            result = diagnostics.run_rotation_axes(experiment, origin)
        elif args.stage == "camera_mount":
            diagnostics = importlib.import_module(
                "benchmark.waypoint_stage1_diagnostics"
            )
            result = diagnostics.run_camera_mount(experiment, origin)
        elif args.stage == "actions":
            result = experiment.run_actions(
                selected_case, repeat_action, origin, reset_comparison
            )
        else:
            result = experiment.follow(origin)
        summary = {"status": "finished", **result}
    except BaseException as exc:
        summary = {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
            "error": str(exc),
        }
        raise
    finally:
        try:
            experiment.close()
        finally:
            summary["samples"] = experiment.sample_count
            write_json(output / "summary.json", summary)
            print(f"Artifacts: {output}", flush=True)


def main():
    """Expose a runtime command and an explicit-file offline report command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one stage; --preview never imports UE")
    run.add_argument(
        "--stage",
        choices=(
            "pose",
            "position",
            "yaw",
            "head",
            "yaw_diagnosis",
            "rotation_axes",
            "camera_mount",
            "actions",
            "follow",
        ),
        required=True,
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, help="New run directory; must not exist")
    run.add_argument("--level", type=int, help="Override plan level for this run")
    run.add_argument("--point-id", type=int, help="Override plan point_id for this run")
    run.add_argument(
        "--case", default="forward", help="One named action case from plan"
    )
    run.add_argument(
        "--repeat",
        action="store_true",
        help="Resend selected action every sampling cycle",
    )
    run.add_argument(
        "--preview", action="store_true", help="Print resolved input only; no simulator"
    )
    report = commands.add_parser(
        "report", help="Plot one explicitly selected samples.jsonl"
    )
    report.add_argument("--samples", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_stage(args)
    else:
        reporting = importlib.import_module("benchmark.waypoint_report")
        reporting.report(args.samples, args.output)


if __name__ == "__main__":
    main()
