"""CLI: run one manually selected stage of a model-independent control experiment."""

import argparse
import importlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULTS = {
    "level": 0,
    "point_id": 0,
    "resolution": [320, 320],
    "period_s": 0.2,
    "settle_s": 2.0,
    "observe_s": 3.0,
    "stop_observe_s": 2.0,
    "waypoint_timeout_s": 15.0,
    "frame": "start_local_cm",
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
    plan["controller"] = {**DEFAULTS["controller"], **supplied.get("controller", {})}
    if plan["frame"] not in ("start_local_cm", "world_cm"):
        raise ValueError("frame must be start_local_cm or world_cm")
    for name in (
        "period_s",
        "settle_s",
        "observe_s",
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


def check_stage(plan, args):
    """Check the selected stage's required inputs before launching UE."""
    if args.stage == "actions":
        case = plan["actions"][args.case]
        turn, forward = case["move"]
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


def run_stage(args):
    """Run exactly one stage; future stages require separate invocations."""
    plan = read_plan(args.plan)
    check_stage(plan, args)
    if args.preview:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "case": args.case,
                    "repeat": args.repeat,
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
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    write_json(
        output / "run.json",
        {
            "stage": args.stage,
            "case": args.case,
            "repeat": args.repeat,
            "plan_path": str(args.plan.resolve()),
            "effective_plan": plan,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "git_head": git_head.stdout.strip(),
            "git_status": git_status.stdout,
            "time_basis": "wall_clock_perf_counter",
            "pose_order_assumption": ["x", "y", "z", "rotation0", "yaw", "rotation2"],
            "position_unit": "cm",
            "angle_unit": "degree",
            "controller": "independent geometric controller; not NoMaD adapter",
        },
    )
    runtime = importlib.import_module("benchmark.waypoint_runtime")
    experiment = runtime.Experiment(plan, output)
    summary = {"status": "running"}
    try:
        origin = experiment.start()
        if args.stage == "pose":
            experiment.observe(plan["observe_s"], "stationary")
            result = {
                "origin_pose": origin["actor_pose"],
                "final_pose": experiment.sample("final")["actor_pose"],
            }
        elif args.stage == "actions":
            result = experiment.run_actions(args.case, args.repeat)
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
    run.add_argument("--stage", choices=("pose", "actions", "follow"), required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, help="New run directory; must not exist")
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
