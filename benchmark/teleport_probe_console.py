"""Terminal presentation for the RescueBench teleport capture probe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.teleport_probe_data import (
    SCENES,
    PoseSpec,
    ProbeInputs,
    TaskSelection,
    configured_pose,
    format_pose,
    load_level_points,
    scene_for_env_id,
)


def print_scene_catalog() -> None:
    """Print the canonical scene catalog without launching UE."""

    print("RescueBench benchmark scenes (7 canonical scenes):")
    for scene in SCENES:
        print(f"  [{scene.index}/7] {scene.name}")
        for env_id in scene.env_ids:
            variant = " (door-open variant)" if env_id.endswith("_dooropen") else ""
            print(f"        {env_id}{variant}")


def print_level_points(level: int) -> None:
    """Print all zero-based task points for one level."""

    points = load_level_points(level)
    print(f"Level {level} test points ({len(points)}, zero-based):")
    for point_id, point in enumerate(points):
        env_id = str(point.get("env_id", ""))
        scene = scene_for_env_id(env_id)
        print(f"  P{point_id:<2d} scene=[{scene.index}/7] {scene.name} env_id={env_id}")
        for label, key in (
            ("agent", "agent_loc"),
            ("injured", "injured_player_loc"),
            ("stretcher", "stretcher_loc"),
            ("ambulance", "ambulance_loc"),
        ):
            print(f"       {label:<9s} {format_pose(point[key])}")


def print_task_configuration(selection: TaskSelection) -> None:
    """Print the selected task point and its configured object poses."""

    point = selection.raw_point
    env_id = str(selection.task_context["env_id"])
    variant = "dooropen" if env_id.endswith("_dooropen") else "default"
    reference_path = selection.task_context.get("reference_image_path") or "NONE"

    print("\n[Selection]")
    print(
        f"  Scene                : [{selection.scene.index}/7] {selection.scene.name}"
    )
    print(f"  Runtime env_id        : {env_id}")
    print(f"  Scene variant         : {variant}")
    print(f"  Level / Point         : L{selection.level} / P{selection.point_id}")
    print("  Point numbering       : zero-based")
    print(f"  Source                : {selection.source_path}")
    print(f"  Source line           : {selection.source_line}")

    print("\n[Task configuration from JSONL]")
    print(
        "  Agent configured      : " + format_pose(configured_pose(point, "agent_loc"))
    )
    print(
        "  Injured configured    : "
        + format_pose(configured_pose(point, "injured_player_loc"))
    )
    print(
        "  Stretcher configured  : "
        + format_pose(configured_pose(point, "stretcher_loc"))
    )
    print(
        "  Ambulance configured  : "
        + format_pose(configured_pose(point, "ambulance_loc"))
    )
    print(f"  Injured appearance ID : {point.get('injured_agent_id')}")
    print(f"  Reference image       : {reference_path}")
    print(f"  Timeout               : {selection.task_context['timeout']} s")


def print_probe_inputs(inputs: ProbeInputs, run_dir: Path) -> None:
    """Print validated probe inputs and the owned output directory."""

    print_task_configuration(inputs.selection)
    print(f"\n[Probe input]\n  Pose manifest         : {inputs.poses_path}")
    print(f"  Pose count            : {len(inputs.poses)}")
    width, height = inputs.render.resolution
    print(f"  Requested resolution  : {width} x {height}")
    print(f"  Sample delays         : {list(inputs.sample_delays)} s")
    print(f"  Output directory      : {run_dir}")


def print_runtime_objects(agent_name: str, cam_id: int) -> None:
    """Print the resolved runtime actor and camera identifiers."""

    print("\n[Runtime objects]")
    print(f"  Agent actor           : {agent_name}")
    print(f"  Robot camera ID       : {cam_id}")
    print("  Reset observation     : discarded; baseline is captured again")


def print_teleport_request(
    pose_index: int,
    pose_count: int,
    pose: PoseSpec,
    before_actor_pose: tuple[float, ...],
    before_camera_pose: list[float],
) -> None:
    """Print one teleport request and the hard-read state preceding it."""

    print(f"\n[Teleport {pose_index}/{pose_count}] {pose.pose_id}")
    print(f"  Note                  : {pose.note or 'NONE'}")
    print(f"  Requested actor pose  : {format_pose(pose.pose)}")
    print(f"  Before actor pose     : {format_pose(before_actor_pose)}")
    print(f"  Before camera pose    : {format_pose(before_camera_pose)}")


def print_sample(row: dict[str, Any], requested_resolution: tuple[int, int]) -> None:
    """Print one durable sample record and any real-resolution mismatch."""

    actual = row["actual_poses"]
    frame = row["frame"]
    print(
        f"\n[Sample {row['pose_id']} t={row['sample_delay_s']:.3f}s "
        f"elapsed={row['sample_elapsed_s']:.3f}s]"
    )
    print(f"  Agent actor actual    : {format_pose(actual['agent_actor'])}")
    print(f"  Robot camera actual   : {format_pose(actual['robot_camera'])}")
    print("  Camera - actor xyz    : " + format_pose(row["camera_minus_actor_xyz"]))
    print(f"  Injured actual        : {format_pose(actual['injured'])}")
    print(f"  Stretcher actual      : {format_pose(actual['stretcher'])}")
    print(f"  Ambulance actual      : {format_pose(actual['ambulance'])}")
    if row["requested_to_actual_error"] is not None:
        error = row["requested_to_actual_error"]
        print(f"  Request error xyz     : {format_pose(error['xyz'])}")
        print("  Request error rotation: " + format_pose(error["rotation_rpy_deg"]))
    print(f"  Frame shape           : {frame['shape']} dtype={frame['dtype']}")
    print(f"  Saved                 : {frame['path']}")
    actual_height, actual_width = frame["shape"][:2]
    requested_width, requested_height = requested_resolution
    if (actual_width, actual_height) != (requested_width, requested_height):
        print(
            "  [Resolution mismatch] "
            f"requested={requested_width}x{requested_height}, "
            f"actual={actual_width}x{actual_height}"
        )


def print_completed(run_dir: Path) -> None:
    """Print the final successful output path."""

    print(f"\n[Completed]\n  Output directory      : {run_dir}")
