"""Plot actual poses, issued commands and target errors from an explicit run log."""

import csv
import importlib
import json


def report(samples_path, output):
    """Create CSV and PNG without importing Gym, UnrealCV, Torch or a model."""
    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = [row for row in rows if row["event"] == "sample"]
    decisions = [row for row in rows if row["event"] == "decision"]
    commands = [row for row in rows if row["event"] == "command_end"]
    if not samples:
        raise ValueError("No pose samples in the selected log")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "poses.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "label",
                "x_cm",
                "y_cm",
                "z_cm",
                "rotation0",
                "yaw_deg",
                "rotation2",
                "camera_x_cm",
                "camera_y_cm",
                "camera_z_cm",
                "camera_rotation0",
                "camera_yaw_deg",
                "camera_rotation2",
                "active_turn",
                "active_forward",
                "active_head_index",
                "active_head_rotation0",
                "active_head_rotation1",
                "active_head_rotation2",
                "read_duration_s",
            ]
        )
        for row in samples:
            writer.writerow(
                [
                    row["time_s"],
                    row["label"],
                    *row["actor_pose"],
                    *row["camera_pose"],
                    *row["active_move"],
                    row.get("active_head_index", 0),
                    *row.get("active_head_rotation", [0.0, 0.0, 0.0]),
                    row["time_s"] - row["read_started_s"],
                ]
            )
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    path_axes, yaw_axes, command_axes, distance_axes = axes.flat
    path_axes.plot(
        [r["actor_pose"][0] for r in samples],
        [r["actor_pose"][1] for r in samples],
        ".-",
        label="actor",
    )
    path_axes.plot(
        [r["camera_pose"][0] for r in samples],
        [r["camera_pose"][1] for r in samples],
        alpha=0.6,
        label="camera",
    )
    for route in (row for row in rows if row["event"] == "route"):
        points = route["targets_world_cm"]
        origin = route["origin_pose"]
        path_axes.plot(
            [origin[0]] + [p[0] for p in points],
            [origin[1]] + [p[1] for p in points],
            "x--",
            label="requested order",
        )
        for index, point in enumerate(points):
            path_axes.annotate(str(index), point)
    path_axes.set(
        xlabel="World X (cm)",
        ylabel="World Y (cm)",
        title="XY path (segments are visual guides)",
    )
    path_axes.set_aspect("equal", adjustable="datalim")
    path_axes.legend()
    for key, label in (("actor_pose", "actor"), ("camera_pose", "camera")):
        yaw_axes.plot(
            [r["time_s"] for r in samples], [r[key][4] for r in samples], label=label
        )
    yaw_axes.set(xlabel="Wall time (s)", ylabel="Yaw (degrees, wrapped)")
    yaw_axes.legend()
    for index, label in enumerate(("turn command", "forward command")):
        command_axes.step(
            [r["started_s"] for r in commands],
            [r["move"][index] for r in commands],
            where="post",
            label=label,
        )
    command_axes.set(
        xlabel="Wall time (s)", ylabel="Command value (not physical speed)"
    )
    command_axes.legend()
    indices = sorted({r["waypoint_index"] for r in decisions})
    for index in indices:
        selected = [r for r in decisions if r["waypoint_index"] == index]
        distance_axes.plot(
            [r["sample_time_s"] for r in selected],
            [r["distance_cm"] for r in selected],
            label=f"waypoint {index}",
        )
    if indices:
        distance_axes.legend()
    distance_axes.set(xlabel="Wall time (s)", ylabel="Target distance (cm)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "overview.png", dpi=160)
    plt.close(figure)
    print(f"Report: {output.resolve()}")
