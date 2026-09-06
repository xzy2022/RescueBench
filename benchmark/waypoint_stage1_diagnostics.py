"""Stage-one diagnostics for Actor yaw, rotation slots, and camera mounting."""

import statistics
import time

from benchmark.waypoint_control import wrap_degrees


def _median_values(samples, key):
    return [
        statistics.median(row[key][index] for row in samples)
        for index in range(len(samples[0][key]))
    ]


def _tail_samples(samples, seconds=1.0):
    threshold = samples[-1]["time_s"] - seconds
    return [row for row in samples if row["time_s"] >= threshold]


def _pose_difference(actual, reference):
    return [
        *(actual[index] - reference[index] for index in range(3)),
        *(wrap_degrees(actual[index] - reference[index]) for index in range(3, 6)),
    ]


def _direct_command(experiment, label, command, action):
    started = experiment.now()
    experiment.record("direct_command_start", label=label, command=command)
    result = action()
    experiment.record(
        "direct_command_end",
        label=label,
        command=command,
        started_s=started,
        response=None if result is None else str(result),
    )
    return result


def _trace_yaw(experiment, seconds, arm, phase, force_pose=None):
    deadline = experiment.now() + seconds
    rows = []
    while experiment.now() < deadline:
        tick = experiment.now()
        if force_pose is not None:
            experiment.base.unrealcv.set_obj_rotation(
                experiment.names[0], force_pose[3:]
            )
        rows.append(
            experiment.sample(
                f"yaw_diagnosis_{arm}_{phase}",
                diagnostic_arm=arm,
                diagnostic_phase=phase,
                requested_yaw_deg=float(experiment.plan["diagnostic_target_yaw_deg"]),
            )
        )
        time.sleep(
            max(
                0.0,
                min(deadline, tick + experiment.plan["period_s"]) - experiment.now(),
            )
        )
    return rows


def run_yaw_diagnosis(experiment, origin, arm):
    """Isolate commands that may overwrite an Actor world rotation."""
    actor = experiment.names[0]
    target = [*origin["actor_pose"]]
    target[4] = float(experiment.plan["diagnostic_target_yaw_deg"])
    if arm == "move_before":
        _direct_command(
            experiment,
            arm,
            "set_move 0 0",
            lambda: experiment.base.unrealcv.set_move_bp(actor, [0, 0]),
        )
    elif arm == "stand_before":
        _direct_command(
            experiment,
            arm,
            "set_standup",
            lambda: experiment.base.unrealcv.set_standup(actor),
        )
    elif arm == "full_step_before":
        experiment.send([0.0, 0.0], arm)
    _direct_command(
        experiment,
        "set_target_yaw",
        f"set_obj_rotation {target[3:]}",
        lambda: experiment.base.unrealcv.set_obj_rotation(actor, target[3:]),
    )
    if arm == "force_hold":
        hold = _trace_yaw(
            experiment,
            experiment.plan["force_hold_s"],
            arm,
            "forced",
            force_pose=target,
        )
        released = _trace_yaw(
            experiment, experiment.plan["release_observe_s"], arm, "released"
        )
        trace = hold + released
    else:
        hold = []
        released = []
        trace = _trace_yaw(
            experiment, experiment.plan["diagnostic_observe_s"], arm, "observe"
        )
    tail = _tail_samples(released or trace)
    result = {
        "arm": arm,
        "configured_start_pose": [
            float(value) for value in experiment.task_context["agent_pose"]
        ],
        "fresh_origin_pose": origin["actor_pose"],
        "requested_actor_pose": target,
        "first_read_actor_pose": trace[0]["actor_pose"],
        "first_read_camera_pose": trace[0]["camera_pose"],
        "settled_actor_pose_median": _median_values(tail, "actor_pose"),
        "settled_camera_pose_median": _median_values(tail, "camera_pose"),
        "forced_actor_pose_median": (
            _median_values(hold, "actor_pose") if hold else None
        ),
        "forced_camera_pose_median": (
            _median_values(hold, "camera_pose") if hold else None
        ),
        "settled_minus_origin_yaw_deg": wrap_degrees(
            statistics.median(row["actor_pose"][4] for row in tail)
            - origin["actor_pose"][4]
        ),
        "settled_minus_requested_yaw_deg": wrap_degrees(
            statistics.median(row["actor_pose"][4] for row in tail) - target[4]
        ),
        "forced_samples": len(hold),
        "released_samples": len(released),
    }
    experiment.record("yaw_diagnosis_result", **result)
    return result


def _raw_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _raw_rotation(experiment, path):
    response = experiment.base.unrealcv.client.request(f"vget /{path}/rotation")
    text = _raw_text(response)
    return text, [float(value) for value in text.split()]


def _set_raw_rotation(experiment, path, values):
    joined = " ".join(str(value) for value in values)
    return experiment.base.unrealcv.client.request(
        f"vset /{path}/rotation {joined}", -1
    )


def _decoded_rotation(experiment, object_name=None, camera_id=None):
    objects = [] if object_name is None else [object_name]
    cameras = [] if camera_id is None else [camera_id]
    actor_poses, camera_poses, _, _, _ = experiment.base.unrealcv.get_pose_img_batch(
        objects, cameras, [True, False, False, False]
    )
    if object_name is not None:
        return [float(value) for value in actor_poses[0][3:]]
    return [float(value) for value in camera_poses[0][3:]]


def _rotation_axis_cases(experiment, path, object_name=None, camera_id=None):
    value = float(experiment.plan["rotation_axis_deg"])
    cases = []
    for index in range(3):
        requested = [0.0, 0.0, 0.0]
        requested[index] = value
        _set_raw_rotation(experiment, path, requested)
        time.sleep(experiment.plan["period_s"])
        raw_reply, raw_values = _raw_rotation(experiment, path)
        result = {
            "id": f"raw_axis_{index}",
            "requested_raw_rotation": requested,
            "raw_reply": raw_reply,
            "raw_values": raw_values,
            "decoded_rotation": _decoded_rotation(experiment, object_name, camera_id),
        }
        cases.append(result)
        experiment.record("rotation_axis_case", target=path, **result)
    return cases


def run_rotation_axes(experiment, origin):
    """Compare protocol slots and decoded rotations on an object and camera 0."""
    object_name = experiment.base.ambulance
    camera_id = 0
    object_path = f"object/{object_name}"
    camera_path = f"camera/{camera_id}"
    object_raw, object_initial = _raw_rotation(experiment, object_path)
    camera_raw, camera_initial = _raw_rotation(experiment, camera_path)
    experiment.base.unrealcv.set_phy(object_name, 0)
    try:
        object_cases = _rotation_axis_cases(
            experiment, object_path, object_name=object_name
        )
        camera_cases = _rotation_axis_cases(
            experiment, camera_path, camera_id=camera_id
        )
    finally:
        _set_raw_rotation(experiment, object_path, object_initial)
        _set_raw_rotation(experiment, camera_path, camera_initial)
    return {
        "origin_pose": origin["actor_pose"],
        "object": {
            "name": object_name,
            "initial_raw_reply": object_raw,
            "cases": object_cases,
        },
        "camera": {
            "id": camera_id,
            "initial_raw_reply": camera_raw,
            "cases": camera_cases,
        },
    }


def _trace_direct_camera(
    experiment, origin_pose, case_type, case_id, location, rotation
):
    actor = experiment.names[0]
    _direct_command(
        experiment,
        f"{case_type}_{case_id}",
        f"set_cam location={location} rotation={rotation}",
        lambda: experiment.base.unrealcv.set_cam(actor, location, rotation),
    )
    samples = experiment.observe(
        experiment.plan["camera_mount_observe_s"],
        f"camera_mount_{case_type}_{case_id}",
        camera_case_type=case_type,
        camera_case_id=case_id,
        configured_camera_location=location,
        configured_camera_rotation=rotation,
    )
    tail = _tail_samples(samples)
    actor_median = _median_values(tail, "actor_pose")
    camera_median = _median_values(tail, "camera_pose")
    return {
        "id": case_id,
        "configured_location": location,
        "configured_rotation": rotation,
        "actor_pose_median": actor_median,
        "actor_pose_delta_from_origin": _pose_difference(actor_median, origin_pose),
        "camera_pose_median": camera_median,
        "camera_rotation_minus_actor_deg": [
            wrap_degrees(camera_median[index] - actor_median[index])
            for index in range(3, 6)
        ],
        "camera_offset_local_xy_cm_median": _median_values(
            tail, "camera_offset_local_xy_cm"
        ),
        "camera_offset_z_cm_median": statistics.median(
            row["camera_offset_z_cm"] for row in tail
        ),
        "camera_yaw_offset_deg_median": statistics.median(
            row["camera_yaw_offset_deg"] for row in tail
        ),
    }


def run_camera_mount(experiment, origin):
    """Separate Actor root pose from direct camera location/head commands."""
    default_location = [
        float(value)
        for value in experiment.base.agents[experiment.names[0]]["relative_location"]
    ]
    neutral_rotation = [0.0, 0.0, 0.0]
    location_results = []
    for case in experiment.plan["camera_location_cases"]:
        location_results.append(
            _trace_direct_camera(
                experiment,
                origin["actor_pose"],
                "location",
                case["id"],
                [float(value) for value in case["relative_location"]],
                neutral_rotation,
            )
        )
    configured_reference = location_results[0]["configured_location"]
    measured_reference = location_results[0]["camera_offset_local_xy_cm_median"] + [
        location_results[0]["camera_offset_z_cm_median"]
    ]
    for result in location_results:
        measured = result["camera_offset_local_xy_cm_median"] + [
            result["camera_offset_z_cm_median"]
        ]
        result["configured_delta_from_first"] = [
            result["configured_location"][index] - configured_reference[index]
            for index in range(3)
        ]
        result["measured_local_delta_from_first"] = [
            measured[index] - measured_reference[index] for index in range(3)
        ]
    rotation_results = []
    for case in experiment.plan["camera_rotation_cases"]:
        rotation_results.append(
            _trace_direct_camera(
                experiment,
                origin["actor_pose"],
                "rotation",
                case["id"],
                default_location,
                [float(value) for value in case["head_rotation"]],
            )
        )
    experiment.base.unrealcv.set_cam(
        experiment.names[0], default_location, neutral_rotation
    )
    return {
        "origin_pose": origin["actor_pose"],
        "default_relative_location": default_location,
        "location_cases": location_results,
        "rotation_cases": rotation_results,
    }
