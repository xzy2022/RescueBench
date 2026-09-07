"""Measure one raw waypoint pulse without pose-feedback correction."""

import math

from benchmark.vint_waypoint_control import waypoint_to_move
from benchmark.waypoint_control import local_xy, wrap_degrees


def _execute_pulse(experiment, case_name, waypoint, move, pulse_s):
    """Send one move, sample without resending, then explicitly stop."""
    before = experiment.sample("open_loop_before", case=case_name)
    experiment.send(move, f"open_loop_{case_name}")
    move_times = (experiment.last_command_start, experiment.last_command_end)
    experiment.observe(
        pulse_s,
        "open_loop_pulse",
        period_s=experiment.plan["open_loop_sample_period_s"],
        case=case_name,
        waypoint=waypoint,
    )
    pulse_end = experiment.sample("open_loop_pulse_end", case=case_name)

    experiment.send([0.0, 0.0], "open_loop_stop")
    stop_times = (experiment.last_command_start, experiment.last_command_end)
    stopped = experiment.sample("open_loop_stopped", case=case_name)
    experiment.observe(
        experiment.plan["stop_observe_s"],
        "open_loop_after_stop",
        period_s=experiment.plan["open_loop_sample_period_s"],
        case=case_name,
    )
    final = experiment.sample("open_loop_final", case=case_name)
    return {
        "before": before,
        "pulse_end": pulse_end,
        "stopped": stopped,
        "final": final,
        "move_start": move_times[0],
        "move_end": move_times[1],
        "stop_start": stop_times[0],
        "stop_end": stop_times[1],
    }


def _summarize_response(context, trace):
    """Express pulse, stop-transition and residual motion in local frames."""
    before_pose = trace["before"]["actor_pose"]
    pulse_end_pose = trace["pulse_end"]["actor_pose"]
    stopped_pose = trace["stopped"]["actor_pose"]
    final_pose = trace["final"]["actor_pose"]
    pulse_displacement = local_xy(before_pose, pulse_end_pose[:2])
    stop_displacement = local_xy(pulse_end_pose, stopped_pose[:2])
    residual_displacement = local_xy(stopped_pose, final_pose[:2])
    return {
        **context,
        "before_pose": before_pose,
        "pulse_end_pose": pulse_end_pose,
        "stopped_pose": stopped_pose,
        "final_pose": final_pose,
        "move_command_duration_s": (
            trace["move_end"]["time_s"] - trace["move_start"]["time_s"]
        ),
        "actual_hold_after_move_command_s": (
            trace["stop_start"]["time_s"] - trace["move_end"]["time_s"]
        ),
        "move_start_to_stop_start_s": (
            trace["stop_start"]["time_s"] - trace["move_start"]["time_s"]
        ),
        "stop_command_duration_s": (
            trace["stop_end"]["time_s"] - trace["stop_start"]["time_s"]
        ),
        "pulse_displacement_local_cm": pulse_displacement,
        "pulse_displacement_cm": math.hypot(*pulse_displacement),
        "pulse_yaw_delta_deg": wrap_degrees(pulse_end_pose[4] - before_pose[4]),
        "stop_transition_displacement_local_cm": stop_displacement,
        "stop_transition_displacement_cm": math.hypot(*stop_displacement),
        "stop_transition_yaw_delta_deg": wrap_degrees(
            stopped_pose[4] - pulse_end_pose[4]
        ),
        "residual_displacement_local_cm": residual_displacement,
        "residual_displacement_cm": math.hypot(*residual_displacement),
        "residual_yaw_delta_deg": wrap_degrees(final_pose[4] - stopped_pose[4]),
    }


def run_open_loop(experiment, origin, case_name, reset_comparison):
    """Convert and apply exactly one selected raw waypoint pulse."""
    case = experiment.plan["open_loop_cases"][case_name]
    waypoint = [float(value) for value in case["waypoint"]]
    conversion = experiment.plan["waypoint_conversion"]
    move = waypoint_to_move(waypoint, **conversion)
    waypoint_norm = math.hypot(*waypoint)
    bearing_deg = (
        math.degrees(math.atan2(waypoint[1], waypoint[0])) if waypoint_norm else None
    )
    context = {
        "case": case_name,
        "reset_comparison": reset_comparison,
        "operational_origin_pose": origin["actor_pose"],
        "waypoint": waypoint,
        "waypoint_norm": waypoint_norm,
        "waypoint_bearing_deg": bearing_deg,
        "waypoint_conversion": conversion,
        "move": move,
        "requested_pulse_s": case["pulse_s"],
    }
    experiment.record(
        "open_loop_input",
        normalize=conversion["normalize"],
        max_v=conversion["max_v"],
        rate_hz=conversion["rate_hz"],
        **context,
    )
    trace = _execute_pulse(experiment, case_name, waypoint, move, case["pulse_s"])
    result = _summarize_response(context, trace)
    experiment.record("open_loop_result", **result)
    return result
