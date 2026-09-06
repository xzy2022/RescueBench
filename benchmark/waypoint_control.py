"""Geometry and an explicit, model-independent planar waypoint controller."""

import math


def wrap_degrees(angle):
    """Return a signed angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def local_xy(pose, world_xy):
    """Express a world XY point relative to actor XY and yaw (pose index 4)."""
    yaw = math.radians(pose[4])
    delta_x, delta_y = world_xy[0] - pose[0], world_xy[1] - pose[1]
    return [
        math.cos(yaw) * delta_x + math.sin(yaw) * delta_y,
        -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y,
    ]


def world_waypoints(plan, origin):
    """Freeze all input points in world coordinates using one measured origin."""
    if plan["frame"] == "world_cm":
        return [list(point) for point in plan["waypoints"]]
    yaw = math.radians(origin[4])
    return [
        [
            origin[0] + math.cos(yaw) * x - math.sin(yaw) * y,
            origin[1] + math.sin(yaw) * x + math.cos(yaw) * y,
        ]
        for x, y in plan["waypoints"]
    ]


def waypoint_action(pose, target, controller):
    """Return Mixed move commands and geometry; command units are not SI speed."""
    forward, lateral = local_xy(pose, target)
    distance = math.hypot(forward, lateral)
    heading = math.degrees(math.atan2(lateral, forward))
    turn = controller["turn_sign"] * controller["turn_gain"] * heading
    turn = max(-controller["max_turn"], min(controller["max_turn"], turn))
    velocity = min(controller["max_forward"], controller["forward_gain"] * distance)
    velocity *= max(0.0, math.cos(math.radians(heading)))
    if abs(heading) > controller["turn_in_place_deg"]:
        velocity = 0.0
    reached = distance <= controller["arrival_radius_cm"]
    return {
        "target_world_cm": list(target),
        "local_error_cm": [forward, lateral],
        "distance_cm": distance,
        "heading_error_deg": heading,
        "reached": reached,
        "move": [0.0, 0.0] if reached else [turn, velocity],
    }
