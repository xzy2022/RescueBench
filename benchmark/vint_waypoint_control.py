"""Experiment-side copy of the ViNT/NoMaD waypoint conversion formula."""

import math


def waypoint_to_move(waypoint, normalize, max_v=0.2, rate_hz=4.0):
    """Reproduce VINTAgent's finite waypoint conversion for isolated tests."""
    values = [float(value) for value in waypoint]
    dx, dy = values[:2]
    hx = hy = None
    if len(values) >= 4:
        hx, hy = values[2:4]

    if normalize:
        scale = max_v / rate_hz
        dx *= scale
        dy *= scale

    epsilon = 1e-8
    if hx is not None and abs(dx) < epsilon and abs(dy) < epsilon:
        angle_rad = math.atan2(hy, hx)
    else:
        angle_rad = math.atan2(dy, dx)
    angle = max(-30.0, min(30.0, math.degrees(angle_rad)))

    if hx is not None and abs(dx) < epsilon and abs(dy) < epsilon:
        velocity = 0.0
    else:
        velocity = dx * rate_hz / max_v * 100.0
    velocity = max(-100.0, min(100.0, velocity))
    return [angle, velocity]
