"""Validated inputs and geometry for stretcher semantic-goal capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.teleport_probe_data import (
    DEFAULT_RESOLUTION,
    TIME_LIMITS,
    ProbeError,
    RenderSettings,
    TaskSelection,
    load_level_points,
    load_task_selection,
    validate_resolution,
)

DEFAULT_CAPTURE_DISTANCES_UU = (200.0, 300.0, 350.0)
DEFAULT_HEIGHT_OFFSET_UU = 160.0
DEFAULT_HEIGHT_RETRY_STEP_UU = 30.0
DEFAULT_MIN_DELTA_Z_UU = 70.0
DEFAULT_MAX_DELTA_Z_UU = 140.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_SETTLE_SAMPLE_INTERVAL_S = 1.0
DEFAULT_STRETCHER_SETTLE_TIMEOUT_S = 20.0
DEFAULT_AGENT_SETTLE_TIMEOUT_S = 12.0
DEFAULT_STABLE_WINDOW_SAMPLES = 3
DEFAULT_POSITION_EPSILON_UU = 1.0
DEFAULT_ROTATION_EPSILON_DEG = 1.0
DEFAULT_MAX_AGENT_XY_ERROR_UU = 10.0
DEFAULT_MAX_AGENT_YAW_ERROR_DEG = 1.0


@dataclass(frozen=True)
class CapturePolicy:
    """Store geometry, height, retry, and stability policy values."""

    capture_distances_uu: tuple[float, ...] = DEFAULT_CAPTURE_DISTANCES_UU
    initial_height_offset_uu: float = DEFAULT_HEIGHT_OFFSET_UU
    height_retry_step_uu: float = DEFAULT_HEIGHT_RETRY_STEP_UU
    min_delta_z_uu: float = DEFAULT_MIN_DELTA_Z_UU
    max_delta_z_uu: float = DEFAULT_MAX_DELTA_Z_UU
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    settle_sample_interval_s: float = DEFAULT_SETTLE_SAMPLE_INTERVAL_S
    stretcher_settle_timeout_s: float = DEFAULT_STRETCHER_SETTLE_TIMEOUT_S
    agent_settle_timeout_s: float = DEFAULT_AGENT_SETTLE_TIMEOUT_S
    stable_window_samples: int = DEFAULT_STABLE_WINDOW_SAMPLES
    position_epsilon_uu: float = DEFAULT_POSITION_EPSILON_UU
    rotation_epsilon_deg: float = DEFAULT_ROTATION_EPSILON_DEG
    max_agent_xy_error_uu: float = DEFAULT_MAX_AGENT_XY_ERROR_UU
    max_agent_yaw_error_deg: float = DEFAULT_MAX_AGENT_YAW_ERROR_DEG


@dataclass(frozen=True)
class CaptureInputs:
    """Collect one validated batch configuration."""

    selections: tuple[TaskSelection, ...]
    topomap_dir: Path
    render: RenderSettings
    policy: CapturePolicy
    resume: bool
    overwrite: bool


@dataclass(frozen=True)
class CaptureGeometry:
    """Describe the requested horizontal capture position and yaw."""

    capture_x: float
    capture_y: float
    yaw_deg: float
    source_injured_xy: tuple[float, float]
    source_stretcher_xy: tuple[float, float]
    distance_uu: float


@dataclass(frozen=True)
class StabilityResult:
    """Report whether the latest pose window is stable."""

    stable: bool
    tail_count: int
    max_position_span_uu: float
    max_rotation_span_deg: float


@dataclass(frozen=True)
class PoseValidationResult:
    """Report final horizontal-position and yaw request errors."""

    valid: bool
    xy_valid: bool
    yaw_valid: bool
    xy_error_uu: float
    yaw_error_deg: float


def _finite_positive(value: float, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ProbeError(f"{field_name} must be a positive finite number")
    return parsed


def _capture_distances(values: Iterable[float]) -> tuple[float, ...]:
    distances = []
    for raw_value in values:
        distance = _finite_positive(raw_value, "capture distance")
        if not distance.is_integer():
            raise ProbeError("capture distances must be whole UU values")
        distances.append(distance)
    if not distances:
        raise ProbeError("at least one capture distance is required")
    if len(set(distances)) != len(distances):
        raise ProbeError("capture distances must be unique")
    return tuple(distances)


def validate_capture_policy(
    *,
    capture_distances_uu: Iterable[float],
    initial_height_offset_uu: float,
    height_retry_step_uu: float,
    min_delta_z_uu: float,
    max_delta_z_uu: float,
    max_attempts: int,
    settle_sample_interval_s: float,
    stretcher_settle_timeout_s: float,
    agent_settle_timeout_s: float,
    stable_window_samples: int,
    position_epsilon_uu: float,
    rotation_epsilon_deg: float,
    max_agent_xy_error_uu: float,
    max_agent_yaw_error_deg: float,
) -> CapturePolicy:
    """Validate all user-facing capture policy values."""

    distances = _capture_distances(capture_distances_uu)
    height_offset = _finite_positive(
        initial_height_offset_uu,
        "initial height offset",
    )
    retry_step = _finite_positive(height_retry_step_uu, "height retry step")
    min_delta = float(min_delta_z_uu)
    max_delta = float(max_delta_z_uu)
    if not all(math.isfinite(value) for value in (min_delta, max_delta)):
        raise ProbeError("height delta bounds must be finite")
    if min_delta <= 0 or min_delta >= max_delta:
        raise ProbeError("height delta bounds must satisfy 0 < MIN < MAX")
    if max_attempts not in (1, 2):
        raise ProbeError("max attempts must be 1 or 2")

    sample_interval = _finite_positive(
        settle_sample_interval_s,
        "settle sample interval",
    )
    stretcher_timeout = _finite_positive(
        stretcher_settle_timeout_s,
        "stretcher settle timeout",
    )
    agent_timeout = _finite_positive(
        agent_settle_timeout_s,
        "agent settle timeout",
    )
    if stable_window_samples <= 0:
        raise ProbeError("stable window samples must be positive")
    minimum_timeout = sample_interval * (stable_window_samples - 1)
    if stretcher_timeout < minimum_timeout or agent_timeout < minimum_timeout:
        raise ProbeError(
            "settle timeouts must allow the requested stable sample window"
        )
    position_epsilon = _finite_positive(
        position_epsilon_uu,
        "position epsilon",
    )
    rotation_epsilon = _finite_positive(
        rotation_epsilon_deg,
        "rotation epsilon",
    )
    max_xy_error = _finite_positive(
        max_agent_xy_error_uu,
        "maximum agent xy error",
    )
    max_yaw_error = _finite_positive(
        max_agent_yaw_error_deg,
        "maximum agent yaw error",
    )
    return CapturePolicy(
        capture_distances_uu=distances,
        initial_height_offset_uu=height_offset,
        height_retry_step_uu=retry_step,
        min_delta_z_uu=min_delta,
        max_delta_z_uu=max_delta,
        max_attempts=max_attempts,
        settle_sample_interval_s=sample_interval,
        stretcher_settle_timeout_s=stretcher_timeout,
        agent_settle_timeout_s=agent_timeout,
        stable_window_samples=stable_window_samples,
        position_epsilon_uu=position_epsilon,
        rotation_epsilon_deg=rotation_epsilon,
        max_agent_xy_error_uu=max_xy_error,
        max_agent_yaw_error_deg=max_yaw_error,
    )


def resolve_task_selections(
    levels: Iterable[int],
    point_ids: Iterable[int] | None,
) -> tuple[TaskSelection, ...]:
    """Resolve ordered levels and optional zero-based point IDs."""

    ordered_levels = tuple(dict.fromkeys(int(level) for level in levels))
    if not ordered_levels:
        raise ProbeError("at least one level is required")
    invalid_levels = [level for level in ordered_levels if level not in TIME_LIMITS]
    if invalid_levels:
        raise ProbeError(
            f"levels must be selected from {sorted(TIME_LIMITS)}: {invalid_levels}"
        )

    selected_ids = None
    if point_ids is not None:
        selected_ids = tuple(dict.fromkeys(int(point_id) for point_id in point_ids))
        if not selected_ids:
            raise ProbeError("point IDs cannot be empty when provided")
        if any(point_id < 0 for point_id in selected_ids):
            raise ProbeError("point IDs must be non-negative")

    selections = []
    for level in ordered_levels:
        point_count = len(load_level_points(level))
        ids_for_level = selected_ids or tuple(range(point_count))
        invalid_ids = [
            point_id for point_id in ids_for_level if point_id >= point_count
        ]
        if invalid_ids:
            raise ProbeError(
                f"point IDs out of range for level {level} (0-{point_count - 1}): "
                f"{invalid_ids}"
            )
        selections.extend(
            load_task_selection(level, point_id) for point_id in ids_for_level
        )
    return tuple(selections)


def build_capture_inputs(
    *,
    levels: Iterable[int],
    point_ids: Iterable[int] | None,
    topomap_dir: Path,
    resolution: Sequence[int] = DEFAULT_RESOLUTION,
    offscreen: bool = True,
    render_quality: int = 2,
    policy: CapturePolicy,
    resume: bool = True,
    overwrite: bool = False,
) -> CaptureInputs:
    """Build one validated batch input object."""

    if render_quality < 0:
        raise ProbeError("render quality must be non-negative")
    output_root = topomap_dir.expanduser().resolve()
    return CaptureInputs(
        selections=resolve_task_selections(levels, point_ids),
        topomap_dir=output_root,
        render=RenderSettings(
            resolution=validate_resolution(resolution),
            offscreen=bool(offscreen),
            quality=int(render_quality),
        ),
        policy=policy,
        resume=bool(resume),
        overwrite=bool(overwrite),
    )


def calculate_capture_geometry(
    injured_pose: Sequence[float],
    stretcher_pose: Sequence[float],
    distance_uu: float,
) -> CaptureGeometry:
    """Place the camera distance from the stretcher toward the injured actor."""

    if len(injured_pose) < 2 or len(stretcher_pose) < 2:
        raise ProbeError("injured and stretcher poses must contain x and y")
    injured_x, injured_y = float(injured_pose[0]), float(injured_pose[1])
    stretcher_x, stretcher_y = float(stretcher_pose[0]), float(stretcher_pose[1])
    delta_x = injured_x - stretcher_x
    delta_y = injured_y - stretcher_y
    length = math.hypot(delta_x, delta_y)
    if not math.isfinite(length) or length <= 1e-6:
        raise ProbeError("injured and stretcher horizontal positions are degenerate")
    distance = _finite_positive(distance_uu, "capture distance")
    capture_x = stretcher_x + distance * delta_x / length
    capture_y = stretcher_y + distance * delta_y / length
    yaw_deg = math.degrees(math.atan2(stretcher_y - capture_y, stretcher_x - capture_x))
    return CaptureGeometry(
        capture_x=capture_x,
        capture_y=capture_y,
        yaw_deg=yaw_deg,
        source_injured_xy=(injured_x, injured_y),
        source_stretcher_xy=(stretcher_x, stretcher_y),
        distance_uu=distance,
    )


def _rotation_span(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    reference = float(values[0])
    normalized = [
        (float(value) - reference + 180.0) % 360.0 - 180.0 for value in values
    ]
    return max(normalized) - min(normalized)


def evaluate_pose_stability(
    poses: Sequence[Sequence[float]],
    *,
    tail_samples: int,
    position_epsilon_uu: float,
    rotation_epsilon_deg: float,
) -> StabilityResult:
    """Evaluate position and wrapped rotation spans over the pose tail."""

    if tail_samples <= 0 or len(poses) < tail_samples:
        return StabilityResult(False, tail_samples, math.inf, math.inf)
    tail = poses[-tail_samples:]
    if any(len(pose) < 6 for pose in tail):
        return StabilityResult(False, tail_samples, math.inf, math.inf)
    position_spans = [
        max(float(pose[index]) for pose in tail)
        - min(float(pose[index]) for pose in tail)
        for index in range(3)
    ]
    rotation_spans = [
        _rotation_span([float(pose[index]) for pose in tail]) for index in range(3, 6)
    ]
    max_position_span = max(position_spans)
    max_rotation_span = max(rotation_spans)
    return StabilityResult(
        stable=(
            max_position_span <= position_epsilon_uu
            and max_rotation_span <= rotation_epsilon_deg
        ),
        tail_count=tail_samples,
        max_position_span_uu=max_position_span,
        max_rotation_span_deg=max_rotation_span,
    )


def wrapped_angle_error_deg(actual_deg: float, requested_deg: float) -> float:
    """Return the absolute shortest signed-angle difference in degrees."""

    difference = (float(actual_deg) - float(requested_deg) + 180.0) % 360.0 - 180.0
    return abs(difference)


def validate_agent_pose(
    requested_pose: Sequence[float],
    actual_pose: Sequence[float],
    policy: CapturePolicy,
) -> PoseValidationResult:
    """Validate actual actor x/y and yaw against one teleport request."""

    if len(requested_pose) < 5 or len(actual_pose) < 5:
        raise ProbeError("requested and actual agent poses must contain x, y, and yaw")
    xy_error = math.hypot(
        float(actual_pose[0]) - float(requested_pose[0]),
        float(actual_pose[1]) - float(requested_pose[1]),
    )
    yaw_error = wrapped_angle_error_deg(actual_pose[4], requested_pose[4])
    xy_valid = math.isfinite(xy_error) and xy_error <= policy.max_agent_xy_error_uu
    yaw_valid = math.isfinite(yaw_error) and yaw_error <= policy.max_agent_yaw_error_deg
    return PoseValidationResult(
        valid=xy_valid and yaw_valid,
        xy_valid=xy_valid,
        yaw_valid=yaw_valid,
        xy_error_uu=xy_error,
        yaw_error_deg=yaw_error,
    )


def classify_height_delta(delta_z_uu: float, policy: CapturePolicy) -> str:
    """Classify one stable agent-to-stretcher actor z difference."""

    delta = float(delta_z_uu)
    if not math.isfinite(delta):
        return "invalid"
    if delta < policy.min_delta_z_uu:
        return "too_low"
    if delta > policy.max_delta_z_uu:
        return "too_high"
    return "valid"


def retry_height_offset(
    current_height_offset_uu: float,
    classification: str,
    policy: CapturePolicy,
) -> float:
    """Apply the one-step retry rule for a stable but invalid landing."""

    if classification == "too_low":
        return current_height_offset_uu + policy.height_retry_step_uu
    if classification == "too_high":
        adjusted = current_height_offset_uu - policy.height_retry_step_uu
        if adjusted <= 0:
            raise ProbeError("height retry would produce a non-positive offset")
        return adjusted
    raise ProbeError(f"height classification is not retryable: {classification}")


def policy_as_dict(policy: CapturePolicy) -> dict[str, Any]:
    """Return a JSON-serializable policy record."""

    return {
        "capture_distances_uu": list(policy.capture_distances_uu),
        "initial_height_offset_uu": policy.initial_height_offset_uu,
        "height_retry_step_uu": policy.height_retry_step_uu,
        "valid_delta_z_uu": [policy.min_delta_z_uu, policy.max_delta_z_uu],
        "max_attempts": policy.max_attempts,
        "settle_sample_interval_s": policy.settle_sample_interval_s,
        "stretcher_settle_timeout_s": policy.stretcher_settle_timeout_s,
        "agent_settle_timeout_s": policy.agent_settle_timeout_s,
        "stable_window_samples": policy.stable_window_samples,
        "position_epsilon_uu": policy.position_epsilon_uu,
        "rotation_epsilon_deg": policy.rotation_epsilon_deg,
        "max_agent_xy_error_uu": policy.max_agent_xy_error_uu,
        "max_agent_yaw_error_deg": policy.max_agent_yaw_error_deg,
    }
