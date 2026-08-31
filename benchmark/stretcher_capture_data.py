"""Validated inputs and geometry for stretcher semantic-goal capture."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmark.teleport_probe_data import (
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_DELAYS,
    TIME_LIMITS,
    ProbeError,
    RenderSettings,
    TaskSelection,
    load_level_points,
    load_task_selection,
    validate_resolution,
    validate_sample_delays,
)

DEFAULT_CAPTURE_DISTANCE_UU = 200.0
DEFAULT_HEIGHT_OFFSET_UU = 160.0
DEFAULT_HEIGHT_RETRY_STEP_UU = 30.0
DEFAULT_MIN_DELTA_Z_UU = 70.0
DEFAULT_MAX_DELTA_Z_UU = 140.0
DEFAULT_MAX_ATTEMPTS = 2
DEFAULT_STABLE_TAIL_SAMPLES = 3
DEFAULT_POSITION_EPSILON_UU = 1.0
DEFAULT_ROTATION_EPSILON_DEG = 1.0


@dataclass(frozen=True)
class CapturePolicy:
    """Store geometry, height, retry, and stability policy values."""

    capture_distance_uu: float = DEFAULT_CAPTURE_DISTANCE_UU
    initial_height_offset_uu: float = DEFAULT_HEIGHT_OFFSET_UU
    height_retry_step_uu: float = DEFAULT_HEIGHT_RETRY_STEP_UU
    min_delta_z_uu: float = DEFAULT_MIN_DELTA_Z_UU
    max_delta_z_uu: float = DEFAULT_MAX_DELTA_Z_UU
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    sample_delays: tuple[float, ...] = DEFAULT_SAMPLE_DELAYS
    stable_tail_samples: int = DEFAULT_STABLE_TAIL_SAMPLES
    position_epsilon_uu: float = DEFAULT_POSITION_EPSILON_UU
    rotation_epsilon_deg: float = DEFAULT_ROTATION_EPSILON_DEG


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
    """Report whether the tail of a pose sequence is stable."""

    stable: bool
    tail_count: int
    max_position_span_uu: float
    max_rotation_span_deg: float


def _finite_positive(value: float, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ProbeError(f"{field_name} must be a positive finite number")
    return parsed


def validate_capture_policy(
    *,
    capture_distance_uu: float,
    initial_height_offset_uu: float,
    height_retry_step_uu: float,
    min_delta_z_uu: float,
    max_delta_z_uu: float,
    max_attempts: int,
    sample_delays: Iterable[float],
    stable_tail_samples: int,
    position_epsilon_uu: float,
    rotation_epsilon_deg: float,
) -> CapturePolicy:
    """Validate all user-facing capture policy values."""

    distance = _finite_positive(capture_distance_uu, "capture distance")
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

    delays = validate_sample_delays(sample_delays)
    if stable_tail_samples <= 0 or stable_tail_samples > len(delays):
        raise ProbeError(
            "stable tail samples must be between 1 and the number of sample delays"
        )
    position_epsilon = _finite_positive(
        position_epsilon_uu,
        "position epsilon",
    )
    rotation_epsilon = _finite_positive(
        rotation_epsilon_deg,
        "rotation epsilon",
    )
    return CapturePolicy(
        capture_distance_uu=distance,
        initial_height_offset_uu=height_offset,
        height_retry_step_uu=retry_step,
        min_delta_z_uu=min_delta,
        max_delta_z_uu=max_delta,
        max_attempts=max_attempts,
        sample_delays=delays,
        stable_tail_samples=stable_tail_samples,
        position_epsilon_uu=position_epsilon,
        rotation_epsilon_deg=rotation_epsilon,
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
        "capture_distance_uu": policy.capture_distance_uu,
        "initial_height_offset_uu": policy.initial_height_offset_uu,
        "height_retry_step_uu": policy.height_retry_step_uu,
        "valid_delta_z_uu": [policy.min_delta_z_uu, policy.max_delta_z_uu],
        "max_attempts": policy.max_attempts,
        "sample_delays_s": list(policy.sample_delays),
        "stable_tail_samples": policy.stable_tail_samples,
        "position_epsilon_uu": policy.position_epsilon_uu,
        "rotation_epsilon_deg": policy.rotation_epsilon_deg,
    }
