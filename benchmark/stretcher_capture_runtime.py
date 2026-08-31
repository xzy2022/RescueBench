"""Runtime state machine for deterministic stretcher goal-image capture."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

from benchmark.stretcher_capture_data import (
    CaptureGeometry,
    CapturePolicy,
    PoseValidationResult,
    StabilityResult,
    calculate_capture_geometry,
    classify_height_delta,
    evaluate_pose_stability,
    retry_height_offset,
    validate_agent_pose,
)
from benchmark.teleport_probe_artifacts import utc_now
from benchmark.teleport_probe_capture import (
    RuntimeHandles,
    RuntimeSnapshot,
    read_runtime_snapshot,
)
from benchmark.teleport_probe_data import ProbeError, TaskSelection

SampleSink = Callable[[dict], None]


@dataclass(frozen=True)
class SequenceResult:
    """Hold rolling-settle rows and the final image-bearing snapshot."""

    rows: tuple[dict[str, Any], ...]
    final_snapshot: RuntimeSnapshot
    stability: StabilityResult
    stop_reason: str
    elapsed_s: float


@dataclass(frozen=True)
class PointPreparationOutcome:
    """Describe the shared stretcher-settle result for one task point."""

    status: str
    reason: str
    baseline: SequenceResult


@dataclass(frozen=True)
class DistanceCaptureOutcome:
    """Describe one distance-qualified candidate view outcome."""

    status: str
    reason: str
    geometry: CaptureGeometry | None
    attempts: tuple[dict[str, Any], ...]
    final_snapshot: RuntimeSnapshot
    requested_agent_pose: tuple[float, ...] | None
    delta_z_uu: float | None
    baseline_stability: StabilityResult
    pose_validation: PoseValidationResult | None


def _stability_dict(result: StabilityResult) -> dict[str, Any]:
    return {
        "stable": result.stable,
        "window_count": result.tail_count,
        "max_position_span_uu": (
            result.max_position_span_uu
            if math.isfinite(result.max_position_span_uu)
            else None
        ),
        "max_rotation_span_deg": (
            result.max_rotation_span_deg
            if math.isfinite(result.max_rotation_span_deg)
            else None
        ),
    }


def _pose_validation_dict(result: PoseValidationResult) -> dict[str, Any]:
    return {
        "valid": result.valid,
        "xy_valid": result.xy_valid,
        "yaw_valid": result.yaw_valid,
        "actual_xy_error_uu": result.xy_error_uu,
        "actual_yaw_error_deg": result.yaw_error_deg,
    }


def _sample_row(
    *,
    snapshot: RuntimeSnapshot,
    selection: TaskSelection,
    policy: CapturePolicy,
    event: str,
    pose_id: str,
    attempt_index: int | None,
    requested_pose: tuple[float, ...] | None,
    capture_distance_uu: float | None,
    sample_index: int,
    elapsed_s: float,
    timeout_s: float,
    stability: StabilityResult,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "recorded_at_utc": utc_now(),
        "event": event,
        "coordinate_frame": "unreal_world",
        "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
        "rotation_unit": "degree",
        "level": selection.level,
        "point_id": selection.point_id,
        "env_id": selection.task_context["env_id"],
        "capture_distance_uu": capture_distance_uu,
        "pose_id": pose_id,
        "attempt_index": attempt_index,
        "sample_index": sample_index,
        "sample_interval_s": policy.settle_sample_interval_s,
        "sample_elapsed_s": elapsed_s,
        "settle_timeout_s": timeout_s,
        "current_window_stable": stability.stable,
        "current_stability": _stability_dict(stability),
        "stop_reason": stop_reason,
        "requested_actor_pose": (
            list(requested_pose) if requested_pose is not None else None
        ),
        "actual_poses": {
            **snapshot.actual,
            "robot_camera": snapshot.camera_pose,
        },
        "agent_minus_stretcher_z_uu": (
            float(snapshot.actual["agent_actor"][2])
            - float(snapshot.actual["stretcher"][2])
        ),
        "frame": {
            "shape": [int(value) for value in snapshot.image.shape],
            "dtype": str(snapshot.image.dtype),
            "saved": False,
        },
    }


def wait_until_stable(
    *,
    runtime: RuntimeHandles,
    selection: TaskSelection,
    policy: CapturePolicy,
    timeout_s: float,
    event: str,
    pose_id: str,
    tracked_label: str,
    attempt_index: int | None,
    requested_pose: tuple[float, ...] | None,
    capture_distance_uu: float | None,
    sample_sink: SampleSink,
) -> SequenceResult:
    """Poll until the latest pose window is stable or the deadline is reached."""

    rows = []
    poses = []
    start = time.monotonic()
    sample_index = 0
    while True:
        scheduled_elapsed = min(
            sample_index * policy.settle_sample_interval_s,
            timeout_s,
        )
        remaining = scheduled_elapsed - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
        snapshot = read_runtime_snapshot(runtime)
        elapsed = time.monotonic() - start
        poses.append(snapshot.actual[tracked_label])
        stability = evaluate_pose_stability(
            poses,
            tail_samples=policy.stable_window_samples,
            position_epsilon_uu=policy.position_epsilon_uu,
            rotation_epsilon_deg=policy.rotation_epsilon_deg,
        )
        stop_reason = None
        if stability.stable:
            stop_reason = "stable_window"
        elif elapsed >= timeout_s:
            stop_reason = "timeout"
        row = _sample_row(
            snapshot=snapshot,
            selection=selection,
            policy=policy,
            event=event,
            pose_id=pose_id,
            attempt_index=attempt_index,
            requested_pose=requested_pose,
            capture_distance_uu=capture_distance_uu,
            sample_index=sample_index,
            elapsed_s=elapsed,
            timeout_s=timeout_s,
            stability=stability,
            stop_reason=stop_reason,
        )
        sample_sink(row)
        rows.append(row)
        if stop_reason is not None:
            return SequenceResult(
                rows=tuple(rows),
                final_snapshot=snapshot,
                stability=stability,
                stop_reason=stop_reason,
                elapsed_s=elapsed,
            )
        sample_index += 1


def prepare_capture_point(
    *,
    runtime: RuntimeHandles,
    selection: TaskSelection,
    policy: CapturePolicy,
    sample_sink: SampleSink,
) -> PointPreparationOutcome:
    """Wait once for the point's stretcher before any candidate teleports."""

    baseline = wait_until_stable(
        runtime=runtime,
        selection=selection,
        policy=policy,
        timeout_s=policy.stretcher_settle_timeout_s,
        event="stretcher_settle_sample",
        pose_id="baseline",
        tracked_label="stretcher",
        attempt_index=None,
        requested_pose=None,
        capture_distance_uu=None,
        sample_sink=sample_sink,
    )
    if not baseline.stability.stable:
        return PointPreparationOutcome(
            status="skipped",
            reason="stretcher_settle_timeout",
            baseline=baseline,
        )
    return PointPreparationOutcome(
        status="ready",
        reason="stretcher_stable",
        baseline=baseline,
    )


def _attempt_summary(
    *,
    attempt_index: int,
    height_offset_uu: float,
    requested_pose: tuple[float, ...],
    sequence: SequenceResult,
    classification: str,
    delta_z_uu: float,
    pose_validation: PoseValidationResult,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "height_offset_uu": height_offset_uu,
        "requested_agent_pose": list(requested_pose),
        "settling": {
            "stop_reason": sequence.stop_reason,
            "elapsed_s": sequence.elapsed_s,
            "sample_count": len(sequence.rows),
            "stability": _stability_dict(sequence.stability),
        },
        "height_classification": classification,
        "agent_minus_stretcher_z_uu": delta_z_uu,
        "pose_validation": _pose_validation_dict(pose_validation),
        "actual_agent_pose": sequence.final_snapshot.actual["agent_actor"],
        "actual_stretcher_pose": sequence.final_snapshot.actual["stretcher"],
    }


def _outcome(
    *,
    status: str,
    reason: str,
    geometry: CaptureGeometry | None,
    attempts: list[dict[str, Any]],
    sequence: SequenceResult,
    baseline: SequenceResult,
    requested_pose: tuple[float, ...] | None,
    delta_z_uu: float | None,
    pose_validation: PoseValidationResult | None,
) -> DistanceCaptureOutcome:
    return DistanceCaptureOutcome(
        status=status,
        reason=reason,
        geometry=geometry,
        attempts=tuple(attempts),
        final_snapshot=sequence.final_snapshot,
        requested_agent_pose=requested_pose,
        delta_z_uu=delta_z_uu,
        baseline_stability=baseline.stability,
        pose_validation=pose_validation,
    )


def capture_distance_candidate(
    *,
    runtime: RuntimeHandles,
    selection: TaskSelection,
    policy: CapturePolicy,
    baseline: SequenceResult,
    capture_distance_uu: float,
    sample_sink: SampleSink,
) -> DistanceCaptureOutcome:
    """Capture one distance candidate from the point's shared stable baseline."""

    try:
        geometry = calculate_capture_geometry(
            baseline.final_snapshot.actual["injured"],
            baseline.final_snapshot.actual["stretcher"],
            capture_distance_uu,
        )
    except ProbeError:
        return _outcome(
            status="skipped",
            reason="degenerate_capture_direction",
            geometry=None,
            attempts=[],
            sequence=baseline,
            baseline=baseline,
            requested_pose=None,
            delta_z_uu=None,
            pose_validation=None,
        )

    height_offset = policy.initial_height_offset_uu
    attempts = []
    last_sequence = baseline
    last_requested_pose = None
    last_delta_z = None
    last_pose_validation = None

    for attempt_index in range(1, policy.max_attempts + 1):
        live_stretcher_z = float(last_sequence.final_snapshot.actual["stretcher"][2])
        requested_pose = (
            geometry.capture_x,
            geometry.capture_y,
            live_stretcher_z + height_offset,
            0.0,
            geometry.yaw_deg,
            0.0,
        )
        last_requested_pose = requested_pose
        runtime.env_unwrapped.unrealcv.set_obj_rotation(
            runtime.agent_name,
            list(requested_pose[3:6]),
        )
        runtime.env_unwrapped.unrealcv.set_obj_location(
            runtime.agent_name,
            list(requested_pose[:3]),
        )
        sequence = wait_until_stable(
            runtime=runtime,
            selection=selection,
            policy=policy,
            timeout_s=policy.agent_settle_timeout_s,
            event="agent_settle_sample",
            pose_id=f"{int(capture_distance_uu)}UU-attempt-{attempt_index}",
            tracked_label="agent_actor",
            attempt_index=attempt_index,
            requested_pose=requested_pose,
            capture_distance_uu=capture_distance_uu,
            sample_sink=sample_sink,
        )
        last_sequence = sequence
        actual_agent_pose = sequence.final_snapshot.actual["agent_actor"]
        delta_z = float(actual_agent_pose[2]) - float(
            sequence.final_snapshot.actual["stretcher"][2]
        )
        last_delta_z = delta_z
        pose_validation = validate_agent_pose(
            requested_pose,
            actual_agent_pose,
            policy,
        )
        last_pose_validation = pose_validation
        classification = (
            classify_height_delta(delta_z, policy)
            if sequence.stability.stable
            else "not_stable"
        )
        attempts.append(
            _attempt_summary(
                attempt_index=attempt_index,
                height_offset_uu=height_offset,
                requested_pose=requested_pose,
                sequence=sequence,
                classification=classification,
                delta_z_uu=delta_z,
                pose_validation=pose_validation,
            )
        )
        if not sequence.stability.stable:
            return _outcome(
                status="skipped",
                reason="agent_settle_timeout",
                geometry=geometry,
                attempts=attempts,
                sequence=sequence,
                baseline=baseline,
                requested_pose=requested_pose,
                delta_z_uu=delta_z,
                pose_validation=pose_validation,
            )
        if classification == "valid":
            if not pose_validation.xy_valid:
                reason = "agent_xy_error"
            elif not pose_validation.yaw_valid:
                reason = "agent_yaw_error"
            else:
                reason = "pose_and_height_valid"
            return _outcome(
                status="captured" if pose_validation.valid else "skipped",
                reason=reason,
                geometry=geometry,
                attempts=attempts,
                sequence=sequence,
                baseline=baseline,
                requested_pose=requested_pose,
                delta_z_uu=delta_z,
                pose_validation=pose_validation,
            )
        if attempt_index < policy.max_attempts:
            height_offset = retry_height_offset(
                height_offset,
                classification,
                policy,
            )

    return _outcome(
        status="skipped",
        reason="height_invalid_after_retry",
        geometry=geometry,
        attempts=attempts,
        sequence=last_sequence,
        baseline=baseline,
        requested_pose=last_requested_pose,
        delta_z_uu=last_delta_z,
        pose_validation=last_pose_validation,
    )


def preparation_as_dict(outcome: PointPreparationOutcome) -> dict[str, Any]:
    """Serialize one point's shared stretcher-settle outcome."""

    snapshot = outcome.baseline.final_snapshot
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "baseline_settling": {
            "stop_reason": outcome.baseline.stop_reason,
            "elapsed_s": outcome.baseline.elapsed_s,
            "sample_count": len(outcome.baseline.rows),
            "stability": _stability_dict(outcome.baseline.stability),
        },
        "actual_poses": {
            **snapshot.actual,
            "robot_camera": snapshot.camera_pose,
        },
    }


def outcome_as_dict(outcome: DistanceCaptureOutcome) -> dict[str, Any]:
    """Serialize a distance outcome without embedding its image array."""

    geometry = None
    if outcome.geometry is not None:
        geometry = {
            "capture_xy": [
                outcome.geometry.capture_x,
                outcome.geometry.capture_y,
            ],
            "requested_yaw_deg": outcome.geometry.yaw_deg,
            "source_injured_xy": list(outcome.geometry.source_injured_xy),
            "source_stretcher_xy": list(outcome.geometry.source_stretcher_xy),
            "distance_uu": outcome.geometry.distance_uu,
        }
    pose_validation = (
        _pose_validation_dict(outcome.pose_validation)
        if outcome.pose_validation is not None
        else None
    )
    return {
        "status": outcome.status,
        "reason": outcome.reason,
        "capture_geometry": geometry,
        "attempts": list(outcome.attempts),
        "requested_agent_pose": (
            list(outcome.requested_agent_pose)
            if outcome.requested_agent_pose is not None
            else None
        ),
        "agent_minus_stretcher_z_uu": outcome.delta_z_uu,
        "pose_validation": pose_validation,
        "baseline_stability": _stability_dict(outcome.baseline_stability),
        "actual_poses": {
            **outcome.final_snapshot.actual,
            "robot_camera": outcome.final_snapshot.camera_pose,
        },
    }
