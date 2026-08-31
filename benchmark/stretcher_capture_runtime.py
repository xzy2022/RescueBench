"""Runtime state machine for deterministic stretcher goal-image capture."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from benchmark.stretcher_capture_data import (
    CaptureGeometry,
    CapturePolicy,
    StabilityResult,
    calculate_capture_geometry,
    classify_height_delta,
    evaluate_pose_stability,
    retry_height_offset,
)
from benchmark.teleport_probe_artifacts import utc_now
from benchmark.teleport_probe_capture import (
    RuntimeHandles,
    RuntimeSnapshot,
    read_runtime_snapshot,
)
from benchmark.teleport_probe_data import ProbeError, TaskSelection

SampleSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SequenceResult:
    """Hold JSON sample rows and the final image-bearing snapshot."""

    rows: tuple[dict[str, Any], ...]
    final_snapshot: RuntimeSnapshot
    stability: StabilityResult


@dataclass(frozen=True)
class PointCaptureOutcome:
    """Describe one point's terminal runtime outcome."""

    status: str
    reason: str
    geometry: CaptureGeometry | None
    attempts: tuple[dict[str, Any], ...]
    final_snapshot: RuntimeSnapshot
    requested_agent_pose: tuple[float, ...] | None
    delta_z_uu: float | None
    baseline_stability: StabilityResult


def _stability_dict(result: StabilityResult) -> dict[str, Any]:
    return {
        "stable": result.stable,
        "tail_count": result.tail_count,
        "max_position_span_uu": result.max_position_span_uu,
        "max_rotation_span_deg": result.max_rotation_span_deg,
    }


def _sample_sequence(
    *,
    runtime: RuntimeHandles,
    selection: TaskSelection,
    policy: CapturePolicy,
    event: str,
    pose_id: str,
    tracked_label: str,
    attempt_index: int | None,
    requested_pose: tuple[float, ...] | None,
    sample_sink: SampleSink,
) -> SequenceResult:
    rows = []
    final_snapshot = None
    start = time.monotonic()
    for sample_index, delay in enumerate(policy.sample_delays):
        remaining = delay - (time.monotonic() - start)
        if remaining > 0:
            time.sleep(remaining)
        snapshot = read_runtime_snapshot(runtime)
        final_snapshot = snapshot
        elapsed = time.monotonic() - start
        row = {
            "schema_version": 1,
            "recorded_at_utc": utc_now(),
            "event": event,
            "coordinate_frame": "unreal_world",
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "rotation_unit": "degree",
            "level": selection.level,
            "point_id": selection.point_id,
            "env_id": selection.task_context["env_id"],
            "pose_id": pose_id,
            "attempt_index": attempt_index,
            "sample_index": sample_index,
            "sample_delay_s": delay,
            "sample_elapsed_s": elapsed,
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
        sample_sink(row)
        rows.append(row)
    if final_snapshot is None:
        raise RuntimeError("sample sequence produced no snapshot")
    poses = [row["actual_poses"][tracked_label] for row in rows]
    stability = evaluate_pose_stability(
        poses,
        tail_samples=policy.stable_tail_samples,
        position_epsilon_uu=policy.position_epsilon_uu,
        rotation_epsilon_deg=policy.rotation_epsilon_deg,
    )
    return SequenceResult(tuple(rows), final_snapshot, stability)


def _attempt_summary(
    *,
    attempt_index: int,
    height_offset_uu: float,
    requested_pose: tuple[float, ...],
    sequence: SequenceResult,
    classification: str,
    delta_z_uu: float,
) -> dict[str, Any]:
    return {
        "attempt_index": attempt_index,
        "height_offset_uu": height_offset_uu,
        "requested_agent_pose": list(requested_pose),
        "stability": _stability_dict(sequence.stability),
        "height_classification": classification,
        "agent_minus_stretcher_z_uu": delta_z_uu,
        "actual_agent_pose": sequence.final_snapshot.actual["agent_actor"],
        "actual_stretcher_pose": sequence.final_snapshot.actual["stretcher"],
    }


def capture_point(
    *,
    runtime: RuntimeHandles,
    selection: TaskSelection,
    policy: CapturePolicy,
    sample_sink: SampleSink,
) -> PointCaptureOutcome:
    """Capture one task point using one initial drop and at most one retry."""

    baseline = _sample_sequence(
        runtime=runtime,
        selection=selection,
        policy=policy,
        event="stretcher_settle_sample",
        pose_id="baseline",
        tracked_label="stretcher",
        attempt_index=None,
        requested_pose=None,
        sample_sink=sample_sink,
    )
    if not baseline.stability.stable:
        return PointCaptureOutcome(
            status="skipped",
            reason="stretcher_not_stable",
            geometry=None,
            attempts=(),
            final_snapshot=baseline.final_snapshot,
            requested_agent_pose=None,
            delta_z_uu=None,
            baseline_stability=baseline.stability,
        )

    try:
        geometry = calculate_capture_geometry(
            baseline.final_snapshot.actual["injured"],
            baseline.final_snapshot.actual["stretcher"],
            policy.capture_distance_uu,
        )
    except ProbeError:
        return PointCaptureOutcome(
            status="skipped",
            reason="degenerate_capture_direction",
            geometry=None,
            attempts=(),
            final_snapshot=baseline.final_snapshot,
            requested_agent_pose=None,
            delta_z_uu=None,
            baseline_stability=baseline.stability,
        )
    height_offset = policy.initial_height_offset_uu
    attempts = []
    last_sequence = baseline
    last_requested_pose = None
    last_delta_z = None

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
        sequence = _sample_sequence(
            runtime=runtime,
            selection=selection,
            policy=policy,
            event="agent_settle_sample",
            pose_id=f"attempt-{attempt_index}",
            tracked_label="agent_actor",
            attempt_index=attempt_index,
            requested_pose=requested_pose,
            sample_sink=sample_sink,
        )
        last_sequence = sequence
        delta_z = float(sequence.final_snapshot.actual["agent_actor"][2]) - float(
            sequence.final_snapshot.actual["stretcher"][2]
        )
        last_delta_z = delta_z
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
            )
        )
        if not sequence.stability.stable:
            return PointCaptureOutcome(
                status="skipped",
                reason="agent_not_stable",
                geometry=geometry,
                attempts=tuple(attempts),
                final_snapshot=sequence.final_snapshot,
                requested_agent_pose=requested_pose,
                delta_z_uu=delta_z,
                baseline_stability=baseline.stability,
            )
        if classification == "valid":
            return PointCaptureOutcome(
                status="captured",
                reason="height_valid",
                geometry=geometry,
                attempts=tuple(attempts),
                final_snapshot=sequence.final_snapshot,
                requested_agent_pose=requested_pose,
                delta_z_uu=delta_z,
                baseline_stability=baseline.stability,
            )
        if attempt_index < policy.max_attempts:
            height_offset = retry_height_offset(
                height_offset,
                classification,
                policy,
            )

    return PointCaptureOutcome(
        status="skipped",
        reason="height_invalid_after_retry",
        geometry=geometry,
        attempts=tuple(attempts),
        final_snapshot=last_sequence.final_snapshot,
        requested_agent_pose=last_requested_pose,
        delta_z_uu=last_delta_z,
        baseline_stability=baseline.stability,
    )


def outcome_as_dict(outcome: PointCaptureOutcome) -> dict[str, Any]:
    """Serialize a point outcome without embedding its image array."""

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
        "baseline_stability": _stability_dict(outcome.baseline_stability),
        "actual_poses": {
            **outcome.final_snapshot.actual,
            "robot_camera": outcome.final_snapshot.camera_pose,
        },
    }
