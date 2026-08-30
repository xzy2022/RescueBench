"""Runtime state capture for the RescueBench teleport capture probe."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchmark.teleport_probe_artifacts import (
    ProbeArtifacts,
    append_jsonl,
    sample_filename,
    utc_now,
)
from benchmark.teleport_probe_console import print_sample
from benchmark.teleport_probe_data import ProbeError, TaskSelection


@dataclass(frozen=True)
class RuntimeHandles:
    """Hold validated environment objects needed during capture."""

    env_unwrapped: Any
    agent_name: str
    cam_id: int
    object_names: dict[str, str]


@dataclass(frozen=True)
class SequenceRequest:
    """Describe one baseline or post-teleport sampling sequence."""

    sequence_index: int
    pose_id: str
    delays: tuple[float, ...]
    requested_pose: tuple[float, ...] | None
    before_actor_pose: tuple[float, ...] | None

    @property
    def event(self) -> str:
        """Return the durable event name for this sequence."""

        return "baseline" if self.requested_pose is None else "teleport_sample"


@dataclass(frozen=True)
class SampleTiming:
    """Identify one sample within a sequence."""

    sample_index: int
    delay: float
    elapsed: float


@dataclass(frozen=True)
class RuntimeSnapshot:
    """Hold one hard-read runtime state and color image."""

    actual: dict[str, list[float]]
    camera_pose: list[float]
    image: Any


def runtime_handles(env: Any) -> RuntimeHandles:
    """Resolve and validate the protagonist and tracked runtime objects."""

    env_unwrapped = env.unwrapped
    protagonist_id = int(getattr(env_unwrapped, "protagonist_id", 0))
    if protagonist_id < 0 or protagonist_id >= len(env_unwrapped.player_list):
        raise ProbeError(
            "Invalid protagonist_id after reset: "
            f"{protagonist_id} for {len(env_unwrapped.player_list)} players"
        )
    if protagonist_id >= len(env_unwrapped.cam_list):
        raise ProbeError(
            "No protagonist camera after reset: "
            f"id={protagonist_id}, cameras={len(env_unwrapped.cam_list)}"
        )

    object_names = {
        "agent_actor": env_unwrapped.player_list[protagonist_id],
        "injured": env_unwrapped.injured_agent,
        "stretcher": env_unwrapped.stretcher,
        "ambulance": env_unwrapped.ambulance,
    }
    return RuntimeHandles(
        env_unwrapped=env_unwrapped,
        agent_name=object_names["agent_actor"],
        cam_id=int(env_unwrapped.cam_list[protagonist_id]),
        object_names=object_names,
    )


def _vector_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    if len(left) != len(right):
        raise ProbeError(
            f"Cannot subtract vectors with lengths {len(left)} and {len(right)}"
        )
    return [float(left[index]) - float(right[index]) for index in range(len(left))]


def _rotation_difference(
    actual_rotation: Sequence[float],
    requested_rotation: Sequence[float],
) -> list[float]:
    if len(actual_rotation) != len(requested_rotation):
        raise ProbeError(
            "Cannot compare rotations with lengths "
            f"{len(actual_rotation)} and {len(requested_rotation)}"
        )
    return [
        (float(actual_rotation[index]) - float(requested_rotation[index]) + 180.0)
        % 360.0
        - 180.0
        for index in range(len(actual_rotation))
    ]


@dataclass(frozen=True)
class CaptureWriter:
    """Persist one probe sequence's frames and sample records."""

    cv2: Any
    artifacts: ProbeArtifacts
    selection: TaskSelection
    requested_resolution: tuple[int, int]

    def save_sample(
        self,
        request: SequenceRequest,
        timing: SampleTiming,
        snapshot: RuntimeSnapshot,
    ) -> dict[str, Any]:
        """Persist one PNG and its JSONL record."""

        filename = sample_filename(
            request.sequence_index,
            timing.sample_index,
            request.pose_id,
            timing.delay,
        )
        absolute_frame_path = self.artifacts.run_dir / "frames" / filename
        if not self.cv2.imwrite(str(absolute_frame_path), snapshot.image):
            raise ProbeError(f"Failed to save PNG frame: {absolute_frame_path}")

        agent_pose = snapshot.actual["agent_actor"]
        camera_minus_actor = _vector_difference(
            snapshot.camera_pose[:3],
            agent_pose[:3],
        )
        pose_error = None
        if request.requested_pose is not None:
            pose_error = {
                "xyz": _vector_difference(
                    agent_pose[:3],
                    request.requested_pose[:3],
                ),
                "rotation_rpy_deg": _rotation_difference(
                    agent_pose[3:6],
                    request.requested_pose[3:6],
                ),
            }

        relative_frame_path = absolute_frame_path.relative_to(self.artifacts.run_dir)
        row = {
            "schema_version": 1,
            "recorded_at_utc": utc_now(),
            "event": request.event,
            "coordinate_frame": "unreal_world",
            "pose_order": ["x", "y", "z", "roll", "yaw", "pitch"],
            "rotation_unit": "degree",
            "scene": self.selection.scene.name,
            "scene_index": self.selection.scene.index,
            "env_id": self.selection.task_context["env_id"],
            "level": self.selection.level,
            "point_id": self.selection.point_id,
            "sequence_index": request.sequence_index,
            "sample_index": timing.sample_index,
            "pose_id": request.pose_id,
            "sample_delay_s": timing.delay,
            "sample_elapsed_s": timing.elapsed,
            "requested_actor_pose": (
                list(request.requested_pose)
                if request.requested_pose is not None
                else None
            ),
            "before_actor_pose": (
                list(request.before_actor_pose)
                if request.before_actor_pose is not None
                else None
            ),
            "actual_poses": {
                **snapshot.actual,
                "robot_camera": snapshot.camera_pose,
            },
            "camera_minus_actor_xyz": camera_minus_actor,
            "requested_to_actual_error": pose_error,
            "frame": {
                "path": relative_frame_path.as_posix(),
                "shape": [int(value) for value in snapshot.image.shape],
                "dtype": str(snapshot.image.dtype),
            },
        }
        append_jsonl(self.artifacts.samples_path, row)
        print_sample(row, self.requested_resolution)
        return row


@dataclass(frozen=True)
class CaptureSession:
    """Read runtime state and capture timed sequences."""

    runtime: RuntimeHandles
    writer: CaptureWriter

    def capture_state(self) -> RuntimeSnapshot:
        """Hard-read tracked object poses, the camera pose, and one frame."""

        labels = list(self.runtime.object_names)
        names = [self.runtime.object_names[label] for label in labels]
        obj_poses, cam_poses, images, _masks, _depths = (
            self.runtime.env_unwrapped.unrealcv.get_pose_img_batch(
                names,
                [self.runtime.cam_id],
                [True, True, False, False],
            )
        )
        if len(obj_poses) != len(names):
            raise ProbeError(
                f"Expected {len(names)} object poses, received {len(obj_poses)}"
            )
        if len(cam_poses) != 1:
            raise ProbeError(f"Expected one camera pose, received {len(cam_poses)}")
        if len(images) != 1 or images[0] is None:
            raise ProbeError("UnrealCV did not return one color frame")

        for index, label in enumerate(labels):
            pose = obj_poses[index]
            if len(pose) != 6:
                raise ProbeError(f"Expected a 6D pose for {label}, received {pose!r}")
        if len(cam_poses[0]) != 6:
            raise ProbeError(f"Expected a 6D camera pose, received {cam_poses[0]!r}")
        if not hasattr(images[0], "shape") or len(images[0].shape) < 2:
            raise ProbeError("UnrealCV color frame has no valid image shape")

        actual = {
            label: [float(value) for value in obj_poses[index]]
            for index, label in enumerate(labels)
        }
        return RuntimeSnapshot(
            actual=actual,
            camera_pose=[float(value) for value in cam_poses[0]],
            image=images[0],
        )

    def capture_sequence(self, request: SequenceRequest) -> list[dict[str, Any]]:
        """Capture all requested delays relative to one sequence start."""

        rows = []
        start = time.monotonic()
        for sample_index, delay in enumerate(request.delays):
            remaining = delay - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
            snapshot = self.capture_state()
            timing = SampleTiming(
                sample_index=sample_index,
                delay=delay,
                elapsed=time.monotonic() - start,
            )
            rows.append(self.writer.save_sample(request, timing, snapshot))
        return rows
