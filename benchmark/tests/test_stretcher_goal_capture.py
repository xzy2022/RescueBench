"""Focused tests for deterministic stretcher semantic-goal capture."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from benchmark.stretcher_capture_artifacts import (
    CaptureTargetPaths,
    inspect_existing_capture,
    save_capture,
)
from benchmark.stretcher_capture_data import (
    CapturePolicy,
    calculate_capture_geometry,
    classify_height_delta,
    evaluate_pose_stability,
    retry_height_offset,
    validate_capture_policy,
)
from benchmark.stretcher_capture_runtime import capture_point
from benchmark.teleport_probe_capture import RuntimeHandles, RuntimeSnapshot
from benchmark.teleport_probe_data import SCENES, TaskSelection


class _EncodedImage:
    def __init__(self, payload: bytes):
        self.payload = payload

    def tobytes(self) -> bytes:
        return self.payload


class _FakeCv2:
    @staticmethod
    def imencode(extension, image):
        if extension != ".png":
            raise AssertionError(f"unexpected extension: {extension}")
        return True, _EncodedImage(b"png:" + image.tobytes())


def _selection() -> TaskSelection:
    return TaskSelection(
        level=0,
        point_id=7,
        source_path=Path("level_0.jsonl"),
        source_line=8,
        raw_point={
            "agent_loc": [-237, 377, 190, 0, 0, 0],
            "injured_player_loc": [0, 0, 193, 0, 0, 0],
            "stretcher_loc": [10, 0, 100, 0, 0, 0],
            "ambulance_loc": [20, 0, 90, 0, 0, 0],
        },
        task_context={"env_id": "UnrealRescue-FlexibleRoom"},
        scene=SCENES[0],
    )


def _snapshot(agent_z: float, stretcher_z: float = 100.0) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        actual={
            "agent_actor": [-237.0, 377.0, agent_z, 0.0, 0.0, 0.0],
            "injured": [0.0, 0.0, 193.0, 0.0, 0.0, 0.0],
            "stretcher": [10.0, 0.0, stretcher_z, 0.0, 0.0, 0.0],
            "ambulance": [20.0, 0.0, 90.0, 0.0, 0.0, 0.0],
        },
        camera_pose=[-237.0, 377.0, agent_z + 67.0, 0.0, 0.0, 0.0],
        image=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def _policy(**overrides) -> CapturePolicy:
    values = {
        "capture_distance_uu": 200.0,
        "initial_height_offset_uu": 160.0,
        "height_retry_step_uu": 30.0,
        "min_delta_z_uu": 70.0,
        "max_delta_z_uu": 140.0,
        "max_attempts": 2,
        "sample_delays": (0.0,),
        "stable_tail_samples": 1,
        "position_epsilon_uu": 1.0,
        "rotation_epsilon_deg": 1.0,
    }
    values.update(overrides)
    return CapturePolicy(**values)


class CapturePolicyTests(unittest.TestCase):
    """Validate geometry, stability, and retry boundaries."""

    def test_geometry_is_between_stretcher_and_injured_and_faces_stretcher(self):
        geometry = calculate_capture_geometry(
            injured_pose=[0, 0, 193, 0, 0, 0],
            stretcher_pose=[500, 0, 100, 0, 0, 0],
            distance_uu=200,
        )

        self.assertEqual(geometry.capture_x, 300.0)
        self.assertEqual(geometry.capture_y, 0.0)
        self.assertEqual(geometry.yaw_deg, 0.0)

    def test_degenerate_geometry_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "degenerate"):
            calculate_capture_geometry(
                injured_pose=[1, 2, 3, 0, 0, 0],
                stretcher_pose=[1, 2, 9, 0, 0, 0],
                distance_uu=200,
            )

    def test_height_bounds_are_inclusive_and_retry_changes_h_once(self):
        policy = _policy()

        self.assertEqual(classify_height_delta(70.0, policy), "valid")
        self.assertEqual(classify_height_delta(140.0, policy), "valid")
        self.assertEqual(classify_height_delta(69.9, policy), "too_low")
        self.assertEqual(classify_height_delta(140.1, policy), "too_high")
        self.assertEqual(retry_height_offset(160, "too_low", policy), 190)
        self.assertEqual(retry_height_offset(160, "too_high", policy), 130)

    def test_stability_handles_wrapped_yaw(self):
        poses = [
            [1, 2, 3, 0, 179.8, 0],
            [1.2, 2, 3, 0, -179.9, 0],
            [1.1, 2, 3, 0, 179.9, 0],
        ]

        result = evaluate_pose_stability(
            poses,
            tail_samples=3,
            position_epsilon_uu=1.0,
            rotation_epsilon_deg=1.0,
        )

        self.assertTrue(result.stable)

    def test_policy_rejects_more_than_one_retry(self):
        with self.assertRaisesRegex(RuntimeError, "1 or 2"):
            validate_capture_policy(
                capture_distance_uu=200,
                initial_height_offset_uu=160,
                height_retry_step_uu=30,
                min_delta_z_uu=70,
                max_delta_z_uu=140,
                max_attempts=3,
                sample_delays=(0, 1, 2),
                stable_tail_samples=3,
                position_epsilon_uu=1,
                rotation_epsilon_deg=1,
            )


class RuntimeStateMachineTests(unittest.TestCase):
    """Exercise successful and retrying drops without Unreal Engine."""

    def _runtime(self):
        unrealcv = SimpleNamespace(
            set_obj_rotation=Mock(),
            set_obj_location=Mock(),
        )
        return RuntimeHandles(
            env_unwrapped=SimpleNamespace(unrealcv=unrealcv),
            agent_name="agent",
            cam_id=0,
            object_names={
                "agent_actor": "agent",
                "injured": "injured",
                "stretcher": "stretcher",
                "ambulance": "ambulance",
            },
        )

    def test_valid_first_drop_uses_actual_stretcher_z_plus_160(self):
        runtime = self._runtime()
        snapshots = [_snapshot(193.0), _snapshot(193.0)]
        rows = []

        with patch(
            "benchmark.stretcher_capture_runtime.read_runtime_snapshot",
            side_effect=snapshots,
        ):
            outcome = capture_point(
                runtime=runtime,
                selection=_selection(),
                policy=_policy(),
                sample_sink=rows.append,
            )

        self.assertEqual(outcome.status, "captured")
        self.assertEqual(outcome.delta_z_uu, 93.0)
        requested_location = runtime.env_unwrapped.unrealcv.set_obj_location.call_args[
            0
        ][1]
        self.assertEqual(requested_location, [-190.0, 0.0, 260.0])
        self.assertEqual(len(rows), 2)

    def test_too_low_drop_retries_at_h190(self):
        runtime = self._runtime()
        snapshots = [_snapshot(193.0), _snapshot(150.0), _snapshot(193.0)]

        with patch(
            "benchmark.stretcher_capture_runtime.read_runtime_snapshot",
            side_effect=snapshots,
        ):
            outcome = capture_point(
                runtime=runtime,
                selection=_selection(),
                policy=_policy(),
                sample_sink=lambda _row: None,
            )

        self.assertEqual(outcome.status, "captured")
        self.assertEqual(len(outcome.attempts), 2)
        locations = [
            call.args[1]
            for call in runtime.env_unwrapped.unrealcv.set_obj_location.call_args_list
        ]
        self.assertEqual(locations[0][2], 260.0)
        self.assertEqual(locations[1][2], 290.0)


class ArtifactTests(unittest.TestCase):
    """Validate PNG/sidecar publication and resume integrity."""

    def test_capture_sidecar_records_actual_shape_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = CaptureTargetPaths(
                image_path=root / "level_0_7.png",
                sidecar_path=root / "level_0_7.json",
            )
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            metadata = save_capture(
                cv2=_FakeCv2(),
                paths=paths,
                image=image,
                metadata={"level": 0, "point_id": 7},
            )
            loaded = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
            status, _existing = inspect_existing_capture(paths, _selection())

        self.assertEqual(metadata["image_size"], [6, 4])
        self.assertEqual(metadata["image_shape"], [4, 6, 3])
        self.assertEqual(loaded["sha256"], metadata["sha256"])
        self.assertEqual(status, "valid")

    def test_missing_sidecar_is_a_resume_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = CaptureTargetPaths(
                image_path=root / "level_0_7.png",
                sidecar_path=root / "level_0_7.json",
            )
            paths.image_path.write_bytes(b"not-a-real-png")
            status, _metadata = inspect_existing_capture(paths, _selection())

        self.assertEqual(status, "conflict")


class CompatibilityTests(unittest.TestCase):
    """Keep all new production sources parseable by Python 3.8."""

    def test_sources_parse_with_python_38_grammar(self):
        benchmark_dir = Path(__file__).resolve().parents[1]
        paths = sorted(benchmark_dir.glob("stretcher_capture*.py"))
        paths.append(benchmark_dir / "capture_stretcher_goal_images.py")

        for path in paths:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(path), feature_version=(3, 8))
                self.assertNotIn("cast(tuple[", source)


if __name__ == "__main__":
    unittest.main()
