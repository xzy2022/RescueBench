"""Focused tests for deterministic stretcher candidate capture."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from benchmark.stretcher_capture_artifacts import (
    BatchArtifacts,
    CaptureTargetPaths,
    format_capture_distance,
    inspect_existing_capture,
    save_capture,
)
from benchmark.stretcher_capture_cli import _run_selection, build_parser
from benchmark.stretcher_capture_data import (
    CapturePolicy,
    calculate_capture_geometry,
    classify_height_delta,
    evaluate_pose_stability,
    retry_height_offset,
    validate_agent_pose,
    validate_capture_policy,
    wrapped_angle_error_deg,
)
from benchmark.stretcher_capture_runtime import (
    capture_distance_candidate,
    prepare_capture_point,
)
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


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


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


def _snapshot(
    agent_z: float,
    *,
    agent_x: float = -190.0,
    agent_y: float = 0.0,
    agent_yaw: float = 0.0,
    stretcher_z: float = 100.0,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        actual={
            "agent_actor": [
                agent_x,
                agent_y,
                agent_z,
                0.0,
                agent_yaw,
                0.0,
            ],
            "injured": [0.0, 0.0, 193.0, 0.0, 0.0, 0.0],
            "stretcher": [10.0, 0.0, stretcher_z, 0.0, 0.0, 0.0],
            "ambulance": [20.0, 0.0, 90.0, 0.0, 0.0, 0.0],
        },
        camera_pose=[agent_x, agent_y, agent_z + 67.0, 0.0, agent_yaw, 0.0],
        image=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def _policy(**overrides) -> CapturePolicy:
    values = {
        "capture_distances_uu": (200.0, 300.0, 350.0),
        "initial_height_offset_uu": 160.0,
        "height_retry_step_uu": 30.0,
        "min_delta_z_uu": 70.0,
        "max_delta_z_uu": 140.0,
        "max_attempts": 2,
        "settle_sample_interval_s": 1.0,
        "stretcher_settle_timeout_s": 20.0,
        "agent_settle_timeout_s": 12.0,
        "stable_window_samples": 3,
        "position_epsilon_uu": 1.0,
        "rotation_epsilon_deg": 1.0,
        "max_agent_xy_error_uu": 10.0,
        "max_agent_yaw_error_deg": 1.0,
    }
    values.update(overrides)
    return CapturePolicy(**values)


def _runtime() -> RuntimeHandles:
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


def _run_with_clock(snapshots, function, **kwargs):
    clock = _FakeClock()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "benchmark.stretcher_capture_runtime.read_runtime_snapshot",
                side_effect=snapshots,
            )
        )
        stack.enter_context(
            patch(
                "benchmark.stretcher_capture_runtime.time.monotonic",
                side_effect=clock.monotonic,
            )
        )
        stack.enter_context(
            patch(
                "benchmark.stretcher_capture_runtime.time.sleep",
                side_effect=clock.sleep,
            )
        )
        return function(**kwargs)


class CapturePolicyTests(unittest.TestCase):
    """Validate geometry, stability, pose errors, and policy boundaries."""

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

    def test_stability_and_yaw_error_handle_wrapping(self):
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
        self.assertAlmostEqual(wrapped_angle_error_deg(-179.9, 179.8), 0.3)

    def test_pose_validation_has_inclusive_xy_and_yaw_limits(self):
        result = validate_agent_pose(
            [0, 0, 260, 0, 179.8, 0],
            [6, 8, 193, 0, -179.2, 0],
            _policy(),
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.xy_error_uu, 10.0)
        self.assertAlmostEqual(result.yaw_error_deg, 1.0)

    def test_policy_accepts_three_distances_and_rejects_duplicates(self):
        values = {
            "capture_distances_uu": (200, 300, 350),
            "initial_height_offset_uu": 160,
            "height_retry_step_uu": 30,
            "min_delta_z_uu": 70,
            "max_delta_z_uu": 140,
            "max_attempts": 2,
            "settle_sample_interval_s": 1,
            "stretcher_settle_timeout_s": 20,
            "agent_settle_timeout_s": 12,
            "stable_window_samples": 3,
            "position_epsilon_uu": 1,
            "rotation_epsilon_deg": 1,
            "max_agent_xy_error_uu": 10,
            "max_agent_yaw_error_deg": 1,
        }
        policy = validate_capture_policy(**values)
        self.assertEqual(policy.capture_distances_uu, (200.0, 300.0, 350.0))

        values["capture_distances_uu"] = (200, 200)
        with self.assertRaisesRegex(RuntimeError, "unique"):
            validate_capture_policy(**values)

    def test_policy_rejects_timeout_too_short_for_window(self):
        with self.assertRaisesRegex(RuntimeError, "allow the requested"):
            validate_capture_policy(
                capture_distances_uu=(200,),
                initial_height_offset_uu=160,
                height_retry_step_uu=30,
                min_delta_z_uu=70,
                max_delta_z_uu=140,
                max_attempts=2,
                settle_sample_interval_s=1,
                stretcher_settle_timeout_s=1,
                agent_settle_timeout_s=12,
                stable_window_samples=3,
                position_epsilon_uu=1,
                rotation_epsilon_deg=1,
                max_agent_xy_error_uu=10,
                max_agent_yaw_error_deg=1,
            )


class RuntimeStateMachineTests(unittest.TestCase):
    """Exercise rolling settling and distance-qualified drops without Unreal."""

    def test_rolling_stretcher_wait_stops_after_three_stable_samples(self):
        rows = []
        preparation = _run_with_clock(
            [_snapshot(193.0)] * 3,
            prepare_capture_point,
            runtime=_runtime(),
            selection=_selection(),
            policy=_policy(),
            sample_sink=rows.append,
        )

        self.assertEqual(preparation.status, "ready")
        self.assertEqual(preparation.baseline.stop_reason, "stable_window")
        self.assertEqual(preparation.baseline.elapsed_s, 2.0)
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[0]["current_stability"]["max_position_span_uu"])

    def test_stretcher_timeout_is_terminal(self):
        snapshots = [
            _snapshot(193.0, stretcher_z=100.0),
            _snapshot(193.0, stretcher_z=90.0),
            _snapshot(193.0, stretcher_z=80.0),
        ]
        preparation = _run_with_clock(
            snapshots,
            prepare_capture_point,
            runtime=_runtime(),
            selection=_selection(),
            policy=_policy(stretcher_settle_timeout_s=2.0),
            sample_sink=lambda _row: None,
        )

        self.assertEqual(preparation.status, "skipped")
        self.assertEqual(preparation.reason, "stretcher_settle_timeout")
        self.assertEqual(preparation.baseline.stop_reason, "timeout")

    def test_valid_drop_uses_actual_stretcher_z_plus_160(self):
        runtime = _runtime()
        preparation = _run_with_clock(
            [_snapshot(193.0)] * 3,
            prepare_capture_point,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            sample_sink=lambda _row: None,
        )
        rows = []
        outcome = _run_with_clock(
            [_snapshot(193.0)] * 3,
            capture_distance_candidate,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            baseline=preparation.baseline,
            capture_distance_uu=200.0,
            sample_sink=rows.append,
        )

        self.assertEqual(outcome.status, "captured")
        self.assertEqual(outcome.delta_z_uu, 93.0)
        requested_location = runtime.env_unwrapped.unrealcv.set_obj_location.call_args[
            0
        ][1]
        self.assertEqual(requested_location, [-190.0, 0.0, 260.0])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["capture_distance_uu"], 200.0)

    def test_too_low_drop_retries_at_h190(self):
        runtime = _runtime()
        preparation = _run_with_clock(
            [_snapshot(193.0)] * 3,
            prepare_capture_point,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            sample_sink=lambda _row: None,
        )
        snapshots = [_snapshot(150.0)] * 3 + [_snapshot(193.0)] * 3
        outcome = _run_with_clock(
            snapshots,
            capture_distance_candidate,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            baseline=preparation.baseline,
            capture_distance_uu=200.0,
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

    def test_stable_xy_error_is_skipped_without_height_retry(self):
        runtime = _runtime()
        preparation = _run_with_clock(
            [_snapshot(193.0)] * 3,
            prepare_capture_point,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            sample_sink=lambda _row: None,
        )
        outcome = _run_with_clock(
            [_snapshot(193.0, agent_x=-170.0)] * 3,
            capture_distance_candidate,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            baseline=preparation.baseline,
            capture_distance_uu=200.0,
            sample_sink=lambda _row: None,
        )

        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.reason, "agent_xy_error")
        self.assertEqual(len(outcome.attempts), 1)

    def test_agent_timeout_is_skipped_without_height_retry(self):
        runtime = _runtime()
        preparation = _run_with_clock(
            [_snapshot(193.0)] * 3,
            prepare_capture_point,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(),
            sample_sink=lambda _row: None,
        )
        outcome = _run_with_clock(
            [_snapshot(250.0), _snapshot(220.0), _snapshot(190.0)],
            capture_distance_candidate,
            runtime=runtime,
            selection=_selection(),
            policy=_policy(agent_settle_timeout_s=2.0),
            baseline=preparation.baseline,
            capture_distance_uu=200.0,
            sample_sink=lambda _row: None,
        )

        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.reason, "agent_settle_timeout")
        self.assertEqual(len(outcome.attempts), 1)


class ArtifactTests(unittest.TestCase):
    """Validate distance-qualified publication and resume integrity."""

    def test_target_paths_include_distance_label(self):
        root = Path("topomaps/stretcher")
        artifacts = BatchArtifacts(
            stretcher_dir=root,
            run_dir=root / "_runs/run",
            run_path=root / "_runs/run/run.json",
            points_path=root / "_runs/run/points.jsonl",
            samples_path=root / "_runs/run/samples.jsonl",
            diagnostics_dir=root / "_runs/run/diagnostics",
        )

        paths = artifacts.target_paths(_selection(), 300.0)

        self.assertEqual(paths.image_path.name, "level_0_7_300UU.png")
        self.assertEqual(paths.sidecar_path.name, "level_0_7_300UU.json")
        self.assertEqual(format_capture_distance(350), "350UU")

    def test_capture_sidecar_records_distance_shape_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = CaptureTargetPaths(
                image_path=root / "level_0_7_200UU.png",
                sidecar_path=root / "level_0_7_200UU.json",
            )
            image = np.zeros((4, 6, 3), dtype=np.uint8)
            metadata = save_capture(
                cv2=_FakeCv2(),
                paths=paths,
                image=image,
                metadata={
                    "schema_version": 2,
                    "level": 0,
                    "point_id": 7,
                    "capture_distance_uu": 200.0,
                },
            )
            loaded = json.loads(paths.sidecar_path.read_text(encoding="utf-8"))
            status, _existing = inspect_existing_capture(
                paths,
                _selection(),
                200.0,
            )

        self.assertEqual(metadata["image_size"], [6, 4])
        self.assertEqual(metadata["image_shape"], [4, 6, 3])
        self.assertEqual(loaded["sha256"], metadata["sha256"])
        self.assertEqual(status, "valid")

    def test_sidecar_distance_mismatch_is_a_resume_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = CaptureTargetPaths(
                image_path=root / "level_0_7_200UU.png",
                sidecar_path=root / "level_0_7_200UU.json",
            )
            save_capture(
                cv2=_FakeCv2(),
                paths=paths,
                image=np.zeros((4, 6, 3), dtype=np.uint8),
                metadata={
                    "schema_version": 2,
                    "level": 0,
                    "point_id": 7,
                    "capture_distance_uu": 300.0,
                },
            )
            status, _metadata = inspect_existing_capture(
                paths,
                _selection(),
                200.0,
            )

        self.assertEqual(status, "conflict")


class CliTests(unittest.TestCase):
    """Keep the direct CLI defaults aligned with the three-candidate contract."""

    def test_parser_defaults_to_three_distances_and_rolling_timeouts(self):
        args = build_parser().parse_args(["--levels", "2", "--topomap-dir", "topomaps"])

        self.assertEqual(args.capture_distances_uu, (200.0, 300.0, 350.0))
        self.assertEqual(args.settle_sample_interval_s, 1.0)
        self.assertEqual(args.stretcher_settle_timeout_s, 20.0)
        self.assertEqual(args.agent_settle_timeout_s, 12.0)

    def test_one_point_prepares_once_then_runs_all_three_distances(self):
        baseline = SimpleNamespace(
            elapsed_s=2.0,
            rows=(1, 2, 3),
            final_snapshot=_snapshot(193.0),
        )
        preparation = SimpleNamespace(
            status="ready",
            reason="stretcher_stable",
            baseline=baseline,
        )
        manager = SimpleNamespace(
            env=SimpleNamespace(reset=Mock()),
            ensure_env=Mock(),
            apply_task_context=Mock(),
        )
        artifacts = Mock()
        artifacts.target_paths.side_effect = lambda _selection, distance: (
            CaptureTargetPaths(
                image_path=Path(f"level_0_7_{int(distance)}UU.png"),
                sidecar_path=Path(f"level_0_7_{int(distance)}UU.json"),
            )
        )
        inputs = SimpleNamespace(
            policy=_policy(),
            resume=True,
            overwrite=False,
        )
        runtime = object()

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "benchmark.stretcher_capture_cli.inspect_existing_capture",
                    return_value=("missing", None),
                )
            )
            stack.enter_context(
                patch("benchmark.stretcher_capture_cli.print_task_configuration")
            )
            stack.enter_context(
                patch(
                    "benchmark.stretcher_capture_cli.runtime_handles",
                    return_value=runtime,
                )
            )
            prepare_mock = stack.enter_context(
                patch(
                    "benchmark.stretcher_capture_cli.prepare_capture_point",
                    return_value=preparation,
                )
            )
            stack.enter_context(
                patch(
                    "benchmark.stretcher_capture_cli.preparation_as_dict",
                    return_value={"status": "ready", "reason": "stretcher_stable"},
                )
            )
            run_distance_mock = stack.enter_context(
                patch(
                    "benchmark.stretcher_capture_cli._run_distance",
                    return_value="captured",
                )
            )
            statuses = _run_selection(
                manager=manager,
                cv2=_FakeCv2(),
                inputs=inputs,
                artifacts=artifacts,
                selection=_selection(),
            )

        self.assertEqual(statuses, ["captured", "captured", "captured"])
        manager.env.reset.assert_called_once_with()
        prepare_mock.assert_called_once()
        self.assertEqual(run_distance_mock.call_count, 3)
        self.assertTrue(
            all(
                call.kwargs["baseline"] is baseline
                for call in run_distance_mock.call_args_list
            )
        )


class CompatibilityTests(unittest.TestCase):
    """Keep all capture production sources parseable by Python 3.8."""

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
