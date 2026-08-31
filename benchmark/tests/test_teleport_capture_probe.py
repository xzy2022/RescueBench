"""Unit tests for UE-independent teleport probe contracts."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from benchmark import teleport_probe_cli
from benchmark.teleport_capture_probe import main
from benchmark.teleport_probe_artifacts import ProbeArtifacts, sample_filename
from benchmark.teleport_probe_capture import (
    CaptureSession,
    CaptureWriter,
    RuntimeHandles,
    SequenceRequest,
)
from benchmark.teleport_probe_data import (
    SCENES,
    ProbeError,
    TaskSelection,
    create_run_directory,
    load_pose_manifest,
    load_task_selection,
    scene_for_env_id,
    validate_sample_delays,
)


class PoseManifestTests(unittest.TestCase):
    """Validate the external pose-manifest contract."""

    def _write_manifest(self, root: Path, payload: object) -> Path:
        path = root / "poses.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_valid_manifest(self) -> None:
        """A valid manifest should produce one normalized pose."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_manifest(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "coordinate_frame": "unreal_world",
                    "poses": [
                        {
                            "id": "near-stretcher-z190",
                            "location_xyz": [500, 200.5, 190],
                            "rotation_rpy_deg": [0, -90, 0],
                            "note": "height probe",
                        }
                    ],
                },
            )

            poses = load_pose_manifest(path)

        self.assertEqual(len(poses), 1)
        self.assertEqual(poses[0].pose_id, "near-stretcher-z190")
        self.assertEqual(poses[0].pose, [500.0, 200.5, 190.0, 0.0, -90.0, 0.0])

    def test_rejects_duplicate_ids(self) -> None:
        """Duplicate pose IDs should fail before runtime startup."""

        with tempfile.TemporaryDirectory() as temp_dir:
            pose = {
                "id": "duplicate",
                "location_xyz": [1, 2, 3],
                "rotation_rpy_deg": [0, 0, 0],
            }
            path = self._write_manifest(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "coordinate_frame": "unreal_world",
                    "poses": [pose, pose],
                },
            )

            with self.assertRaisesRegex(ProbeError, "Duplicate pose id"):
                load_pose_manifest(path)

    def test_rejects_non_finite_coordinate(self) -> None:
        """NaN coordinates should not enter an UnrealCV command."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_manifest(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "coordinate_frame": "unreal_world",
                    "poses": [
                        {
                            "id": "bad-coordinate",
                            "location_xyz": [1, 2, float("nan")],
                            "rotation_rpy_deg": [0, 0, 0],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ProbeError, "must be finite"):
                load_pose_manifest(path)


class SceneAndTaskSelectionTests(unittest.TestCase):
    """Validate scene resolution and zero-based task selection."""

    def _make_gym_root(self, root: Path) -> Path:
        jsonl_dir = root / "envs" / "setting" / "test_jsonl"
        jsonl_dir.mkdir(parents=True)
        rows = [
            {
                "env_id": "UnrealRescue-FlexibleRoom",
                "agent_loc": [1, 2, 3, 0, 10, 0],
                "injured_player_loc": [4, 5, 6, 0, 0, 0],
                "injured_agent_id": 4,
                "stretcher_loc": [7, 8, 9, 0, 0, 0],
                "ambulance_loc": [10, 11, 12, 0, 90, 0],
                "reference_text": ["test"],
                "reference_image_path": ["test.png"],
                "timeout": 180,
            },
            {
                "env_id": "UnrealRescue-SuburbNeighborhood_Day_dooropen",
                "agent_loc": [11, 12, 13, 0, 20, 0],
                "injured_player_loc": [14, 15, 16, 0, 0, 0],
                "injured_agent_id": 5,
                "stretcher_loc": [17, 18, 19, 0, 0, 0],
                "ambulance_loc": [20, 21, 22, 0, 90, 0],
                "reference_text": ["test 2"],
                "reference_image_path": ["test2.png"],
                "timeout": 300,
            },
        ]
        (jsonl_dir / "level_0.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return root

    def test_dooropen_is_variant_of_second_canonical_scene(self) -> None:
        """The door-open ID should retain its canonical scene index."""

        scene = scene_for_env_id("UnrealRescue-SuburbNeighborhood_Day_dooropen")
        self.assertEqual(scene.index, 2)
        self.assertEqual(scene.name, "SuburbNeighborhood_Day")

    def test_load_task_selection_uses_zero_based_point(self) -> None:
        """Point IDs should map directly to the TaskLoader list index."""

        with tempfile.TemporaryDirectory() as temp_dir:
            gym_root = self._make_gym_root(Path(temp_dir))
            selection = load_task_selection(0, 1, gym_root)

        self.assertEqual(selection.point_id, 1)
        self.assertEqual(selection.source_line, 2)
        self.assertEqual(selection.scene.index, 2)
        self.assertEqual(
            selection.task_context["env_id"],
            "UnrealRescue-SuburbNeighborhood_Day_dooropen",
        )

    def test_rejects_negative_and_out_of_range_points(self) -> None:
        """Negative and oversized zero-based IDs should fail clearly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            gym_root = self._make_gym_root(Path(temp_dir))
            with self.assertRaisesRegex(ProbeError, "non-negative"):
                load_task_selection(0, -1, gym_root)
            with self.assertRaisesRegex(ProbeError, "valid range is 0-1"):
                load_task_selection(0, 2, gym_root)


class ProbeConfigurationTests(unittest.TestCase):
    """Validate timing and output-directory configuration."""

    def test_sample_delays_must_be_strictly_increasing(self) -> None:
        """Repeated or descending delays should be rejected."""

        self.assertEqual(
            validate_sample_delays([0, 0.2, 1]),
            (0.0, 0.2, 1.0),
        )
        with self.assertRaisesRegex(ProbeError, "strictly increasing"):
            validate_sample_delays([0, 0.2, 0.2])

    def test_sample_filename_uses_index_to_avoid_rounding_collision(self) -> None:
        """Distinct samples must not collide after delay formatting."""

        first = sample_filename(1, 0, "pose-001", 0.0)
        second = sample_filename(1, 1, "pose-001", 0.0004)

        self.assertNotEqual(first, second)
        self.assertEqual(first, "001_000_pose-001_t0.000s.png")
        self.assertEqual(second, "001_001_pose-001_t0.000s.png")

    def test_run_directory_never_overwrites_existing_run(self) -> None:
        """Runs with the same timestamp should receive unique suffixes."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = create_run_directory(root, 0, 7, "20260830-120000")
            second = create_run_directory(root, 0, 7, "20260830-120000")

            self.assertNotEqual(first, second)
            self.assertEqual(second.name, "teleport-probe-20260830-120000-L0-P7-01")
            self.assertTrue((first / "frames").is_dir())
            self.assertTrue((second / "frames").is_dir())

    def test_probe_sources_keep_python_38_runtime_compatibility(self) -> None:
        """Probe modules must remain importable by the AutoDL Python 3.8 runtime."""

        benchmark_dir = Path(__file__).resolve().parents[1]
        paths = sorted(benchmark_dir.glob("teleport_probe*.py"))
        paths.append(benchmark_dir / "teleport_capture_probe.py")

        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                ast.parse(source, filename=str(path), feature_version=(3, 8))
                self.assertNotIn("strict=", source)
                self.assertNotIn("from datetime import UTC", source)
                self.assertNotIn("cast(tuple[", source)


class CaptureRecordTests(unittest.TestCase):
    """Validate hard-read pose and frame records without UE."""

    @staticmethod
    def _get_pose_img_batch(_objects, _cameras, flags):
        if flags != [True, True, False, False]:
            raise AssertionError(f"unexpected flags: {flags}")
        object_poses = [
            [10, 20, 30, 0, 90, 0],
            [40, 50, 60, 0, 0, 0],
            [70, 80, 90, 0, 0, 0],
            [100, 110, 120, 0, 0, 0],
        ]
        camera_poses = [[11, 22, 63, 0, 90, 0]]
        images = [np.zeros((4, 6, 3), dtype=np.uint8)]
        return object_poses, camera_poses, images, [], []

    @staticmethod
    def _write_image(path, image):
        Path(path).write_bytes(image.tobytes())
        return True

    def test_capture_sequence_records_hard_poses_and_actual_shape(self) -> None:
        """A captured sample should contain hard poses and actual image shape."""

        selection = TaskSelection(
            level=0,
            point_id=7,
            source_path=Path("level_0.jsonl"),
            source_line=8,
            raw_point={
                "agent_loc": [1, 2, 3, 0, 0, 0],
                "injured_player_loc": [4, 5, 6, 0, 0, 0],
                "stretcher_loc": [7, 8, 9, 0, 0, 0],
                "ambulance_loc": [10, 11, 12, 0, 0, 0],
            },
            task_context={"env_id": "UnrealRescue-FlexibleRoom"},
            scene=SCENES[0],
        )
        runtime = RuntimeHandles(
            env_unwrapped=SimpleNamespace(
                unrealcv=SimpleNamespace(get_pose_img_batch=self._get_pose_img_batch)
            ),
            agent_name="agent",
            cam_id=0,
            object_names={
                "agent_actor": "agent",
                "injured": "injured",
                "stretcher": "stretcher",
                "ambulance": "ambulance",
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "frames").mkdir()
            artifacts = ProbeArtifacts(
                run_dir=run_dir,
                samples_path=run_dir / "samples.jsonl",
                metadata_path=run_dir / "run.json",
            )
            writer = CaptureWriter(
                cv2=SimpleNamespace(imwrite=self._write_image),
                artifacts=artifacts,
                selection=selection,
                requested_resolution=(640, 640),
            )
            session = CaptureSession(runtime=runtime, writer=writer)
            with redirect_stdout(StringIO()):
                rows = session.capture_sequence(
                    SequenceRequest(
                        sequence_index=1,
                        pose_id="pose-001",
                        delays=(0.0,),
                        requested_pose=(10, 20, 30, 0, 90, 0),
                        before_actor_pose=(1, 2, 3, 0, 0, 0),
                    )
                )
            saved_row = json.loads(artifacts.samples_path.read_text(encoding="utf-8"))

        self.assertEqual(rows[0]["actual_poses"]["agent_actor"][2], 30.0)
        self.assertEqual(rows[0]["actual_poses"]["robot_camera"][2], 63.0)
        self.assertEqual(rows[0]["camera_minus_actor_xyz"], [1.0, 2.0, 33.0])
        self.assertEqual(saved_row["frame"]["shape"], [4, 6, 3])
        self.assertEqual(saved_row["requested_to_actual_error"]["xyz"], [0, 0, 0])
        self.assertEqual(saved_row["sample_index"], 0)
        self.assertEqual(
            saved_row["frame"]["path"],
            "frames/001_000_pose-001_t0.000s.png",
        )


class LauncherTests(unittest.TestCase):
    """Validate the direct-script launcher boundary without UE."""

    def test_list_scenes_uses_package_cli(self) -> None:
        """The launcher should delegate list mode to the package CLI."""

        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["--list-scenes"])

        self.assertEqual(exit_code, 0)
        self.assertIn("RescueBench benchmark scenes", output.getvalue())


class ProbeLifecycleTests(unittest.TestCase):
    """Validate cleanup behavior independently of UE startup."""

    def test_close_env_runs_when_failure_recording_also_fails(self) -> None:
        """A metadata write failure must not prevent UE process cleanup."""

        inputs = SimpleNamespace(
            render=SimpleNamespace(
                resolution=(640, 640),
                quality=2,
                offscreen=True,
            )
        )
        manager = Mock()
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            artifacts = ProbeArtifacts(
                run_dir=run_dir,
                samples_path=run_dir / "samples.jsonl",
                metadata_path=run_dir / "run.json",
            )
            with (
                patch.object(
                    teleport_probe_cli,
                    "_resolve_inputs",
                    return_value=inputs,
                ),
                patch.object(
                    teleport_probe_cli.ProbeArtifacts,
                    "create",
                    return_value=artifacts,
                ),
                patch.object(
                    teleport_probe_cli,
                    "build_run_metadata",
                    return_value={"status": "running"},
                ),
                patch.object(teleport_probe_cli, "write_json_atomic"),
                patch.object(teleport_probe_cli, "print_probe_inputs"),
                patch.object(
                    teleport_probe_cli,
                    "EnvManager",
                    return_value=manager,
                ),
                patch.object(
                    teleport_probe_cli,
                    "_execute_probe",
                    side_effect=RuntimeError("probe failed"),
                ),
                patch.object(
                    teleport_probe_cli,
                    "_record_failure",
                    side_effect=OSError("metadata failed"),
                ),
            ):
                with self.assertRaisesRegex(OSError, "metadata failed"):
                    teleport_probe_cli.run_probe(SimpleNamespace())

        manager.close_env.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
