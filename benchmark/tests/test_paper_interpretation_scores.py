"""Focused tests for paper-interpretation episode scores."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from core.benchmark_runner import BenchmarkRunner  # noqa: E402
from core.metrics import EpisodeMetrics  # noqa: E402
from core.result_writer import ResultWriter  # noqa: E402
from utils.paper_interpretation_scores import (  # noqa: E402
    calculate_paper_interpretation_scores,
)
from wrappers.progress_tracking_wrapper import ProgressTrackingController  # noqa: E402


def _calculate(**overrides):
    stage_near_radius = overrides.pop("stage_near_radius", 750.0)
    eps = overrides.pop("eps", 1e-6)
    inputs = {
        "stage1_initial_distance": 1750.0,
        "stage1_best_distance": 1750.0,
        "stage2_initial_distance": 1750.0,
        "stage2_best_distance": 1750.0,
        "stage1_success": False,
    }
    inputs.update(overrides)
    return calculate_paper_interpretation_scores(
        inputs,
        stage_near_radius=stage_near_radius,
        eps=eps,
    )


class PaperInterpretationScoreTests(unittest.TestCase):
    def test_scores_cover_outer_and_inner_progress(self):
        no_progress = _calculate()
        self.assertEqual(no_progress["paper_interpretation_task_score"], 0.0)

        outer_half = _calculate(stage1_best_distance=1250.0)
        self.assertEqual(outer_half["paper_interpretation_s1_score"], 12.5)
        self.assertEqual(outer_half["paper_interpretation_s2_score"], 0.0)

        at_boundary = _calculate(stage1_best_distance=750.0)
        self.assertEqual(at_boundary["paper_interpretation_s1_score"], 25.0)
        self.assertEqual(at_boundary["paper_interpretation_s2_score"], 0.0)

        inner_half = _calculate(stage1_best_distance=375.0)
        self.assertEqual(inner_half["paper_interpretation_s1_score"], 25.0)
        self.assertEqual(inner_half["paper_interpretation_s2_score"], 12.5)

        at_target = _calculate(stage1_best_distance=0.0)
        self.assertEqual(at_target["paper_interpretation_s1_score"], 25.0)
        self.assertEqual(at_target["paper_interpretation_s2_score"], 25.0)

    def test_stage2_is_gated_by_stage1_success(self):
        gated = _calculate(stage2_best_distance=0.0)
        self.assertEqual(gated["paper_interpretation_s3_score"], 0.0)
        self.assertEqual(gated["paper_interpretation_s4_score"], 0.0)

        active = _calculate(
            stage1_success=True,
            stage2_initial_distance=1750.0,
            stage2_best_distance=375.0,
        )
        self.assertEqual(active["paper_interpretation_s3_score"], 25.0)
        self.assertEqual(active["paper_interpretation_s4_score"], 12.5)

    def test_initial_distance_inside_boundary_uses_eps_and_clips(self):
        scores = _calculate(
            stage1_initial_distance=500.0,
            stage1_best_distance=500.0,
        )
        self.assertEqual(scores["paper_interpretation_s1_score"], 25.0)
        self.assertAlmostEqual(scores["paper_interpretation_s2_score"], 25.0 / 3.0)

    def test_task_score_is_component_sum_and_scores_are_bounded(self):
        scores = _calculate(
            stage1_best_distance=100.0,
            stage1_success=True,
            stage2_initial_distance=1200.0,
            stage2_best_distance=600.0,
        )
        component_keys = (
            "paper_interpretation_s1_score",
            "paper_interpretation_s2_score",
            "paper_interpretation_s3_score",
            "paper_interpretation_s4_score",
        )
        components = [scores[key] for key in component_keys]
        self.assertTrue(all(0.0 <= score <= 25.0 for score in components))
        self.assertEqual(scores["paper_interpretation_task_score"], sum(components))

    def test_wrapper_adds_scores_without_changing_source_scores(self):
        benchmark = SimpleNamespace(
            rescue_distance=120.0,
            place_distance=100.0,
            stage2_success_radius=200.0,
            interaction_z_threshold=220.0,
        )
        context = {
            "injured_pose": [0.0, 0.0, 0.0],
            "stretcher_pose": [0.0, 1000.0, 0.0],
        }
        controller = ProgressTrackingController(benchmark, context)
        controller.reset([1750.0, 0.0, 0.0])
        controller.update_after_env_step([1250.0, 0.0, 0.0], carrying_now=False)

        metrics = controller.finalize()

        self.assertEqual(metrics["s1_score"], 0.0)
        self.assertEqual(metrics["s2_score"], 0.0)
        self.assertEqual(metrics["task_score"], 0.0)
        self.assertEqual(metrics["paper_interpretation_s1_score"], 12.5)
        self.assertEqual(metrics["paper_interpretation_task_score"], 12.5)

    def test_optional_scores_serialize_as_json_null(self):
        metrics = EpisodeMetrics(
            episode_id=0,
            level=0,
            point_id=0,
            success=False,
            time_cost=0.0,
            steps=0,
            collision_count=0,
            failure_reason="EXCEPTION",
            final_state="FAILED",
        )

        payload = json.loads(
            json.dumps(ResultWriter.metrics_to_episode_record(metrics))
        )

        self.assertIsNone(payload["paper_interpretation_s1_score"])
        self.assertIsNone(payload["paper_interpretation_s2_score"])
        self.assertIsNone(payload["paper_interpretation_s3_score"])
        self.assertIsNone(payload["paper_interpretation_s4_score"])
        self.assertIsNone(payload["paper_interpretation_task_score"])

    def test_numeric_scores_survive_json_serialization(self):
        metrics = EpisodeMetrics(
            episode_id=0,
            level=0,
            point_id=0,
            success=False,
            time_cost=1.0,
            steps=1,
            collision_count=0,
            paper_interpretation_s1_score=12.5,
            paper_interpretation_s2_score=0.0,
            paper_interpretation_s3_score=0.0,
            paper_interpretation_s4_score=0.0,
            paper_interpretation_task_score=12.5,
        )

        payload = json.loads(
            json.dumps(ResultWriter.metrics_to_episode_record(metrics))
        )

        self.assertEqual(payload["paper_interpretation_s1_score"], 12.5)
        self.assertEqual(payload["paper_interpretation_s2_score"], 0.0)
        self.assertEqual(payload["paper_interpretation_s3_score"], 0.0)
        self.assertEqual(payload["paper_interpretation_s4_score"], 0.0)
        self.assertEqual(payload["paper_interpretation_task_score"], 12.5)

    def test_outer_episode_exception_records_existing_failure_fields(self):
        captured = []
        result_writer = SimpleNamespace(
            append_episode_result=captured.append,
            build_level_metrics=Mock(return_value="level-metrics"),
            print_level_summary=Mock(),
        )
        benchmark = SimpleNamespace(
            task_loader=SimpleNamespace(
                get_point_count=Mock(return_value=1),
                get_level_time_limit_text=Mock(return_value="1s"),
            ),
            resume_manager=SimpleNamespace(get=Mock(return_value=None)),
            run_episode=Mock(side_effect=RuntimeError("episode failed")),
            result_writer=result_writer,
        )

        _, episodes = BenchmarkRunner(benchmark).evaluate_level(
            0,
            point_ids=[0],
            close_env=False,
        )

        self.assertEqual(len(episodes), 1)
        self.assertIs(episodes[0], captured[0])
        self.assertFalse(episodes[0].success)
        self.assertEqual(episodes[0].failure_reason, "EXCEPTION")
        self.assertEqual(episodes[0].final_state, "FAILED")
        self.assertIsNone(episodes[0].paper_interpretation_task_score)


if __name__ == "__main__":
    unittest.main()
