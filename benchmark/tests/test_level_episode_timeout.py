"""Focused tests for per-level episode timeout CLI overrides."""

from __future__ import annotations

import json
import sys
import tempfile
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

cli = import_module("core.cli")
metrics_module = import_module("core.metrics")
result_writer_module = import_module("core.result_writer")
resume_module = import_module("core.resume")
task_loader_module = import_module("core.task_loader")

create_base_parser = cli.create_base_parser
run_benchmark_from_args = cli.run_benchmark_from_args
EpisodeMetrics = metrics_module.EpisodeMetrics
ResultWriter = result_writer_module.ResultWriter
ResumeManager = resume_module.ResumeManager
TaskLoader = task_loader_module.TaskLoader


def _test_point(timeout: int) -> dict:
    return {
        "env_id": "UnrealRescue-FlexibleRoom",
        "injured_player_loc": [0, 0, 0, 0, 0, 0],
        "stretcher_loc": [1, 0, 0, 0, 0, 0],
        "agent_loc": [2, 0, 0, 0, 0, 0],
        "ambulance_loc": [3, 0, 0, 0, 0, 0],
        "timeout": timeout,
    }


def _write_level(root: Path, level: int, timeout: int) -> None:
    level_dir = root / "envs" / "setting" / "test_jsonl"
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / f"level_{level}.jsonl").write_text(
        json.dumps(_test_point(timeout)) + "\n",
        encoding="utf-8",
    )


def test_parser_reads_positionally_aligned_timeout_values() -> None:
    args = create_base_parser().parse_args(
        [
            "--levels",
            "0",
            "1",
            "2",
            "--levels-episode-timeout",
            "40",
            "60",
            "80",
        ]
    )

    assert args.levels == [0, 1, 2]
    assert args.levels_episode_timeout == [40, 60, 80]


def test_cli_passes_level_timeout_mapping_to_benchmark() -> None:
    captured = {}

    class FakeBenchmark:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.env = None

        def run_benchmark(self, **kwargs):
            captured["run"] = kwargs
            return "result"

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SimpleNamespace(
            levels=[0, 1, 2],
            levels_episode_timeout=[40, 60, 80],
            output=temp_dir,
        )
        fake_module = SimpleNamespace(RescueBenchmark=FakeBenchmark)
        with patch.object(cli.signal, "signal"), patch.dict(
            sys.modules, {"rescue_benchmark": fake_module}
        ):
            result = run_benchmark_from_args(args, Mock(), model_name="test")

    assert result == "result"
    assert captured["init"]["level_episode_timeouts"] == {0: 40, 1: 60, 2: 80}
    assert captured["run"]["levels"] == [0, 1, 2]


def test_task_loader_cli_timeout_overrides_jsonl_timeout() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_level(root, level=0, timeout=180)
        loader = TaskLoader(
            gym_rescue_root=str(root),
            fallback_env_id="UnrealRescue-FlexibleRoom",
            time_limits={0: 180},
            level_episode_timeouts={0: 40},
        )

        context = loader.build_task_context(0, 0)

    assert context["timeout"] == 40
    assert loader.get_level_time_limit_text(0) == "40s"


def test_task_loader_keeps_jsonl_timeout_without_cli_override() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _write_level(root, level=0, timeout=180)
        loader = TaskLoader(
            gym_rescue_root=str(root),
            fallback_env_id="UnrealRescue-FlexibleRoom",
            time_limits={0: 999},
        )

        context = loader.build_task_context(0, 0)

    assert context["timeout"] == 180
    assert loader.get_level_time_limit_text(0) == "180s"


def test_episode_record_persists_timeout_and_old_records_still_load() -> None:
    metrics = EpisodeMetrics(
        episode_id=0,
        level=0,
        point_id=0,
        success=False,
        time_cost=40.0,
        steps=100,
        collision_count=0,
        episode_timeout=40,
    )

    record = ResultWriter.metrics_to_episode_record(metrics)
    old_record = dict(record)
    old_record.pop("episode_timeout")
    restored = ResumeManager.record_to_episode_metrics(old_record)

    assert record["episode_timeout"] == 40
    assert restored.episode_timeout is None
