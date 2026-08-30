"""Input models and validation for the RescueBench teleport capture probe."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from benchmark.core.task_loader import TaskLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
GYM_RESCUE_ROOT = REPO_ROOT / "gym_rescue"
TIME_LIMITS = {0: 180, 1: 180, 2: 240, 3: 300, 4: 300}
DEFAULT_RESOLUTION = (640, 640)
DEFAULT_SAMPLE_DELAYS = (0.0, 0.2, 0.5, 1.0, 2.0)
MAX_SAMPLE_DELAY_SECONDS = 60.0
POSE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


@dataclass(frozen=True)
class SceneDefinition:
    """Describe one canonical benchmark scene and its runtime IDs."""

    index: int
    name: str
    env_ids: tuple[str, ...]


SCENES = (
    SceneDefinition(1, "FlexibleRoom", ("UnrealRescue-FlexibleRoom",)),
    SceneDefinition(
        2,
        "SuburbNeighborhood_Day",
        (
            "UnrealRescue-SuburbNeighborhood_Day",
            "UnrealRescue-SuburbNeighborhood_Day_dooropen",
        ),
    ),
    SceneDefinition(3, "Forglar_Map", ("UnrealRescue-Forglar_Map",)),
    SceneDefinition(4, "HongKongStreet", ("UnrealRescue-HongKongStreet",)),
    SceneDefinition(5, "DesertMap", ("UnrealRescue-DesertMap",)),
    SceneDefinition(6, "DowntownWest", ("UnrealRescue-DowntownWest",)),
    SceneDefinition(7, "Tokyo", ("UnrealRescue-Tokyo",)),
)
SCENE_BY_ENV_ID = {env_id: scene for scene in SCENES for env_id in scene.env_ids}


class ProbeError(RuntimeError):
    """Report invalid probe input or runtime state."""


@dataclass(frozen=True)
class PoseSpec:
    """Store one requested actor pose from the external manifest."""

    pose_id: str
    location_xyz: tuple[float, float, float]
    rotation_rpy_deg: tuple[float, float, float]
    note: str = ""

    @property
    def pose(self) -> list[float]:
        """Return the pose in RescueBench six-value order."""

        return [*self.location_xyz, *self.rotation_rpy_deg]


@dataclass(frozen=True)
class TaskSelection:
    """Store a resolved zero-based task point and its source row."""

    level: int
    point_id: int
    source_path: Path
    source_line: int
    raw_point: dict[str, Any]
    task_context: dict[str, Any]
    scene: SceneDefinition


@dataclass(frozen=True)
class RenderSettings:
    """Store validated rendering settings for the probe runtime."""

    resolution: tuple[int, int]
    offscreen: bool
    quality: int


@dataclass(frozen=True)
class ProbeInputs:
    """Collect validated inputs shared by probe setup and execution."""

    selection: TaskSelection
    poses_path: Path
    poses: tuple[PoseSpec, ...]
    output_root: Path
    render: RenderSettings
    sample_delays: tuple[float, ...]


def _parse_vector(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProbeError(f"{field_name} must be a JSON array with exactly 3 numbers")

    parsed = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProbeError(f"{field_name}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ProbeError(f"{field_name}[{index}] must be finite")
        parsed.append(number)
    return cast(tuple[float, float, float], tuple(parsed))


def load_pose_manifest(path: Path) -> tuple[PoseSpec, ...]:
    """Load and strictly validate an external JSON pose manifest."""

    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ProbeError(f"Pose manifest does not exist: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8-sig") as stream:
            manifest = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            f"Pose manifest is not valid JSON: {manifest_path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ProbeError("Pose manifest root must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ProbeError("Pose manifest schema_version must be 1")
    if manifest.get("coordinate_frame") != "unreal_world":
        raise ProbeError("Pose manifest coordinate_frame must be 'unreal_world'")

    raw_poses = manifest.get("poses")
    if not isinstance(raw_poses, list) or not raw_poses:
        raise ProbeError("Pose manifest poses must be a non-empty JSON array")

    poses = []
    seen_ids = set()
    for index, raw_pose in enumerate(raw_poses):
        prefix = f"poses[{index}]"
        if not isinstance(raw_pose, dict):
            raise ProbeError(f"{prefix} must be a JSON object")

        pose_id = raw_pose.get("id")
        if not isinstance(pose_id, str) or not POSE_ID_PATTERN.fullmatch(pose_id):
            raise ProbeError(f"{prefix}.id must match {POSE_ID_PATTERN.pattern!r}")
        if pose_id in seen_ids:
            raise ProbeError(f"Duplicate pose id: {pose_id}")
        seen_ids.add(pose_id)

        note = raw_pose.get("note", "")
        if not isinstance(note, str):
            raise ProbeError(f"{prefix}.note must be a string when provided")

        poses.append(
            PoseSpec(
                pose_id=pose_id,
                location_xyz=_parse_vector(
                    raw_pose.get("location_xyz"), f"{prefix}.location_xyz"
                ),
                rotation_rpy_deg=_parse_vector(
                    raw_pose.get("rotation_rpy_deg"),
                    f"{prefix}.rotation_rpy_deg",
                ),
                note=note,
            )
        )
    return tuple(poses)


def scene_for_env_id(env_id: str) -> SceneDefinition:
    """Resolve an environment ID to one canonical scene."""

    try:
        return SCENE_BY_ENV_ID[env_id]
    except KeyError as exc:
        raise ProbeError(
            f"Test point uses env_id outside the seven-scene catalog: {env_id}"
        ) from exc


def load_level_points(
    level: int,
    gym_rescue_root: Path = GYM_RESCUE_ROOT,
) -> tuple[dict[str, Any], ...]:
    """Load all configured points for one benchmark level."""

    if level not in TIME_LIMITS:
        raise ProbeError(f"Level must be one of {sorted(TIME_LIMITS)}: {level}")
    loader = TaskLoader(
        gym_rescue_root=str(gym_rescue_root),
        fallback_env_id="",
        time_limits=TIME_LIMITS,
    )
    points = loader.load_level_test_points(level)
    if not points:
        raise ProbeError(f"No test points found for level {level}")
    return tuple(points)


def load_task_selection(
    level: int,
    point_id: int,
    gym_rescue_root: Path = GYM_RESCUE_ROOT,
) -> TaskSelection:
    """Resolve a zero-based point ID using the benchmark TaskLoader."""

    if level not in TIME_LIMITS:
        raise ProbeError(f"Level must be one of {sorted(TIME_LIMITS)}: {level}")
    if point_id < 0:
        raise ProbeError(f"Point ID must be non-negative: {point_id}")

    loader = TaskLoader(
        gym_rescue_root=str(gym_rescue_root),
        fallback_env_id="",
        time_limits=TIME_LIMITS,
    )
    points = load_level_points(level, gym_rescue_root)
    if point_id >= len(points):
        raise ProbeError(
            f"Point ID {point_id} is out of range for level {level}; "
            f"valid range is 0-{len(points) - 1}"
        )

    raw_point = points[point_id]
    task_context = loader.build_task_context(level, point_id)
    env_id = str(task_context["env_id"])
    source_path = (
        gym_rescue_root / "envs" / "setting" / "test_jsonl" / f"level_{level}.jsonl"
    ).resolve()
    return TaskSelection(
        level=level,
        point_id=point_id,
        source_path=source_path,
        source_line=point_id + 1,
        raw_point=raw_point,
        task_context=task_context,
        scene=scene_for_env_id(env_id),
    )


def validate_sample_delays(values: Iterable[float]) -> tuple[float, ...]:
    """Validate strictly increasing finite capture delays."""

    delays = tuple(float(value) for value in values)
    if not delays:
        raise ProbeError("At least one sample delay is required")
    if any(not math.isfinite(value) for value in delays):
        raise ProbeError("Sample delays must be finite")
    if any(value < 0 for value in delays):
        raise ProbeError("Sample delays must be non-negative")
    if any(value > MAX_SAMPLE_DELAY_SECONDS for value in delays):
        raise ProbeError(
            f"Sample delays cannot exceed {MAX_SAMPLE_DELAY_SECONDS:g} seconds"
        )
    if any(delays[index] <= delays[index - 1] for index in range(1, len(delays))):
        raise ProbeError("Sample delays must be strictly increasing")
    return delays


def validate_resolution(values: Sequence[int]) -> tuple[int, int]:
    """Validate the requested width and height."""

    if len(values) != 2 or any(value <= 0 for value in values):
        raise ProbeError("Resolution must contain two positive integers: WIDTH HEIGHT")
    return int(values[0]), int(values[1])


def create_run_directory(
    output_root: Path,
    level: int,
    point_id: int,
    timestamp: str | None = None,
) -> Path:
    """Create a unique run directory without overwriting prior captures."""

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"teleport-probe-{stamp}-L{level}-P{point_id}"
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        (candidate / "frames").mkdir()
        return candidate
    raise ProbeError(f"Could not create a unique run directory under: {root}")


def format_pose(pose: Sequence[float]) -> str:
    """Format a numeric pose for terminal output."""

    return "[" + ", ".join(f"{float(value):.6f}" for value in pose) + "]"


def configured_pose(raw_point: dict[str, Any], key: str) -> list[float]:
    """Read one six-value configured pose from a task point."""

    value = raw_point.get(key)
    if not isinstance(value, list) or len(value) != 6:
        raise ProbeError(f"Test point field {key!r} must contain a 6D pose")
    return [float(item) for item in value]
