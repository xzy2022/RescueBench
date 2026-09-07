"""Opt-in observation, inference and action records using one monotonic clock."""

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def json_value(value):
    """Convert numpy values without losing action structure."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class NomadDiagnostics:
    """One episode's records; image writes occur after the action returns."""

    def __init__(self, benchmark, level, point_id, episode_id):
        self.env = benchmark.env.unwrapped
        self.root = (
            Path(benchmark.output_dir)
            / "_nomad_diagnostics"
            / (f"level_{level}/point_{point_id}/episode_{episode_id:04d}")
        )
        self.root.mkdir(parents=True, exist_ok=False)
        for folder in ("observations", "goals", "model_inputs"):
            (self.root / folder).mkdir()
        self.pending = []
        self.history = []
        self.goals = set()
        self.record = {}
        self.previous_send = None
        self.tensor_every = benchmark.diagnostic_tensor_every
        agent = benchmark.agent
        cam_id = self.env.cam_list[self.env.protagonist_id]
        revision = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata = {
            "schema_version": 1,
            "time_basis": "perf_counter_seconds",
            "pose_order": ["x", "y", "z", "rotation0", "yaw", "rotation2"],
            "position_unit": "cm",
            "camera_horizontal_fov_deg": float(
                self.env.unrealcv.client.request(f"vget /camera/{cam_id}/fov")
            ),
            "model_params": agent.model_params,
            "argv": sys.argv,
            "git_head": revision.stdout.strip(),
            "git_error": revision.stderr.strip(),
            "waypoint_idx": agent.waypoint_idx,
            "conversion": {
                "max_v": agent._deploy_max_v,
                "rate_hz": agent._deploy_rate_hz,
            },
            "tensor_capture_every": benchmark.diagnostic_tensor_every,
            "tensor_preview": (
                "joint min/max linear display; exact float tensors stored in npy"
            ),
        }
        (self.root / "run.json").write_text(
            json.dumps(metadata, default=json_value, indent=2), encoding="utf-8"
        )

    def pose(self):
        """Read live poses; actor and camera requests have separate time bounds."""
        index = self.env.protagonist_id
        start = time.perf_counter()
        actor = self.read_pose(f"object/{self.env.player_list[index]}")
        actor_end = time.perf_counter()
        camera = self.read_pose(f"camera/{self.env.cam_list[index]}")
        return {
            "actor": actor,
            "camera": camera,
            "read_start_s": start,
            "actor_read_end_s": actor_end,
            "read_end_s": time.perf_counter(),
        }

    def read_pose(self, target):
        """Direct UnrealCV queries avoid cached get-pose implementations."""
        values = []
        for component in ("location", "rotation"):
            reply = self.env.unrealcv.client.request(f"vget /{target}/{component}")
            values.extend(float(value) for value in reply.split())
        if len(values) != 6:
            raise ValueError(f"Unexpected live pose for {target}: {values}")
        return values

    def begin(self, step, observation, observation_context):
        """Retain the actual act() input, not the post-action render frame."""
        self.pending = []
        image_path = f"observations/frame_{step:06d}.png"
        rgb = np.asarray(observation)[..., :3][..., ::-1].copy()
        self.pending.append((image_path, Image.fromarray(rgb.astype(np.uint8))))
        self.record = {
            "step_id": step,
            "observation": image_path,
            "observation_context": observation_context,
            "inference_before_pose": self.pose(),
        }
        self.record["act_start_s"] = time.perf_counter()

    def model_input(self, goal, source, count, reset_history):
        """Record actual padded context references and the chosen goal image."""
        if reset_history:
            self.history = []
        self.history.append(
            {
                "step_id": self.record["step_id"],
                "image": self.record["observation"],
                "context": self.record["observation_context"],
            }
        )
        self.history = self.history[-count:]
        context = [self.history[0]] * (count - len(self.history)) + self.history
        digest = hashlib.sha256(str(goal.size).encode() + goal.tobytes()).hexdigest()
        path = f"goals/{digest}.png"
        if digest not in self.goals:
            self.pending.append((path, goal.copy()))
            self.goals.add(digest)
        self.record.update(history=context, goal=path, goal_source=source)

    def tensors(self, observation, goal):
        """Capture tensors already produced by the real preprocessing path."""
        self.record["model_tensor_shapes"] = [list(observation.shape), list(goal.shape)]
        if (self.record["step_id"] - 1) % self.tensor_every:
            return
        for name, tensor in [("observation", observation), ("goal", goal)]:
            path = f"model_inputs/step_{self.record['step_id']:06d}_{name}.npy"
            self.pending.append((path, tensor.detach().cpu().numpy().copy()))
            self.record[f"{name}_tensor"] = path

    def result(self, waypoints, index, chosen, error, started):
        """Associate the full prediction and fallback status with the input."""
        self.record.update(
            waypoints=waypoints,
            selected_index=index,
            waypoint=chosen,
            raw_bearing_deg=math.degrees(math.atan2(chosen[1], chosen[0])),
            inference_failed=error is not None,
            error=error,
            model_start_s=started,
            model_end_s=time.perf_counter(),
        )

    def after_inference(self, nav_action, extra_info):
        """Measure the pose after inference, including old-command movement."""
        self.record.update(
            act_end_s=time.perf_counter(),
            nav_action=nav_action,
            converted_move=nav_action[0],
            agent_info=extra_info,
            inference_after_pose=self.pose(),
        )

    def before_send(self, final_action, env_action):
        """Snapshot the actual command about to enter env.step."""
        self.record.update(
            final_action=final_action, env_action=env_action, send_pose=self.pose()
        )
        now = time.perf_counter()
        self.record["send_start_s"] = now
        self.record["actual_send_interval_s"] = (
            None if self.previous_send is None else now - self.previous_send
        )
        self.previous_send = now

    def finish(self, observation_context):
        """Flush each completed step, accounting for diagnostic disk-write time."""
        self.record["after_step"] = observation_context
        start = time.perf_counter()
        for path, data in self.pending:
            if path.endswith(".npy"):
                np.save(str(self.root / path), data)
            else:
                data.save(self.root / path)
        self.record["artifact_write_duration_s"] = time.perf_counter() - start
        with (self.root / "steps.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(self.record, default=json_value) + "\n")
