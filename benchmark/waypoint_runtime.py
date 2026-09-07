"""Run independent pose, action-response and waypoint experiments in RescueBench."""

import importlib
import json
import math
import time

from benchmark.core.env_manager import EnvManager
from benchmark.teleport_probe_data import load_task_selection
from benchmark.vint_waypoint_control import waypoint_to_move
from benchmark.waypoint_control import (
    local_xy,
    waypoint_action,
    world_waypoints,
    wrap_degrees,
)


def pose_difference(actual, reference):
    """Subtract two six-value poses and wrap all three rotation differences."""
    return [
        *(actual[index] - reference[index] for index in range(3)),
        *(wrap_degrees(actual[index] - reference[index]) for index in range(3, 6)),
    ]


def cached_actor_pose(value, actor_id):
    """Extract one actor pose from a reset/cache collection when available."""
    try:
        pose = value[actor_id]
        if len(pose) != 6:
            return None
        return [float(item) for item in pose]
    except (IndexError, KeyError, TypeError):
        return None


class Experiment:
    """Own one environment and a flushed timeline of commands and fresh poses."""

    def __init__(self, plan, output):
        self.plan = plan
        self.output = output
        self.manager = EnvManager(tuple(plan["resolution"]), offscreen=True)
        self.base = None
        self.clock = time.perf_counter()
        self.stream = None
        self.active_move = [0.0, 0.0]
        self.active_head = 0
        self.command_id = 0
        self.sample_count = 0
        self.names = []
        self.camera_id = None
        self.last_sample = None
        self.task_context = None

    def now(self):
        """Return elapsed wall-clock seconds, not simulated time."""
        return time.perf_counter() - self.clock

    def record(self, event, **fields):
        """Flush each event so an interrupted experiment retains its timeline."""
        row = {"event": event, "time_s": self.now(), **fields}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        return row

    def start(self, prepare_action=True):
        """Create the normal Mixed environment without a model or task controller."""
        # Imports stay here so help, planning and reporting work without UE/Gym.
        np = importlib.import_module("numpy")
        np.bool8 = np.bool_
        importlib.import_module("gym_rescue")
        self.stream = (self.output / "samples.jsonl").open("w", encoding="utf-8")
        selection = load_task_selection(self.plan["level"], self.plan["point_id"])
        context = dict(selection.task_context)
        if "start_pose" in self.plan:
            context["agent_pose"] = self.plan["start_pose"]
        self.task_context = context
        self.record("task", task_context=context)
        self.manager.ensure_env(context["env_id"], self.plan["level"])
        self.manager.apply_task_context(context)
        _, reset_info = self.manager.env.reset()
        self.base = self.manager.env.unwrapped
        actor_id = self.base.protagonist_id
        self.names = [self.base.player_list[actor_id]]
        self.camera_id = self.base.cam_list[actor_id]
        self.record(
            "objects",
            actor=self.names[0],
            camera_id=self.camera_id,
            camera_configuration=self.base.agents[self.names[0]],
            requested_start_pose=context["agent_pose"],
        )
        reset_info_pose = cached_actor_pose(
            reset_info.get("Pose", reset_info.get("pose")), actor_id
        )
        reset_cache_pose = cached_actor_pose(self.base.obj_poses, actor_id)
        fresh_after_reset = self.sample("post_reset_fresh")
        reset_comparison = {
            "requested_start_pose": [float(item) for item in context["agent_pose"]],
            "reset_info_pose": reset_info_pose,
            "reset_cache_pose": reset_cache_pose,
            "fresh_after_reset_pose": fresh_after_reset["actor_pose"],
        }
        for name in ("reset_info_pose", "reset_cache_pose", "fresh_after_reset_pose"):
            value = reset_comparison[name]
            reset_comparison[f"{name}_error"] = (
                pose_difference(value, reset_comparison["requested_start_pose"])
                if value is not None
                else None
            )
        self.record("reset_comparison", **reset_comparison)
        if prepare_action:
            self.send([0.0, 0.0], "initial_stop")
            self.observe(self.plan["settle_s"], "settle")
        return self.sample("origin"), reset_comparison

    def head_value(self, head_index):
        """Resolve the integer Mixed head action through the active scene config."""
        choices = self.base.agents[self.names[0]]["head_action"]
        if head_index < 0 or head_index >= len(choices):
            raise ValueError(
                f"head_index {head_index} is outside the configured range "
                f"0-{len(choices) - 1}"
            )
        return [float(value) for value in choices[head_index]]

    def send(self, move, label, head_index=0):
        """Use the same env.step Mixed action boundary used by benchmark agents."""
        np = importlib.import_module("numpy")

        self.command_id += 1
        started = self.now()
        configured_head = self.head_value(head_index)
        self.record(
            "command_start",
            command_id=self.command_id,
            label=label,
            move=move,
            head_index=head_index,
            configured_head_rotation=configured_head,
        )
        _, _, terminated, truncated, _ = self.manager.env.step(
            [(np.asarray(move, dtype=np.float32), head_index, 0)]
        )
        self.active_move = list(move)
        self.active_head = head_index
        self.record(
            "command_end",
            command_id=self.command_id,
            label=label,
            move=move,
            head_index=head_index,
            configured_head_rotation=configured_head,
            started_s=started,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if terminated or truncated:
            raise RuntimeError(
                "Environment terminated/truncated; inspect samples.jsonl"
            )
        return configured_head

    def sample(self, label, **fields):
        """Read actor and camera, without requesting images or using pose caches."""
        started = self.now()
        actors, cameras, _, _, _ = self.base.unrealcv.get_pose_img_batch(
            self.names, [self.camera_id], [True, False, False, False]
        )
        self.sample_count += 1
        actor = [float(value) for value in actors[0]]
        camera = [float(value) for value in cameras[0]]
        self.last_sample = self.record(
            "sample",
            read_started_s=started,
            label=label,
            actor_pose=actor,
            camera_pose=camera,
            camera_offset_local_xy_cm=local_xy(actor, camera[:2]),
            camera_offset_z_cm=camera[2] - actor[2],
            camera_yaw_offset_deg=wrap_degrees(camera[4] - actor[4]),
            command_id=self.command_id,
            active_move=self.active_move,
            active_head_index=self.active_head,
            active_head_rotation=self.head_value(self.active_head),
            **fields,
        )
        return self.last_sample

    def observe(self, seconds, label, repeat_move=None, period_s=None, **fields):
        """Sample until a wall-clock deadline; optionally resend at sample rate."""
        deadline = self.now() + seconds
        period = self.plan["period_s"] if period_s is None else period_s
        first_sample = True
        samples = []
        while self.now() < deadline:
            tick = self.now()
            if repeat_move is not None and not first_sample:
                self.send(repeat_move, label)
            samples.append(self.sample(label, **fields))
            first_sample = False
            time.sleep(max(0.0, min(deadline, tick + period) - self.now()))
        return samples

    def run_pose(self, origin, reset_comparison):
        """Finish the static baseline and retain reset cache versus hard-read data."""
        self.observe(self.plan["observe_s"], "stationary")
        final = self.sample("final")
        return {
            "reset_comparison": reset_comparison,
            "origin_pose": origin["actor_pose"],
            "final_pose": final["actor_pose"],
            "static_pose_delta": pose_difference(
                final["actor_pose"], origin["actor_pose"]
            ),
        }

    def apply_actor_pose(self, case_type, case_id, requested_pose):
        """Set one controlled Actor pose and collect its settled hard-read result."""
        label = f"{case_type}_{case_id}"
        self.send([0.0, 0.0], f"{label}_stop")
        started = self.now()
        self.record(
            "pose_request",
            case_type=case_type,
            case_id=case_id,
            requested_actor_pose=requested_pose,
        )
        self.base.unrealcv.set_obj_rotation(self.names[0], requested_pose[3:])
        self.base.unrealcv.set_obj_location(self.names[0], requested_pose[:3])
        self.record(
            "pose_request_complete",
            case_type=case_type,
            case_id=case_id,
            requested_actor_pose=requested_pose,
            started_s=started,
        )
        self.observe(self.plan["case_observe_s"], label)
        actual = self.sample(f"{label}_final")
        result = {
            "id": case_id,
            "requested_actor_pose": requested_pose,
            "actual_actor_pose": actual["actor_pose"],
            "pose_error": pose_difference(actual["actor_pose"], requested_pose),
            "camera_pose": actual["camera_pose"],
            "camera_offset_local_xy_cm": actual["camera_offset_local_xy_cm"],
            "camera_offset_z_cm": actual["camera_offset_z_cm"],
            "camera_yaw_offset_deg": actual["camera_yaw_offset_deg"],
        }
        self.record("pose_case_result", case_type=case_type, **result)
        return result

    def run_position(self, origin):
        """Apply configured world-X/world-Y offsets around one measured origin."""
        origin_pose = origin["actor_pose"]
        results = []
        for case in self.plan["position_cases"]:
            offset_x, offset_y = case["offset_world_xy_cm"]
            requested = [
                origin_pose[0] + offset_x,
                origin_pose[1] + offset_y,
                *origin_pose[2:],
            ]
            result = self.apply_actor_pose("position", case["id"], requested)
            result["requested_offset_world_xy_cm"] = [offset_x, offset_y]
            result["actual_offset_world_xy_cm"] = [
                result["actual_actor_pose"][0] - origin_pose[0],
                result["actual_actor_pose"][1] - origin_pose[1],
            ]
            results.append(result)
        return {"origin_pose": origin_pose, "position_cases": results}

    def run_yaw(self, origin):
        """Apply configured yaw values at one fixed measured world location."""
        origin_pose = origin["actor_pose"]
        results = []
        for case in self.plan["yaw_cases"]:
            requested = [
                *origin_pose[:3],
                origin_pose[3],
                float(case["yaw_deg"]),
                origin_pose[5],
            ]
            results.append(self.apply_actor_pose("yaw", case["id"], requested))
        return {"origin_pose": origin_pose, "yaw_cases": results}

    def run_head(self, origin):
        """Apply absolute configured head actions while retaining a fixed Actor pose."""
        origin_pose = origin["actor_pose"]
        self.apply_actor_pose("head", "fixed_actor_origin", origin_pose)
        results = []
        for case in self.plan["head_cases"]:
            case_id = case["id"]
            head_index = case["head_index"]
            configured_head = self.send(
                [0.0, 0.0], f"head_{case_id}", head_index=head_index
            )
            self.observe(self.plan["case_observe_s"], f"head_{case_id}")
            actual = self.sample(f"head_{case_id}_final")
            result = {
                "id": case_id,
                "head_index": head_index,
                "configured_head_rotation": configured_head,
                "actual_actor_pose": actual["actor_pose"],
                "actor_pose_delta_from_origin": pose_difference(
                    actual["actor_pose"], origin_pose
                ),
                "actual_camera_pose": actual["camera_pose"],
                "camera_offset_local_xy_cm": actual["camera_offset_local_xy_cm"],
                "camera_offset_z_cm": actual["camera_offset_z_cm"],
                "camera_yaw_offset_deg": actual["camera_yaw_offset_deg"],
            }
            results.append(result)
            self.record("head_case_result", **result)
        self.send([0.0, 0.0], "head_restore_neutral", head_index=0)
        return {"origin_pose": origin_pose, "head_cases": results}

    def run_actions(self, case_name, repeat, origin, reset_comparison):
        """Convert one waypoint, measure its pulse, then send zero motion."""
        case = self.plan["actions"][case_name]
        conversion = self.plan["waypoint_conversion"]
        move = waypoint_to_move(case["waypoint"], **conversion)
        before = self.sample("before_pulse")
        self.record(
            "waypoint_conversion",
            case=case_name,
            waypoint=case["waypoint"],
            normalize=conversion["normalize"],
            max_v=conversion["max_v"],
            rate_hz=conversion["rate_hz"],
            move=move,
        )
        self.send(move, case_name)
        self.observe(case["hold_s"], "pulse", move if repeat else None)
        pulse_end = self.sample("pulse_end")
        self.send([0.0, 0.0], "pulse_stop")
        self.observe(self.plan["stop_observe_s"], "after_stop")
        after = self.sample("final")
        return {
            "case": case_name,
            "repeat": repeat,
            "reset_comparison": reset_comparison,
            "operational_origin_pose": origin["actor_pose"],
            "before_pulse_pose": before["actor_pose"],
            "pulse_end_pose": pulse_end["actor_pose"],
            "final_pose": after["actor_pose"],
            "waypoint": case["waypoint"],
            "waypoint_conversion": conversion,
            "move": move,
            "pulse_displacement_local_cm": local_xy(
                before["actor_pose"], pulse_end["actor_pose"]
            ),
            "pulse_yaw_delta_deg": wrap_degrees(
                pulse_end["actor_pose"][4] - before["actor_pose"][4]
            ),
            "after_stop_displacement_cm": math.dist(
                after["actor_pose"][:2], pulse_end["actor_pose"][:2]
            ),
        }

    def follow(self, origin):
        """Track fixed points in order; end the sequence at the first timeout."""
        targets = world_waypoints(self.plan, origin["actor_pose"])
        self.record("route", origin_pose=origin["actor_pose"], targets_world_cm=targets)
        results = []
        for index, target in enumerate(targets):
            started = self.now()
            while True:
                tick = self.now()
                sample = self.sample("tracking", waypoint_index=index)
                decision = waypoint_action(
                    sample["actor_pose"], target, self.plan["controller"]
                )
                self.record(
                    "decision",
                    waypoint_index=index,
                    sample_time_s=sample["time_s"],
                    **decision,
                )
                timed_out = self.now() - started >= self.plan["waypoint_timeout_s"]
                if decision["reached"] or timed_out:
                    status = "reached" if decision["reached"] else "timeout"
                    result = {
                        "index": index,
                        "status": status,
                        "elapsed_s": self.now() - started,
                        **decision,
                    }
                    results.append(result)
                    self.record("waypoint_result", **result)
                    print(
                        f"Waypoint {index}: {status}, "
                        f"distance={decision['distance_cm']:.2f} cm",
                        flush=True,
                    )
                    self.send([0.0, 0.0], "waypoint_stop")
                    break
                self.send(decision["move"], f"waypoint_{index}")
                time.sleep(max(0.0, tick + self.plan["period_s"] - self.now()))
            if status == "timeout":
                break
        self.observe(self.plan["stop_observe_s"], "after_stop")
        final = self.sample("final")
        return {
            "waypoints": results,
            "all_reached": len(results) == len(targets)
            and all(row["status"] == "reached" for row in results),
            "final_distance_to_last_target_cm": math.dist(
                final["actor_pose"][:2], targets[-1]
            ),
        }

    def close(self):
        """Attempt a final zero-motion command, then release UE and the log."""
        try:
            if self.base is not None:
                try:
                    self.send([0.0, 0.0], "cleanup_stop")
                except Exception as exc:
                    # Cleanup must still close UE after a socket failure.
                    print(f"Cleanup stop failed: {exc}", flush=True)
        finally:
            self.manager.close_env()
            if self.stream is not None:
                self.stream.close()
