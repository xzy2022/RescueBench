"""Run independent pose, action-response and waypoint experiments in RescueBench."""

import importlib
import json
import math
import time

from benchmark.core.env_manager import EnvManager
from benchmark.teleport_probe_data import load_task_selection
from benchmark.waypoint_control import (
    local_xy,
    waypoint_action,
    world_waypoints,
    wrap_degrees,
)


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
        self.command_id = 0
        self.sample_count = 0
        self.names = []
        self.camera_id = None
        self.last_sample = None

    def now(self):
        """Return elapsed wall-clock seconds, not simulated time."""
        return time.perf_counter() - self.clock

    def record(self, event, **fields):
        """Flush each event so an interrupted experiment retains its timeline."""
        row = {"event": event, "time_s": self.now(), **fields}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        return row

    def start(self):
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
        self.record("task", task_context=context)
        self.manager.ensure_env(context["env_id"], self.plan["level"])
        self.manager.apply_task_context(context)
        self.manager.env.reset()
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
        self.send([0.0, 0.0], "initial_stop")
        self.observe(self.plan["settle_s"], "settle")
        return self.sample("origin")

    def send(self, move, label):
        """Use the same env.step Mixed action boundary used by benchmark agents."""
        np = importlib.import_module("numpy")

        self.command_id += 1
        started = self.now()
        self.record("command_start", command_id=self.command_id, label=label, move=move)
        _, _, terminated, truncated, _ = self.manager.env.step(
            [(np.asarray(move, dtype=np.float32), 0, 0)]
        )
        self.active_move = list(move)
        self.record(
            "command_end",
            command_id=self.command_id,
            label=label,
            move=move,
            started_s=started,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if terminated or truncated:
            raise RuntimeError(
                "Environment terminated/truncated; inspect samples.jsonl"
            )

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
            **fields,
        )
        return self.last_sample

    def observe(self, seconds, label, repeat_move=None):
        """Sample until a wall-clock deadline; optionally resend at sample rate."""
        deadline = self.now() + seconds
        first_sample = True
        while self.now() < deadline:
            tick = self.now()
            if repeat_move is not None and not first_sample:
                self.send(repeat_move, label)
            self.sample(label)
            first_sample = False
            time.sleep(
                max(0.0, min(deadline, tick + self.plan["period_s"]) - self.now())
            )

    def run_actions(self, case_name, repeat):
        """Measure one selected pulse from a fresh reset, followed by zero motion."""
        case = self.plan["actions"][case_name]
        before = self.sample("before_pulse")
        self.send(case["move"], case_name)
        self.observe(case["hold_s"], "pulse", case["move"] if repeat else None)
        pulse_end = self.sample("pulse_end")
        self.send([0.0, 0.0], "pulse_stop")
        self.observe(self.plan["stop_observe_s"], "after_stop")
        after = self.sample("final")
        return {
            "case": case_name,
            "repeat": repeat,
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
