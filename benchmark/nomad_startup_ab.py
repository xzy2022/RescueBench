#!/usr/bin/env python3
"""One-shot NoMaD/UE startup-order A/B experiment."""

import os
import random
import sys
import traceback

import numpy as np

from rescue_benchmark import RescueBenchmark
from agents.factory import get_agent_from_cli_args
from agents.profiles import apply_model_profile_defaults
from core.cli import add_model_args, create_base_parser


EXIT_FIRST_INFERENCE_FAILED = 86
EXIT_NO_INFERENCE = 87


def parse_args():
    parser = create_base_parser(
        description="NoMaD / UE startup-order one-shot A/B experiment"
    )
    parser.add_argument(
        "--model",
        default="nomad",
        choices=["nomad"],
    )
    add_model_args(parser)
    apply_model_profile_defaults(parser, "nomad")

    parser.add_argument(
        "--startup-order",
        required=True,
        choices=["agent-first", "ue-first"],
        help="A=agent-first reproduces current order; B=ue-first launches UE first",
    )

    args = parser.parse_args()

    if len(args.levels) != 1:
        parser.error("This one-shot experiment requires exactly one --levels value")
    if args.point_ids is None or len(args.point_ids) != 1:
        parser.error("This one-shot experiment requires exactly one --point-ids value")
    if args.episodes != 1:
        parser.error("This one-shot experiment requires --episodes 1")

    return args


def make_benchmark(args):
    output_dir = os.path.join(os.path.abspath(args.output), args.model)

    return RescueBenchmark(
        env_id=args.env,
        agent=None,
        resolution=tuple(args.resolution),
        render=args.render,
        output_dir=output_dir,
        enable_collision_detection=not args.no_collision,
        enable_trajectory_recording=args.enable_trajectory,
        enable_path_similarity=args.enable_similarity,
        similarity_method=args.similarity_method,
        rescue_distance=args.rescue_distance,
        place_distance=args.place_distance,
        interaction_z_threshold=args.interaction_z_threshold,
        stage2_success_radius=args.stage2_success_radius,
        passthrough=args.passthrough,
        save_frame_every=args.save_frame_every,
        save_video=args.save_video,
        video_fps=args.video_fps,
        resume_jsonl=args.resume_jsonl,
        resume_skip=args.resume_skip,
        resume_append=args.resume_append,
        passthrough_env_term_geometry_sync=(
            args.passthrough_env_term_geometry_sync
        ),
        multiagent_env=args.multiagent_env,
    )


def prelaunch_ue(benchmark, level, point_id):
    """Execute the launch/init part normally performed by the first env.reset()."""

    task_context = benchmark.task_loader.build_task_context(level, point_id)

    print(
        f"[AB] Preparing UE before importing/loading NoMaD: "
        f"env={task_context['env_id']} level={level} point={point_id}",
        flush=True,
    )
    print(
        f"[AB] torch_imported_before_ue={'torch' in sys.modules}",
        flush=True,
    )

    benchmark._ensure_env(task_context["env_id"], level)
    benchmark.env_manager.apply_task_context(task_context)

    raw_env = benchmark.env.unwrapped
    if raw_env.launched:
        raise RuntimeError("UE was unexpectedly already launched")

    try:
        # This is the first-time initialization block from BaseEnv.reset(),
        # deliberately moved before NoMaD/CUDA initialization.
        raw_env.launch_ue_env()
        raw_env.init_agents()
        raw_env.init_objects()
        raw_env.launched = True
    except BaseException:
        # Handle partial startup, because BaseEnv.close() only closes when
        # raw_env.launched is already True.
        try:
            raw_env.ue_binary.close()
        except Exception:
            pass
        raise

    ue_process = getattr(raw_env.ue_binary, "env", None)
    ue_pid = getattr(ue_process, "pid", None)

    print(
        f"[AB] UE_READY pid={ue_pid}; UnrealCV connection and object init completed",
        flush=True,
    )


def make_nomad_agent(args):
    print("[AB] Constructing NOMADAgent and initializing CUDA now", flush=True)

    agent = get_agent_from_cli_args("nomad", args)

    # The normal factory silently falls back to RandomAgent on model-load errors.
    # That fallback would invalidate this experiment.
    if agent.__class__.__name__ != "NOMADAgent":
        raise RuntimeError(
            f"Expected NOMADAgent, got {agent.__class__.__name__}; "
            "model initialization failed or factory used fallback"
        )

    if getattr(agent, "device", None).type != "cuda":
        raise RuntimeError(f"Expected CUDA device, got {agent.device}")

    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.synchronize()

    print(
        "[AB] CUDA_READY "
        f"torch={torch.__version__} "
        f"torch_cuda={torch.version.cuda} "
        f"cudnn={torch.backends.cudnn.version()} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    return agent


def install_one_shot_fail_fast(agent):
    """Stop immediately after the first complete inference, success or failure."""

    original_infer_nomad = agent._infer_nomad

    def one_shot_infer(*args, **kwargs):
        try:
            distance, waypoints = original_infer_nomad(*args, **kwargs)
        except Exception as exc:
            print(
                f"[AB_RESULT] FIRST_INFERENCE_CUDA_ERROR: {exc!r}",
                flush=True,
            )
            traceback.print_exc()
            # SystemExit is not caught by VINTAgent's `except Exception`.
            raise SystemExit(EXIT_FIRST_INFERENCE_FAILED) from exc

        print(
            "[AB_RESULT] FIRST_INFERENCE_OK "
            f"distance_shape={np.asarray(distance).shape} "
            f"waypoints_shape={np.asarray(waypoints).shape}",
            flush=True,
        )
        # This is a startup probe, not a 180-second navigation evaluation.
        raise SystemExit(0)

    agent._infer_nomad = one_shot_infer


def main():
    args = parse_args()
    level = args.levels[0]
    point_id = args.point_ids[0]

    print(
        f"[AB] START order={args.startup_order} "
        f"level={level} point={point_id}",
        flush=True,
    )

    benchmark = make_benchmark(args)

    try:
        if args.startup_order == "ue-first":
            prelaunch_ue(benchmark, level, point_id)
        else:
            print(
                "[AB] A control: NoMaD/CUDA will initialize before UE",
                flush=True,
            )

        agent = make_nomad_agent(args)
        install_one_shot_fail_fast(agent)
        benchmark.agent = agent

        benchmark.run_benchmark(
            levels=args.levels,
            episodes_per_point=args.episodes,
            model_name=args.model,
            point_ids=args.point_ids,
        )

        print("[AB_RESULT] NO_INFERENCE_REACHED", flush=True)
        return EXIT_NO_INFERENCE

    finally:
        benchmark._close_env()
        print("[AB] CLEANUP_COMPLETE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())