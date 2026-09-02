"""Paper-interpretation scores derived from recorded episode distances."""

from typing import Any, Dict, Mapping

MAX_STAGE_SCORE = 25.0


def _clip_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _outer_progress_score(
    best_distance: float,
    initial_distance: float,
    stage_near_radius: float,
    eps: float,
) -> float:
    denominator = max(float(initial_distance) - float(stage_near_radius), float(eps))
    progress = 1.0 - (float(best_distance) - float(stage_near_radius)) / denominator
    return MAX_STAGE_SCORE * _clip_unit(progress)


def _inner_progress_score(
    best_distance: float,
    stage_near_radius: float,
    eps: float,
) -> float:
    denominator = max(float(stage_near_radius), float(eps))
    progress = 1.0 - float(best_distance) / denominator
    return MAX_STAGE_SCORE * _clip_unit(progress)


def calculate_paper_interpretation_scores(
    progress_metrics: Mapping[str, Any],
    *,
    stage_near_radius: float,
    eps: float,
) -> Dict[str, float]:
    """Calculate the four interpreted stage scores and their total."""
    s1_score = _outer_progress_score(
        progress_metrics["stage1_best_distance"],
        progress_metrics["stage1_initial_distance"],
        stage_near_radius,
        eps,
    )
    s2_score = _inner_progress_score(
        progress_metrics["stage1_best_distance"],
        stage_near_radius,
        eps,
    )

    if progress_metrics["stage1_success"]:
        s3_score = _outer_progress_score(
            progress_metrics["stage2_best_distance"],
            progress_metrics["stage2_initial_distance"],
            stage_near_radius,
            eps,
        )
        s4_score = _inner_progress_score(
            progress_metrics["stage2_best_distance"],
            stage_near_radius,
            eps,
        )
    else:
        s3_score = 0.0
        s4_score = 0.0

    return {
        "paper_interpretation_s1_score": s1_score,
        "paper_interpretation_s2_score": s2_score,
        "paper_interpretation_s3_score": s3_score,
        "paper_interpretation_s4_score": s4_score,
        "paper_interpretation_task_score": s1_score + s2_score + s3_score + s4_score,
    }
