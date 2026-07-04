"""LLM-based validation for edited video segments."""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, unquote

from .interaction_utils import SegmentGuidance, prompt_manager
from .editing_utils import Score
from .llm_interface import Message, chat_with_llm

logger = logging.getLogger("director_validation")


@dataclass
class BeamCandidate:
    """One candidate video from beam search."""

    video_path: str
    score: Score


@dataclass
class ValidationResult:
    """Outcome of an LLM validation call."""

    is_success: bool
    verdict: str
    issues: list[str]
    suggestions: list[str]
    best_candidate: int | None
    raw_response: str


MIN_DASHSCOPE_VIDEO_SECONDS = 2.0


def _candidate_local_path(video_path: str) -> Path | None:
    parsed = urlparse(video_path)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if not parsed.scheme:
        return Path(video_path)
    return None


def _video_duration_seconds(video_path: str) -> float | None:
    local_path = _candidate_local_path(video_path)
    if local_path is None or not local_path.exists():
        return None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(local_path),
            ],
            text=True,
        ).strip()
        return float(out)
    except Exception as exc:
        logger.debug("Unable to probe candidate duration for %s: %s", video_path, exc)
        return None


def _top_candidate_fallback(reason: str) -> ValidationResult:
    logger.warning("Skipping LLM validation: %s", reason)
    return ValidationResult(
        is_success=True,
        verdict=reason,
        issues=[reason],
        suggestions=[],
        best_candidate=0,
        raw_response=json.dumps({
            "success": True,
            "best_candidate": 0,
            "verdict": reason,
        }),
    )


def _build_validation_prompt(guidance: SegmentGuidance, candidates: list[BeamCandidate]) -> str:
    """Compose the textual instruction for the LLM across multiple beam candidates."""

    guidance_block = (
        f"Retrieval Query: {guidance.retrieval_query}\n"
        f"Editing Mode: {guidance.weight_profile}\n"
        f"Pacing Control (beat duration for each shot): {guidance.pacing_control}\n"
    )

    candidate_lines = []
    for idx, cand in enumerate(candidates):
        score = cand.score
        candidate_lines.append(
            f"Candidate {idx}: video={cand.video_path}\n"
            f"  Automated Scores -> Retrieval Query Alignment: {score.prompt:.2f}; Semantic Relevance: {score.semantic:.2f}; Composition Similarity: {score.saliency:.2f}; "
            f"Motion Continuity: {score.motion:.2f}; Energy Alignment: {score.energy:.2f}; Combined Score: {score.combined:.2f}"
        )

    instruction = prompt_manager.get_prompt(
        "Director", "Validation",
        guidance_block=guidance_block,
        candidate_lines=candidate_lines,
    )
    
    return instruction


def _parse_json_response(response: str) -> dict:
    """Best-effort JSON extraction from the LLM response."""

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def validate_edit(
    guidance: SegmentGuidance,
    candidates: list[BeamCandidate],
    fps: int = 2,
) -> ValidationResult:
    """Use the LLM to select the best beam candidate or fail all if off-query.

    Args:
        guidance: Guidance for this segment.
        candidates: Beam candidates with their edit metadata and video paths.
        fps: Sampling fps when sending video to the LLM.
    Returns:
        ValidationResult with parsed LLM feedback.
    """

    if not candidates:
        raise ValueError("No beam candidates provided for validation.")

    prompt_text = _build_validation_prompt(guidance, candidates)

    logger.debug("Validation prompt prepared for LLM (beam mode).")

    messages = [Message(role="user").add_text(prompt_text).to_dict()]
    video_messages_added = 0
    for idx, cand in enumerate(candidates):
        duration = _video_duration_seconds(cand.video_path)
        if duration is not None and duration < MIN_DASHSCOPE_VIDEO_SECONDS:
            logger.warning(
                "Skipping Candidate %s video validation because it is too short for DashScope: %.3fs < %.3fs (%s)",
                idx, duration, MIN_DASHSCOPE_VIDEO_SECONDS, cand.video_path,
            )
            continue
        messages.append(
            Message(role="user")
            .add_text(f"Candidate {idx} video")
            .add_video(cand.video_path, fps=fps)
            .to_dict()
        )
        video_messages_added += 1

    if video_messages_added == 0:
        return _top_candidate_fallback(
            "all beam candidate videos are shorter than DashScope video minimum; selected top-scoring candidate"
        )

    try:
        response = chat_with_llm(messages)
    except RuntimeError as exc:
        if "video file is too short" in str(exc):
            return _top_candidate_fallback(
                "DashScope rejected a validation video as too short; selected top-scoring candidate"
            )
        raise

    parsed = _parse_json_response(response)

    is_success = bool(parsed.get("success", False))
    verdict = str(parsed.get("verdict", "")) or response.strip()
    issues = parsed.get("issues") or []
    suggestions = parsed.get("suggestions") or []

    if not isinstance(issues, list):
        issues = [str(issues)]
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]

    best_candidate = parsed.get("best_candidate")
    if best_candidate is not None:
        try:
            best_candidate = int(best_candidate)
        except (TypeError, ValueError):
            best_candidate = None

    if best_candidate is not None and not (0 <= best_candidate < len(candidates)):
        best_candidate = None

    result = ValidationResult(
        is_success=is_success,
        verdict=verdict,
        issues=[str(item) for item in issues],
        suggestions=[str(item) for item in suggestions],
        best_candidate=best_candidate,
        raw_response=response,
    )

    logger.info(
        "LLM validation %s. Verdict: %s", "passed" if result.is_success else "failed", result.verdict
    )

    return result

