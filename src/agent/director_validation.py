"""LLM-based validation for edited video segments."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

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
    for idx, cand in enumerate(candidates):
        messages.append(
            Message(role="user")
            .add_text(f"Candidate {idx} video")
            .add_video(cand.video_path, fps=fps)
            .to_dict()
        )

    response = chat_with_llm(messages)

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

