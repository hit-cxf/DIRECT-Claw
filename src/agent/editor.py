"""
Implement the core functionality of the video editor to generate video segments based on segment planning.
"""

import logging
from copy import deepcopy
import numpy as np

from .retriever import ShotCandidate
from .editing_utils import ScoreConfig, EditResult
from .retriever import get_candidates_pool, retrieve

logger = logging.getLogger('editor')

def generate_segment_video(
    video_features, 
    prompt_embed: np.ndarray, 
    cut_points: list[float], 
    visual_energies: list[float],
    config: ScoreConfig, 
    prev_shots: EditResult,
    beam_size: int = 10, 
    exploration: int = 5, 
    ret_num: int = 3,
    pool_size: int = 100, 
    fps: int = 24
) -> list[EditResult]:
    """Generate video segments based on segment planning.

    Parameters:
        video_features: Dictionary of video features
        prompt_embed: Segment prompt embeddings
        cutpoints: Cut points (in seconds)
        config: Scoring weights
        prev_shots: List of previous shots
        beam_size: Beam search size
        exploration: Number of explorations
        pool_size: Candidate pool size
        fps: Frame rate
    """
    done_frames = prev_shots.total_frames
    logger.info(f"Generating video for the segment, cut points: {cut_points}, visual energies: {visual_energies}, completed frames: {done_frames}.")

    beam_state = [
        EditResult(video_candidates=[], total_frames=0, total_score=0.0)
    ]
    
    candidate_pool = get_candidates_pool(
        video_features,
        prompt_embed,
        prev_shots.video_candidates,
        pool_size,
    )

    for cut_point, visual_energy in zip(cut_points, visual_energies):
        shot_len = round(cut_point * fps) - done_frames
        config.energy_value = visual_energy
        if shot_len <= 0:
            logger.warning(f"Cut point {cut_point}s corresponds to fewer frames than completed frames, skipping this cut point.")
            continue
        new_beam_state = []
        for segment in beam_state:
            retrieved_shots = retrieve(
                video_features,
                candidate_pool,
                prev_shots.video_candidates + segment.video_candidates,
                config,
                shot_len,
                top_k=exploration
            )

            for shot in retrieved_shots:
                new_segment = deepcopy(segment)
                new_segment.append_shot(shot)
                new_beam_state.append(new_segment)

        # Sort by total score and keep the top beam_size
        new_beam_state.sort(key=lambda x: x.total_score, reverse=True)
        beam_state = new_beam_state[:beam_size]
        done_frames += shot_len
    
    beam_state = beam_state[:ret_num]
    logger.info(f"Video generation for the segment completed, generated {len(beam_state)} candidate results.")
    for idx, beam in enumerate(beam_state):
        score = beam.get_score()
        logger.debug(f"Candidate {idx}: Pro {score.prompt}; Sem {score.semantic}; Mot {score.motion}; Sal {score.saliency}; Ene {score.energy}; Total {score.combined}.")

    return beam_state

