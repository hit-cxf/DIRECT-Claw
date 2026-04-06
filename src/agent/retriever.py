"""
Implement retrieval algorithm
Three-stage retrieval: Fast indexing → Precise interval scoring → Ranking and returning
"""

import numpy as np
from copy import deepcopy
import logging
import os
from pathlib import Path
import pickle

from ..features.features import VideoFeatures, ShotFeatures
from .editing_utils import Score, ScoreConfig, ShotCandidate
from ..eval.metrics import get_semantic_score

logger = logging.getLogger('retriever')

# ==================== Stage 1: Fast Retrieval ====================

def get_candidates_pool(
    video_features: dict[str, VideoFeatures],
    query_embed: np.ndarray,
    prev_shots: list[ShotCandidate],
    pool_size,
) -> list[ShotCandidate]:
    """Fast retrieval of candidate pool (based on CLIP similarity) - Internal implementation
    
    Args:
        video_features: Dictionary of video features
        query_embed: Query embedding vector
        pool_size: Size of the candidate pool
    
    Returns:
        List of candidate shots (sorted by CLIP score)
    """
    candidates_pool = []
    
    for video_id, features in video_features.items():
        for shot_idx, shot in enumerate(features.shots):

            shot_embed = features.get_clip_embed(shot["start"], shot["end"])
            prompt_score = get_semantic_score(query_embed, shot_embed)
            
            candidates_pool.append(ShotCandidate(
                video_id=video_id,
                shot_idx=shot_idx,
                start_frame=shot["start"],
                end_frame=shot["end"],
                score=Score(prompt=prompt_score, semantic=0.0, saliency=0.0, motion=0.0, energy=0.0, combined=prompt_score)
            ))

    banned_shots = set([(shot.video_id, shot.shot_idx) for shot in prev_shots])
    filtered_pool = [
        c for c in candidates_pool 
        if not ((c.video_id, c.shot_idx) in banned_shots)
    ]
    filtered_pool.sort(key=lambda x: x.score.prompt, reverse=True)
    return filtered_pool[:pool_size]


# ==================== Stage 2: Precise Scoring ====================

def stage2_precise_scoring(video_features: dict[str, VideoFeatures],
                           candidates: list[ShotCandidate],
                           query_features: ShotFeatures | None,
                           score_config: ScoreConfig,
                           next_shot_len: int,
                           frame_step: int = 4) -> list[ShotCandidate]:
    """Precise scoring: Calculate weighted saliency/semantic/motion scores
    
    Args:
        video_features: Dictionary of video features
        candidates: List of candidate shots
        query_features: Features of the query shot
        score_config: Scoring configuration object
        next_shot_len: Sliding window length
    
    Returns:
        List of candidates with updated scores
    """
    result_candidates = []
    
    for candidate in candidates:
        candidate = deepcopy(candidate)
        
        features = video_features[candidate.video_id]
        
        candidate_len = candidate.end_frame - candidate.start_frame + 1
        if candidate_len > next_shot_len:
            # Sliding window search for the optimal position
            best_score = -1.0
            best_start = candidate.start_frame
            
            for start in range(candidate.start_frame, candidate.end_frame - next_shot_len + 2, frame_step):
                end = start + next_shot_len - 1
                interval_features = features.get_shot_features(start, end)
                
                # Use get_score() to calculate the combined score
                score = score_config.get_score(query_features, interval_features)
                
                if score.combined > best_score:
                    best_score = score.combined
                    best_start = start
                    candidate.score = score
            
            # Update the frame range of the candidate
            candidate.start_frame = best_start
            candidate.end_frame = best_start + next_shot_len - 1
            result_candidates.append(candidate)
    
    return result_candidates


# ==================== Stage 3: Ranking and Returning ====================

def stage3_rank_and_return(candidates: list[ShotCandidate],
                           top_k: int = 10) -> list[ShotCandidate]:
    """Rank and return Top-K candidates
    
    Args:
        candidates: List of candidate shots
        top_k: Number of results to return
    
    Returns:
        Top-K candidate shots
    """
    candidates.sort(key=lambda x: x.score.combined, reverse=True)
    return candidates[:top_k]

def retrieve(video_features: dict[str, VideoFeatures],
             candidates_pool: list[ShotCandidate],
             prev_shots: list[ShotCandidate],
             score_config: ScoreConfig,
             next_shot_len: int,
             top_k: int = 100) -> list[ShotCandidate]:
    """Complete process for retrieving similar shots
    
    Args:
        video_features: Dictionary of video features
        candidates_pool: Candidate pool (obtained via get_candidates_pool())
        prev_shots: Current list of shots
        score_config: Scoring configuration object
        next_shot_len: Sliding window length
        top_k: Number of Top-K results to return
    
    Returns:
        Top-K similar candidate shots
    """
    
    logger.debug(f"Retrieving similar shots, current number of shots: {len(prev_shots)}, candidate pool size: {len(candidates_pool)}, required shot length: {next_shot_len}, returning Top-{top_k}")

    if not candidates_pool:
        logger.warning("No candidate shots found")
        return []
    
    if not prev_shots or prev_shots[-1].video_id == "BLACK_SCREEN":
        logger.debug("prev_shots is empty, using None for retrieval")
        query_features = None
        filtered_pool = candidates_pool
    else:
        # Use the last shot as the query feature
        query_shot = prev_shots[-1]
        query_features = video_features[query_shot.video_id].get_shot_features(
            query_shot.start_frame, query_shot.end_frame
        )
        
        banned_shots = set([(shot.video_id, shot.shot_idx) for shot in prev_shots])
        filtered_pool = [
            c for c in candidates_pool 
            if not ((c.video_id, c.shot_idx) in banned_shots)
        ]
    
    # Stage 2: Precise Scoring
    ranked = stage2_precise_scoring(video_features, filtered_pool.copy(), query_features,
                                    score_config, next_shot_len)
    
    # Stage 3: Ranking and Returning
    result = stage3_rank_and_return(ranked, top_k)
    logger.debug(f"Returning Top-{top_k} results")

    # Add a black screen shot as a fallback if no results are found
    if not result:
        logger.warning("No suitable candidate shots found, adding a black screen shot as fallback")
        black_screen_candidate = ShotCandidate(
            video_id="BLACK_SCREEN",
            shot_idx=-1,
            start_frame=0,
            end_frame=next_shot_len - 1,
            score=Score(0, 0, 0, 0, 0, 0)
        )
        result.append(black_screen_candidate)

    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


