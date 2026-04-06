from typing import Optional
import numpy as np
from scipy.stats import spearmanr
from ..agent.music_utils import MusicResult
from .metrics import get_saliency_score, get_semantic_score, get_motion_score
from ..features import VideoFeatures, ShotFeatures

def get_beat_alignment_score(cut_times: list[float], beats: list[float]) -> dict:
    """
    Calculate the alignment score between video cut points and music beats.
    
    Formula: m_5(v_i, M) = exp(-||T_cut - T_beat||)
    where T_cut is the cut point time, and T_beat is the nearest music beat time.
    
    Find the best alignment by enumerating offsets.
    
    Parameters:
        cut_times: List of video cut point times (in seconds)
        beats: List of music beat times (in seconds)
        fps: Video frame rate
    
    Returns:
        A dictionary containing best_offset and score
    """
    if len(cut_times) == 0 or len(beats) == 0:
        return {
            'best_offset': 0.0,
            'score': 0.0
        }
    
    beats_array = np.array(beats)
    
    # Enumerate offsets from -0.5 seconds to 0.5 seconds
    offset_range = np.linspace(-0.5, 0.5, 41)  # 41 sampling points
    
    best_offset = 0.0
    best_score = 0.0
    
    for offset in offset_range:
        adjusted_cut_times = [t + offset for t in cut_times]
        
        scores = []
        for cut_time in adjusted_cut_times:
            # Find the nearest beat to the current cut point
            distances = np.abs(beats_array - cut_time)
            min_distance = np.min(distances)
            
            # Calculate alignment score: exp(-distance)
            score = np.exp(-min_distance)
            scores.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_offset = offset
    
    return {
        'best_offset': float(best_offset),
        'score': float(best_score)
    }

def get_energy_correspondence_score(shots: list[ShotFeatures], music_prof: MusicResult, fps: float) -> dict:
    """
    Calculate the audiovisual energy correspondence score.
    
    Formula: m_6(V, M) = corr(||F(v_i)||_{i=1}^m, RMS(M))
    Compute the correlation coefficient between video optical flow intensity and audio RMS energy.
    
    Parameters:
        shots: List of video shot features
        music_prof: Music analysis result (MusicInfo)
        fps: Video frame rate
    
    Returns:
        A dictionary containing pearson, spearman, and spearman_p
    """
    if len(shots) == 0:
        return {
            'pearson': 0.0,
            'spearman': 0.0,
            'spearman_p': 1.0
        }
    
    # Collect optical flow energy for each shot and audio RMS
    visual_energies = []
    audio_rms = []
    
    for shot in shots:
        # Video optical flow energy (precomputed in ShotFeatures)
        visual_energies.append(shot.energy_value)
        
        # Calculate audio RMS for the corresponding time period
        start_time = shot.start_frame / fps
        end_time = shot.end_frame / fps
        rms = music_prof.get_rms(start_time, end_time)
        audio_rms.append(rms)
    
    # Calculate correlation coefficients
    if len(visual_energies) < 2:
        return {
            'pearson': 0.0,
            'spearman': 0.0,
            'spearman_p': 1.0
        }
    
    # Pearson correlation coefficient
    pearson_corr = np.corrcoef(visual_energies, audio_rms)[0, 1]
    if np.isnan(pearson_corr):
        pearson_corr = 0.0
    
    # Spearman rank correlation coefficient
    spearman_corr, spearman_p = spearmanr(visual_energies, audio_rms)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
        spearman_p = 1.0
    
    return {
        'pearson': float(pearson_corr),
        'spearman': float(spearman_corr),
        'spearman_p': float(spearman_p)
    }

def get_prompt_relevance_scores(shots: list[ShotFeatures], prompt_embed: np.ndarray) -> float:
    """
    Calculate the average semantic relevance score m_1 between each shot and the user prompt.
    
    Formula: m_1(v_i, I) = cos(E(v_i), E(I))
    where E(v_i) is the CLIP embedding of the shot (average of keyframes), and E(I) is the CLIP embedding of the user prompt.
    
    Parameters:
        shots: List of video shot features
        prompt_embed: CLIP embedding of the user prompt (normalized)
    
    Returns:
        Average relevance score for all shots
    """
    if len(shots) == 0:
        return 0.0
    
    scores = []
    for shot in shots:
        score = get_semantic_score(shot.clip_embed, prompt_embed)
        scores.append(score)
    
    return float(np.mean(scores))

def evaluate_scores(features: VideoFeatures, 
                   music_prof: Optional[MusicResult] = None, 
                   fps: float = 24,
                   prompt_embed: Optional[np.ndarray] = None):
    """
    Evaluate video feature scores, including saliency, semantics, motion, music beat alignment, and prompt relevance.
    
    Parameters:
        features: Video features
        music_prof: Music analysis result (optional)
        fps: Video frame rate
        prompt_embed: CLIP embedding of the user prompt (optional, used for calculating m_1 metric)
    
    Returns:
        (prompt_relevance_avg, semantic_scores, motion_scores, saliency_scores, 
         beat_alignment_score, energy_correspondence_score)
    """
    shots: list[ShotFeatures] = []

    for curr_shot in features.shots:
        shots.append(features.get_shot_features(curr_shot["start"], curr_shot["end"]))
    
    saliency_scores = []
    semantic_scores = []
    motion_scores = []

    for i in range(len(shots) - 1):
        saliency_scores.append(get_saliency_score(shots[i].end_saliency, shots[i+1].start_saliency))
        semantic_scores.append(get_semantic_score(shots[i].clip_embed, shots[i+1].clip_embed))
        motion_scores.append(get_motion_score(shots[i].end_flow, shots[i+1].start_flow))

    # Calculate music beat alignment score
    beat_alignment_score = None
    energy_correspondence_score = None
    if music_prof is not None:
        # Get all cut point times (in seconds)
        cut_times = [shot.end_frame / fps for shot in shots[:-1]]  # The last shot has no cut
        beats = music_prof.beats  # List of music beat times
        
        beat_alignment_score = get_beat_alignment_score(cut_times, beats)
        
        # Calculate audiovisual energy correspondence score
        energy_correspondence_score = get_energy_correspondence_score(shots, music_prof, fps)
    
    # Calculate prompt relevance score m_1
    prompt_relevance_avg = None
    if prompt_embed is not None:
        prompt_relevance_avg = get_prompt_relevance_scores(shots, prompt_embed)
    
    return prompt_relevance_avg, semantic_scores, motion_scores, saliency_scores, beat_alignment_score, energy_correspondence_score, 