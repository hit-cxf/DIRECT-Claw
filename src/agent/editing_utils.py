from dataclasses import dataclass
import numpy as np
import cv2
from pathlib import Path
import logging
import pickle
import os

from ..features.features import VideoFeatures, ShotFeatures
from ..eval.metrics import get_saliency_score, get_semantic_score, get_motion_score, get_energy_score
from ..utils.path import load_video_list

logger = logging.getLogger('retriever')

@dataclass
class Score:
    """single shot score"""
    prompt: float
    semantic: float
    saliency: float
    motion: float
    energy: float
    combined: float

@dataclass
class ScoreConfig:
    """score weights"""
    prompt_embed: np.ndarray | None
    prompt_weight: float
    semantic_weight: float
    saliency_weight: float
    motion_weight: float
    energy_weight: float
    energy_value: float

    def get_score(self, last_shot: ShotFeatures | None, new_shot: ShotFeatures) -> Score:
        """calculate weighted score"""
        assert self.prompt_embed is not None, "Prompt embedding is None in ScoreConfig."
        prompt_score = get_semantic_score(self.prompt_embed, new_shot.clip_embed)
        energy_score = get_energy_score(self.energy_value, new_shot.energy_value)

        if last_shot is None:
            semantic_score = 0.0
            saliency_score = 0.0
            motion_score = 0.0
            
        else:
            semantic_score = get_semantic_score(last_shot.clip_embed, new_shot.clip_embed)
            saliency_score = get_saliency_score(last_shot.end_saliency, new_shot.start_saliency)
            motion_score = get_motion_score(last_shot.end_flow, new_shot.start_flow)
        
        combined_score = float(
            self.prompt_weight * prompt_score +
            self.semantic_weight * semantic_score +
            self.saliency_weight * saliency_score +
            self.motion_weight * motion_score +
            self.energy_weight * energy_score
        )
        return Score(
            prompt=prompt_score,
            semantic=semantic_score,
            saliency=saliency_score,
            motion=motion_score,
            energy=energy_score,
            combined=combined_score
        )

@dataclass
class ShotCandidate:
    video_id: str
    shot_idx: int
    start_frame: int
    end_frame: int
    score: Score

@dataclass
class EditResult:
    video_candidates: list[ShotCandidate]
    total_frames: int
    total_score: float

    def append_shot(self, shot: ShotCandidate) -> None:
        self.video_candidates.append(shot)
        self.total_frames += (shot.end_frame - shot.start_frame + 1)
        self.total_score += shot.score.combined
    
    def extend(self, other: "EditResult") -> None:
        self.video_candidates.extend(other.video_candidates)
        self.total_frames += other.total_frames
        self.total_score += other.total_score

    def get_score(self) -> Score:
        """get average score"""
        num_shots = len(self.video_candidates)
        if num_shots == 0:
            return Score(0, 0, 0, 0, 0, 0)
        
        total_prompt = sum(shot.score.prompt for shot in self.video_candidates)
        total_semantic = sum(shot.score.semantic for shot in self.video_candidates)
        total_saliency = sum(shot.score.saliency for shot in self.video_candidates)
        total_motion = sum(shot.score.motion for shot in self.video_candidates)
        total_energy = sum(shot.score.energy for shot in self.video_candidates)
        total_combined = sum(shot.score.combined for shot in self.video_candidates)
        
        return Score(
            prompt=total_prompt / num_shots,
            semantic=total_semantic / num_shots,
            saliency=total_saliency / num_shots,
            motion=total_motion / num_shots,
            energy=total_energy / num_shots,
            combined=total_combined / num_shots
        )

    def generate_video(self, video_paths: dict[str, Path], output_path: Path, fps: int = 24, show_logger: bool = True) -> None:
        """generate video output
        
        args:
            output_path: output video path
            fps: frames per second
        """
        if not self.video_candidates:
            logger.warning("No candidate segments available, unable to generate video")
            return
        
        # Get video resolution (from the first non-black-screen candidate's video)
        index = min([i for i, c in enumerate(self.video_candidates) if c.video_id != "BLACK_SCREEN"], default=None)
        if index is not None:
            first_candidate = self.video_candidates[index]
            first_video_path = video_paths[first_candidate.video_id]
            
            cap = cv2.VideoCapture(str(first_video_path))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            frame_width, frame_height = 1280, 720
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        if show_logger:
            logger.info(f"Start generating video, output path: {output_path}")
        
        # Iterate through all candidate segments, extracting and writing frames sequentially
        for candidate in self.video_candidates:
            if candidate.video_id == "BLACK_SCREEN":
            # Generate black screen segment
                num_frames = candidate.end_frame - candidate.start_frame + 1
                black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                for _ in range(num_frames):
                    out.write(black_frame)
                if show_logger:
                    logger.debug(f"Added black screen segment, total {num_frames} frames")
                continue

            video_path = video_paths[candidate.video_id]
            start_frame = candidate.start_frame
            end_frame = candidate.end_frame
            
            if show_logger:
                logger.debug(f"Processing segment: {candidate.video_id} [{start_frame}:{end_frame}]")
            
            cap = cv2.VideoCapture(str(video_path))
            
            # Jump to the starting frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and write frames sequentially
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    if show_logger:
                        logger.warning(f"Failed to read frame: {video_path} frame {frame_idx}")
                    break
                
                # Resize frame to match the video writer's dimensions
                if frame.shape[:2] != (frame_height, frame_width):
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                out.write(frame)
            
            cap.release()
        
        out.release()
        if show_logger:
            logger.info(f"Video generated successfully: {output_path}")
