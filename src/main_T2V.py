"""
T2V Baseline: Pure text-driven video retrieval and editing

Workflow:
1. Use LLM to generate N storyboard scripts based on user_prompt (default 30).
2. Each storyboard has a fixed duration of 4 seconds, no music synchronization.
3. Use Beam Search to retrieve the best segments, keeping only the "semantic" weights:
   - prompt_weight (CLIP similarity between query and candidate segments)
   - semantic_weight (CLIP similarity between the previous segment and candidate segment)
   - saliency_weight = 0, motion_weight = 0, energy_weight = 0
"""

import os
from pathlib import Path
from copy import deepcopy

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

import argparse
import logging
import json
import re
import yaml
import subprocess
import numpy as np

from .utils.path import load_video_footages, get_data_dir
from .agent.llm_interface import chat_with_llm
from .agent.interaction_utils import encode_text
from .agent.editor import generate_segment_video
from .agent.editing_utils import Score, ScoreConfig, EditResult

logger = logging.getLogger("main_T2V")

# ==================== Constants ====================

NUM_SHOTS: int = 30       # Number of storyboards
SHOT_DURATION: float = 4.0  # Duration of each storyboard (seconds)
FPS: int = 24

# T2V-specific weights: Keep only semantic-related weights, set others to 0
T2V_SCORE_CONFIG = ScoreConfig(
    prompt_embed=None,  # Dynamically set during runtime for each storyboard
    prompt_weight=48,   # CLIP similarity between query and candidate segments
    semantic_weight=2,  # CLIP similarity between the previous segment and candidate segment (visual coherence)
    saliency_weight=0,
    motion_weight=0,
    energy_weight=0,
    energy_value=0,
)

# ==================== Logging Configuration ====================

def set_logger(log_path: str = "output/tmp_t2v.log") -> None:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s")
    )

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
    )

    logger_names = ["editor", "retriever", "path", "main_T2V"]
    for name in logger_names:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.addHandler(console_handler)
        lg.addHandler(file_handler)
        lg.propagate = False


# ==================== LLM Storyboard Generation ====================

def generate_storyboard(user_prompt: str, num_shots: int = NUM_SHOTS) -> list[str]:
    """Call LLM to generate num_shots storyboard retrieval descriptions based on user_prompt.

    Returns:
        list[str]: A list of retrieval queries with length num_shots, corresponding to each storyboard in order.
    """
    prompt = f"""# Role
You are a professional video editor and storyboard artist.

# Task
Given the following user prompt, generate exactly {num_shots} shot descriptions for a video montage.
Each shot is approximately {int(SHOT_DURATION)} seconds long, and together they form a coherent visual narrative.

# Guidelines
1. Each shot description must be a concise, visually concrete CLIP-style retrieval query (10–20 words).
2. The sequence of shots should reflect a natural narrative arc (opening → development → climax → ending).
3. Vary shot types (close-up, wide shot, action, reaction, establishing, etc.) to create visual rhythm.
4. Use concrete, descriptive visual language. Avoid abstract or metaphorical phrasing.
5. Each query must be distinct — do not repeat the same description.
6. Do NOT mention character names, movie titles, or specific IP references.

# User Prompt
{user_prompt}

# Output Format
Output ONLY a valid JSON array of exactly {num_shots} strings. No extra text, no markdown fences.
Example (3 shots): ["wide shot of a city skyline at dusk", "close-up of a woman's determined face", "two figures running through a dark alley"]
"""

    logger.info("Calling LLM to generate storyboard scripts...")
    messages = [{"role": "user", "content": prompt}]
    response = chat_with_llm(messages)
    logger.debug(f"Raw LLM response:\n{response}")

    # Extract JSON array
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return a valid JSON list. Raw response:\n{response}")

    shots: list[str] = json.loads(match.group())

    if len(shots) != num_shots:
        logger.warning(
            f"LLM returned {len(shots)} storyboards, expected {num_shots}. Adjusting automatically..."
        )
        if len(shots) < num_shots:
            # If insufficient, loop to supplement
            while len(shots) < num_shots:
                shots += shots[: num_shots - len(shots)]
        shots = shots[:num_shots]

    logger.info(f"Storyboard generation complete. Total {len(shots)} shots:")
    for i, q in enumerate(shots):
        logger.info(f"  Shot {i + 1:02d}: {q}")

    return shots


# ==================== Main Logic ====================

def main_T2V() -> None:
    parser = argparse.ArgumentParser(description="T2V Baseline: Text-driven video retrieval and editing")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to benchmark YAML configuration file")
    parser.add_argument(
        "--result_path",
        type=str,
        default="output/results/result_t2v.mp4",
        help="Path to save the output video",
    )
    parser.add_argument("--log_path", type=str, default="output/tmp_t2v.log", help="Path to log file")
    parser.add_argument("--num_shots", type=int, default=NUM_SHOTS, help=f"Number of storyboards (default {NUM_SHOTS})")
    parser.add_argument(
        "--no_music",
        action="store_true",
        default=False,
        help="Do not add background music, output a silent video",
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam Search width (default 3)"
    )
    parser.add_argument(
        "--exploration", type=int, default=5, help="Top-K candidates per step (default 5)"
    )
    args = parser.parse_args()

    set_logger(args.log_path)

    # ---------- Load Configuration ----------
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_path = Path(config["video_csv"])
    user_prompt: str = config["user_prompt"]
    music_path = config.get("music_path")

    logger.info(f"Loading video footage features: {csv_path}")
    video_paths, video_features = load_video_footages(csv_path)

    # ---------- Generate Storyboard ----------
    shot_queries = generate_storyboard(user_prompt, args.num_shots)

    # ---------- Beam Search for Each Storyboard ----------
    result = EditResult(video_candidates=[], total_frames=0, total_score=0.0)

    for i, query in enumerate(shot_queries):
        logger.info(f"[{i + 1}/{len(shot_queries)}] Retrieving storyboard: '{query}'")

        # Encode text
        prompt_embed = encode_text(query)

        # Build scoring configuration (deep copy to avoid modifying shared constants)
        score_cfg = deepcopy(T2V_SCORE_CONFIG)
        score_cfg.prompt_embed = prompt_embed
        score_cfg.energy_value = 50.0  # No music-driven energy, use medium energy value

        # Cut-off time for the current storyboard (seconds) = completed frames/fps + 4s
        cut_point = result.total_frames / FPS + SHOT_DURATION

        beam_results = generate_segment_video(
            video_features=video_features,
            prompt_embed=prompt_embed,
            cut_points=[cut_point],
            visual_energies=[score_cfg.energy_value],
            config=score_cfg,
            prev_shots=result,
            beam_size=args.beam_size,
            exploration=args.exploration,
        )

        # Select the best candidate from Beam Search
        best = beam_results[0]
        result.extend(best)

    logger.info(
        f"Retrieval complete. Total {len(result.video_candidates)} segments, "
        f"total frames {result.total_frames} (≈{result.total_frames / FPS:.1f}s)"
    )

    # ---------- Generate Video ----------
    tmp_video = Path("tmp_t2v.mp4")
    result.generate_video(video_paths, tmp_video)

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    if music_path and not args.no_music:
        logger.info(f"Merging background music: {music_path}")
        command = [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-i", str(get_data_dir() / music_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(result_path),
        ]
        subprocess.run(command, check=True)
        tmp_video.unlink(missing_ok=True)
    else:
        tmp_video.replace(result_path)

    logger.info(f"T2V result saved to: {result_path}")


if __name__ == "__main__":
    try:
        main_T2V()
    except Exception as e:
        logger.exception(f"main_T2V encountered an error: {e}")
        raise
