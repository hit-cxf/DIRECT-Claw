import pickle
import math
import gc
from pathlib import Path

import torch

from ..features import VideoFeatures
from .shot_detect import detect_shots
from .calc import calc_saliency, calc_clip_embedding, calc_optical_flow
from ..utils.decoder import PaddedVideoDecoder, KeyframeFrameCache

MAX_KEYFRAME_CHUNK_SECONDS = 3600


def _safe_fps(decoder: PaddedVideoDecoder) -> float:
    fps = float(decoder.cap.get(5))
    if fps <= 0 or math.isnan(fps):
        return 30.0
    return fps


def _iter_keyframe_chunks(total_keyframes: int, keyframes_per_chunk: int):
    for start in range(0, total_keyframes, keyframes_per_chunk):
        yield start, min(start + keyframes_per_chunk, total_keyframes)


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess_video(video_id: str, video_path: Path, output_path: Path, recalc: bool = False):
    if output_path.exists() and not recalc:
        print(f"\n=== Features already exist: {video_id}: {str(video_path)} ===")
        return

    if not video_path.exists():
        print(f"\n=== Video not found: {video_id}: {str(video_path)} ===")
        return

    print(f"\n=== Processing {video_id}: {str(video_path)} ===")
    decoder = PaddedVideoDecoder(video_path)
    num_frames = len(decoder)
    fps = _safe_fps(decoder)

    print(f"Decoder ready. Total frames = {num_frames}, fps = {fps:.3f}")

    features = VideoFeatures(video_id, num_frames)
    total_keyframes = len(features.keyframes)
    keyframes_per_chunk = max(
        1,
        int(math.ceil(fps * MAX_KEYFRAME_CHUNK_SECONDS / features.keyframe_interval)),
    )
    chunk_count = int(math.ceil(total_keyframes / keyframes_per_chunk))

    print(
        "Processing keyframes in chunks: "
        f"interval={features.keyframe_interval} frames, "
        f"max_chunk={MAX_KEYFRAME_CHUNK_SECONDS}s, "
        f"keyframes_per_chunk={keyframes_per_chunk}, chunks={chunk_count}"
    )

    for chunk_idx, (start_idx, end_idx) in enumerate(
        _iter_keyframe_chunks(total_keyframes, keyframes_per_chunk), start=1
    ):
        start_frame = features.keyframes[start_idx].frame_idx
        end_frame = features.keyframes[end_idx - 1].frame_idx
        print(
            f"\n--- Keyframe chunk {chunk_idx}/{chunk_count}: "
            f"kf[{start_idx}:{end_idx}) frames {start_frame}..{end_frame} ---"
        )
        frame_cache = None
        try:
            frame_cache = KeyframeFrameCache(
                video_path,
                num_frames,
                features.keyframe_interval,
                primary_start_idx=start_idx,
                primary_end_idx=end_idx,
            )

            print(f"Calculating saliency for chunk {chunk_idx}/{chunk_count}...")
            calc_saliency(video_path, num_frames, features, frame_cache=frame_cache)
            print(f"Calculating CLIP embedding for chunk {chunk_idx}/{chunk_count}...")
            calc_clip_embedding(video_path, num_frames, features, frame_cache=frame_cache)
            print(f"Calculating optical flow for chunk {chunk_idx}/{chunk_count}...")
            calc_optical_flow(video_path, num_frames, features, frame_cache=frame_cache)
            print(
                f"Chunk {chunk_idx}/{chunk_count} features stored: "
                f"kf[{start_idx}:{end_idx})"
            )
        finally:
            print(f"Cleaning chunk {chunk_idx}/{chunk_count} keyframe cache...")
            if frame_cache is not None:
                del frame_cache
            _cleanup_memory()

    print(f"Detecting shots...")
    detect_shots(video_path, features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {video_id}'s features to {str(output_path)}...")
    with open(output_path, "wb") as f:
        pickle.dump(features, f)
