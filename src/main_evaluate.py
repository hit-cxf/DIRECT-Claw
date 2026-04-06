import argparse
import pickle
import subprocess
import tempfile
from typing import Optional
import yaml
from pathlib import Path
import cv2
import numpy as np

from .preprocess import preprocess_video
from .features import VideoFeatures
from .eval.evaluate import evaluate_scores
from .agent.music_utils import analyze_music
from .agent.interaction_utils import model as _clip_model

def get_video_fps(video_path: Path) -> float:
    """
    Automatically extract the frame rate from a video file.
    
    Parameters:
        video_path: Path to the video file
    
    Returns:
        Frame rate (fps) of the video
    """
    try:
        # Try to use ffprobe to get the precise fps
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        fps_str = result.stdout.strip()
        
        if '/' in fps_str:
            # Handle fractional fps, e.g., "24000/1001"
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        return fps
    except Exception as e:
        print(f"Warning: Failed to get fps using ffprobe: {e}")
        
        # Fallback to using cv2
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps
        except Exception as e2:
            print(f"Warning: Failed to get fps using cv2: {e2}")
            return 24.0  # Default to 24 fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to YAML config file (for keywords/prompt)")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root folder for raw video files")
    parser.add_argument("--output_root", type=str, default="./output",
                        help="Root folder for output features")
    parser.add_argument("--no_music", action='store_true', help="Skip music analysis")
    parser.add_argument("--recalc", action='store_true', help="Recalculate all features")

    args = parser.parse_args()

    FilePath = Path(args.file_path)
    video_path = args.data_root / FilePath
    output_path = args.output_root / FilePath.parent / "features.pkl"

    if not output_path.exists() or args.recalc:
        print(f"Cached features not found / recalc enabled. Re-calculating...")
        preprocess_video("temp_video", video_path, output_path, args.recalc)

    with open(output_path, 'rb') as f:
        features: VideoFeatures = pickle.load(f)
        features.pre_calc()
        print(f"Features loaded successfully.")
    
    # Read YAML config (if provided) and encode keywords
    prompt_embed = None
    keywords_text = None
    yaml_music_path = None
    if args.yaml_path:
        config_path = Path(args.yaml_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:  
                config = yaml.safe_load(f)
            
            # Read keywords
            keywords_text = config.get('keywords', None)
            if keywords_text:
                print(f"\nFound keywords in config: {keywords_text}")
                print(f"Encoding keywords with CLIP for m_1 metric...")

                # Encode keywords
                import torch
                import open_clip
                model_name = "ViT-B-32"
                with torch.no_grad():
                    tokenizer = open_clip.get_tokenizer(model_name)
                    text_tokens = tokenizer([keywords_text]).to("cuda")
                    prompt_embed = _clip_model.encode_text(text_tokens).cpu().numpy()[0]
                
                # Normalize
                prompt_embed = prompt_embed / np.linalg.norm(prompt_embed)
                
                print(f"Keywords encoded successfully.")
            else:
                print(f"Warning: No 'keywords' field found in {config_path}")
            
            # Read music_path
            yaml_music_path = config.get('music_path', None)
            if yaml_music_path:
                print(f"Found music_path in config: {yaml_music_path}")
        else:
            print(f"Warning: YAML config file not found: {config_path}")
    
    # Auto-detect fps 
    fps = get_video_fps(video_path)
    print(f"Auto-detected video FPS: {fps:.2f}")
    
    # Analyze music
    music_prof = None
    temp_audio_file = None
    
    if not args.no_music:
        # Determine music file path: read from YAML config or extract from video
        if yaml_music_path:
            music_source_path: Optional[Path] = Path(yaml_music_path)
            print(f"Using music file from YAML config: {music_source_path}")
        else:
            music_source_path = None
        
        if music_source_path:
            # Use the specified music file
            music_prof = analyze_music(music_source_path)
        else:
            # Extract audio from video to a temporary file
            print(f"Extracting audio from video: {video_path}")
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            temp_audio_path = Path(temp_audio_file.name)
            temp_audio_file.close()
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                "-y", str(temp_audio_path)
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Audio extracted to: {temp_audio_path}")
                music_prof = analyze_music(temp_audio_path, is_absolute=True)
            except subprocess.CalledProcessError as e:
                print(f"Error extracting audio: {e}")
                temp_audio_path.unlink(missing_ok=True)
                temp_audio_file = None
        
        if music_prof:
            print(f"Music analysis completed.")
    
    try:
        prompt_relevance_avg, semantic_scores, motion_scores, saliency_scores, beat_alignment, energy_correspondence = evaluate_scores(
            features, music_prof, fps, prompt_embed
        )
    finally:
        # Clean up temporary files
        if temp_audio_file is not None:
            temp_audio_path = Path(temp_audio_file.name)
            temp_audio_path.unlink(missing_ok=True)

    # Calculate basic information
    num_shots = len(features.shots)
    shot_durations = [(shot['end'] - shot['start']) / fps for shot in features.shots]
    avg_shot_duration = sum(shot_durations) / len(shot_durations) if shot_durations else 0
    total_duration = sum(shot_durations)
    num_frames = features.shots[-1]['end'] if features.shots else 0

    print("\n========== Video Basic Information ==========")
    print(f"Total frames: {num_frames}")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"FPS: {fps:.2f}")
    print(f"Number of shots: {num_shots}")
    print(f"Average shot duration: {avg_shot_duration:.2f}s")

    print("\n========== Evaluation Results ==========")
    if prompt_relevance_avg is not None:
        print(f"(m1) Prompt Relevance Score: {prompt_relevance_avg:.4f}")

    print(f"(m2) Average semantic score: {sum(semantic_scores) / len(semantic_scores):.4f}")
    print(f"(m3) Average motion score: {sum(motion_scores) / len(motion_scores):.4f}")
    print(f"(m4) Average saliency score: {sum(saliency_scores) / len(saliency_scores):.4f}")
    
    
    if beat_alignment is not None:
        print(f"\n--- Beat Alignment Score ---")
        print(f"Best offset: {beat_alignment['best_offset']:.3f} seconds")
        print(f"(m5) Sync score: {beat_alignment['score']:.4f}")
    
    if energy_correspondence is not None:
        print(f"\n--- Energy Correspondence Score ---")
        print(f"Pearson Correlation: {energy_correspondence['pearson']:.4f}")
        print(f"Spearman's Rank Correlation: {energy_correspondence['spearman']:.4f} (p-value: {energy_correspondence['spearman_p']:.4e})")
        print(f"(m6) Energy score: {(energy_correspondence['spearman'] + 1) / 2:.4f}")