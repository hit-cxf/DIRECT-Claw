import argparse
from pathlib import Path
import csv

from .preprocess import preprocess_video
from .utils.path import load_video_list, get_data_dir, get_output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True,
                        help="CSV file with video_id, filepath")
    parser.add_argument("--recalc", action='store_true', help="Recalculate all features")
    args = parser.parse_args()

    video_list = load_video_list(args.csv)
    print(f"Loaded {len(video_list)} videos from CSV.")

    for video_id, filepath in video_list:
        FilePath = Path(filepath)

        video_path = get_data_dir() / FilePath
        output_path = get_output_dir() / FilePath.with_suffix(".pkl")

        preprocess_video(video_id, video_path, output_path, args.recalc)

    print("\nAll videos processed.")