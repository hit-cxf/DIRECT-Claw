from pathlib import Path
import os
import csv
import yaml
import logging
import pickle
from ..features.features import VideoFeatures

logger = logging.getLogger("path")

def get_root_dir() -> Path:
    """Get the root directory of the project.

    Returns:
        Path: Full path of the project root directory.
    """
    return Path(__file__).parent.parent.parent

def get_config_path(name) -> Path:
    """Get the path of a configuration file.

    Args:
        name (str): Configuration file name.
    Returns:
        Path: Full path of the configuration file.
    """
    config_dir = get_root_dir() / "configs"
    return config_dir / name

def get_config(name: str) -> dict:
    """Load and return the content of a configuration file.

    Args:
        name (str): Configuration file name.
    Returns:
        dict: Dictionary representation of the configuration file content.
    """
    config_path = get_config_path(name)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_data_dir() -> Path:
    """Get the path of the data directory.

    Returns:
        Path: Full path of the data directory.
    """
    data_dir = get_root_dir() / "data"
    return data_dir

def get_output_dir() -> Path:
    """Get the path of the output directory.

    Returns:
        Path: Full path of the output directory.
    """
    output_dir = get_root_dir() / "output"
    return output_dir

def load_video_list(csv_path: Path) -> list[tuple[str, str]]:
    """Load the video list from a CSV file.

    Args:
        csv_path (Path): Path to the CSV file.
    Returns:
        list[tuple[str, str]]: List of tuples containing video_id and filepath.
    """
    videos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id, filepath = row["video_id"], row["filepath"]
            videos.append((video_id, filepath))
    return videos

def load_video_footages(csv_path: Path):
    """Load video features from a CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Loaded video features dictionary.

    Notes:
        The CSV file should contain the following columns:
        - video_id: Video ID
        - filepath: Video file path
        
        video_path = data_root / filepath

        Feature file path construction: get_output_dir() / filepath.with_suffix(".pkl")
    """
    csv_path = get_data_dir() / csv_path

    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    video_paths: dict[str, Path] = {}
    video_features: dict[str, VideoFeatures] = {}
    
    for video_id, filepath in load_video_list(csv_path):
        feature_path = get_output_dir() / Path(filepath).with_suffix(".pkl")

        if not feature_path.exists():
            logger.error(f"Feature file not found: {str(feature_path)}")
            raise FileNotFoundError(f"Feature file not found: {str(feature_path)}")

        video_feat = pickle.load(open(feature_path, "rb"))
        video_feat.pre_calc()

        video_path = get_data_dir() / filepath
        video_paths[video_id] = video_path
        video_features[video_id] = video_feat

    logger.info(f"Successfully loaded {len(video_paths)} videos")

    return video_paths, video_features
