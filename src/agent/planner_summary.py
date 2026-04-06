import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import hashlib
import json
from tqdm import tqdm
import logging
from sklearn.cluster import HDBSCAN
from umap import UMAP
from typing import Any


logger = logging.getLogger('planner_summary')

from .llm_interface import chat_with_llm, Message
from ..features import VideoFeatures
from ..utils.path import get_data_dir, get_output_dir, load_video_footages
from .interaction_utils import prompt_manager

summary_dir = get_output_dir() / "summary"
os.makedirs(summary_dir, exist_ok=True)

# Configure parameters
N_CLUSTERS = 32   # Number of clusters
SHOTS_PER_CLUSTER = 16 # Number of representative shots per cluster
HDBSCAN_MIN_CLUSTER_SIZE = 16
UMAP_N_COMPONENTS = 16
UMAP_MIN_DIST = 0.0

def pil_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 80) -> str:
    """Convert a PIL image object to a base64 string"""
    buffered = BytesIO()
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def _read_frame_at_index(path: Path | str, frame_index: int) -> np.ndarray | None:
    """Read a frame at the specified index"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def process_vlm_task_sequential(cluster_id: int, frames: list[Image.Image], genre: str, share: float) -> dict[str, Any]:
    """Process a single cluster's VLM request"""
    try:
        msg = Message(role="user")
        
        # Add each image sequentially
        for frame_pil in frames:
            # Compress quality to prevent overly large Base64 strings for multiple images
            b64_str = pil_to_base64(frame_pil, quality=70)
            msg = msg.add_image_base64(b64_str)
            
        msg = msg.add_text(prompt_manager.get_prompt(
            "Screenwriter", "Cluster_Caption", 
            genre = genre, 
            n_shots = len(frames), 
            share = share,
        ))
        
        res = chat_with_llm([msg.to_dict()])
        return {"id": cluster_id, "txt": res.strip()}
    except Exception as e:
        logger.error(f"Error processing cluster {cluster_id}: {e}")
        return {"id": cluster_id, "error": str(e)}

def run_clustering_pipeline(video_paths: dict[str, Path], video_features: dict[str, VideoFeatures], genre: str) -> str:
    # 1. Collect all shot information and embeddings
    shots: list[dict[str, int | str]] = []
    embeddings_list: list[np.ndarray] = []

    logger.info("Collecting shots and embeddings...")
    for vid, feature in video_features.items():
        if vid in video_paths:
            path = str(video_paths[vid])
            for i, shot in enumerate(feature.shots):
                # Directly get embedding
                emb = feature.get_clip_embed(shot['start'], shot['end'])
                if emb is not None:
                    embeddings_list.append(emb)
                    mid_frame: int = int((shot['start'] + shot['end']) // 2)
                    shots.append({
                        'vid': vid,
                        'path': path,
                        'start': shot['start'],
                        'end': shot['end'],
                        'mid': mid_frame,
                        'shot_idx': i,
                        'unique_id': f"{vid}_{i}"
                    })

    if not shots:
        return "No footage found."

    embeddings = np.stack(embeddings_list)

    # 2. UMAP dimensionality reduction (reduce dimensions before density clustering)
    logger.info(f"Running UMAP to {UMAP_N_COMPONENTS} dims (min_dist={UMAP_MIN_DIST}) on {len(embeddings)} embeddings...")
    umap_model = UMAP(n_components=UMAP_N_COMPONENTS, min_dist=UMAP_MIN_DIST, metric='cosine')
    reduced = umap_model.fit_transform(embeddings)

    # 3. HDBSCAN clustering
    logger.info(f"Running HDBSCAN (min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}) on {len(shots)} shots...")
    clusterer = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, copy=True)
    labels = clusterer.fit_predict(reduced)

    label_counts = {
        label: int(np.sum(labels == label))
        for label in set(labels)
        if label >= 0
    }
    cluster_infos = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:N_CLUSTERS]
    if not cluster_infos:
        cluster_infos = [(-1, len(shots))]
    
    vlm_results = []
    
    logger.info(f"Analyzing {len(cluster_infos)} visual clusters sequentially...")
    
    # 5. Sequentially execute VLM analysis
    for cluster_id, size in tqdm(cluster_infos, desc="VLM Analysis"):
        # Get representative frames for the cluster
        indices = np.where(labels == cluster_id)[0]
        center = np.mean(reduced[indices], axis=0)
        dists = np.linalg.norm(reduced[indices] - center, axis=1)
        
        # Select the top N closest to the center
        top_k_local_indices = np.argsort(dists)[:SHOTS_PER_CLUSTER]
        top_k_global_indices = indices[top_k_local_indices]
        
        representative_frames: list[Image.Image] = []
        for idx in top_k_global_indices:
            shot_info = shots[idx]
            frame_img = _read_frame_at_index(shot_info['path'], shot_info['mid']) # type: ignore
            if frame_img is not None:
                representative_frames.append(Image.fromarray(frame_img))

        # Execute analysis immediately
        share = size / len(shots)
        res = process_vlm_task_sequential(cluster_id, representative_frames, genre, share)
        res["share"] = share
        vlm_results.append(res)
    
    # 6. Aggregate results
    vlm_results.sort(key=lambda x: x.get("share", 0.0), reverse=True)
    
    segments = []
    for res in vlm_results:
        if "txt" in res:
            share_pct = res.get("share", 0.0)
            header = f"[Visual Cluster ({share_pct:%} of total shots)]"
            segments.append(f"{header}\n{res['txt']}")
            
    full_corpus = "\n\n".join(segments)
    
    # 7. Final LLM summary
    agg_prompt = prompt_manager.get_prompt(
        "Screenwriter", "Footage_Summary",
        full_corpus = full_corpus, 
        total_shots = len(shots), 
        genre = genre,
    )
    logger.debug(agg_prompt)
    final_summary = chat_with_llm([
        Message(role="user").add_text(agg_prompt).to_dict()
    ])
    
    return final_summary

def get_summary_cache_key(csv_path: Path, genre: str) -> str:
    """Generate a unique key for the summary cache"""
    csv_path = get_data_dir() / csv_path
    csv_bytes = csv_path.read_bytes()
    cache_hash = hashlib.md5(csv_bytes + genre.encode("utf-8")).hexdigest()
    return cache_hash

def get_summary_cache(csv_path: Path, genre: str = "Movie Collection") -> str | None:
    """Attempt to retrieve the summary from the cache"""
    summary_root = summary_dir
    summary_root.mkdir(parents=True, exist_ok=True)

    csv_path = get_data_dir() / csv_path
    cache_hash = get_summary_cache_key(csv_path, genre)
    cache_path = summary_root / f"{cache_hash}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.info(f"Using cached summary from {cache_path}")
            return cached["summary"]
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_path} is corrupted. Ignoring cache.")
            return None
    return None

def get_summary(csv_path: Path, video_paths, video_features, summary_root: Path = summary_dir, genre: str = "Movie Collection", use_cache: bool = True) -> str:
    """Generate the Planner Summary for video footage (Sequential Clustering)"""
    summary_root.mkdir(parents=True, exist_ok=True)

    csv_path = get_data_dir() / csv_path
    cache_hash = get_summary_cache_key(csv_path, genre)
    cache_path = summary_root / f"{cache_hash}.json"

    if use_cache:
        cached_summary = get_summary_cache(csv_path, genre)
        if cached_summary:
            return cached_summary
            
    logger.info("Generating new summary (Sequential Clustering Strategy)...")
    
    summary = run_clustering_pipeline(video_paths, video_features, genre)

    cache_payload = {
        "csv_path": str(csv_path),
        "genre": genre,
        "summary": summary
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary