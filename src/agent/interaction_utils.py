from dataclasses import dataclass
import open_clip
import yaml
import os
import numpy as np

from .music_utils import MusicResult
from .editing_utils import ScoreConfig

class PromptManager:
    def __init__(self, config_path="configs/prompts.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    def get_prompt(self, arg1, arg2, **kwargs):
        try:
            template = self.prompts[arg1][arg2]
            return template.format(**kwargs)
        except KeyError:
            raise Exception(f"Prompt for {arg1} {arg2} not found")

prompt_manager = PromptManager()

@dataclass
class SectionInfo:
    """Music section information"""
    label: str
    energy_level: str
    visual_tags: list[str]
    rationale: str

@dataclass
class SegmentGuidance:
    """Segment guidance information"""
    retrieval_query: str
    weight_profile: str
    pacing_control: list[int]

    
# ==================== Predefined weight templates ====================

class ConfigPresets:
    # Motion continuity priority
    ACTION_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=16,
        semantic_weight=1,
        saliency_weight=2,
        motion_weight=8,  
        energy_weight=4,
        energy_value=0,
    )

    # Semantic priority
    SEMANTIC_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=48,
        semantic_weight=2,
        saliency_weight=2,
        motion_weight=2,  
        energy_weight=4,
        energy_value=0,
    )

    # Saliency priority
    SALIENCY_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=16,
        semantic_weight=1,
        saliency_weight=12,
        motion_weight=2,  
        energy_weight=4,
        energy_value=0,
    )

    # Visual complexity priority
    VISUAL_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=16,
        semantic_weight=1,
        saliency_weight=8,
        motion_weight=6,  
        energy_weight=4,
        energy_value=0,
    )

    # Balanced
    BALANCED_PRIORITY = ScoreConfig(
        prompt_embed=None,
        prompt_weight=16,
        semantic_weight=1,
        saliency_weight=4,
        motion_weight=3,  
        energy_weight=4,
        energy_value=0,
    )

def get_weight_profile_by_name(profile_name: str) -> ScoreConfig:
    """Retrieve scoring configuration by weight profile name"""
    if profile_name == "Motion_Continuity_Priority":
        return ConfigPresets.ACTION_PRIORITY
    elif profile_name == "Semantic_Priority":
        return ConfigPresets.SEMANTIC_PRIORITY
    elif profile_name == "Composition_Similarity_Priority":
        return ConfigPresets.SALIENCY_PRIORITY
    elif profile_name == "Visual_Complexity_Priority":
        return ConfigPresets.VISUAL_PRIORITY
    elif profile_name == "Default_Priority":
        return ConfigPresets.BALANCED_PRIORITY
    else:
        return ConfigPresets.BALANCED_PRIORITY


model_name = "ViT-B-32"
model, _, _ = open_clip.create_model_and_transforms(
    model_name,
    pretrained="laion2b-s34b-b79k",
    device="cuda"
)
model.eval()


def encode_text(text: str) -> np.ndarray:
    """Encode text into CLIP embedding vectors."""
    return model.encode_text(
        open_clip.tokenize([text]).to("cuda")
    ).detach().cpu().numpy()[0]


def parse_segment_guidance(segment_guidance: SegmentGuidance) -> tuple[np.ndarray, list[int], ScoreConfig]:
    """Parse raw segment guidance information into standard format."""
    prompt_embed = encode_text(segment_guidance.retrieval_query)

    pacing_control = segment_guidance.pacing_control
    
    config = get_weight_profile_by_name(segment_guidance.weight_profile)
    config.prompt_embed = prompt_embed

    return prompt_embed, pacing_control, config