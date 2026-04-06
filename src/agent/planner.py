import allin1
import json
import os
import re
from pathlib import Path
import argparse
import logging
from dataclasses import dataclass

from .llm_interface import chat_with_llm
from .interaction_utils import SectionInfo, prompt_manager

logger = logging.getLogger('planner')

def get_plan_with_llm(
    user_prompt: str, 
    footage_summary: str,
    music_info: str,
    music_section_count: int,
    retry_count: int=3
):
    """
    Generate a JSON-formatted video section plan using LLM.
    """
    logger.info("Generating global plan...")
    full_prompt = prompt_manager.get_prompt(
        "Screenwriter", "Structural_Plan",
        user_prompt=user_prompt,
        footage_summary=footage_summary,
        music_info=music_info
    )
    logger.debug("Full Prompt:\n%s", full_prompt)

    for attempt in range(retry_count):
        try:
            # 1. Call the LLM
            response = chat_with_llm([
                {"role": "user", "content": full_prompt},
            ])
            
            # 2. Extract the JSON block using Regular Expression
            # This looks for content inside ```json ... ``` or just the first [ or {
            json_match = re.search(r'\[\s*{.*}\s*\]', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: if no brackets found, the whole response might be raw JSON
                json_str = response.strip()

            # 3. Parse JSON
            plan_data = json.loads(json_str)

            section_list: list[SectionInfo] = []
            for plan in plan_data:
                section_info = SectionInfo(
                    label = plan.get("section_name", ""),
                    energy_level = plan.get("energy_level", ""),
                    visual_tags = plan.get("visual_tags", []),
                    rationale = plan.get("rationale", "")
                )
                section_list.append(section_info)
            
            if len(section_list) != music_section_count:
                logger.warning(f"Expected {music_section_count} sections based on music info, but got {len(section_list)} from LLM. Retrying...")
                continue

            # 4. Optional: Extract the Narrative Flow text separately if needed
            narrative_flow = ""
            if "## Global Narrative Flow" in response:
                narrative_flow = response.split("## Global Narrative Flow")[-1].split("##")[0].strip()

            logger.info(f"Global Narrative Flow: {narrative_flow}")
            logger.debug("Planner raw response:\n%s", response)

            return section_list

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Attempt {attempt + 1} failed to parse JSON: {e}")
            continue
    
    raise ValueError("Failed to get a valid plan from LLM after multiple attempts.")
