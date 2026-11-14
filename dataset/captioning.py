"""
Multimodal Caption & Text Preparation Service.
Generates structured, visual-language descriptions covering subjects, actions,
environments, camera angles, lighting, and artistic style while preserving original prompts.
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from dataset.models import DatasetItem, CaptionMetadata


class CaptionService:
    """Manages structured caption synthesis, prompt preservation, and manual review."""

    @staticmethod
    def synthesize_structured_caption(item: DatasetItem) -> str:
        """
        Synthesizes a detailed, objective training caption from structured visual metadata.
        Explicitly mentions subjects, actions, environment, lighting, camera perspective,
        and visual/animation style without inventing unsupported details.
        """
        parts = []

        # 1. Subject & Character intro
        chars = ", ".join(item.characters) if item.characters else None
        subject = item.subject if item.subject and item.subject != "general" else None

        if chars and subject:
            parts.append(f"A scene featuring {chars} ({subject})")
        elif chars:
            parts.append(f"A scene featuring {chars}")
        elif subject:
            parts.append(f"A scene depicting {subject}")
        else:
            parts.append("A visual sequence")

        # 2. Action / Motion
        if item.actions:
            action_desc = ", ".join(item.actions)
            parts.append(f"engaged in {action_desc}")
        elif item.description:
            # First sentence of description if available
            first_sent = item.description.split(".")[0].strip()
            if first_sent:
                parts.append(f"showing {first_sent.lower()}")

        # 3. Environment & Setting
        if item.environment and item.environment != "general":
            parts.append(f"set within a {item.environment} environment")

        # 4. Scene type & Lighting
        details = []
        if item.scene_type and item.scene_type != "general":
            details.append(f"{item.scene_type} framing")
        if item.lighting and item.lighting != "natural":
            details.append(f"{item.lighting} illumination")
        if details:
            parts.append(f"with {', '.join(details)}")

        # 5. Camera movement
        if item.camera_movement and item.camera_movement != "static":
            parts.append(f"captured via {item.camera_movement} camera motion")

        # 6. Animation / Visual Style
        style_tokens = []
        if item.animation_style and item.animation_style != "live_action":
            style_tokens.append(item.animation_style.replace("_", " "))
        if item.visual_style and item.visual_style != "cinematic":
            style_tokens.append(item.visual_style.replace("_", " "))
        if style_tokens:
            parts.append(f"rendered in {' '.join(style_tokens)} style")

        full_caption = ", ".join(parts).strip() + "."
        # Capitalize first letter
        if full_caption:
            full_caption = full_caption[0].upper() + full_caption[1:]
        return full_caption

    def enrich_item(self, item: DatasetItem, force: bool = False) -> DatasetItem:
        """
        Enriches a DatasetItem with a structured generated caption.
        Preserves original user-provided caption or prompt in separate fields.
        """
        if not item.generated_caption or force:
            synth = self.synthesize_structured_caption(item)
            item.generated_caption = synth
            if not item.caption_source or item.caption_source == "user_supplied":
                item.caption_source = "structured_synthesis"
        return item

    def update_manual_caption(self, item: DatasetItem, manual_caption: str) -> DatasetItem:
        """
        Applies human-reviewed or manual caption correction.
        """
        item.original_caption = manual_caption.strip()
        item.caption_source = "manual_correction"
        return item
