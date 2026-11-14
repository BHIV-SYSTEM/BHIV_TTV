"""
Dataset Validation and Duplicate Detection Engine.
Performs exact SHA-256 duplicate detection, perceptual hash (pHash) visual similarity analysis,
and comprehensive training-eligibility validation without deleting or modifying original files.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

from dataset.models import DatasetItem, ValidationStatus, PermissionState
from dataset.quality import hamming_distance
from app.core.logging import logger


class DatasetValidator:
    """Validates dataset items and detects exact and perceptual duplicates."""

    def __init__(
        self,
        min_width: int = 480,
        min_height: int = 320,
        min_duration: float = 1.0,
        max_phash_distance: int = 6,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_duration = min_duration
        self.max_phash_distance = max_phash_distance

    def validate_item(self, item: DatasetItem) -> DatasetItem:
        """
        Runs comprehensive validation across an individual DatasetItem.
        Updates validation_status, validation_reasons, and training_eligible.
        """
        reasons: List[str] = []

        # 1. File existence & readability
        if item.storage_path:
            p = Path(item.storage_path)
            if not p.exists() or p.stat().st_size < 100:
                reasons.append(f"Storage file missing or empty: {item.storage_path}")
        elif item.media_type != "text":
            reasons.append("Missing storage path for media item")

        # 2. Resolution checks for video/image
        if item.media_type in ("video", "image") and item.resolution:
            try:
                w, h = map(int, item.resolution.lower().split("x"))
                if w < self.min_width or h < self.min_height:
                    reasons.append(f"Resolution {item.resolution} is below minimum threshold ({self.min_width}x{self.min_height})")
            except Exception:
                reasons.append(f"Unparseable resolution string: '{item.resolution}'")

        # 3. Video duration checks
        if item.media_type == "video":
            if item.duration <= 0.0:
                reasons.append("Video duration is 0 or negative")
            elif item.duration < self.min_duration:
                reasons.append(f"Video duration ({item.duration}s) is below minimum required ({self.min_duration}s)")

        # 4. Caption & Prompt checks
        final_caption = item.final_caption.strip()
        if not final_caption or len(final_caption) < 10:
            reasons.append("Missing or insufficient text caption/prompt (< 10 characters)")

        # 5. Licensing & Training Permission checks
        if item.permission_state in (PermissionState.RESTRICTED.value, PermissionState.UNKNOWN.value):
            reasons.append(f"License permission state is '{item.permission_state}' (not eligible for automated training)")
            item.training_eligible = False
        elif item.permission_state == PermissionState.NEEDS_REVIEW.value:
            reasons.append("License requires manual verification before training admission")
            item.training_eligible = False

        # Determine overall status
        if reasons:
            item.validation_reasons = reasons
            # If failed on corruption/path/resolution -> INVALID
            if any("missing" in r.lower() or "corrupt" in r.lower() for r in reasons):
                item.validation_status = ValidationStatus.INVALID.value
                item.training_eligible = False
            elif any("below minimum" in r.lower() or "requires manual" in r.lower() or "insufficient" in r.lower() for r in reasons):
                item.validation_status = ValidationStatus.NEEDS_REVIEW.value
                item.training_eligible = False
            elif item.permission_state == PermissionState.RESTRICTED.value:
                item.validation_status = ValidationStatus.NEEDS_REVIEW.value
                item.training_eligible = False
        else:
            item.validation_status = ValidationStatus.VALID.value
            item.validation_reasons = []
            if item.permission_state in (PermissionState.LICENSE_VERIFIED.value, PermissionState.PERMISSION_RECORDED.value):
                item.training_eligible = True

        return item

    def detect_duplicates(self, items: List[DatasetItem]) -> None:
        """
        Identifies exact duplicates via SHA-256 and visually similar media via pHash.
        Updates item duplicate_status, duplicate_of, and validation_status without deleting files.
        """
        seen_sha256: Dict[str, str] = {}  # sha256 -> media_id
        seen_phash: List[Tuple[str, str]] = []  # list of (phash_str, media_id)

        for item in items:
            # 1. Exact SHA-256 duplicate check
            if item.sha256:
                if item.sha256 in seen_sha256:
                    item.duplicate_status = "duplicate_exact"
                    item.duplicate_of = seen_sha256[item.sha256]
                    item.validation_status = ValidationStatus.DUPLICATE.value
                    item.validation_reasons.append(f"Exact byte duplicate of item {item.duplicate_of}")
                    item.training_eligible = False
                    continue
                seen_sha256[item.sha256] = item.media_id

            # 2. Perceptual duplicate check (pHash)
            if item.phash:
                match_id = None
                for ref_phash, ref_id in seen_phash:
                    dist = hamming_distance(item.phash, ref_phash)
                    if dist <= self.max_phash_distance:
                        match_id = ref_id
                        break

                if match_id:
                    item.duplicate_status = "duplicate_perceptual"
                    item.duplicate_of = match_id
                    item.validation_status = ValidationStatus.DUPLICATE.value
                    item.validation_reasons.append(f"Perceptual visual duplicate (pHash dist <= {self.max_phash_distance}) of item {match_id}")
                    item.training_eligible = False
                else:
                    seen_phash.append((item.phash, item.media_id))

    def validate_dataset(self, items: List[DatasetItem]) -> Tuple[List[DatasetItem], Dict[str, int]]:
        """
        Orchestrates full dataset validation and deduplication.
        Returns (validated_items, counts_summary).
        """
        for item in items:
            self.validate_item(item)

        self.detect_duplicates(items)

        counts = {
            "total": len(items),
            "valid": sum(1 for it in items if it.validation_status == ValidationStatus.VALID.value),
            "invalid": sum(1 for it in items if it.validation_status == ValidationStatus.INVALID.value),
            "needs_review": sum(1 for it in items if it.validation_status == ValidationStatus.NEEDS_REVIEW.value),
            "duplicates": sum(1 for it in items if it.validation_status == ValidationStatus.DUPLICATE.value),
            "processing_failed": sum(1 for it in items if it.validation_status == ValidationStatus.PROCESSING_FAILED.value),
            "training_eligible": sum(1 for it in items if it.training_eligible),
        }
        return items, counts
