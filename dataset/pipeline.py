"""
Unified Dataset Pipeline Orchestrator.
Coordinates ingestion, licensing verification, quality assurance, deduplication, captioning,
and manifest building for general-purpose Text-to-Video datasets across diverse visual domains.
"""
from __future__ import annotations

import os
import json
import shutil
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random

from dataset.models import (
    AssetMetadata, QualityReport, DatasetItem, DatasetVersionMetadata,
    PermissionState, ValidationStatus
)
from dataset.quality import (
    verify_license,
    calculate_sha256,
    assess_image_quality,
    hamming_distance,
)
from dataset.caption import auto_captioner
from dataset.captioning import CaptionService
from dataset.preprocessor import MediaPreprocessor
from dataset.ingest import DatasetIngestionService
from dataset.deduplicate import DatasetValidator
from dataset.manifest import DatasetManager
from dataset.sources.wikimedia import wikimedia_source
from dataset.sources.archive_org import archive_org_source
from dataset.sources.synthetic import synthetic_source
from dataset.downloader import download_resumable, extension_for_url, DownloadError
from dataset.video_processor import video_processor
from dataset.registry import DatasetPolicyError, assert_candidate_allowed
from app.core.logging import logger


class DatasetPipeline:
    """
    Centralized, extensible orchestrator for collecting, preprocessing,
    captioning, validating, and preparing datasets for Text-to-Video training.
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.images_dir = self.base_dir / "images"
        self.videos_dir = self.base_dir / "videos"
        self.frames_dir = self.base_dir / "frames"
        self.staging_dir = self.base_dir / "staging"
        self.processed_dir = self.base_dir / "processed"
        self.rejected_dir = self.base_dir / "rejected"
        self.manifests_dir = self.base_dir / "manifests"
        self.metadata_dir = self.base_dir / "metadata"
        self.embeddings_dir = self.base_dir / "embeddings"
        self.captions_dir = self.base_dir / "captions"
        self.annotations_dir = self.base_dir / "annotations"

        self.ensure_directories()

        # Integrated sub-services
        self.ingest_service = DatasetIngestionService(storage_root=str(self.base_dir / "datasets"))
        self.preprocessor = MediaPreprocessor(storage_root=str(self.base_dir / "datasets"))
        self.caption_service = CaptionService()
        self.validator = DatasetValidator()
        self.dataset_manager = DatasetManager(storage_root=str(self.base_dir / "datasets"))

    def ensure_directories(self):
        for d in [
            self.raw_dir,
            self.images_dir,
            self.videos_dir,
            self.frames_dir,
            self.staging_dir,
            self.processed_dir,
            self.rejected_dir,
            self.manifests_dir,
            self.metadata_dir,
            self.embeddings_dir,
            self.captions_dir,
            self.annotations_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Create dataset category subsets
        dataset_subsets = Path("datasets")
        for cat in [
            "human_actions",
            "talking_faces",
            "environments",
            "objects",
            "vehicles",
            "nature",
            "cinematic",
            "multilingual_talking",
        ]:
            (dataset_subsets / cat).mkdir(parents=True, exist_ok=True)
            (dataset_subsets / "manifests").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Unified Ingestion APIs (Phase 4)
    # ------------------------------------------------------------------

    def collect_uploaded_file(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """Collects and preprocesses user-uploaded media with flexible metadata."""
        item = self.ingest_service.ingest_uploaded_file(file_bytes, filename, metadata, dataset_id=dataset_id)
        item = self.caption_service.enrich_item(item)
        item = self.validator.validate_item(item)
        self.dataset_manager.save_items(dataset_id, [item])
        return item

    def collect_from_url(
        self,
        url: str,
        metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """Collects media from an allowlisted web URL with SSRF protection."""
        item = self.ingest_service.ingest_web_url(url, metadata, dataset_id=dataset_id)
        item = self.caption_service.enrich_item(item)
        item = self.validator.validate_item(item)
        self.dataset_manager.save_items(dataset_id, [item])
        return item

    def collect_from_youtube(
        self,
        youtube_url: str,
        metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """Collects video from YouTube via yt-dlp, tracking CC-BY vs copyrighted license states."""
        item = self.ingest_service.ingest_youtube_url(youtube_url, metadata, dataset_id=dataset_id)
        item = self.caption_service.enrich_item(item)
        item = self.validator.validate_item(item)
        self.dataset_manager.save_items(dataset_id, [item])
        return item

    def collect_text_story(
        self,
        story_text: str,
        metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """Collects cultural story descriptions or narratives."""
        item = self.ingest_service.ingest_text_story(story_text, metadata, dataset_id=dataset_id)
        item = self.caption_service.enrich_item(item)
        item = self.validator.validate_item(item)
        self.dataset_manager.save_items(dataset_id, [item])
        return item

    def validate_and_deduplicate(self, dataset_id: str = "default") -> Dict[str, Any]:
        """Runs deduplication and validation across all items in a dataset."""
        items = self.dataset_manager.load_items(dataset_id)
        items, counts = self.validator.validate_dataset(items)
        self.dataset_manager.save_items(dataset_id, items)
        return counts

    def prepare_training_dataset(
        self,
        dataset_id: str = "default",
        version: str = "v1.0",
        name: str = "Training Dataset",
        description: str = "",
    ) -> DatasetVersionMetadata:
        """Partitions valid items and exports a reproducible training manifest."""
        return self.dataset_manager.create_version_snapshot(
            dataset_id=dataset_id,
            name=name,
            version=version,
            description=description,
        )

    def prepare_sample_openvid(self) -> Dict[str, Any]:
        """Prepares the existing OpenVid-1M sample dataset without modifying original files."""
        return self.dataset_manager.prepare_openvid_sample_dataset()

    # ------------------------------------------------------------------
    # Legacy & Registry-Gated Ingestion APIs
    # ------------------------------------------------------------------

    def ingest_assets(
        self,
        source: str = "synthetic",
        query: str = "farmer speaking",
        limit: int = 10,
        license_filter: Optional[str] = None,
        media_type: str = "image",
        min_resolution: Optional[tuple[int, int]] = None,
        max_download_gb: int = 100,
    ) -> List[AssetMetadata]:
        ingested = []

        if source == "wikimedia":
            raw_candidates = wikimedia_source.search_assets(query, limit=limit, media_type=media_type)
        elif source == "archive_org":
            raw_candidates = archive_org_source.search_assets(query, limit=limit, media_type=media_type)
        else:
            raw_candidates = []
            categories = ["talking_faces", "human_actions", "environments", "nature", "multilingual_talking"]
            for i in range(1, limit + 1):
                cat = categories[(i - 1) % len(categories)]
                dest = str(self.raw_dir / f"syn_{cat}_{i:03d}.jpg")
                info = synthetic_source.generate_synthetic_image(cat, dest, index=i)
                raw_candidates.append(info)

        for cand in raw_candidates:
            if source not in {"wikimedia", "synthetic"}:
                self._reject(cand.get("asset_id", "unknown"), "manual_review_required", source)
                continue
            asset = self.process_candidate(
                cand, query=query, license_filter=license_filter,
                min_resolution=min_resolution, max_download_bytes=max_download_gb * 1024 ** 3,
            )
            if asset:
                ingested.append(asset)

        return ingested

    def process_candidate(
        self, cand: Dict[str, Any], query: str = "", license_filter: Optional[str] = None,
        min_resolution: Optional[tuple[int, int]] = None, max_download_bytes: int = 100 * 1024 ** 3,
    ) -> Optional[AssetMetadata]:
        asset_id = cand.get("asset_id", "asset_unknown")
        src_url = cand.get("source_url", "")
        license_str = cand.get("license", "unknown")

        if cand.get("source_name") != "internal-synthetic":
            try:
                assert_candidate_allowed(cand, "wikimedia_commons")
            except DatasetPolicyError as exc:
                self._reject(asset_id, "provenance_policy_failed", str(exc))
                return None

        lic_info = verify_license(license_str)
        requested_license = (license_filter or "").strip().lower()
        matches_requested_license = not requested_license or requested_license in license_str.lower()
        if not lic_info.training_eligible or not matches_requested_license:
            rej_path = self.rejected_dir / f"{asset_id}_rejected.json"
            rej_path.write_text(json.dumps({"asset_id": asset_id, "reason": "unsupported_license", "details": lic_info.__dict__}))
            return None

        media_type = cand.get("media_type", "image")
        suffix = extension_for_url(cand.get("download_url") or src_url, media_type)
        local_path = cand.get("local_path") or str(self.raw_dir / f"{asset_id}{suffix}")
        if not os.path.exists(local_path):
            download_url = cand.get("download_url") or src_url
            if cand.get("source_name") == "internal-synthetic":
                synthetic_source.generate_synthetic_image(cand.get("category", "general"), local_path)
            elif not download_url or not download_url.startswith(("http://", "https://")):
                self._reject(asset_id, "missing_download_url")
                return None
            else:
                try:
                    download_resumable(download_url, Path(local_path), max_download_bytes)
                except DownloadError as exc:
                    self._reject(asset_id, "download_failed", str(exc))
                    return None

        sha256 = calculate_sha256(local_path)

        if media_type == "video":
            video_info = video_processor.process_video(
                local_path, str(self.staging_dir / asset_id), target_duration=4
            )
            if not video_info.get("valid"):
                self._reject(asset_id, "invalid_video", video_info.get("error"))
                return None
            width, height = int(video_info["width"]), int(video_info["height"])
            quality_metrics = None
            if min_resolution and (width < min_resolution[0] or height < min_resolution[1]):
                self._reject(asset_id, "below_minimum_resolution", f"{width}x{height}")
                return None
            phash = None
            duration = float(video_info["duration"])
        else:
            quality_metrics = assess_image_quality(
                local_path,
                min_width=(min_resolution or (512, 512))[0],
                min_height=(min_resolution or (512, 512))[1],
            )
            width, height = quality_metrics.width, quality_metrics.height
            phash = quality_metrics.phash
            duration = 0.0

        if quality_metrics and not quality_metrics.passed_qc:
            rej_path = self.rejected_dir / f"{asset_id}_qc_failed.json"
            rej_path.write_text(json.dumps({"asset_id": asset_id, "reasons": quality_metrics.failure_reasons}))
            return None

        caption_meta = auto_captioner.generate_caption(
            prompt_or_description=query,
            context={"subject": cand.get("category", "general")},
        )

        final_path = self.processed_dir / f"{asset_id}{suffix}"
        if not final_path.exists():
            shutil.copy2(local_path, str(final_path))

        meta = AssetMetadata(
            asset_id=asset_id,
            source_url=src_url,
            source_name=cand.get("source_name", "unknown"),
            license=license_str,
            license_url=cand.get("license_url"),
            creator=cand.get("creator"),
            media_type=media_type,
            resolution=f"{width}x{height}",
            duration=duration,
            sha256=sha256,
            phash=phash,
            allowed_for_training=True,
            category=cand.get("category", "general"),
            local_path=str(final_path),
            quality_metrics=quality_metrics.__dict__ if quality_metrics else video_info,
            caption_metadata=caption_meta.__dict__,
        )

        meta_path = self.metadata_dir / f"{asset_id}.json"
        meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return meta

    def _reject(self, asset_id: str, reason: str, details: Any = None):
        rej_file = self.rejected_dir / f"{asset_id}_{reason}.json"
        rej_file.write_text(json.dumps({"asset_id": asset_id, "reason": reason, "details": details}), encoding="utf-8")
