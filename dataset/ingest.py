"""
Unified Dataset Ingestion Service.
Supports:
1. User-uploaded videos and images.
2. Explicitly submitted, legally permitted direct URLs (with SSRF protection).
3. Supported YouTube URLs through yt-dlp with permission/license extraction.
4. Legally usable text and story descriptions.
Maintains source provenance, licensing, and permission states.
"""
from __future__ import annotations

import os
import ipaddress
import socket
import urllib.parse
import urllib.request
import subprocess
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from dataset.models import DatasetItem, PermissionState, ValidationStatus
from dataset.preprocessor import MediaPreprocessor
from dataset.quality import calculate_sha256, calculate_phash
from app.core.logging import logger


# SSRF & Protocol Safety
ALLOWED_URL_SCHEMES = {"http", "https"}
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("10.0.0.0/8"),       # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),    # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),   # RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local / Cloud metadata
    ipaddress.ip_network("fc00::/7"),         # IPv6 ULA
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
]

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_TEXT_EXTENSIONS = {".txt", ".json", ".md"}


def validate_url_safety(url: str) -> Tuple[bool, str]:
    """
    Validates URL safety against SSRF, internal network scanning, and invalid schemes.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        return False, f"Prohibited URL scheme '{parsed.scheme}'. Only http and https are allowed."

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL."

    # Prevent direct loopback hostnames
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
        return False, f"Access to '{hostname}' is prohibited (SSRF prevention)."

    # Check if hostname itself is an IP address
    try:
        ip_obj = ipaddress.ip_address(hostname)
        for net in BLOCKED_IP_NETWORKS:
            if ip_obj in net:
                return False, f"IP {hostname} belongs to a prohibited private network ({net})."
        return True, "URL is safe"
    except ValueError:
        pass  # It's a domain name, not an IP literal

    # Resolve IP and verify against private / link-local subnets
    try:
        ip_addr_str = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_addr_str)
        for net in BLOCKED_IP_NETWORKS:
            if ip_obj in net:
                return False, f"Resolved IP {ip_addr_str} belongs to a prohibited private network ({net})."
    except Exception:
        # In offline or restricted network environments, check domain suffix for local patterns
        if hostname.endswith(".local") or hostname.endswith(".internal") or hostname.endswith(".lan"):
            return False, f"Access to local domain '{hostname}' is prohibited."

    return True, "URL is safe"


class DatasetIngestionService:
    """Orchestrates asset ingestion from uploads, direct URLs, YouTube, and text stories."""

    def __init__(self, storage_root: str = "data/datasets"):
        self.storage_root = Path(storage_root)
        self.raw_dir = self.storage_root / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = MediaPreprocessor(storage_root=storage_root)

    def _determine_permission_state(self, license_str: str, source_type: str) -> Tuple[str, bool]:
        """
        Determines permission state and training eligibility based on license and source.
        Returns (permission_state, training_eligible).
        """
        norm = (license_str or "").strip().lower()
        permissive = ("cc0", "public domain", "cc-by", "cc-by-4.0", "cc-by-3.0", "mit", "apache", "user-authorized", "creative commons attribution")
        restricted = ("all rights reserved", "standard youtube license", "copyrighted", "cc-by-nc", "nc", "non-commercial")

        if any(p in norm for p in permissive):
            return PermissionState.LICENSE_VERIFIED.value, True
        elif any(r in norm for r in restricted):
            return PermissionState.RESTRICTED.value, False
        elif source_type == "upload":
            return PermissionState.PERMISSION_RECORDED.value, True
        elif norm and norm != "unknown":
            return PermissionState.NEEDS_REVIEW.value, False
        else:
            return PermissionState.UNKNOWN.value, False

    def ingest_uploaded_file(
        self,
        file_bytes: bytes,
        original_filename: str,
        user_metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """
        Ingests a user-uploaded file (video, image, or text) with flexible metadata.
        """
        media_id = f"item_{uuid.uuid4().hex[:12]}"
        safe_name = self.preprocessor.sanitize_filename(original_filename)
        ext = os.path.splitext(safe_name)[1].lower()

        if ext in ALLOWED_VIDEO_EXTENSIONS:
            media_type = "video"
        elif ext in ALLOWED_IMAGE_EXTENSIONS:
            media_type = "image"
        elif ext in ALLOWED_TEXT_EXTENSIONS:
            media_type = "text"
        else:
            raise ValueError(f"Unsupported file extension '{ext}'.")

        # Save to raw storage safely
        dest_filename = f"{media_id}_{safe_name}"
        dest_path = self.raw_dir / dest_filename
        dest_path.write_bytes(file_bytes)

        # Compute SHA-256
        sha256_hash = calculate_sha256(str(dest_path))

        # Licensing & permission
        license_str = user_metadata.get("license") or "user-authorized"
        perm_state, training_eligible = self._determine_permission_state(license_str, "upload")

        # Preprocess based on media type
        phash = None
        keyframe_paths = []
        thumbnail_path = None
        duration = 0.0
        resolution = "720x480"
        fps = 24.0
        frame_count = 0
        validation_status = ValidationStatus.VALID.value
        validation_reasons = []

        if media_type == "video":
            info = self.preprocessor.inspect_and_preprocess_video(str(dest_path), media_id)
            if not info["valid"]:
                validation_status = ValidationStatus.PROCESSING_FAILED.value
                validation_reasons.append(info.get("error") or "Video preprocessing failed")
                training_eligible = False
            else:
                duration = info["duration"]
                resolution = info["resolution"]
                fps = info["fps"]
                frame_count = info["frame_count"]
                keyframe_paths = info["keyframe_paths"]
                thumbnail_path = info["thumbnail_path"]
                if keyframe_paths:
                    phash = calculate_phash(keyframe_paths[0])

        elif media_type == "image":
            info = self.preprocessor.inspect_and_preprocess_image(str(dest_path), media_id)
            if not info["valid"]:
                validation_status = ValidationStatus.PROCESSING_FAILED.value
                validation_reasons.append(info.get("error") or "Image preprocessing failed")
                training_eligible = False
            else:
                resolution = info["resolution"]
                thumbnail_path = info["thumbnail_path"]
                phash = calculate_phash(str(dest_path))

        elif media_type == "text":
            info = self.preprocessor.inspect_text(dest_path.read_text(encoding="utf-8", errors="ignore"))
            if not info["valid"]:
                validation_status = ValidationStatus.INVALID.value
                validation_reasons.append(info.get("error") or "Text content invalid")
                training_eligible = False

        item = DatasetItem(
            media_id=media_id,
            dataset_id=dataset_id,
            subject=user_metadata.get("subject", "general"),
            topic=user_metadata.get("topic", "general"),
            category=user_metadata.get("category", "general"),
            subcategory=user_metadata.get("subcategory", "general"),
            content_title=user_metadata.get("content_title") or safe_name,
            description=user_metadata.get("description", ""),
            characters=user_metadata.get("characters", []),
            environment=user_metadata.get("environment", "general"),
            actions=user_metadata.get("actions", []),
            animation_style=user_metadata.get("animation_style", "live_action"),
            visual_style=user_metadata.get("visual_style", "cinematic"),
            scene_type=user_metadata.get("scene_type", "landscape"),
            camera_movement=user_metadata.get("camera_movement", "static"),
            lighting=user_metadata.get("lighting", "natural"),
            source_type="upload",
            source_url="local://upload",
            license=license_str,
            license_url=user_metadata.get("license_url", ""),
            creator=user_metadata.get("creator", "user"),
            permission_state=perm_state,
            commercial_allowed=user_metadata.get("commercial_allowed", True),
            training_eligible=training_eligible,
            attribution=user_metadata.get("attribution", ""),
            media_type=media_type,
            original_prompt=user_metadata.get("prompt", ""),
            original_caption=user_metadata.get("caption", ""),
            duration=duration,
            resolution=resolution,
            fps=fps,
            frame_count=frame_count,
            keyframe_paths=keyframe_paths,
            thumbnail_path=thumbnail_path,
            storage_path=str(dest_path),
            sha256=sha256_hash,
            phash=phash,
            validation_status=validation_status,
            validation_reasons=validation_reasons,
            dataset_split=user_metadata.get("split", "train"),
            metadata=user_metadata,
        )
        return item

    def ingest_web_url(
        self,
        url: str,
        user_metadata: Dict[str, Any],
        dataset_id: str = "default",
        max_bytes: int = 250 * 1024 * 1024,  # 250MB limit
    ) -> DatasetItem:
        """
        Ingests media from an approved direct web URL with strict SSRF protection.
        """
        is_safe, reason = validate_url_safety(url)
        if not is_safe:
            raise ValueError(f"SSRF Safety Rejection: {reason}")

        media_id = f"item_{uuid.uuid4().hex[:12]}"
        parsed = urllib.parse.urlparse(url)
        raw_name = os.path.basename(parsed.path) or "downloaded_media.mp4"
        safe_name = self.preprocessor.sanitize_filename(raw_name)
        ext = os.path.splitext(safe_name)[1].lower() or ".mp4"

        dest_filename = f"{media_id}_{safe_name}"
        dest_path = self.raw_dir / dest_filename

        # Download with timeout and byte limit
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BHIV-TTV-DatasetCollector/2.0"},
        )
        downloaded = 0
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out_f:
            while chunk := resp.read(65536):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    out_f.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise ValueError(f"Downloaded media exceeded maximum limit of {max_bytes / (1024*1024):.0f} MB.")
                out_f.write(chunk)

        content_bytes = dest_path.read_bytes()
        dest_path.unlink()  # Processed via ingest_uploaded_file into canonical storage
        return self.ingest_uploaded_file(content_bytes, safe_name, user_metadata, dataset_id=dataset_id)

    def ingest_youtube_url(
        self,
        url: str,
        user_metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """
        Ingests video from YouTube using yt-dlp, capturing authentic metadata and license status.
        Respects platform terms: non-CC / copyrighted material is marked RESTRICTED and excluded from training.
        """
        is_safe, reason = validate_url_safety(url)
        if not is_safe:
            raise ValueError(f"SSRF Safety Rejection: {reason}")

        media_id = f"item_{uuid.uuid4().hex[:12]}"

        # 1. Probe video metadata with yt-dlp JSON dump
        probe_cmd = [
            "yt-dlp",
            "--dump-json",
            "--skip-download",
            "--no-warnings",
            url,
        ]
        try:
            res = subprocess.run(probe_cmd, capture_output=True, text=True, check=False, timeout=30)
            if res.returncode != 0 or not res.stdout.strip():
                raise RuntimeError(f"yt-dlp probe failed: {res.stderr.strip() or 'Unknown error'}")
            yt_meta = json.loads(res.stdout.strip().splitlines()[0])
        except Exception as exc:
            raise RuntimeError(f"Failed extracting YouTube metadata for {url}: {exc}") from exc

        title = yt_meta.get("title", "")
        description = yt_meta.get("description", "")
        uploader = yt_meta.get("uploader", "")
        yt_license = yt_meta.get("license") or "Standard YouTube License"
        duration = float(yt_meta.get("duration") or 0.0)

        # 2. Determine permission state
        perm_state, training_eligible = self._determine_permission_state(yt_license, "youtube")

        # 3. Download clip (capped at 720p to conserve bandwidth)
        output_tmpl = str(self.raw_dir / f"{media_id}_%(title).50s.%(ext)s")
        dl_cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_tmpl,
            "--no-playlist",
            "--max-filesize", "100M",
            url,
        ]
        try:
            dl_res = subprocess.run(dl_cmd, capture_output=True, text=True, check=False, timeout=120)
            if dl_res.returncode != 0:
                raise RuntimeError(f"yt-dlp download failed: {dl_res.stderr.strip()}")
        except Exception as exc:
            raise RuntimeError(f"Failed downloading YouTube video: {exc}") from exc

        # Find downloaded file
        candidates = list(self.raw_dir.glob(f"{media_id}_*"))
        if not candidates:
            raise FileNotFoundError(f"yt-dlp completed but output file not found in {self.raw_dir}")
        downloaded_path = candidates[0]

        # 4. Preprocess video
        sha256_hash = calculate_sha256(str(downloaded_path))
        info = self.preprocessor.inspect_and_preprocess_video(str(downloaded_path), media_id)
        phash = calculate_phash(info["keyframe_paths"][0]) if info.get("keyframe_paths") else None

        # Build merged metadata
        merged_meta = {**yt_meta, **user_metadata}

        item = DatasetItem(
            media_id=media_id,
            dataset_id=dataset_id,
            subject=user_metadata.get("subject", "general"),
            topic=user_metadata.get("topic", "general"),
            category=user_metadata.get("category", "general"),
            content_title=title,
            description=description,
            characters=user_metadata.get("characters", []),
            environment=user_metadata.get("environment", "general"),
            actions=user_metadata.get("actions", []),
            animation_style=user_metadata.get("animation_style", "live_action"),
            visual_style=user_metadata.get("visual_style", "cinematic"),
            source_type="youtube",
            source_url=url,
            license=yt_license,
            license_url="https://www.youtube.com/t/terms",
            creator=uploader,
            permission_state=perm_state,
            commercial_allowed=False,
            training_eligible=training_eligible,
            attribution=f"Video by {uploader} on YouTube ({url})",
            media_type="video",
            original_prompt=user_metadata.get("prompt", title),
            original_caption=description[:300] if description else title,
            duration=info.get("duration", duration),
            resolution=info.get("resolution", "1280x720"),
            fps=info.get("fps", 24.0),
            frame_count=info.get("frame_count", 0),
            keyframe_paths=info.get("keyframe_paths", []),
            thumbnail_path=info.get("thumbnail_path"),
            storage_path=str(downloaded_path),
            sha256=sha256_hash,
            phash=phash,
            validation_status=ValidationStatus.VALID.value if info.get("valid") else ValidationStatus.PROCESSING_FAILED.value,
            validation_reasons=[] if info.get("valid") else [info.get("error", "Processing error")],
            dataset_split=user_metadata.get("split", "train"),
            metadata=merged_meta,
        )
        return item

    def ingest_text_story(
        self,
        story_text: str,
        user_metadata: Dict[str, Any],
        dataset_id: str = "default",
    ) -> DatasetItem:
        """
        Ingests cultural stories, mythology narratives, or detailed prompts as dataset text samples.
        """
        media_id = f"item_{uuid.uuid4().hex[:12]}"
        clean_text = story_text.strip()
        safe_title = self.preprocessor.sanitize_filename(user_metadata.get("content_title") or "story_sample")
        dest_filename = f"{media_id}_{safe_title}.txt"
        dest_path = self.raw_dir / dest_filename
        dest_path.write_text(clean_text, encoding="utf-8")

        sha256_hash = calculate_sha256(str(dest_path))
        info = self.preprocessor.inspect_text(clean_text)

        license_str = user_metadata.get("license") or "user-authorized"
        perm_state, training_eligible = self._determine_permission_state(license_str, "text_story")

        item = DatasetItem(
            media_id=media_id,
            dataset_id=dataset_id,
            subject=user_metadata.get("subject", "mythology"),
            topic=user_metadata.get("topic", "cultural_story"),
            category=user_metadata.get("category", "narrative"),
            content_title=user_metadata.get("content_title", "Cultural Narrative"),
            description=clean_text[:200],
            characters=user_metadata.get("characters", []),
            environment=user_metadata.get("environment", "historical"),
            actions=user_metadata.get("actions", []),
            animation_style=user_metadata.get("animation_style", "cinematic_3d"),
            visual_style=user_metadata.get("visual_style", "mythological"),
            source_type="text_story",
            source_url=user_metadata.get("source_url", "local://text"),
            license=license_str,
            license_url=user_metadata.get("license_url", ""),
            creator=user_metadata.get("creator", "author"),
            permission_state=perm_state,
            commercial_allowed=True,
            training_eligible=training_eligible and info["valid"],
            attribution=user_metadata.get("attribution", ""),
            media_type="text",
            original_prompt=user_metadata.get("prompt") or clean_text[:250],
            original_caption=user_metadata.get("caption") or clean_text,
            storage_path=str(dest_path),
            sha256=sha256_hash,
            validation_status=ValidationStatus.VALID.value if info["valid"] else ValidationStatus.INVALID.value,
            validation_reasons=[] if info["valid"] else [info["error"]],
            dataset_split=user_metadata.get("split", "train"),
            metadata=user_metadata,
        )
        return item
