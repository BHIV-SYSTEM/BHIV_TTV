"""Reference upload and URL processing for the TTV pipeline."""
from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
from PIL import Image

from app.adapters.reference import get_reference_adapter
from app.adapters.reference.validator import validate_media_file, validate_public_url
from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import telemetry
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
ALLOWED_VIDEO_EXT = {".mp4", ".webm", ".mov"}
ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg"}
ALLOWED_VIDEO_MIME = {"video/mp4", "video/webm", "video/quicktime"}


class ReferenceService:
    def __init__(self):
        self.root = settings.get_absolute_path(settings.REFERENCES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    def _safe_ext(self, filename: str) -> str:
        ext = Path(filename or "").suffix.lower()
        if ext in ALLOWED_IMAGE_EXT or ext in ALLOWED_VIDEO_EXT:
            return ext
        raise ValidationError("Unsupported reference file type. Upload an image (JPG/PNG/WEBP) or video (MP4/WEBM).")

    def save_upload(self, filename: str, content: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        if not content:
            raise ValidationError("The uploaded reference file is empty.")
        if len(content) > settings.REFERENCE_MAX_BYTES:
            raise ValidationError("The uploaded reference file is too large.")
        ext = self._safe_ext(filename)
        mime = (content_type or "").split(";")[0].strip().lower()
        if mime:
            if ext in ALLOWED_IMAGE_EXT and mime not in ALLOWED_IMAGE_MIME and not mime.startswith("image/"):
                raise ValidationError("The uploaded file MIME type does not match a supported image format.")
            if ext in ALLOWED_VIDEO_EXT and mime not in ALLOWED_VIDEO_MIME and not mime.startswith("video/"):
                raise ValidationError("The uploaded file MIME type does not match a supported video format.")

        ref_id = f"ref_{uuid.uuid4().hex[:12]}"
        dest_dir = self.root / ref_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"source{ext}"
        try:
            dest_path.write_bytes(content)
            inspected = validate_media_file(
                str(dest_path),
                max_bytes=settings.REFERENCE_MAX_BYTES,
                max_duration=settings.REFERENCE_MAX_DURATION,
            )
            media_type = "image" if ext in ALLOWED_IMAGE_EXT else "video"
            if inspected["media_type"] != media_type:
                raise ValidationError("The uploaded file extension does not match its media format.")
            if media_type == "video":
                self.extract_still(str(dest_path), media_type, ref_id)
            payload = self._describe(str(dest_path), media_type, source="upload", reference_id=ref_id)
        except Exception:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        telemetry.emit("reference_uploaded", ref_id, {"media_type": media_type, "bytes": len(content)})
        return payload

    async def ingest_url(self, url: str, reference_id: Optional[str] = None) -> Dict[str, Any]:
        safe_url, source = validate_public_url(url, resolve_dns=True)
        ref_id = reference_id or f"ref_{uuid.uuid4().hex[:12]}"
        dest_dir = str(self.root / ref_id)
        os.makedirs(dest_dir, exist_ok=True)
        adapter = get_reference_adapter(safe_url)
        retrieved = await adapter.retrieve(safe_url, dest_dir)
        inspected = validate_media_file(
            retrieved["path"],
            max_bytes=settings.REFERENCE_MAX_BYTES,
            max_duration=settings.REFERENCE_MAX_DURATION,
        )
        if inspected["media_type"] != retrieved["media_type"]:
            raise ValidationError("The downloaded file's media format did not match its metadata.")
        if retrieved["media_type"] == "video":
            self.extract_still(retrieved["path"], retrieved["media_type"], ref_id)
        payload = self._describe(
            retrieved["path"],
            retrieved["media_type"],
            source=retrieved.get("source", source),
            reference_id=ref_id,
        )
        payload["source_url_host"] = re.sub(r"^www\.", "", (safe_url.split("/")[2] if "://" in safe_url else ""))
        telemetry.emit("reference_url_ingested", ref_id, {"source": payload["source"], "media_type": payload["media_type"]})
        return payload

    def load_existing(self, reference_id: str) -> Dict[str, Any]:
        if not reference_id or not re.fullmatch(r"ref_[a-f0-9]{12}", reference_id):
            raise ValidationError("Invalid reference identifier.")
        dest_dir = self.root / reference_id
        if not dest_dir.exists():
            raise ValidationError("The referenced upload could not be found. Please upload it again.")
        files = list(dest_dir.glob("source.*")) + list(dest_dir.glob("reference.*"))
        if not files:
            raise ValidationError("The referenced upload is missing its media file.")
        path = str(files[0])
        ext = Path(path).suffix.lower()
        media_type = "image" if ext in ALLOWED_IMAGE_EXT else "video"
        return self._describe(path, media_type, source="upload", reference_id=reference_id)

    async def resolve(
        self,
        reference_url: Optional[str] = None,
        reference_id: Optional[str] = None,
        reference_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if reference_id:
            payload = self.load_existing(reference_id)
        elif reference_url:
            payload = await self.ingest_url(reference_url)
        else:
            return None
        if reference_type in {"image", "video"}:
            payload["requested_type"] = reference_type
        payload["still_path"] = self.extract_still(payload["path"], payload["media_type"], payload["reference_id"])
        return payload

    def extract_still(self, media_path: str, media_type: str, reference_id: str) -> str:
        if media_type == "image":
            return media_path
        dest = str(self.root / reference_id / "still.jpg")
        if os.path.exists(dest):
            return dest
        import subprocess
        try:
            cmd = ['ffmpeg', '-y', '-ss', '0.5', '-i', media_path, '-vframes', '1', '-q:v', '2', dest]
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            if res.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
                return dest
        except Exception:
            pass
        cap = cv2.VideoCapture(media_path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise ValidationError("Could not extract a still frame from the reference video.")
        if not cv2.imwrite(dest, frame):
            raise ValidationError("Could not save a still frame from the reference video.")
        return dest

    def _describe(self, path: str, media_type: str, source: str, reference_id: str) -> Dict[str, Any]:
        width = height = None
        duration = 0.0
        fps = 24
        if media_type == "image":
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                pass
        else:
            try:
                from training.preprocessing.long_video_processor import LongVideoProcessor
                processor = LongVideoProcessor(max_duration=getattr(settings, 'REFERENCE_MAX_DURATION', 3600.0))
                meta = processor.validate_and_extract_metadata(path)
                width = meta.get('width') or 1280
                height = meta.get('height') or 720
                fps = int(meta.get('fps') or 24)
                duration = meta.get('duration') or 5.0
            except Exception:
                try:
                    cap = cv2.VideoCapture(path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
                    fps_cap = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
                    fps = int(fps_cap) if fps_cap > 0 else 24
                    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    duration = round(frame_count / max(1.0, fps_cap), 2) if frame_count > 0 else 5.0
                    cap.release()
                except Exception:
                    pass

        fname = os.path.basename(path)
        still_file = self.root / reference_id / "still.jpg"
        if still_file.exists():
            preview_url = f"/generated/references/{reference_id}/still.jpg"
            still_path = str(still_file)
        else:
            preview_url = f"/generated/references/{reference_id}/{fname}"
            still_path = path if media_type == "image" else None

        return {
            "reference_id": reference_id,
            "path": path,
            "media_type": media_type,
            "source": source,
            "filename": fname,
            "width": width or 1280,
            "height": height or 720,
            "duration": duration,
            "fps": fps,
            "still_path": still_path,
            "preview_url": preview_url,
        }


    def cleanup(self, reference_id: Optional[str]) -> None:
        if not reference_id:
            return
        dest = self.root / reference_id
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)


reference_service = ReferenceService()
