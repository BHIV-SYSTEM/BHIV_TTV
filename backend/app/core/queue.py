import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from app.core.exceptions import JobNotFoundError

class JobManager:
    """Manages asynchronous generation jobs and their execution states."""
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._lock is None or (loop and getattr(self._lock, "_loop", None) not in (None, loop)):
            self._lock = asyncio.Lock()
        return self._lock

    async def create_job(
        self,
        job_id: str,
        request_data: Dict[str, Any],
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with self._get_lock():
            resolved_settings = settings or {}
            aspect_ratio = resolved_settings.get("aspect_ratio") or request_data.get("aspect_ratio") or "16:9"
            visual_style = (
                resolved_settings.get("visual_style")
                or resolved_settings.get("style")
                or request_data.get("visual_style")
                or request_data.get("style")
                or "cinematic"
            )
            language = resolved_settings.get("language") or request_data.get("language") or "en"
            reference_type = request_data.get("reference_type")
            reference_url = request_data.get("reference_url")
            voice_enabled = request_data.get("voice", True)
            lipsync_enabled = request_data.get("lipsync", True)

            job = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "stage": "queued",
                "request": request_data,
                "settings": settings,
                "aspect_ratio": aspect_ratio,
                "visual_style": visual_style,
                "language": language,
                "reference_type": reference_type,
                "reference_url": reference_url,
                "tts_status": "pending" if voice_enabled else "skipped",
                "video_status": "pending",
                "lip_sync_status": "pending" if (lipsync_enabled and voice_enabled) else "skipped",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "result": None,
                "error": None
            }
            self._jobs[job_id] = job
            return job

    async def update_progress(self, job_id: str, stage: str, progress: int):
        async with self._get_lock():
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "processing"
                self._jobs[job_id]["stage"] = stage
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def update_stage_status(
        self,
        job_id: str,
        tts_status: Optional[str] = None,
        video_status: Optional[str] = None,
        lip_sync_status: Optional[str] = None,
    ):
        async with self._get_lock():
            if job_id in self._jobs:
                if tts_status is not None:
                    self._jobs[job_id]["tts_status"] = tts_status
                if video_status is not None:
                    self._jobs[job_id]["video_status"] = video_status
                if lip_sync_status is not None:
                    self._jobs[job_id]["lip_sync_status"] = lip_sync_status
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def complete_job(self, job_id: str, result_data: Dict[str, Any]):
        async with self._get_lock():
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["stage"] = "completed"
                self._jobs[job_id]["progress"] = 100
                self._jobs[job_id]["result"] = result_data
                if self._jobs[job_id]["tts_status"] != "skipped":
                    self._jobs[job_id]["tts_status"] = "completed"
                self._jobs[job_id]["video_status"] = "completed"
                if self._jobs[job_id]["lip_sync_status"] != "skipped":
                    self._jobs[job_id]["lip_sync_status"] = "completed"
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    async def fail_job(self, job_id: str, error_message: str):
        async with self._get_lock():
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["stage"] = "failed"
                self._jobs[job_id]["error"] = error_message
                if self._jobs[job_id]["tts_status"] == "processing":
                    self._jobs[job_id]["tts_status"] = "failed"
                if self._jobs[job_id]["video_status"] == "processing":
                    self._jobs[job_id]["video_status"] = "failed"
                if self._jobs[job_id]["lip_sync_status"] == "processing":
                    self._jobs[job_id]["lip_sync_status"] = "failed"
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self._jobs:
            raise JobNotFoundError(f"Job '{job_id}' not found.")
        return self._jobs[job_id]

    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        return self._jobs

job_manager = JobManager()
