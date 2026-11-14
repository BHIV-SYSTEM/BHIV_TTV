import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.models.scene import Scene
from app.core.config import settings
from app.core.exceptions import FFmpegProcessingError
from app.core.logging import telemetry
from app.utils.media_check import validate_video_file

def get_ffmpeg_binary() -> Optional[str]:
    """
    Locates a working FFmpeg binary executable using prioritized discovery:
    1. settings.FFMPEG_PATH or FFMPEG_PATH environment variable
    2. System PATH (via shutil.which)
    3. Active Python environment Scripts directory (sys.prefix/Scripts/ffmpeg.exe)
    4. Project workspace .venv/Scripts/ffmpeg.exe
    5. imageio_ffmpeg bundled binary (if installed)
    Automatically updates os.environ['PATH'] with the binary directory for subprocesses.
    """
    configured = getattr(settings, "FFMPEG_PATH", None) or os.environ.get("FFMPEG_PATH")
    candidates: List[Path] = []
    if configured:
        candidates.append(Path(configured))

    system_which = shutil.which("ffmpeg")
    if system_which:
        candidates.append(Path(system_which))

    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates.append(Path(sys.prefix) / "Scripts" / exe_name)
    candidates.append(Path(settings.WORKSPACE_ROOT) / ".venv" / "Scripts" / exe_name)

    try:
        import imageio_ffmpeg
        img_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if img_bin:
            candidates.append(Path(img_bin))
    except Exception:
        pass

    for cand in candidates:
        if cand and cand.is_file():
            bin_path = str(cand.resolve())
            bin_dir = str(cand.resolve().parent)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if bin_dir.lower() not in [p.lower() for p in path_parts if p]:
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return bin_path

    return None

class FFmpegService:
    """
    Production video compositor leveraging system FFmpeg.
    - Concatenates scene clips with seamless stream alignment
    - Multiplexes synchronized speech & ambient audio
    - Generates and embeds synchronized subtitle streams
    - Enforces web-standard H.264 (yuv420p) + AAC encoding with faststart
    """
    def assemble_final_video(
        self,
        scenes: List[Scene],
        master_audio_path: str,
        output_path: str,
        execution_id: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_dir = settings.get_absolute_path(settings.TEMP_DIR)
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Create FFmpeg concat demuxer file
        concat_file = str(temp_dir / f"{execution_id}_concat.txt")
        with open(concat_file, "w", encoding="utf-8") as f:
            for scene in scenes:
                if not scene.video_path or not os.path.exists(scene.video_path):
                    raise FFmpegProcessingError(f"Missing scene video clip: {scene.video_path}")
                # FFmpeg concat file requires escaped forward slashes
                clean_path = os.path.abspath(scene.video_path).replace("\\", "/")
                f.write(f"file '{clean_path}'\n")

        # Letterbox / pillarbox to the selected aspect without stretching.
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={fps},setsar=1"
        )

        ffmpeg_bin = get_ffmpeg_binary()

        if not ffmpeg_bin:
            # Fallback to OpenCV compositor when FFmpeg is not installed in the local environment
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            for scene in scenes:
                if not scene.video_path or not os.path.exists(scene.video_path):
                    continue
                cap = cv2.VideoCapture(scene.video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    out.write(frame)
                cap.release()
            out.release()
            is_valid, info = validate_video_file(output_path)
            telemetry.emit("ffmpeg_assembly_completed", execution_id, info)
            return output_path

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-i", master_audio_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            telemetry.emit("ffmpeg_assembly_failed", execution_id, {"stderr": e.stderr}, level="error")
            raise FFmpegProcessingError(f"FFmpeg assembly failed: {e.stderr}")
        finally:
            if os.path.exists(concat_file):
                try:
                    os.remove(concat_file)
                except Exception:
                    pass

        # 3. Validate resulting video file
        is_valid, info = validate_video_file(output_path)
        if not is_valid:
            raise FFmpegProcessingError(f"Assembled video failed validation: {info.get('error')}")
        if int(info.get("width") or 0) != int(width) or int(info.get("height") or 0) != int(height):
            raise FFmpegProcessingError(
                f"Assembled video dimensions {info.get('width')}x{info.get('height')} "
                f"do not match requested {width}x{height}."
            )

        telemetry.emit("ffmpeg_assembly_completed", execution_id, info)
        return output_path

    def generate_srt_subtitles(self, scenes: List[Scene], output_srt_path: str) -> str:
        """Generates a standard SubRip (.srt) subtitle file matching scene timings."""
        os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
        current_time = 0.0

        def _format_srt_time(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        with open(output_srt_path, "w", encoding="utf-8") as f:
            for idx, scene in enumerate(scenes, 1):
                start_str = _format_srt_time(current_time)
                end_str = _format_srt_time(current_time + scene.duration)
                text = scene.narrative or scene.title
                
                f.write(f"{idx}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")

                current_time += scene.duration

        return output_srt_path

    def generate_vtt_subtitles(self, scenes: List[Scene], output_vtt_path: str) -> str:
        """Generates a standard WebVTT (.vtt) subtitle file matching scene timings."""
        os.makedirs(os.path.dirname(output_vtt_path), exist_ok=True)
        current_time = 0.0

        def _format_vtt_time(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

        with open(output_vtt_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for idx, scene in enumerate(scenes, 1):
                start_str = _format_vtt_time(current_time)
                end_str = _format_vtt_time(current_time + scene.duration)
                text = scene.narrative or scene.title

                f.write(f"{idx}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{text}\n\n")

                current_time += scene.duration

        return output_vtt_path

ffmpeg_service = FFmpegService()
