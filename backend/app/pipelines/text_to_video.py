import time
import json
from typing import Dict, Any, List, Optional, Callable
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry
from app.core.security import governance_guard
from app.core.exceptions import VisualGenerationError
from app.core.queue import job_manager
from app.services.prompt_service import prompt_service
from app.services.story_service import story_service
from app.services.scene_service import scene_service
from app.services.vision_service import vision_service
from app.services.image_service import image_service
from app.services.video_service import video_service
from app.services.tts_service import tts_service
from app.services.audio_service import audio_service
from app.services.ffmpeg_service import ffmpeg_service
from app.services.storage_service import storage_service
from app.services.reference_service import reference_service
from app.services.lipsync_service import lipsync_service
from app.services.localization_service import localization_service
from app.services.character_service import character_service
from app.services.realtime_data_service import realtime_data_service
from app.services.prompt_constraints import prompt_constraint_service
from app.utils.media_check import validate_video_file
from app.utils.hashing import generate_execution_id
from app.utils.video_settings import resolve_video_settings
from app.utils.languages import normalize_language

class TextToVideoPipeline:
    """
    Master Text-to-Video orchestration pipeline.
    TEXT -> SETTINGS -> REFERENCE -> STORY -> SCENES -> VISUALS -> TTS -> LIPSYNC -> FFMPEG -> MP4
    """
    def __init__(self):
        self.governance = governance_guard

    def validate_prompt(self, prompt: str, execution_id: str) -> str:
        telemetry.emit("stage_1_validate_prompt", execution_id, {"prompt": prompt})
        return prompt_service.validate_prompt(prompt)

    async def understand_prompt(self, prompt: str, execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_2_understand_prompt", execution_id)
        return await prompt_service.understand_prompt(prompt)

    async def generate_story(self, prompt: str, analysis: Dict[str, Any], duration: int, execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_3_generate_story", execution_id, {"duration": duration})
        return await story_service.generate_story(prompt, analysis, duration)

    async def generate_script(self, story: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_4_generate_script", execution_id)
        return story

    async def generate_scenes(self, script: Dict[str, Any], duration: int, style: str, execution_id: str) -> List[Scene]:
        telemetry.emit("stage_5_generate_scenes", execution_id, {"style": style})
        return await scene_service.generate_scenes(script, duration, style)

    def generate_visual_prompts(
        self,
        scenes: List[Scene],
        style: str,
        analysis: Dict[str, Any],
        execution_id: str,
        aspect_ratio: str = "16:9",
        reference: Optional[Dict[str, Any]] = None,
    ) -> List[Scene]:
        telemetry.emit("stage_6_generate_visual_prompts", execution_id, {"style": style, "aspect_ratio": aspect_ratio})
        enriched = vision_service.apply_visual_consistency(
            scenes, style, analysis, aspect_ratio=aspect_ratio, reference=reference
        )
        constraints = prompt_constraint_service.build(str(analysis.get("raw_prompt", "")), analysis, style)
        return prompt_constraint_service.apply_to_scenes(enriched, constraints)

    async def generate_visual_assets(
        self,
        scenes: List[Scene],
        execution_id: str,
        width: int = 1280,
        height: int = 720,
        reference: Optional[Dict[str, Any]] = None,
    ) -> List[Scene]:
        telemetry.emit("stage_7_generate_visual_assets", execution_id, {"width": width, "height": height})
        return await image_service.generate_scene_keyframes(scenes, execution_id, width=width, height=height, reference=reference)

    async def generate_scene_videos(
        self,
        scenes: List[Scene],
        execution_id: str,
        fps: int = 24,
        width: int = 1280,
        height: int = 720,
    ) -> List[Scene]:
        telemetry.emit("stage_8_generate_scene_videos", execution_id, {"fps": fps, "width": width, "height": height})
        return await video_service.generate_scene_videos(scenes, execution_id, fps, width=width, height=height)

    async def generate_voice(
        self,
        scenes: List[Scene],
        voice_enabled: bool,
        execution_id: str,
        language: str = "en",
        voice: Optional[str] = None,
    ) -> List[Scene]:
        telemetry.emit("stage_9_generate_voice", execution_id, {"enabled": voice_enabled, "language": language})
        if voice_enabled:
            return await tts_service.generate_scene_narration(scenes, execution_id, language=language, voice=voice)
        return scenes

    def process_audio(self, scenes: List[Scene], total_duration: float, execution_id: str) -> str:
        telemetry.emit("stage_10_process_audio", execution_id)
        temp_dir = settings.get_absolute_path(settings.TEMP_DIR)
        audio_out = str(temp_dir / f"{execution_id}_master_audio.wav")
        return audio_service.mix_complete_audio(scenes, total_duration, audio_out)

    def assemble_video(
        self,
        scenes: List[Scene],
        master_audio_path: str,
        execution_id: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> str:
        telemetry.emit("stage_11_assemble_video", execution_id, {"width": width, "height": height, "fps": fps})
        temp_dir = settings.get_absolute_path(settings.TEMP_DIR)
        raw_final = str(temp_dir / f"{execution_id}_assembled.mp4")
        return ffmpeg_service.assemble_final_video(
            scenes, master_audio_path, raw_final, execution_id, width=width, height=height, fps=fps
        )

    def validate_video(self, video_path: str, execution_id: str, expected_duration: Optional[float] = None) -> Dict[str, Any]:
        telemetry.emit("stage_12_validate_video", execution_id)
        is_valid, info = validate_video_file(video_path)
        if not is_valid:
            raise VisualGenerationError(f"Video validation failed: {info.get('error')}")
        if expected_duration is not None:
            actual_duration = float(info.get("duration") or 0.0)
            if abs(actual_duration - expected_duration) > 1.5:
                raise VisualGenerationError(
                    f"Generated video duration mismatch: actual {actual_duration:.2f}s "
                    f"does not match requested duration {expected_duration:.2f}s."
                )
        return info

    def store_output(
        self,
        video_path: str,
        execution_id: str,
        prompt: str,
        scenes: List[Scene],
        duration: float,
        style: str,
        resolution: str,
        fps: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        telemetry.emit("stage_13_store_output", execution_id)
        return storage_service.persist_video_artifact(
            execution_id=execution_id,
            video_path=video_path,
            prompt=prompt,
            scenes=scenes,
            duration=duration,
            style=style,
            resolution=resolution,
            fps=fps,
            extra=extra,
        )

    def return_result(self, metadata: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        telemetry.emit("stage_14_return_result", execution_id, {
            "status": "success",
            "generation_request_id": execution_id,
            "output_filename": metadata.get("filename"),
            "output_path": metadata.get("video_url"),
            "processing_time_s": metadata.get("processing_time_s")
        })
        return {
            "status": "success",
            "execution_id": execution_id,
            "metadata": metadata
        }

    async def execute(
        self,
        prompt: str,
        dialogue: Optional[str] = None,
        duration: int = 15,
        style: str = "cinematic",
        voice: bool = True,
        token: Optional[str] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        aspect_ratio: Optional[str] = None,
        quality: Optional[str] = None,
        resolution: Optional[str] = None,
        fps: Optional[int] = None,
        language: str = "en",
        voice_id: Optional[str] = None,
        reference_url: Optional[str] = None,
        reference_type: Optional[str] = None,
        reference_id: Optional[str] = None,
        lipsync: bool = True,
        model_mode: Optional[str] = None,
        character_id: Optional[str] = None,
        subtitles: bool = True,
        realtime_data: bool = False,
        music: bool = True,
    ) -> Dict[str, Any]:
        execution_id = job_id or generate_execution_id()
        start_time = time.time()
        video_cfg = resolve_video_settings(aspect_ratio, quality, resolution)
        width = int(video_cfg["width"])
        height = int(video_cfg["height"])
        fps_value = int(fps or settings.DEFAULT_FPS)
        language_code = normalize_language(language)
        style_key = (style or settings.DEFAULT_STYLE).lower()

        models_used = {
            "llm": settings.LLM_PROVIDER,
            "image": settings.IMAGE_PROVIDER,
            "video": settings.VIDEO_PROVIDER,
            "tts": settings.TTS_PROVIDER,
            "lipsync": settings.LIPSYNC_PROVIDER,
        }
        gen_params = {
            "duration": duration,
            "style": style_key,
            "visual_style": style_key,
            "dialogue": dialogue,
            "voice": voice,
            "fps": fps_value,
            "resolution": video_cfg["resolution"],
            "aspect_ratio": video_cfg["aspect_ratio"],
            "quality": video_cfg["quality"],
            "language": language_code,
            "lipsync": lipsync,
            "model_mode": model_mode or settings.MODEL_MODE,
            "character_id": character_id,
            "subtitles": subtitles,
            "realtime_data": realtime_data,
        }
        telemetry.emit("pipeline_execution_started", execution_id, {
            "received_prompt": prompt,
            "dialogue": dialogue,
            "generation_request_id": execution_id,
            "models_used": models_used,
            "generation_parameters": gen_params
        })

        async def _report(stage_name: str, pct: int):
            if progress_callback:
                if callable(progress_callback):
                    res = progress_callback(stage_name, pct)
                    if hasattr(res, "__await__"):
                        await res

        try:
            self.governance.verify_execution(token=token, execution_id=execution_id)

            await _report("prompt_processed", 4)
            clean_prompt = self.validate_prompt(prompt, execution_id)

            await _report("reference_processed", 8)
            reference = await reference_service.resolve(reference_url, reference_id, reference_type)

            # This is inference-time context only. It never changes a training
            # manifest and is omitted entirely if no configured provider succeeds.
            if realtime_data:
                await _report("retrieving_realtime_data", 10)
                realtime_context = await realtime_data_service.fetch_realtime_context(clean_prompt)
                clean_prompt = realtime_data_service.enrich_prompt_with_knowledge(clean_prompt, realtime_context)
            else:
                realtime_context = None

            await _report("understanding_prompt", 12)
            analysis = await self.understand_prompt(clean_prompt, execution_id)

            await _report("generating_story", 20)
            story = await self.generate_story(clean_prompt, analysis, duration, execution_id)

            await _report("generating_script", 28)
            script = await self.generate_script(story, execution_id)

            await _report("generating_scenes", 35)
            scenes = await self.generate_scenes(script, duration, style_key, execution_id)
            # Apply language and user-supplied spoken dialogue
            scenes = localization_service.apply(scenes, language_code, analysis, dialogue=dialogue)
            scenes = character_service.apply_character_consistency(scenes, character_id, language_code)

            profile = character_service.get_profile(character_id)
            if profile and not voice_id and profile.voice_id.lower().startswith(language_code.lower()):
                voice_id = profile.voice_id

            await _report("applying_visual_consistency", 42)
            scenes = self.generate_visual_prompts(
                scenes, style_key, analysis, execution_id, video_cfg["aspect_ratio"], reference
            )

            await _report("generating_keyframes", 55)
            scenes = await self.generate_visual_assets(scenes, execution_id, width, height, reference)

            await _report("rendering_video", 68)
            await job_manager.update_stage_status(execution_id, video_status="processing")
            scenes = await self.generate_scene_videos(scenes, execution_id, fps=fps_value, width=width, height=height)
            await job_manager.update_stage_status(execution_id, video_status="completed")

            await _report("voice_generated", 76)
            if voice:
                await job_manager.update_stage_status(execution_id, tts_status="processing")
            scenes = await self.generate_voice(scenes, voice, execution_id, language=language_code, voice=voice_id)
            if voice:
                await job_manager.update_stage_status(execution_id, tts_status="completed")

            await _report("lip_synchronization", 82)
            if lipsync and voice:
                await job_manager.update_stage_status(execution_id, lip_sync_status="processing")
            scenes = await lipsync_service.apply_to_scenes(scenes, execution_id, enabled=bool(lipsync and voice))
            if lipsync and voice:
                await job_manager.update_stage_status(execution_id, lip_sync_status="completed")

            await _report("mixing_audio", 86)
            original_music_setting = settings.BACKGROUND_MUSIC_ENABLED
            try:
                settings.BACKGROUND_MUSIC_ENABLED = bool(music)
                master_audio = self.process_audio(scenes, float(duration), execution_id)
            finally:
                settings.BACKGROUND_MUSIC_ENABLED = original_music_setting

            await _report("finalizing_video", 92)
            final_raw_video = self.assemble_video(
                scenes, master_audio, execution_id, width=width, height=height, fps=fps_value
            )

            await _report("validating_video", 95)
            validation_info = self.validate_video(final_raw_video, execution_id, expected_duration=float(duration))
            if int(validation_info.get("width") or 0) != width or int(validation_info.get("height") or 0) != height:
                raise VisualGenerationError(
                    f"Final video is {validation_info.get('width')}x{validation_info.get('height')}, "
                    f"expected {width}x{height}."
                )

            await _report("storing_artifacts", 98)
            metadata = self.store_output(
                video_path=final_raw_video,
                execution_id=execution_id,
                prompt=clean_prompt,
                scenes=scenes,
                duration=float(duration),
                style=style_key,
                resolution=video_cfg["resolution"],
                fps=fps_value,
                extra={
                    "aspect_ratio": video_cfg["aspect_ratio"],
                    "quality": video_cfg["quality"],
                    "language": language_code,
                    "dialogue": dialogue,
                    "visual_style": style_key,
                    "lipsync": bool(lipsync and voice),
                    "voice_enabled": voice,
                    "character_id": character_id,
                    "realtime_data": realtime_context,
                    "settings": gen_params,
                },
            )
            if subtitles and voice:
                subtitle_dir = settings.get_absolute_path(settings.TEMP_DIR)
                srt = ffmpeg_service.generate_srt_subtitles(scenes, str(subtitle_dir / f"{execution_id}.srt"))
                vtt = ffmpeg_service.generate_vtt_subtitles(scenes, str(subtitle_dir / f"{execution_id}.vtt"))
                metadata["subtitles"] = {
                    "language": language_code,
                    "srt_url": storage_service.persist_subtitle_artifact(execution_id, srt, "srt"),
                    "vtt_url": storage_service.persist_subtitle_artifact(execution_id, vtt, "vtt"),
                }
                # Persist the newly-added subtitle URLs in the existing sidecar.
                sidecar = settings.get_absolute_path(settings.OUTPUT_DIR) / f"{execution_id}_metadata.json"
                sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            metadata["processing_time_s"] = round(time.time() - start_time, 2)
            metadata["validation"] = validation_info

            await _report("completed", 100)
            return self.return_result(metadata, execution_id)

        except Exception as e:
            telemetry.emit("pipeline_execution_failed", execution_id, {"error": str(e)}, level="error")
            raise

pipeline = TextToVideoPipeline()
