import io
import os
import time
import pytest
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.reference_service import reference_service
from app.services.training_service import training_job_manager
from training.datasets.manifest_builder import prepare_reference_dataset_entry
from training.datasets.loader import create_dataloader

client = TestClient(app)


def _create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (640, 480), color=(255, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_reference_upload_image():
    content = _create_test_image_bytes()
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("test_ref.jpg", content, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["reference_id"].startswith("ref_")
    assert data["media_type"] == "image"
    assert data["source"] == "upload"
    assert data["preview_url"] is not None


def test_reference_upload_invalid_type():
    response = client.post(
        "/api/v1/references/upload",
        files={"file": ("malicious.exe", b"binary content", "application/x-msdownload")}
    )
    assert response.status_code == 400
    assert "Unsupported reference file type" in response.json().get("message", "")


def test_reference_url_invalid():
    response = client.post(
        "/api/v1/references/url",
        json={"url": "http://127.0.0.1:8080/internal-secret"}
    )
    assert response.status_code == 400


def test_training_start_without_reference():
    payload = {
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_")
    assert data["status"] in {"running", "queued"}


def test_training_start_with_image_reference():
    content = _create_test_image_bytes()
    upload_res = client.post(
        "/api/v1/references/upload",
        files={"file": ("train_ref.jpg", content, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert upload_res.status_code == 200
    ref_id = upload_res.json()["reference_id"]

    payload = {
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": "0.0001",
        "batch_size": 1,
        "reference_id": ref_id,
        "reference_type": "image"
    }
    response = client.post("/api/v1/training/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_")
    assert data["reference_id"] == ref_id
    assert any("Attached reference" in log for log in data.get("logs", []))


def test_training_prepare_reference_dataset_manifest(tmp_path):
    content = _create_test_image_bytes()
    saved = reference_service.save_upload("ref_test.jpg", content, "image/jpeg")
    
    out_manifest = str(tmp_path / "reference_train_manifest.jsonl")
    entry = prepare_reference_dataset_entry(saved, output_manifest_path=out_manifest)
    
    assert entry["id"].startswith("sample_ref_")
    assert os.path.exists(out_manifest)
    
    # Test DataLoader compatibility with the reference manifest
    loader = create_dataloader(
        manifest_path=out_manifest,
        batch_size=1,
        num_frames=8,
        height=128,
        width=128,
        shuffle=False
    )
    assert len(loader) >= 1
    batch = next(iter(loader))
    assert batch["frames"].shape == (1, 8, 3, 128, 128)
    assert batch["prompt_tokens"].shape[0] == 1


def _mock_training_queue(monkeypatch):
    submitted = []

    async def create_job(manifest_data, config, manifest_filename=None):
        submitted.append({
            "manifest_data": manifest_data,
            "config": config,
            "manifest_filename": manifest_filename,
        })
        return {
            "job_id": f"queued_{len(submitted)}",
            "status": "queued",
            "stage": "queued",
            "message": "Training job queued.",
            "cuda_available": True,
            "hardware": {"gpu_name": "Test GPU"},
        }

    monkeypatch.setattr(training_job_manager, "create_job", create_job)
    return submitted


def _wait_for_automatic_job(job_id):
    for _ in range(100):
        response = client.get(f"/api/v1/training/auto/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] not in {"preparing", "processing"}:
            return job
        time.sleep(0.02)
    pytest.fail("Automatic training did not finish reference preparation in time.")


def _training_config(reference_id):
    return {
        "reference_id": reference_id,
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 1,
        "learning_rate": 0.0001,
        "batch_size": 1,
    }


def test_uploaded_video_automatically_validates_and_queues_training(monkeypatch):
    submitted = _mock_training_queue(monkeypatch)
    video_path = Path(__file__).resolve().parents[2] / "test_ball_motion.mp4"
    content = video_path.read_bytes()
    upload = client.post(
        "/api/v1/references/upload",
        files={"file": ("training_video.mp4", content, "video/mp4")},
    )
    assert upload.status_code == 200
    reference_id = upload.json()["reference_id"]

    try:
        response = client.post("/api/v1/training/auto", json=_training_config(reference_id))
        assert response.status_code == 200
        job = _wait_for_automatic_job(response.json()["job_id"])
        assert job["status"] == "queued"
        assert job["training_job_id"] == "queued_1"
        assert len(submitted) == 1
        sample = submitted[0]["manifest_data"][0]
        assert sample["video_path"].endswith("source.mp4")
        assert os.path.isfile(sample["video_path"])

        duplicate = client.post("/api/v1/training/auto", json=_training_config(reference_id))
        assert duplicate.status_code == 200
        assert duplicate.json()["job_id"] == job["job_id"]
        assert len(submitted) == 1
    finally:
        reference_service.cleanup(reference_id)


def test_supported_video_url_automatically_validates_and_queues_training(monkeypatch):
    submitted = _mock_training_queue(monkeypatch)
    video_path = Path(__file__).resolve().parents[2] / "test_ball_motion.mp4"
    content = video_path.read_bytes()

    async def authorized_youtube_retrieval(url, reference_id=None):
        payload = reference_service.save_upload("youtube_reference.mp4", content, "video/mp4")
        payload["source"] = "youtube"
        return payload

    monkeypatch.setattr(reference_service, "ingest_url", authorized_youtube_retrieval)
    url_response = client.post(
        "/api/v1/references/url",
        json={"url": "https://www.youtube.com/watch?v=authorized-test"},
    )
    assert url_response.status_code == 200
    reference_id = url_response.json()["reference_id"]

    try:
        response = client.post("/api/v1/training/auto", json=_training_config(reference_id))
        assert response.status_code == 200
        job = _wait_for_automatic_job(response.json()["job_id"])
        assert job["status"] == "queued"
        assert len(submitted) == 1
        sample = submitted[0]["manifest_data"][0]
        assert sample["metadata"]["source"] == "youtube"
        assert sample["video_path"].endswith("source.mp4")
    finally:
        reference_service.cleanup(reference_id)


def test_uploaded_image_automatically_queues_training(monkeypatch):
    submitted = _mock_training_queue(monkeypatch)
    upload = client.post(
        "/api/v1/references/upload",
        files={"file": ("training_image.jpg", _create_test_image_bytes(), "image/jpeg")},
    )
    assert upload.status_code == 200
    reference_id = upload.json()["reference_id"]

    try:
        response = client.post("/api/v1/training/auto", json=_training_config(reference_id))
        assert response.status_code == 200
        job = _wait_for_automatic_job(response.json()["job_id"])
        assert job["status"] == "queued"
        assert len(submitted) == 1
        sample = submitted[0]["manifest_data"][0]
        assert sample["video_path"].endswith("source.jpg")
    finally:
        reference_service.cleanup(reference_id)
