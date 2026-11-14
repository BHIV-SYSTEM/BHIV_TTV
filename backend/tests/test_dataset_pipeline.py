"""
Test suite for BHIV TTV Studio Dataset Collection, Ingestion, Preprocessing,
Deduplication, Validation, Versioning, and Training Manifest Preparation.
"""
from __future__ import annotations

import os
import json
import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import WORKSPACE_ROOT
from app.services.training_service import validate_manifest_records
from dataset.models import (
    DatasetItem,
    PermissionState,
    ValidationStatus,
    DatasetVersionMetadata,
)
from dataset.ingest import DatasetIngestionService, validate_url_safety
from dataset.preprocessor import MediaPreprocessor
from dataset.captioning import CaptionService
from dataset.deduplicate import DatasetValidator
from dataset.manifest import DatasetManager, prepare_openvid_sample_dataset
from dataset.pipeline import DatasetPipeline

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Flexible Domain & Metadata Extensibility Tests
# ---------------------------------------------------------------------------
def test_flexible_domain_and_metadata_extensibility():
    """Confirms system supports arbitrary visual subjects, environments, and styles."""
    subjects = [
        ("mythology", "indian_epic", "Arjuna archery challenge in forest", "cinematic_3d"),
        ("historical", "medieval_battle", "Panipat cavalry charge", "realistic"),
        ("3d_cartoon", "sci_fi_comedy", "Cute helper robot fixing spaceship", "3d_cartoon"),
        ("2d_animation", "folklore", "Panchatantra wise fox and crow", "2d_animation"),
        ("nature_landscape", "geography", "Himalayan peak during golden sunset", "cinematic"),
        ("character_action", "martial_arts", "Kalarippayattu swordsman training", "live_action"),
    ]

    for subj, topic, desc, style in subjects:
        item = DatasetItem(
            media_id=f"test_{subj}",
            storage_path=f"data/samples/{subj}.mp4",
            media_type="video",
            subject=subj,
            topic=topic,
            description=desc,
            animation_style=style,
            visual_style="cinematic",
            environment="natural_outdoor",
            camera_movement="drone_aerial",
            lighting="golden_hour",
        )
        data = item.to_dict()
        assert data["subject"] == subj
        assert data["topic"] == topic
        assert data["animation_style"] == style
        assert data["environment"] == "natural_outdoor"
        # Can be restored losslessly
        restored = DatasetItem.from_dict(data)
        assert restored.subject == subj
        assert restored.topic == topic


# ---------------------------------------------------------------------------
# 2. SSRF Protection & Safe URL Ingestion Tests
# ---------------------------------------------------------------------------
def test_ssrf_protection_blocks_private_and_loopback_ips():
    """Verify that private IP ranges, loopbacks, and local metadata IPs are blocked."""
    unsafe_urls = [
        "http://127.0.0.1/video.mp4",
        "http://localhost:8000/media.mp4",
        "http://10.0.0.1/dataset.zip",
        "http://192.168.1.1/video.mp4",
        "http://172.16.0.1/video.mp4",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/video.mp4",
        "file:///etc/passwd",
        "ftp://example.com/video.mp4",
    ]
    for url in unsafe_urls:
        is_safe, reason = validate_url_safety(url)
        assert not is_safe, f"Expected {url} to be blocked by SSRF check, but it passed: {reason}"


def test_ssrf_allows_public_https_domains():
    """Verify safe public domains pass SSRF filter."""
    safe_urls = [
        "https://example.com/video.mp4",
        "https://wikimedia.org/sample.jpg",
        "https://huggingface.co/datasets/sample.mp4",
    ]
    for url in safe_urls:
        is_safe, _ = validate_url_safety(url)
        assert is_safe, f"Expected {url} to pass SSRF check"


# ---------------------------------------------------------------------------
# 3. YouTube & Provenance / Permission Tracking Tests
# ---------------------------------------------------------------------------
def test_youtube_license_classification_and_manifest_exclusion():
    """
    Standard YouTube licenses must be marked RESTRICTED with training_eligible=False.
    Creative Commons licenses are LICENSE_VERIFIED with training_eligible=True.
    Unknown licenses must be flagged UNKNOWN with training_eligible=False.
    """
    svc = DatasetIngestionService(storage_root=str(WORKSPACE_ROOT / "data" / "test_ingest"))

    # Creative Commons
    perm, eligible = svc._determine_permission_state("Creative Commons Attribution license (reuse allowed)", "youtube")
    assert perm == PermissionState.LICENSE_VERIFIED.value
    assert eligible is True

    # Standard YouTube License
    perm, eligible = svc._determine_permission_state("Standard YouTube License", "youtube")
    assert perm == PermissionState.RESTRICTED.value
    assert eligible is False

    # Unknown
    perm, eligible = svc._determine_permission_state("", "youtube")
    assert perm == PermissionState.UNKNOWN.value
    assert eligible is False


# ---------------------------------------------------------------------------
# 4. Multimodal Caption Synthesis & Original Prompt Preservation Tests
# ---------------------------------------------------------------------------
def test_caption_synthesis_preserves_original_prompt_and_caption():
    """
    Caption service synthesizes a rich multi-field caption without ever
    modifying or discarding original_prompt or original_caption.
    """
    orig_prompt = "A cinematic drone shot flying over snow-covered pine mountains at sunset"
    orig_caption = "Himalayan sunset mountain aerial view"

    item = DatasetItem(
        media_id="test_caption_1",
        storage_path="sample.mp4",
        media_type="video",
        original_prompt=orig_prompt,
        original_caption=orig_caption,
        subject="nature_landscape",
        description="Majestic snow-clad peaks bathed in warm golden light",
        environment="snowy_mountains",
        camera_movement="drone_aerial",
        lighting="golden_hour",
        animation_style="cinematic",
        visual_style="photorealistic",
        actions=["flying_over", "gliding"],
    )

    caption_svc = CaptionService()
    caption_svc.enrich_item(item)

    assert len(item.generated_caption) > 0
    # Must preserve exact original strings
    assert item.original_prompt == orig_prompt
    assert item.original_caption == orig_caption
    assert "snow" in item.generated_caption.lower() or "nature" in item.generated_caption.lower()


# ---------------------------------------------------------------------------
# 5. Media Preprocessing & Quality Verification Tests
# ---------------------------------------------------------------------------
def test_image_preprocessing_and_blur_score(tmp_path: Path):
    """Verifies image inspection, Laplacian blur score, and thumbnail generation."""
    preproc = MediaPreprocessor(storage_root=str(tmp_path))

    # Create sharp synthetic image with clear edges
    sharp_img = np.zeros((480, 720, 3), dtype=np.uint8)
    cv2.rectangle(sharp_img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(sharp_img, (500, 240), 80, (0, 255, 0), -1)
    img_path = tmp_path / "sharp_test.png"
    cv2.imwrite(str(img_path), sharp_img)

    info = preproc.inspect_and_preprocess_image(str(img_path), "test_sharp")
    assert info["valid"] is True
    assert info["width"] == 720
    assert info["height"] == 480
    assert info["blur_score"] > 50.0  # Sharp image has high Laplacian variance
    assert info["thumbnail_path"] is not None
    assert Path(info["thumbnail_path"]).exists()


# ---------------------------------------------------------------------------
# 6. Deduplication (Exact SHA-256 and Perceptual pHash) Tests
# ---------------------------------------------------------------------------
def test_deduplication_exact_sha256(tmp_path: Path):
    """Exact duplicate (identical SHA-256) is marked DUPLICATE without deleting the file."""
    validator = DatasetValidator()

    file_a = tmp_path / "clip_a.txt"
    file_b = tmp_path / "clip_b.txt"
    content = b"Exact identical video frame bytes for deduplication test " * 5
    file_a.write_bytes(content)
    file_b.write_bytes(content)

    item_a = DatasetItem(
        media_id="item_a",
        storage_path=str(file_a),
        media_type="text",
        sha256="abc12345",
        original_caption="Valid test caption text for item A",
        permission_state=PermissionState.LICENSE_VERIFIED.value,
        training_eligible=True,
    )
    item_b = DatasetItem(
        media_id="item_b",
        storage_path=str(file_b),
        media_type="text",
        sha256="abc12345",
        original_caption="Valid test caption text for item B",
        permission_state=PermissionState.LICENSE_VERIFIED.value,
        training_eligible=True,
    )

    items, counts = validator.validate_dataset([item_a, item_b])

    assert item_a.validation_status == ValidationStatus.VALID.value
    assert item_a.training_eligible is True

    # Item B must be flagged duplicate
    assert item_b.validation_status == ValidationStatus.DUPLICATE.value
    assert item_b.training_eligible is False
    assert item_b.duplicate_status == "duplicate_exact"
    assert item_b.duplicate_of == item_a.media_id

    # File must NOT be deleted from disk
    assert file_b.exists()


def test_perceptual_hash_similarity(tmp_path: Path):
    """Near-identical images within pHash Hamming distance <= 6 are detected."""
    validator = DatasetValidator(max_phash_distance=6)

    # Identical or near-identical pHash (distance <= 6)
    phash1 = "1111000011110000"
    phash2 = "1111000011110001"  # hamming distance = 1

    file_a = tmp_path / "img_a.txt"
    file_b = tmp_path / "img_b.txt"
    file_a.write_bytes(b"content a padding for size " * 6)
    file_b.write_bytes(b"content b different bytes padding " * 6)

    item1 = DatasetItem(
        media_id="img_1",
        storage_path=str(file_a),
        media_type="image",
        resolution="720x480",
        original_caption="Valid image description 1",
        permission_state=PermissionState.LICENSE_VERIFIED.value,
        training_eligible=True,
        sha256="sha_1",
        phash=phash1,
    )
    item2 = DatasetItem(
        media_id="img_2",
        storage_path=str(file_b),
        media_type="image",
        resolution="720x480",
        original_caption="Valid image description 2",
        permission_state=PermissionState.LICENSE_VERIFIED.value,
        training_eligible=True,
        sha256="sha_2",
        phash=phash2,
    )

    validator.validate_dataset([item1, item2])

    assert item1.validation_status == ValidationStatus.VALID.value
    assert item2.validation_status == ValidationStatus.DUPLICATE.value
    assert item2.training_eligible is False
    assert item2.duplicate_status == "duplicate_perceptual"
    assert item2.duplicate_of == item1.media_id


# ---------------------------------------------------------------------------
# 7. Partitioning & Manifest Export Tests
# ---------------------------------------------------------------------------
def test_deterministic_split_partitioning_and_zero_leakage(tmp_path: Path):
    """
    Partitioning must produce deterministic 80/10/10 train/val/test splits
    with zero duplicate hashes across splits.
    """
    manager = DatasetManager(storage_root=str(tmp_path))

    # Create 20 distinct valid items
    items = []
    for i in range(20):
        dummy_file = tmp_path / f"video_{i:02d}.mp4"
        dummy_file.write_bytes(f"Unique content payload {i}".encode())
        item = DatasetItem(
            media_id=f"item_{i:02d}",
            storage_path=str(dummy_file),
            media_type="video",
            original_caption=f"Clip description text {i}",
            subject="mythology" if i % 2 == 0 else "nature",
            validation_status=ValidationStatus.VALID.value,
            permission_state=PermissionState.LICENSE_VERIFIED.value,
            training_eligible=True,
            sha256=f"hash_{i:03d}",
        )
        items.append(item)

    # Partition 80/10/10
    manager.partition_splits(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_items = [it for it in items if it.dataset_split == "train"]
    val_items = [it for it in items if it.dataset_split == "validation"]
    test_items = [it for it in items if it.dataset_split == "test"]

    assert len(train_items) == 16
    assert len(val_items) == 2
    assert len(test_items) == 2

    # Zero leakage check
    train_hashes = {it.sha256 for it in train_items}
    val_hashes = {it.sha256 for it in val_items}
    test_hashes = {it.sha256 for it in test_items}

    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0


# ---------------------------------------------------------------------------
# 8. OpenVid-1M Sample Dataset Schema Compliance Verification
# ---------------------------------------------------------------------------
def test_openvid_sample_dataset_full_schema_compliance():
    """
    Verifies that the existing OpenVid-1M sample dataset is fully prepared into
    dataset/manifests/openvid_1m_prepared_manifest.jsonl and passes
    training_service.validate_manifest_records with 100% compliance.
    """
    res = prepare_openvid_sample_dataset()
    assert res["success"] is True
    assert res["train_samples"] == 80
    assert res["val_samples"] == 10
    assert res["test_samples"] == 10

    manifest_path = Path(res["manifest_path"])
    assert manifest_path.exists()

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 100, f"Expected 100 sample records, got {len(records)}"

    # Validate against training_service validator
    val_res = validate_manifest_records(records)
    assert val_res["valid"] is True, f"Manifest validation failed: {val_res.get('error')}"
    assert val_res["total_records"] == 100
    assert val_res["valid_records"] == 100
    assert val_res["rejected_records"] == 0
    assert val_res["splits"]["train"] == 80
    assert val_res["splits"]["validation"] == 10
    assert val_res["splits"]["test"] == 10
    assert "video_path" in records[0]
    assert "caption" in records[0]


# ---------------------------------------------------------------------------
# 9. Dataset API Endpoints Verification Tests
# ---------------------------------------------------------------------------
def test_dataset_dashboard_endpoint():
    """Verify GET /api/v1/dataset/dashboard returns sample dataset and status."""
    response = client.get("/api/v1/dataset/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "sample_dataset" in data
    assert data["sample_dataset"]["present"] is True
    assert data["sample_dataset"]["total_clips"] == 100
    assert data["sample_dataset"]["splits"]["train"] == 80
    assert data["sample_dataset"]["splits"]["validation"] == 10
    assert data["sample_dataset"]["splits"]["test"] == 10


def test_dataset_collect_text_story_endpoint():
    """Verify POST /api/v1/dataset/collect with rich metadata."""
    payload = {
        "source_type": "text_story",
        "story_text": "Lord Krishna imparting timeless wisdom to Arjuna on the battlefield of Kurukshetra amidst churning clouds.",
        "dataset_id": "test_epics",
        "subject": "mythology",
        "topic": "bhagavad_gita",
        "category": "characters",
        "animation_style": "cinematic_3d",
        "visual_style": "dramatic",
        "environment": "battlefield",
        "camera_movement": "orbital",
        "lighting": "dramatic_twilight",
        "prompt": "Arjuna with folded hands listening to Krishna holding chariot reins",
        "caption": "Krishna delivering the Gita to Arjuna",
    }
    response = client.post("/api/v1/dataset/collect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["item"]["subject"] == "mythology"
    assert data["item"]["topic"] == "bhagavad_gita"
    assert data["item"]["original_prompt"] == payload["prompt"]
    assert data["item"]["original_caption"] == payload["caption"]


def test_dataset_list_items_endpoint():
    """Verify GET /api/v1/dataset/items queries items by dataset_id."""
    response = client.get("/api/v1/dataset/items?dataset_id=test_epics")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data


def test_dataset_sample_prepare_endpoint():
    """Verify POST /api/v1/dataset/sample/prepare prepares and returns OpenVid manifest."""
    response = client.post("/api/v1/dataset/sample/prepare")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["train_samples"] == 80
    assert data["validation_report"]["valid"] is True
    assert data["validation_report"]["valid_records"] == 100


def test_dataset_manifests_list_endpoint():
    """Verify GET /api/v1/dataset/manifests lists prepared manifests."""
    response = client.get("/api/v1/dataset/manifests")
    assert response.status_code == 200
    data = response.json()
    assert "manifests" in data
    filenames = [m["filename"] for m in data["manifests"]]
    assert "openvid_1m_prepared_manifest.jsonl" in filenames
