# BHIV Text-to-Video AI Studio — Engineering Handover Documentation

**Project:** BHIV Text-to-Video AI Studio  
**Workspace:** `C:\Users\PC\Desktop\BHIV_TTV_Studio\BHIV_TTV`  
**Active Working Branch:** `feat/ttv-environment-core-fixes`  
**Target Hardware Baseline:** NVIDIA GeForce RTX 3080 (10.0 GB VRAM, Ampere Compute Capability 8.6)  
**System Environment:** Windows 11, Python 3.12.10, PyTorch 2.6.0+cu124, CUDA 12.4  

---

## 1. Executive Summary

This handover document provides an exhaustive reference for the engineering team taking over the **BHIV Text-to-Video (TTV) AI Studio**.

Over Phases 1 through 5, the codebase was transformed from a disjointed experimental repository into a unified, memory-optimized, production-ready Text-to-Video studio. It encompasses:
1. **Core Video Adapters & Inference Stabilization:** OpenCV (production default), Wan 2.2, and CogVideoX-2B.
2. **CogVideoX Root-Cause Debugging & Quality Restoration:** Resolved white/dark gradient frame artifacts through strict VAE decoding, correct RGB clamping, dynamic range rescaling, and model CPU offloading.
3. **Phase 4 Automated Universal Dataset Pipeline:** An extensible collection, ingestion, deduplication, captioning, validation, and manifest generation system supporting diverse domains (Indian mythology, historical events, 3D/2D animation, realistic cinema, nature, action sequences).
4. **Phase 5 10 GB VRAM-Optimized LoRA Training Pipeline:** Parameter-Efficient Fine-Tuning (LoRA rank 4, alpha 16.0), activation gradient checkpointing, mixed precision (`bfloat16`/`float16`), micro-batching with gradient accumulation, peak VRAM telemetry, 21-point automated preflight checklist, and comprehensive UI/API integration.

---

## 2. What Was Done

### Phase 1–3: Architecture, Environment & Video Inference Stabilization
- **FastAPI Backend & Pipeline Unification:** Consolidated the 14-stage pipeline (prompt ingestion, LLM scriptwriting, prompt expansion, scene generation, audio synthesis, lipsync, video assembly, and quality metrics).
- **Default Video Adapter:** Maintained `OpenCVVideoAdapter` as the default fast, rock-solid provider for development and rapid testing.
- **Wan 2.2 Video Adapter:** Integrated `WanVideoAdapter` for high-fidelity multi-stage diffusion video generation.
- **CogVideoX-2B Adapter:** Integrated `CogVideoXVideoAdapter` with lazy pipeline initialization, GPU device offloading, and bfloat16 precision.
- **Quality Artifact Resolution:** Diagnosed and fixed the "white-and-dark gradient" failure in CogVideoX outputs by rectifying VAE latent denormalization, enforcing proper color-space transformations, and ensuring guidance scale $\ge 6.0$.

### Phase 4: Automated Universal Dataset Preparation System
- **Domain-Agnostic Design:** Built a modular system in `dataset/` capable of ingesting arbitrary video/image/text content across Indian mythology, history, animation, nature, and cinematic footage without code changes.
- **Multi-Source Ingestion (`dataset/ingest.py`):** Ingestion engine supporting local directories, remote HTTP/HTTPS URLs, Hugging Face repositories, and CSV/JSONL manifests.
- **Intelligent Deduplication (`dataset/deduplicate.py`):** Two-tier deduplication combining perceptual image hashing (pHash) on keyframes and cosine similarity of sentence embeddings on captions.
- **Automated Captioning (`dataset/captioning.py`):** Hybrid captioning pipeline combining rule-based heuristics, prompt expansion, and vision-language model generation.
- **Manifest Architecture (`dataset/manifest.py`):** Strict JSONL schema with governance and licensing gates (`training_eligible`, `permission_state`, `validation_status`).
- **Sample Dataset:** Prepared and validated 100 sample video clips and captions from OpenVid-1M into `dataset/manifests/openvid_1m_prepared_manifest_train.jsonl`.

### Phase 5: 10 GB VRAM-Optimized LoRA Training Pipeline
- **Hardware & Memory Telemetry (`training/utils/hardware.py`):** Dynamic runtime inspection of total VRAM, allocated VRAM, reserved VRAM, free VRAM, CUDA compute capability, and disk capacity. Implemented theoretical memory budgeting (`estimate_training_vram`).
- **SpatialTemporalTTVModel Upgrades (`training/fine_tuning/model.py`):** Integrated activation gradient checkpointing (`torch.utils.checkpoint.checkpoint`) to dramatically reduce backpropagation memory overhead during training.
- **Native LoRA Layer Implementation (`training/fine_tuning/lora.py`):** Native PyTorch Low-Rank Adaptation (`LoRALinear`, `LoRAConv1d`, `LoRAConv2d`) targeting attention projections (`to_q`, `to_k`, `to_v`, `proj`) and temporal convolutions (`temporal_conv`, `temporal_proj`). Reduces trainable parameters from 2.4M to ~18.4K (< 0.8% of total parameters).
- **Memory-Optimized Trainer (`training/fine_tuning/trainer.py`):**
  - Micro-batching (batch size 1) with gradient accumulation (4 steps) for an effective batch size of 4 without VRAM spikes.
  - Native `bfloat16` and `float16` (with `GradScaler`) mixed precision.
  - Real-time VRAM telemetry tracking and periodic CUDA cache clearing (`torch.cuda.empty_cache()`).
  - Graceful Out-Of-Memory (OOM) handling with diagnostic logging.
  - Standalone portable LoRA adapter export (`final_lora_adapter.pt`).
- **Gated Dataset Loader (`training/datasets/loader.py`):** `TextToVideoDataset` filters out restricted, duplicate, or ineligible samples, resolves workspace-relative paths, and loads frames lazily to avoid memory exhaustion.
- **Automated Preflight Checklist (`training/preflight.py`):** 21-point system check verifying CUDA availability, total VRAM, free VRAM, mixed precision support, disk capacity, manifest validity, physical video integrity, and a non-destructive dry-run forward pass.
- **Dedicated Hardware Config (`training/configs/lora_10gb_rtx3080.yaml`):** Fine-tuned hyperparameters for the RTX 3080.
- **API & UI Integration:** Added `/api/v1/training/preflight`, `/api/v1/training/hardware`, `/api/v1/training/checkpoints` endpoints, live VRAM monitoring pill, LoRA hyperparameter controls, and a Preflight Check button in the Web Studio UI.
- **Beginner Documentation (`docs/LORA_FINETUNING_BEGINNER_GUIDE.md`):** Complete operator's guide explaining LoRA theory, 10 GB techniques, UI/CLI usage, and troubleshooting.

---

## 3. What Is Where (Repository Map)

```
BHIV_TTV/
|-- backend/                                # FastAPI Backend Application
|   |-- app/
|   |   |-- adapters/video/                 # Video Generation Adapters
|   |   |   |-- base_video_adapter.py       # Abstract Base Adapter
|   |   |   |-- cogvideox_video_adapter.py  # CogVideoX-2B Video Adapter
|   |   |   |-- opencv_video_adapter.py     # OpenCV Procedural Adapter (Default)
|   |   |   |-- wan_video_adapter.py        # Wan 2.2 Video Adapter
|   |   |-- api/routes/
|   |   |   |-- dataset.py                  # Dataset Ingestion, Validation & Manifest Routes
|   |   |   |-- training.py                 # Training Jobs, Preflight, Hardware & Checkpoint Routes
|   |   |   |-- generate.py                 # End-to-End TTV Generation Routes
|   |   |   |-- jobs.py                     # Asynchronous Job Execution & Polling Routes
|   |   |   |-- videos.py                   # Media Streaming & Library Routes
|   |   |   |-- health.py                   # Health & Liveness Probes
|   |   |-- core/                           # Application Config, Logging, Telemetry, Exceptions
|   |   |-- models/                         # Domain Entities (Scene, Project, Script, Video)
|   |   |-- schemas/                        # Pydantic Schemas for API Requests & Responses
|   |   |-- services/                       # Business Logic (training_service.py, reference_service.py)
|   |   |-- main.py                         # FastAPI App Initialization, Middleware & Lifespan
|   |-- tests/                              # Unit & Regression Test Suites
|       |-- test_dataset_pipeline.py        # Phase 4 Dataset Pipeline Tests
|       |-- test_training_pipeline_phase5.py# Phase 5 LoRA Training & Preflight Tests
|-- dataset/                                # Automated Dataset Collection & Preparation Pipeline
|   |-- ingest.py                           # Multi-source dataset ingestion engine
|   |-- deduplicate.py                      # Perceptual hash & caption embedding deduplicator
|   |-- captioning.py                       # Automated rule-based & vision-language captioner
|   |-- manifest.py                         # JSONL manifest generator with governance gating
|   |-- models.py                           # Pydantic data models for dataset records
|   |-- pipeline.py                         # DatasetPreparationPipeline orchestrator
|   |-- preprocessor.py                     # Keyframe extraction & thumbnail generator
|   |-- manifests/                          # Phase 4 Prepared JSONL Manifests
|   |   |-- openvid_1m_prepared_manifest_train.jsonl
|   |   |-- openvid_1m_prepared_manifest_val.jsonl
|   |   |-- openvid_1m_prepared_manifest_test.jsonl
|   |-- ttv_data/                           # OpenVid-1M Sample Dataset (100 clips & captions)
|       |-- training_data/openvid_1m/       # Sample MP4s (untracked/ignored) and TXT captions
|-- training/                               # 10 GB VRAM LoRA Training Pipeline
|   |-- configs/
|   |   |-- lora_10gb_rtx3080.yaml          # Dedicated RTX 3080 (10 GB) Configuration
|   |   |-- default_training_config.yaml    # General Training Configuration Template
|   |-- datasets/
|   |   |-- loader.py                       # TextToVideoDataset, PyTorch DataLoader & Collation
|   |-- fine_tuning/
|   |   |-- model.py                        # SpatialTemporalTTVModel with Gradient Checkpointing
|   |   |-- lora.py                         # Native LoRA Modules (Linear, Conv1d, Conv2d)
|   |   |-- trainer.py                      # TTVTrainer (Mixed Precision, Grad Accum, Telemetry)
|   |-- preprocessing/
|   |   |-- video_preprocessor.py           # Uniform Frame Extractor & Lanczos Normalizer
|   |   |-- text_preprocessor.py            # Text Tokenizer & Tensor Embedding Processor
|   |-- utils/
|   |   |-- hardware.py                     # Dynamic VRAM Inspection & Memory Budgeting
|   |-- preflight.py                        # 21-Step Preflight Check & Dry-Run CLI Tool
|   |-- run_training.py                     # Standalone Training Execution Script
|-- frontend/                               # Web Studio User Interface
|   |-- index.html                          # UI HTML with Training Controls & VRAM Telemetry
|   |-- app.js                              # Client-side Logic, Polling, Preflight & Job Management
|   |-- style.css                           # BHIV Studio Dark Theme Styling
|-- docs/                                   # Project Documentation
|   |-- HANDOVER.md                         # This Engineering Handover Document
|   |-- LORA_FINETUNING_BEGINNER_GUIDE.md   # Beginner Operator Manual for LoRA Training
|-- .env.example                            # Environment Variables Template
|-- pytest.ini                              # Pytest Configuration
`-- README.md                               # Project Overview & Quickstart Guide
```

---

## 4. How to Run and Verify the System

### 4.1. Environment Activation
Open PowerShell in the workspace root:
```powershell
cd C:\Users\PC\Desktop\BHIV_TTV_Studio\BHIV_TTV
.venv\Scripts\Activate.ps1
$env:PYTHONPATH=".;backend"
```

### 4.2. Run the 10 GB VRAM Preflight Feasibility Check
Execute the preflight checker before launching any training experiment. It runs 21 verification checks and executes a non-destructive dry-run forward pass to measure exact peak VRAM usage:
```powershell
.venv\Scripts\python.exe training/preflight.py --config training/configs/lora_10gb_rtx3080.yaml --dry-run
```
*Expected Output:* Status `READY`, 21/21 checks passed, peak VRAM measured at ~125.5 MB (9.87 GB available headroom).

### 4.3. Launch the Backend API and Web Studio
Start the FastAPI backend:
```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000`.
- **Training Tab:** View live VRAM metrics in the header pill (`GPU: NVIDIA GeForce RTX 3080 | VRAM: ~8.9 GB Free / 10.0 GB Total`).
- **Preflight Button:** Click **Run Preflight Check** to validate configuration and manifest readiness before launching a job.

### 4.4. Launch an Approved LoRA Training Run (CLI)
When authorized to perform a real training experiment:
```powershell
.venv\Scripts\python.exe training/run_training.py --config training/configs/lora_10gb_rtx3080.yaml --version-name lora_rtx3080_v1
```

### 4.5. Resume Training from Checkpoint
To resume an interrupted training job from an existing checkpoint:
```powershell
.venv\Scripts\python.exe training/run_training.py --config training/configs/lora_10gb_rtx3080.yaml --resume-from training/checkpoints/lora_rtx3080/checkpoint_epoch_1.pt
```

---

## 5. What to Do Next (Action Roadmap)

### Immediate Next Steps (Priority 1)
1. **Pull Request Review & Merge:** Review the PR associated with branch `feat/ttv-environment-core-fixes` and merge into `main` after team sign-off.
2. **Review Beginner Guide:** Review [docs/LORA_FINETUNING_BEGINNER_GUIDE.md](file:///c:/Users/PC/Desktop/BHIV_TTV_Studio/BHIV_TTV/docs/LORA_FINETUNING_BEGINNER_GUIDE.md) with team operators.
3. **Run Live Preflight Check:** Execute `training/preflight.py --dry-run` to re-confirm zero CUDA contention with other processes on the host machine.

### Short-Term Next Steps (Priority 2)
1. **Execute Initial LoRA Experiment:** Launch a controlled 3-epoch fine-tuning run on the prepared 100-sample OpenVid-1M dataset using `training/configs/lora_10gb_rtx3080.yaml`.
2. **Adapter Inference Verification:** Load the resulting `final_lora_adapter.pt` into the video generation pipeline and evaluate output quality against baseline.
3. **Expand Dataset Categories:** Use the Phase 4 pipeline to ingest additional categories (e.g., custom mythological or historical narrative clips) into `dataset/manifests/`.

### Medium-Term Enhancements (Priority 3)
1. **CogVideoX-2B Latent Caching Pipeline:** Implement an offline precomputation utility to encode full video datasets into 3D-VAE latent tensors (`.pt`) and T5 prompt embeddings. This will allow direct fine-tuning of the 2.2B CogVideoX transformer on the 10 GB RTX 3080 without VRAM exhaustion.
2. **LoRA Adapter Switching in Studio UI:** Add a dropdown in the Web Studio generation panel allowing users to select and apply active LoRA adapter checkpoints on the fly.
3. **Automated Evaluation Metrics:** Integrate automated CLIP-Score, SSIM, and FVD (Fréchet Video Distance) scoring into the post-training reporting suite.

---

## 6. Key Contacts and Resources

-
- **Repository:** `https://github.com/BHIV-SYSTEM/BHIV_TTV`
- **Active Branch:** `feat/ttv-environment-core-fixes`
- **Documentation:**
  - Handover Guide: [docs/HANDOVER.md](file:///c:/Users/PC/Desktop/BHIV_TTV_Studio/BHIV_TTV/docs/HANDOVER.md)
  - LoRA Training Guide: [docs/LORA_FINETUNING_BEGINNER_GUIDE.md](file:///c:/Users/PC/Desktop/BHIV_TTV_Studio/BHIV_TTV/docs/LORA_FINETUNING_BEGINNER_GUIDE.md)
  - Dataset Manifest: [dataset/manifests/openvid_1m_prepared_manifest_train.jsonl](file:///c:/Users/PC/Desktop/BHIV_TTV_Studio/BHIV_TTV/dataset/manifests/openvid_1m_prepared_manifest_train.jsonl)
