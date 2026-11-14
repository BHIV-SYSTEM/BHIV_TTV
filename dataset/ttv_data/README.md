# OpenVid-1M Dataset Preparation & CogVideoX-2B LoRA Training Pipeline

This repository provides an automated, end-to-end data engineering and LoRA fine-tuning pipeline for **CogVideoX-2B** using authentic video data from the **OpenVid-1M** (`nkp37/OpenVid-1M`) dataset on Hugging Face.

---

## 1. Directory Structure

```text
ttv_data/
├── training_data/
│   └── openvid_1m/
│       ├── metadata/
│       │   ├── OpenVidHD.json                      # Part-to-video mapping (14.58 MB)
│       │   ├── OpenVidHD.csv                       # Complete metadata records (285.96 MB)
│       │   ├── selected_100_candidates.json        # Curated 100 high-quality candidates
│       │   └── downloaded_verified_videos.json     # Probe-verified download records
│       ├── raw_videos/                             # 100 authentic 1080p MP4 videos (203.16 MB)
│       ├── captions/                               # 100 matching text caption files (.txt)
│       ├── processed_videos/                       # 100 normalized 720x480 @ 8fps 49-frame MP4s (37.93 MB)
│       │   └── frames/                             # 300 visual verification keyframes (3 per clip)
│       ├── train/                                  # 80 training videos, captions, videos.txt, prompts.txt, metadata.csv
│       ├── validation/                             # 10 validation videos, captions, videos.txt, prompts.txt, metadata.csv
│       ├── test/                                   # 10 test videos, captions, videos.txt, prompts.txt, metadata.csv
│       ├── rejected/                               # Corrupt or duplicate files (0 rejected)
│       ├── reports/
│       │   ├── validation_report.json              # Statistical validation metrics
│       │   └── validation_report.md                # Human-readable validation report
│       └── dataset_manifest.jsonl                  # Master structured JSONL manifest
├── training_configs/
│   ├── cogvideox_lora_rtx3060.yaml                 # Optimized for RTX 3060 12GB VRAM
│   └── cogvideox_lora_rtx3080.yaml                 # Optimized for RTX 3080 10GB/12GB VRAM
├── training_scripts/
│   ├── train_cogvideox_lora.py                     # Self-contained PEFT LoRA training & verification script
│   ├── run_training_rtx3060.ps1                    # Launch script for RTX 3060
│   └── run_training_rtx3080.ps1                    # Launch script for RTX 3080
├── training_checkpoints/                           # Dedicated output directory (preserves base model)
└── README.md                                       # Full pipeline documentation
```

---

## 2. Dataset Collection & Validation Methodology

### Zero-Waste Streaming ZIP Extraction
Instead of downloading multi-terabyte dataset archives (over 4.5 TB for OpenVidHD), this pipeline employs a custom seekable HTTP stream reader (`SeekableHttpFile`) that reads the central directory of remote ZIP archives on Hugging Face using standard HTTP Range requests (`Accept-Ranges: bytes`, HTTP 206). It downloads only the exact bytes corresponding to the selected 100 MP4 video files.

### Quality Curation & Diversity
Candidates were filtered from 433,509 metadata records and 26,317 videos in `OpenVidHD_part_1.zip` according to strict thresholds:
- **Duration**: 5.0 to 10.0 seconds (preferring clips with optimal scene coherence).
- **Aesthetic Quality Score**: $\ge 5.6$ (Mean aesthetic score: **6.735 / 10**).
- **Motion Score**: $\ge 1.2$ (Mean motion score: **8.520**, ensuring dynamic action).
- **Temporal Consistency**: $\ge 0.96$ (Mean temporal consistency: **0.9961**, smooth transitions).
- **Domain Balance**:
  - 25 People & Portraits
  - 20 Animals & Wildlife
  - 20 Nature & Landscapes
  - 18 Vehicles & Machinery
  - 17 Urban & Interior Environments

### Integrity Verification
Every downloaded video was checked with `ffprobe`:
- Codec: H.264 / AVC1 or MPEG-4
- Dimensions: 1920x1080 (HD) or 1080x1080
- Frame integrity: No corrupted bitstreams
- Duplicate check: MD5 content hashing confirmed 100 unique videos (zero duplicates).

---

## 3. Preprocessing Specifications for CogVideoX-2B

CogVideoX-2B uses a 3D Variational Autoencoder (VAE) that compresses video spatially by 8x and temporally by 4x.
- **Resolution**: Scaled and center-cropped to `720x480` ($90 \times 60$ latents, divisible by 16).
- **Frame Rate**: Resampled to `8.0 FPS`.
- **Temporal Latent Chunking**: Exactly `49 frames` per video clip ($49 / 8 = 6.125\text{ seconds}$), yielding $(49-1)/4 + 1 = 13$ latent temporal frames.
- **Audio**: Stripped (`-an`) to save memory and processing bandwidth.

---

## 4. Dataset Splits (Zero Data Leakage)

The 100 preprocessed videos are partitioned into:
- **Train (80%)**: 80 videos (21 People, 16 Animals, 16 Nature, 14 Vehicles, 13 Urban)
- **Validation (10%)**: 10 videos (2 People, 2 Animals, 2 Nature, 2 Vehicles, 2 Urban)
- **Test (10%)**: 10 videos (2 People, 2 Animals, 2 Nature, 2 Vehicles, 2 Urban)

Verification:
- `Train ∩ Validation = ∅`
- `Train ∩ Test = ∅`
- `Validation ∩ Test = ∅`

---

## 5. Hardware Audit & Memory Fitting Guide

### Local Host Audit
- **GPU Detected**: Integrated `Intel(R) Iris(R) Xe Graphics` (`PCI\VEN_8086&DEV_9A49`).
- **NVIDIA GPU**: None present locally (nvidia-smi unavailable).
- **PyTorch**: `2.13.0+cpu` (`CUDA available: False`).
- **Conclusion**: Local machine is suitable for data preprocessing and validation. Model fine-tuning must be executed on a CUDA-enabled GPU (RTX 3060, RTX 3080, RTX 4090, or cloud instance).

### Memory Optimization for Target GPUs

| Parameter | RTX 3060 (12GB VRAM) | RTX 3080 (10GB / 12GB VRAM) |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | 16 | 32 |
| **LoRA Alpha ($\alpha$)** | 32 | 64 |
| **Target Modules** | `to_q`, `to_k`, `to_v`, `to_out.0` | `to_q`, `to_k`, `to_v`, `to_out.0` |
| **Precision** | `fp16` | `bf16` |
| **Batch Size** | 1 | 1 |
| **Gradient Accumulation** | 4 steps | 2 steps |
| **Optimizer** | 8-bit AdamW (`bitsandbytes`) | 8-bit AdamW (`bitsandbytes`) |
| **Gradient Checkpointing**| Enabled | Enabled |
| **Text/Latent Caching** | Enabled (eliminates T5/VAE from VRAM) | Enabled |
| **Estimated Peak VRAM** | **~9.2 GB** (Fits in 12GB) | **~8.8 GB** (Fits in 10GB/12GB) |

---

## 6. How to Run Training

### Pre-flight Dry-run (Validation only, no training)
```bash
python training_scripts/train_cogvideox_lora.py --dry_run
```

### Exact Training Command for RTX 3060 (12GB VRAM)
```powershell
python training_scripts/train_cogvideox_lora.py --config training_configs/cogvideox_lora_rtx3060.yaml
```

### Exact Training Command for RTX 3080 (10GB / 12GB VRAM)
```powershell
python training_scripts/train_cogvideox_lora.py --config training_configs/cogvideox_lora_rtx3080.yaml
```

### Alternative: Hugging Face Diffusers Reference Command
```bash
accelerate launch examples/cogvideo/train_cogvideox_lora.py \
  --pretrained_model_name_or_path="THUDM/CogVideoX-2b" \
  --instance_data_root="training_data/openvid_1m/train" \
  --video_column="videos.txt" \
  --caption_column="prompts.txt" \
  --height=480 \
  --width=720 \
  --fps=8 \
  --max_num_frames=49 \
  --rank=16 \
  --lora_alpha=32 \
  --mixed_precision="fp16" \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=4 \
  --enable_gradient_checkpointing \
  --use_8bit_adam \
  --output_dir="training_checkpoints/cogvideox_2b_lora"
```
