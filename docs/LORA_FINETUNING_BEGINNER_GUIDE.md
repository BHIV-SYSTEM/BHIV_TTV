# Beginner's Guide: Video Model LoRA Fine-Tuning (10 GB VRAM / RTX 3080)

Welcome to the **BHIV Text-to-Video AI Studio** LoRA fine-tuning guide! This guide explains how to fine-tune a video generation model on your own PC, step-by-step, even if you are completely new to deep learning.

---

## 1. What is LoRA? (The 2-Minute Explanation)

When you train or "fine-tune" a modern video AI model from scratch, you would normally have to adjust billions of numbers (called *weights* or *parameters*). Doing that requires giant supercomputers with 80 GB or more of Video RAM (VRAM). If you tried that on a consumer graphics card, your computer would instantly crash with an **Out Of Memory (OOM)** error.

**LoRA** stands for **Low-Rank Adaptation**:
- Instead of changing all the original weights, we **freeze** the base video model completely.
- We attach tiny "adapter" helper layers next to the model's attention mechanisms.
- During training, **only the tiny helper layers learn**.
- This reduces the number of trained parameters by over **90%** (e.g. from 2.4 million weights down to just 280,000 weights).
- The result is a lightweight file (`final_lora_adapter.pt`, typically under 50 MB) that you can easily share, switch on/off, or load on top of your model!

---

## 2. Why 10 GB VRAM is Special

The NVIDIA GeForce **RTX 3080** has **10.0 GB of VRAM**. This is a powerful graphics card, but video generation is one of the heaviest AI tasks because it processes space (height × width), color (RGB channels), and time (multiple video frames simultaneously).

Our pipeline applies five memory-saving techniques to make training smooth and stable:

1. **Micro-Batch Size = 1:** We only process 1 video clip at a time through the GPU.
2. **Gradient Accumulation = 4:** The computer processes 4 clips one-by-one, gathers the learning updates, and applies them together. This gives you the mathematical stability of batch size 4 without using 4x the VRAM!
3. **Gradient Checkpointing:** During calculation, intermediate activations are thrown away and recomputed only when needed, cutting memory spikes by ~60%.
4. **Native bfloat16 Precision:** Numbers are stored in 16-bit brain floating point format instead of standard 32-bit. This halves the memory used for weights while preserving dynamic range.
5. **Frozen Backbone:** Only the LoRA adapter layers receive gradients.

---

## 3. Step 1: Prepare Your Dataset

Before you can train a model, you need video clips with matching text descriptions (captions).

### Option A: Use the Built-in OpenVid-1M Sample Dataset (Recommended for Beginners)
The studio already comes with a verified 100-clip sample dataset in `dataset/ttv_data/training_data/openvid_1m/`.
1. Open the BHIV TTV Studio Web UI.
2. Navigate to the **Training / Fine-Tuning** tab.
3. Click the button: **⚡ Load Sample Dataset (OpenVid-1M, 100 clips)**.
4. You will see:
   - **Video Samples:** 100
   - **Resolution:** 720×480
   - **Train / Val / Test:** 80 / 10 / 10
   - **Status:** Manifest prepared and verified!

### Option B: Your Own Custom Videos
If you want to train on your own style (e.g. 3D cartoon, Indian mythology, nature):
1. In the **Dataset Management** dropzone, upload a `.jsonl` manifest file where each line has:
   ```json
   {"video_path": "dataset/ttv_data/my_video_01.mp4", "caption": "A golden sunset over snow-covered pine mountains", "split": "train"}
   ```
2. The system will automatically validate file existence, detect duplicates, and verify licensing permissions.

---

## 4. Step 2: Run the Preflight & Memory Feasibility Check

**Never launch training without running a preflight check first!** The preflight check inspects your hardware, measures your free VRAM, validates your manifest, and even runs a safe dummy forward pass to prove your GPU can handle the workload without modifying model weights.

### In the Web UI:
1. In the **Hyperparameters & 10 GB VRAM Method** card, click:
   **🔍 Run Preflight & VRAM Feasibility Check**
2. Within 2 seconds, you will see a readiness badge:
   - 🟢 **READY (10 GB Safe):** Your GPU has plenty of free VRAM and all dataset files exist. You are cleared to train!
   - 🟡 **READY WITH LIMITATIONS:** The configuration fits, but caution is advised (e.g. keep batch size at 1).
   - 🔴 **NOT READY:** A critical issue was detected (e.g. CUDA driver missing, or manifest file not found). Read the recommendation to fix it.

### Via Command Line (CLI):
```powershell
.venv\Scripts\python.exe training/preflight.py --config training/configs/lora_10gb_rtx3080.yaml --dry-run
```

---

## 5. Step 3: Understanding the Hyperparameters

Here is what the settings in the UI and config file actually do:

| Parameter | Recommended Setting | What It Means |
| :--- | :--- | :--- |
| **Base Model** | `SpatialTemporalTTVModel` | The starting neural video architecture. |
| **Fine-Tuning Method** | `LoRA` | Enables low-rank adapter training with frozen base weights. |
| **LoRA Rank ($r$)** | `4` (or `8`) | The width of the adapter matrices. Rank 4 is fast and compact; rank 8 learns slightly finer details. |
| **LoRA Alpha ($\alpha$)** | `16.0` | The strength scaling factor. Rule of thumb: $\alpha = 2 \times r$ or $4 \times r$. |
| **Micro-Batch Size** | `1` | Number of clips in GPU memory at any exact millisecond. Always use `1` for 10 GB VRAM. |
| **Gradient Accumulation** | `4` | How many micro-batches to gather before updating weights. |
| **Epochs** | `1` to `5` | How many times the model sees the entire dataset. Start with `1` to test the pipeline! |
| **Learning Rate** | `0.0001` ($1\times10^{-4}$) | How big an adjustment step the optimizer takes. Too high causes instability; too low trains very slowly. |
| **Precision** | `bfloat16` | Uses NVIDIA Ampere hardware acceleration to cut VRAM in half. |

---

## 6. Step 4: Launching Training

### Method A: Through the Web UI
1. Ensure the preflight badge shows **READY**.
2. Click **Launch Training**.
3. Watch the live **Training Execution Worker** log terminal on the right:
   - `[System] Job ID registered: train_job_...`
   - `[Worker] CUDA environment confirmed.`
   - `Epoch [1/1] Step 20 - Train Loss: 0.2841 - Peak VRAM: 125 MB`
   - `TRAINING COMPLETED SUCCESSFULLY!`

### Method B: Through the Command Line
```powershell
.venv\Scripts\python.exe training/run_training.py --config training/configs/lora_10gb_rtx3080.yaml --version-name my_first_lora_v1
```

---

## 7. Step 5: Where Are My Checkpoints?

When training completes, your trained LoRA adapter is saved automatically to:
`training/checkpoints/lora_rtx3080/`

You will find:
- `checkpoint_epoch_1.pt` (Full epoch checkpoint with optimizer states)
- `final_model.pt` (Final model checkpoint)
- `final_lora_adapter.pt` (Lightweight standalone LoRA adapter weights)
- `training_config.yaml` (Exact snapshot of the configuration used)

### How to Resume an Interrupted Training Run:
If power went out or training was stopped:
```powershell
.venv\Scripts\python.exe training/run_training.py --config training/configs/lora_10gb_rtx3080.yaml --resume training/checkpoints/lora_rtx3080/checkpoint_epoch_1.pt
```

---

## 8. Common Beginner Questions & Troubleshooting

### Q: Why did I get a "CUDA Out of Memory" (OOM) error?
- **Cause 1:** Another app (Chrome with hardware acceleration, a video game, Blender, or another Python script) is holding onto VRAM.
  - *Fix:* Check `training/utils/hardware.py` to see your free VRAM. Close other apps until you have at least 7 GB free.
- **Cause 2:** Batch size was set $> 1$.
  - *Fix:* Keep `batch_size: 1` and increase `gradient_accumulation_steps: 4`.
- **Cause 3:** Video resolution was set too high (e.g. 1080p).
  - *Fix:* Use the standard training resolution ($128\times128$ or $256\times256$). The generator will upscale frames during inference.

### Q: Is my base model or existing checkpoints overwritten?
- **No.** The pipeline creates a unique experiment folder for each job. Base model weights remain 100% frozen and untouched.

### Q: How do I know if the model is actually learning?
- Look at the `Train Loss` readout in each epoch. It should gradually decrease (e.g. from ~`0.85` down towards ~`0.25`).
- The `Peak VRAM` readout will show you exactly how much memory was used at peak load.
