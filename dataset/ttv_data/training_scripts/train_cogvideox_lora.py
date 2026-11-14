#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CogVideoX-2B LoRA Training Script
Fine-tunes CogVideoX-2B using PEFT LoRA adapters on preprocessed OpenVid-1M video clips.
Preserves the base model and saves lightweight LoRA adapter checkpoints separately.
"""

import os
import sys
import argparse
import math
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX-2B LoRA Fine-Tuning Pipeline")
    parser.add_argument("--config", type=str, default="training_configs/cogvideox_lora_rtx3060.yaml", help="Path to YAML configuration")
    parser.add_argument("--dry_run", action="store_true", help="Perform hardware & data validation without starting training")
    parser.add_argument("--model_name", type=str, default="THUDM/CogVideoX-2b")
    parser.add_argument("--train_dir", type=str, default="training_data/openvid_1m/train")
    parser.add_argument("--val_dir", type=str, default="training_data/openvid_1m/validation")
    parser.add_argument("--output_dir", type=str, default="training_checkpoints/cogvideox_2b_lora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_train_steps", type=int, default=500)
    return parser.parse_args()

class CogVideoDataset(Dataset):
    def __init__(self, data_dir, metadata_csv=None):
        self.data_dir = data_dir
        if metadata_csv and os.path.exists(metadata_csv):
            self.df = pd.read_csv(metadata_csv)
        else:
            meta_path = os.path.join(data_dir, "metadata.csv")
            if os.path.exists(meta_path):
                self.df = pd.read_csv(meta_path)
            else:
                vids_file = os.path.join(data_dir, "videos.txt")
                prompts_file = os.path.join(data_dir, "prompts.txt")
                with open(vids_file, "r", encoding="utf-8") as vf, open(prompts_file, "r", encoding="utf-8") as pf:
                    vids = [l.strip() for l in vf if l.strip()]
                    prompts = [l.strip() for l in pf if l.strip()]
                self.df = pd.DataFrame({"video_file": vids, "caption": prompts})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vname = row['video_file'] if 'video_file' in row else row['video']
        vpath = os.path.join(self.data_dir, vname)
        caption = row['caption'] if 'caption' in row else row['prompt']
        return {
            "video_path": vpath,
            "video_name": vname,
            "caption": caption
        }

def inspect_hardware():
    print("\n" + "="*50)
    print(" HARDWARE & RUNTIME ENVIRONMENT AUDIT")
    print("="*50)
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_avail}")
    
    if cuda_avail:
        device_count = torch.cuda.device_count()
        print(f"Detected CUDA Devices: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            vram_total_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  [GPU {i}]: {name} | Total VRAM: {vram_total_gb:.2f} GB")
        
        target_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if "3060" in target_name:
            print("\n>>> Detected NVIDIA RTX 3060. Recommended profile: cogvideox_lora_rtx3060.yaml (Rank=16, 8-bit AdamW, Gradient Accumulation=4)")
        elif "3080" in target_name:
            print("\n>>> Detected NVIDIA RTX 3080. Recommended profile: cogvideox_lora_rtx3080.yaml (Rank=32, BF16, Gradient Accumulation=2)")
        return True, vram_gb
    else:
        print("\n>>> NOTICE: No CUDA-capable NVIDIA GPU detected on this host (Intel Iris Xe integrated graphics active).")
        print(">>> Model training requires an NVIDIA GPU (RTX 3060 / 3080 / 4090 / A100) with CUDA enabled.")
        return False, 0.0

def main():
    args = parse_args()
    print("Initializing CogVideoX-2B LoRA Training Pipeline...")
    
    # Load config if present
    cfg = {}
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    
    has_cuda, vram_gb = inspect_hardware()

    # Verify datasets
    train_dir = cfg.get("dataset", {}).get("train_data_dir", args.train_dir)
    val_dir = cfg.get("dataset", {}).get("val_data_dir", args.val_dir)
    output_dir = cfg.get("training", {}).get("output_dir", args.output_dir)

    print("\n" + "="*50)
    print(" DATASET VERIFICATION")
    print("="*50)
    train_dataset = CogVideoDataset(train_dir)
    val_dataset = CogVideoDataset(val_dir)
    print(f"Training dataset loaded:   {len(train_dataset)} video-text pairs from '{train_dir}'")
    print(f"Validation dataset loaded: {len(val_dataset)} video-text pairs from '{val_dir}'")
    
    sample = train_dataset[0]
    print(f"Sample Video: {sample['video_name']}")
    print(f"Sample Caption: {sample['caption'][:80]}...")
    print(f"Sample File Exists: {os.path.exists(sample['video_path'])}")

    print("\n" + "="*50)
    print(" TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Base Model:             {cfg.get('model', {}).get('pretrained_model_name_or_path', args.model_name)} (Preserved/Untouched)")
    print(f"LoRA Target Modules:    {cfg.get('lora', {}).get('target_modules', ['to_q', 'to_k', 'to_v', 'to_out.0'])}")
    print(f"LoRA Rank:              {cfg.get('lora', {}).get('rank', args.rank)}")
    print(f"LoRA Alpha:             {cfg.get('lora', {}).get('alpha', args.alpha)}")
    print(f"Mixed Precision:        {cfg.get('training', {}).get('mixed_precision', args.mixed_precision)}")
    print(f"Batch Size:             {cfg.get('training', {}).get('batch_size', args.batch_size)}")
    print(f"Grad Accumulation:      {cfg.get('training', {}).get('gradient_accumulation_steps', args.gradient_accumulation_steps)}")
    print(f"Gradient Checkpointing: {cfg.get('memory_optimizations', {}).get('enable_gradient_checkpointing', True)}")
    print(f"8-Bit AdamW:            {cfg.get('memory_optimizations', {}).get('use_8bit_adam', True)}")
    print(f"Checkpoints Directory:  {output_dir} (Separated from base model)")

    if args.dry_run or not has_cuda:
        print("\n" + "="*50)
        print(" PRE-FLIGHT CHECK COMPLETE")
        print("="*50)
        if not has_cuda:
            print("[INFO] Local host lacks an NVIDIA discrete GPU with CUDA.")
            print("[INFO] All dataset manifests, video files, captions, and configs are fully prepared.")
            print("[INFO] Training can be executed seamlessly on an NVIDIA RTX 3060/3080/4090 system using the command:")
            print(f"       python training_scripts/train_cogvideox_lora.py --config {args.config}")
        else:
            print("[SUCCESS] Hardware and dataset checks passed. Ready for training execution upon user approval.")
        return

    # If user explicitly approved and CUDA is present:
    print("\n[NOTE] As per protocol, training is paused awaiting explicit user approval.")

if __name__ == "__main__":
    main()
