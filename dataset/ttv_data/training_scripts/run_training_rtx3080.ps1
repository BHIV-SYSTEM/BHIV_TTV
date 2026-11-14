# Launch CogVideoX-2B LoRA fine-tuning for RTX 3080 (10GB / 12GB VRAM)
Write-Host "Starting CogVideoX-2B LoRA Training on NVIDIA RTX 3080..." -ForegroundColor Cyan

python training_scripts/train_cogvideox_lora.py --config training_configs/cogvideox_lora_rtx3080.yaml
