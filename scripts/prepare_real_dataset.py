import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.datasets.pipeline import build_and_resolve_dataset

def main():
    manifest_path = Path('data/manifests/ttv_training_manifest_120.json')
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        sys.exit(1)

    print(f"Loading {manifest_path}...")
    with open(manifest_path, 'r') as f:
        data = json.load(f)

    print("Downloading remote videos and building dataset. This may take a while depending on your internet connection...")
    res = build_and_resolve_dataset(
        records=data['records'], 
        dataset_name='ttv_training_manifest_120', 
        base_output_dir='data/training', 
        download_remote_videos=True
    )
    
    # Update config paths
    print("Dataset downloaded and resolved to:", res['dataset_dir'])
    
    # Overwrite the old dataset.jsonl and splits in training/datasets
    target_dir = Path("training/datasets")
    shutil.copy(res['dataset_manifest'], target_dir / "dataset.jsonl")
    shutil.copy(res['train_manifest'], target_dir / "dataset_train.jsonl")
    shutil.copy(res['val_manifest'], target_dir / "dataset_val.jsonl")
    shutil.copy(res['test_manifest'], target_dir / "dataset_test.jsonl")
    
    print("Dataset manifests successfully copied to training/datasets.")
    print("You can now start training using:")
    print("python training/run_training.py --config training/configs/lora_finetune.yaml --version-name ttv_lora_gpu_real_v1")

if __name__ == "__main__":
    main()
