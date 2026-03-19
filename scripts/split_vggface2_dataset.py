"""
Split VGGFace2 dataset into train/test sets while preserving folder structure.

Usage:
    python scripts/split_vggface2_dataset.py \
        --source data/vggface2/test \
        --output_dir data/vggface2 \
        --train_ratio 0.7 \
        --seed 42
"""

import os
import shutil
import random
import argparse
import json
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.7, seed=42):
    """
    Split dataset into train/test maintaining identity folder structure.
    
    Args:
        source_dir: Path to source dataset (contains identity folders)
        output_dir: Path where train/ and test/ folders will be created
        train_ratio: Fraction of identities for training (default 0.7)
        seed: Random seed for reproducibility
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Validate source directory
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_path}")
    
    # Get all identity folders
    identity_folders = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    if not identity_folders:
        raise ValueError(f"No identity folders found in {source_path}")
    
    print(f"Found {len(identity_folders)} identity folders")
    
    # Set seed for reproducibility
    random.seed(seed)
    random.shuffle(identity_folders)
    
    # Calculate split
    num_train = int(len(identity_folders) * train_ratio)
    train_identities = identity_folders[:num_train]
    test_identities = identity_folders[num_train:]
    
    print(f"Train: {len(train_identities)} identities ({train_ratio*100:.0f}%)")
    print(f"Test: {len(test_identities)} identities ({(1-train_ratio)*100:.0f}%)")
    
    # Create output directories
    train_dir = output_path / "train"
    test_dir = output_path / "test_split"  # Renamed to avoid conflict
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating {train_dir}...")
    for i, identity_folder in enumerate(train_identities):
        dest = train_dir / identity_folder.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(identity_folder, dest)
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(train_identities)} identities")
    
    print(f"\nCreating {test_dir}...")
    for i, identity_folder in enumerate(test_identities):
        dest = test_dir / identity_folder.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(identity_folder, dest)
        if (i + 1) % 50 == 0:
            print(f"  Copied {i + 1}/{len(test_identities)} identities")
    
    # Save split metadata for reproducibility
    split_metadata = {
        "seed": seed,
        "train_ratio": train_ratio,
        "total_identities": len(identity_folders),
        "train_identities": len(train_identities),
        "test_identities": len(test_identities),
        "train_identities_list": [f.name for f in train_identities],
        "test_identities_list": [f.name for f in test_identities]
    }
    
    metadata_path = output_path / "split_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"\n✓ Split complete!")
    print(f"✓ Metadata saved to {metadata_path}")
    print(f"\nFolder structure:")
    print(f"  {train_dir}/ → {len(train_identities)} identity folders")
    print(f"  {test_dir}/ → {len(test_identities)} identity folders")
    
    # Count total images
    train_images = sum(len(list((train_dir / d).glob("*.*"))) 
                       for d in os.listdir(train_dir))
    test_images = sum(len(list((test_dir / d).glob("*.*"))) 
                      for d in os.listdir(test_dir))
    
    print(f"\nTotal images:")
    print(f"  Train: ~{train_images:,}")
    print(f"  Test: ~{test_images:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VGGFace2 dataset into train/test")
    parser.add_argument("--source", type=str, default="data/vggface2/test",
                        help="Source directory with identity folders")
    parser.add_argument("--output_dir", type=str, default="data/vggface2",
                        help="Output directory where train/ and test/ will be created")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Fraction of identities for training (default 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_dataset(args.source, args.output_dir, args.train_ratio, args.seed)
