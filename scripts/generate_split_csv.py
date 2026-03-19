"""
Generate CSV metadata from the split VGGFace2 dataset folders.

Creates a reproducible record of the train/test split for GitHub documentation.

Usage:
    python scripts/generate_split_csv.py \
        --train-dir data/vggface2/train \
        --test-dir data/vggface2/test_split \
        --output-csv data/metadata/vggface2_split.csv \
        --metadata-path data/vggface2/split_metadata.json
"""

import os
import json
import csv
import argparse
from pathlib import Path


def count_images_in_folder(folder_path):
    """Count image files in a folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    count = 0
    for f in os.listdir(folder_path):
        if Path(f).suffix.lower() in image_extensions:
            count += 1
    return count


def generate_split_csv(train_dir, test_dir, output_csv, metadata_path=None):
    """
    Generate CSV from split dataset folders.
    
    Args:
        train_dir: Path to training directory (contains identity folders)
        test_dir: Path to test directory (contains identity folders)
        output_csv: Path to output CSV file
        metadata_path: Optional path to split_metadata.json (for reference)
    """
    
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    output_csv = Path(output_csv)
    
    # Validate directories
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    if not test_dir.exists():
        raise ValueError(f"Test directory not found: {test_dir}")
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect split information
    rows = []
    
    # Process training set
    train_identities = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Processing training set: {len(train_identities)} identities...")
    for i, identity_folder in enumerate(train_identities):
        num_images = count_images_in_folder(identity_folder)
        rows.append({
            'identity_id': identity_folder.name,
            'split': 'train',
            'num_images': num_images,
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(train_identities)}")
    
    # Process test set
    test_identities = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    print(f"Processing test set: {len(test_identities)} identities...")
    for i, identity_folder in enumerate(test_identities):
        num_images = count_images_in_folder(identity_folder)
        rows.append({
            'identity_id': identity_folder.name,
            'split': 'test',
            'num_images': num_images,
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_identities)}")
    
    # Write CSV
    print(f"\nWriting CSV to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['identity_id', 'split', 'num_images'])
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary statistics
    train_rows = [r for r in rows if r['split'] == 'train']
    test_rows = [r for r in rows if r['split'] == 'test']
    
    train_images = sum(r['num_images'] for r in train_rows)
    test_images = sum(r['num_images'] for r in test_rows)
    total_images = train_images + test_images
    
    print(f"\n✓ Split CSV created successfully!")
    print(f"\nDataset Summary:")
    print(f"  Training identities: {len(train_rows)}")
    print(f"  Training images: {train_images:,}")
    print(f"  Test identities: {len(test_rows)}")
    print(f"  Test images: {test_images:,}")
    print(f"  Total: {total_images:,} images")
    
    # Optional: Load and print metadata
    if metadata_path and Path(metadata_path).exists():
        print(f"\nMetadata from {metadata_path}:")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            print(f"  Split seed: {metadata['seed']}")
            print(f"  Train ratio: {metadata['train_ratio']:.1%}")
            print(f"  Total identities: {metadata['total_identities']}")
    
    print(f"\n✓ Ready for GitHub! Push the CSV to:")
    print(f"  {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate split CSV from VGGFace2 folders")
    parser.add_argument("--train-dir", type=str, default="data/vggface2/train",
                        help="Path to training directory")
    parser.add_argument("--test-dir", type=str, default="data/vggface2/test_split",
                        help="Path to test directory")
    parser.add_argument("--output-csv", type=str, default="data/metadata/vggface2_split.csv",
                        help="Path to output CSV file")
    parser.add_argument("--metadata-path", type=str, default="data/vggface2/split_metadata.json",
                        help="Path to split metadata JSON (optional, for reference)")
    
    args = parser.parse_args()
    
    generate_split_csv(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_csv=args.output_csv,
        metadata_path=args.metadata_path,
    )
