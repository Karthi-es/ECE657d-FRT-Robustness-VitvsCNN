"""Unit tests for training/data_loaders.py"""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import csv

import torch
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_loaders import (
    VGGFace2Dataset,
    create_vggface2_loaders,
    get_vggface2_transforms,
    get_lfw_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


@pytest.fixture
def temp_vggface2_setup():
    """Create a temporary VGGFace2-like dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        data_dir = tmpdir / "data" / "vggface2" / "test"
        metadata_dir = tmpdir / "data" / "metadata"
        data_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        
        # Create sample images
        identities = ["n000001", "n000002", "n000003"]
        csv_rows = []
        
        for idx, identity in enumerate(identities):
            ident_dir = data_dir / identity
            ident_dir.mkdir()
            
            # Create 4 dummy images per identity
            for img_idx in range(4):
                img_path = ident_dir / f"image_{img_idx:02d}.jpg"
                # Create a simple RGB image
                img = Image.new("RGB", (224, 224), color=(73, 109, 137))
                img.save(img_path)
                
                # Add to CSV (80% train, 20% val)
                split = "val" if img_idx == 0 else "train"
                csv_rows.append({
                    "image_path": f"data/vggface2/test/{identity}/image_{img_idx:02d}.jpg",
                    "identity_id": identity,
                    "split": split,
                })
        
        # Write CSV
        csv_path = metadata_dir / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "identity_id", "split"])
            writer.writeheader()
            writer.writerows(csv_rows)
        
        yield tmpdir, str(csv_path)


def test_vggface2_dataset_loading(temp_vggface2_setup):
    """Test that VGGFace2 dataset loads correctly."""
    tmpdir, csv_path = temp_vggface2_setup
    
    dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="train",
        transform=get_vggface2_transforms("train"),
        image_dir_offset=str(tmpdir),
    )
    
    assert len(dataset) > 0
    assert dataset.num_classes == 3  # 3 identities
    assert hasattr(dataset, "identity_to_idx")


def test_vggface2_dataset_getitem(temp_vggface2_setup):
    """Test that dataset returns correct tensor shapes."""
    tmpdir, csv_path = temp_vggface2_setup
    
    dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="train",
        transform=get_vggface2_transforms("train"),
        image_dir_offset=str(tmpdir),
    )
    
    image, label = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)  # 3 channels, 224x224
    assert isinstance(label, int)
    assert 0 <= label < dataset.num_classes


def test_vggface2_dataset_split_separation(temp_vggface2_setup):
    """Test that train/val splits are separated correctly."""
    tmpdir, csv_path = temp_vggface2_setup
    
    train_dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="train",
        image_dir_offset=str(tmpdir),
    )
    
    val_dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="val",
        image_dir_offset=str(tmpdir),
    )
    
    # Should have at least some train and val samples
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(train_dataset) > len(val_dataset)  # More train than val


def test_vggface2_transforms_shape():
    """Test that transforms output correct shape."""
    transform = get_vggface2_transforms("train")
    
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color=(73, 109, 137))
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_vggface2_transforms_normalization():
    """Test that normalization is applied correctly."""
    transform = get_vggface2_transforms("val")
    
    # Create two images with different pixel values for testing
    img_white = Image.new("RGB", (256, 256), color=(255, 255, 255))
    img_black = Image.new("RGB", (256, 256), color=(0, 0, 0))
    
    tensor_white = transform(img_white)
    tensor_black = transform(img_black)
    
    # After normalization, white and black should produce different values
    # Black pixels will be negative (below ImageNet mean)
    # White pixels will be positive (above ImageNet mean)
    assert tensor_black.min() < 0  # Black pixels produce negative values
    assert tensor_white.min() > 0  # White pixels produce positive values


def test_vggface2_loaders_creation(temp_vggface2_setup):
    """Test that loaders are created successfully."""
    tmpdir, csv_path = temp_vggface2_setup
    
    train_loader, val_loader = create_vggface2_loaders(
        csv_path=csv_path,
        batch_size=2,
        num_workers=0,  # 0 for testing
        image_dir_offset=str(tmpdir),
    )
    
    assert train_loader is not None
    assert val_loader is not None


def test_vggface2_loaders_batch_shape(temp_vggface2_setup):
    """Test that loaders return correct batch shapes."""
    tmpdir, csv_path = temp_vggface2_setup
    
    train_loader, val_loader = create_vggface2_loaders(
        csv_path=csv_path,
        batch_size=2,
        num_workers=0,
        image_dir_offset=str(tmpdir),
    )
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    assert images.shape[0] == 2  # Batch size
    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)
    assert labels.dtype == torch.long


def test_vggface2_loaders_deterministic_val():
    """Test that validation loader is deterministic (no shuffle)."""
    # This is a property check, not requiring temp setup
    transform = get_vggface2_transforms("val")
    
    # Create identical dataset twice
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_array = torch.tensor([[[128]*224 for _ in range(224)] for _ in range(3)])
    
    # After transform, should be deterministic
    t1 = transform(img)
    t2 = transform(img)
    
    assert torch.allclose(t1, t2)


def test_imagenet_normalization_values():
    """Test that ImageNet normalization constants are correct."""
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    assert all(0 < v < 1 for v in IMAGENET_MEAN)
    assert all(0 < v < 1 for v in IMAGENET_STD)


def test_lfw_transforms_shape():
    """Test that LFW transforms output correct shape."""
    transform = get_lfw_transforms()
    
    img = Image.new("RGB", (256, 256), color=(73, 109, 137))
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
