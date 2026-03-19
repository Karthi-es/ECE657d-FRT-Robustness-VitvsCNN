"""
Data loaders for VGGFace2 face identification and LFW face verification.

VGGFace2 loader: Identity classification task (training/validation)
LFW loader: Face verification task (testing with same/different pairs)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ImageNet normalization statistics (for timm pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VGGFace2FolderDataset(Dataset):
    """
    VGGFace2 face identification dataset loader from folder structure.
    
    Expects directory structure:
    data/vggface2/train/
    ├── identity_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── identity_2/
    └── ...
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: Root directory containing identity folders
            transform: Torchvision transforms pipeline
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.identity_to_idx = {}
        
        # Scan directory structure
        identity_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if not identity_folders:
            raise ValueError(f"No identity folders found in {self.root_dir}")
        
        # Load all images from identity folders
        for idx, identity_folder in enumerate(identity_folders):
            self.identity_to_idx[identity_folder.name] = idx
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            image_files = [
                f for f in identity_folder.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            
            for image_file in image_files:
                self.samples.append((image_file, idx))
        
        if not self.samples:
            raise ValueError(f"No images found in {self.root_dir}")
        
        self.num_classes = len(self.identity_to_idx)
        print(f"[VGGFace2Folder] Loaded {len(self.samples)} images from {self.num_classes} identities")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label)
        """
        image_path, label = self.samples[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image: {image_path}") from e
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class VGGFace2Dataset(Dataset):
    """
    VGGFace2 face identification dataset loader.
    
    Loads images from CSV metadata file, assigns identity labels.
    Use case: Identity classification training (predict which of N identities)
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",  # "train" or "val"
        transform: Optional[Callable] = None,
        image_dir_offset: str = "",
    ):
        """
        Args:
            csv_path: Path to metadata CSV (columns: image_path, identity_id, split)
            split: "train" or "val" to load only specified split
            transform: Torchvision transforms pipeline
            image_dir_offset: Prefix for image paths (default: use paths as-is)
        """
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform
        self.image_dir_offset = image_dir_offset
        
        # Load CSV and filter by split
        self.samples = []
        self.identity_to_idx = {}  # Map identity_id → class index
        next_idx = 0
        
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    identity_id = row["identity_id"]
                    
                    # Assign class index to identity (consistent across dataset)
                    if identity_id not in self.identity_to_idx:
                        self.identity_to_idx[identity_id] = next_idx
                        next_idx += 1
                    
                    image_path = row["image_path"]
                    label = self.identity_to_idx[identity_id]
                    self.samples.append((image_path, label, identity_id))
        
        if not self.samples:
            raise ValueError(f"No samples found for split='{split}' in {csv_path}")
        
        self.num_classes = len(self.identity_to_idx)
        print(f"[VGGFace2] Loaded {len(self.samples)} images from {self.num_classes} identities (split={split})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Returns:
            (image_tensor, label)
            - image_tensor: Normalized image [3, 224, 224]
            - label: Class index (0 to num_classes-1)
        """
        image_path, label, _ = self.samples[idx]
        
        # Handle path prefix
        if self.image_dir_offset:
            full_path = Path(self.image_dir_offset) / image_path
        else:
            full_path = Path(image_path)
        
        # Load image
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image: {full_path}") from e
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LFWPairsDataset(Dataset):
    """
    LFW (Labeled Faces in the Wild) face verification dataset.
    
    Loads image pairs and labels for same/different person classification.
    Use case: Face verification (predict if two images are same person)
    """

    def __init__(
        self,
        pairs_file: str | Path,
        lfw_root: str | Path,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            pairs_file: Path to pairs.txt file (LFW standard format)
            lfw_root: Path to lfw-deepfunneled directory
            transform: Torchvision transforms pipeline
        """
        self.pairs_file = Path(pairs_file)
        self.lfw_root = Path(lfw_root)
        self.transform = transform
        
        # Parse pairs.txt
        # Format:
        # - First line: total_pairs = N
        # - Lines 1-N (same person): name image_id1 image_id2
        # - Lines N+1 onwards (different person): name1 image_id1 name2 image_id2
        self.pairs = []
        
        with open(self.pairs_file, "r") as f:
            lines = f.readlines()
        
        for line in lines[1:]:  # Skip first line
            parts = line.strip().split()
            if len(parts) == 3:
                # Same person: name, img1_id, img2_id
                name, img1_id, img2_id = parts
                path1 = self.lfw_root / name / f"{name}_{int(img1_id):04d}.jpg"
                path2 = self.lfw_root / name / f"{name}_{int(img2_id):04d}.jpg"
                label = 1  # Same person
            elif len(parts) == 4:
                # Different person: name1, img1_id, name2, img2_id
                name1, img1_id, name2, img2_id = parts
                path1 = self.lfw_root / name1 / f"{name1}_{int(img1_id):04d}.jpg"
                path2 = self.lfw_root / name2 / f"{name2}_{int(img2_id):04d}.jpg"
                label = 0  # Different person
            else:
                continue
            
            # Only add pair if both images exist
            if path1.exists() and path2.exists():
                self.pairs.append((str(path1), str(path2), label))
        
        print(f"[LFW] Loaded {len(self.pairs)} verification pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            (image1_tensor, image2_tensor, label)
            - image1_tensor: First face image [3, 224, 224]
            - image2_tensor: Second face image [3, 224, 224]
            - label: 1 if same person, 0 if different
        """
        path1, path2, label = self.pairs[idx]
        
        # Load images
        try:
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load LFW pair: {path1}, {path2}") from e
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_vggface2_transforms(split: str = "train") -> transforms.Compose:
    """
    Get image preprocessing pipeline for VGGFace2.
    
    Args:
        split: "train" or "val" - applies augmentation only to train split
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    if split == "train":
        # Training augmentation (light to preserve face identity)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Validation: no augmentation (deterministic)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_lfw_transforms() -> transforms.Compose:
    """Get image preprocessing for LFW (same as validation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_vggface2_loaders(
    csv_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    image_dir_offset: str = "",
) -> tuple[DataLoader, DataLoader]:
    """
    Create VGGFace2 DataLoaders for training and validation.
    
    Args:
        csv_path: Path to VGGFace2 metadata CSV
        batch_size: Batch size for training/validation
        num_workers: Number of data loading workers
        image_dir_offset: Prefix for image paths
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="train",
        transform=get_vggface2_transforms("train"),
        image_dir_offset=image_dir_offset,
    )
    
    val_dataset = VGGFace2Dataset(
        csv_path=csv_path,
        split="val",
        transform=get_vggface2_transforms("val"),
        image_dir_offset=image_dir_offset,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_vggface2_folder_loaders(
    train_dir: str | Path,
    test_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """
    Create VGGFace2 DataLoaders from folder structure.
    
    This function loads identity folders directly without needing a CSV file.
    Perfect for the split created by split_vggface2_dataset.py.
    
    Args:
        train_dir: Path to training directory (contains identity folders)
        test_dir: Path to test directory (contains identity folders)
        batch_size: Batch size for training/validation
        num_workers: Number of data loading workers
        val_split: Fraction of test set to use for validation (default 0.2)
    
    Returns:
        (train_loader, val_loader)
    
    Example:
        train_loader, val_loader = create_vggface2_folder_loaders(
            train_dir="data/vggface2/train",
            test_dir="data/vggface2/test_split",
            batch_size=64,
        )
    """
    # Load full test set first (will be split into val)
    full_test_dataset = VGGFace2FolderDataset(
        root_dir=test_dir,
        transform=get_vggface2_transforms("val"),
    )
    
    # Split test into val
    num_val = int(len(full_test_dataset) * val_split)
    num_test = len(full_test_dataset) - num_val
    val_dataset, _ = torch.utils.data.random_split(
        full_test_dataset,
        [num_val, num_test],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Load training set
    train_dataset = VGGFace2FolderDataset(
        root_dir=train_dir,
        transform=get_vggface2_transforms("train"),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_lfw_loader(
    pairs_file: str | Path,
    lfw_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create LFW verification DataLoader.
    
    Args:
        pairs_file: Path to pairs.txt
        lfw_root: Path to lfw-deepfunneled directory
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        LFW DataLoader
    """
    dataset = LFWPairsDataset(
        pairs_file=pairs_file,
        lfw_root=lfw_root,
        transform=get_lfw_transforms(),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader
