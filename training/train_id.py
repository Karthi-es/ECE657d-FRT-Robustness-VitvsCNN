"""
Training script for identity classification on VGGFace2.

Week 1 Objective: 
- Head-only fine-tuning of ResNet50 and ViT-B/16
- Target: >95% accuracy on clean VGGFace2 subset
- Output: Trained model checkpoints for face verification

Usage:
    python training/train_id.py --model resnet50 --epochs 20 --batch-size 64
    python training/train_id.py --model vit_b16 --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Skipping cloud logging.")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.resnet50_face import create_resnet50_face
from models.vit_b16_face import create_vit_b16_face
from training.data_loaders import create_vggface2_loaders, create_vggface2_folder_loaders


class HeadOnlyTrainer:
    """
    Trainer for head-only fine-tuning on face identification task.
    
    Freezes backbone, trains only classification head.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_wandb: bool = False,
        project_name: str = "ece657d-frt",
        run_name: str = "baseline",
    ):
        """
        Args:
            model: Face identification model (must have .head and .backbone)
            device: torch.device(cuda or cpu)
            learning_rate: LR for head optimizer
            weight_decay: L2 regularization
            use_wandb: Enable W&B logging
            project_name: W&B project name
            run_name: W&B run name
        """
        self.model = model
        self.device = device
        self.use_wandb = use_wandb and HAS_WANDB
        
        # Freeze backbone
        self._freeze_backbone()
        
        # Setup optimizer (only for head)
        self.optimizer = optim.SGD(
            self.model.head.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,  # Decay LR every 10 epochs
            gamma=0.5,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_history = {"loss": [], "accuracy": []}
        self.val_history = {"loss": [], "accuracy": []}
        
        # W&B initialization
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "optimizer": "SGD",
                    "scheduler": "StepLR",
                },
            )
            wandb.watch(self.model.head, log="all", log_freq=100)

    def _freeze_backbone(self):
        """Freeze all backbone parameters (only head is trainable)."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        # Verify only head is trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Training] Trainable parameters: {trainable_params:,} / {total_params:,}")

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (avg_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(images)  # Classification output
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                # Progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct / total:.2%}",
                })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            (avg_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                    # Metrics
                    total_loss += loss.item() * images.size(0)
                    _, predicted = logits.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{correct / total:.2%}",
                    })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_dir: Optional[str | Path] = None,
        save_best_only: bool = True,
    ) -> dict:
        """
        Train for specified number of epochs.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save checkpoint if validation accuracy improves
            
        Returns:
            Training history
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_history["loss"].append(train_loss)
            self.train_history["accuracy"].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            self.val_history["loss"].append(val_loss)
            self.val_history["accuracy"].append(val_acc)
            
            # Learning rate schedule
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Print
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                  f"LR: {current_lr:.2e}")
            
            # W&B logging
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                })
            
            # Save checkpoint
            if checkpoint_dir:
                if save_best_only:
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        checkpoint_path = checkpoint_dir / f"best_model.pth"
                        self._save_checkpoint(checkpoint_path, epoch, val_acc)
                        print(f"  ✓ Best model saved: {checkpoint_path}")
                else:
                    checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:02d}.pth"
                    self._save_checkpoint(checkpoint_path, epoch, val_acc)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed/60:.1f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2%}")
        print(f"{'='*60}\n")
        
        # W&B finish
        if self.use_wandb:
            wandb.finish()
        
        return {
            "train": self.train_history,
            "val": self.val_history,
            "best_val_acc": best_val_acc,
            "elapsed_seconds": elapsed,
        }

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_acc,
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Train face identification model (head-only fine-tuning)")
    parser.add_argument("--model", type=str, choices=["resnet50", "vit_b16"], default="resnet50",
                        help="Model architecture")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to VGGFace2 metadata CSV (use if not using --train-dir)")
    parser.add_argument("--train-dir", type=str, default="data/vggface2/train",
                        help="Path to training directory (folder-based, created by split_vggface2_dataset.py)")
    parser.add_argument("--test-dir", type=str, default="data/vggface2/test_split",
                        help="Path to test directory (folder-based, created by split_vggface2_dataset.py)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training/validation")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for head optimizer")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load data
    if args.csv_path:
        # CSV-based loading (legacy)
        print(f"\nLoading VGGFace2 from CSV: {args.csv_path}...")
        train_loader, val_loader = create_vggface2_loaders(
            csv_path=args.csv_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        # Folder-based loading (recommended)
        print(f"\nLoading VGGFace2 from folders:")
        print(f"  Train: {args.train_dir}")
        print(f"  Test: {args.test_dir}")
        train_loader, val_loader = create_vggface2_folder_loaders(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    
    # Determine number of identities (based on actual dataset)
    num_identities = train_loader.dataset.num_classes
    print(f"Number of identities in training set: {num_identities}")
    print(f"⚠️  Training on {num_identities} identities (if less than expected, dataset may be incomplete)")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == "resnet50":
        model = create_resnet50_face(
            num_identities=num_identities,
            pretrained=True,
        )
    elif args.model == "vit_b16":
        model = create_vit_b16_face(
            num_identities=num_identities,
            pretrained=True,
        )
    
    model.to(device)
    print(f"Model moved to {device}")
    
    # Create trainer
    trainer = HeadOnlyTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_wandb=args.use_wandb,
        project_name="ece657d-frt",
        run_name=f"{args.model}_clean_baseline",
    )
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_best_only=True,
    )
    
    # Save history
    checkpoint_dir = Path(args.checkpoint_dir)
    history_path = checkpoint_dir / f"{args.model}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
