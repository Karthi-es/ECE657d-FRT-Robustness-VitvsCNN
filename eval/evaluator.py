"""
Face verification evaluator for robustness analysis.

Week 3 Task: Evaluate model robustness against makeup/tattoo perturbations.

- Extracts embeddings from trained models
- Computes pairwise similarities
- Calculates EER (Equal Error Rate) on clean and perturbed images
- Compares robustness: ViT vs CNN

Usage:
    # Evaluate on clean test set (baseline)
    python eval/evaluator.py \
        --model-checkpoint models/checkpoints/resnet50_best.pth \
        --model-type resnet50 \
        --test-dir data/vggface2/test_split \
        --output-dir eval/results

    # Evaluate on perturbed images
    python eval/evaluator.py \
        --model-checkpoint models/checkpoints/vit_b16_best.pth \
        --model-type vit_b16 \
        --test-dir data/vggface2_perturbed/makeup_heavy \
        --output-dir eval/results
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

from eval.metrics import compute_eer, compute_roc_curve, print_verification_results, plot_roc_curve


class FaceEmbedder:
    """
    Extract face embeddings from trained identification models.
    Removes the classification head to get embedding vectors.
    """
    
    def __init__(
        self,
        model_checkpoint: str | Path,
        model_type: str = "resnet50",
        device: str = "cuda",
    ):
        """
        Initialize embedder with trained model.
        
        Args:
            model_checkpoint: Path to trained model checkpoint
            model_type: Model architecture ("resnet50" or "vit_b16")
            device: Device for inference
        """
        self.device = torch.device(device)
        self.model_type = model_type
        
        # Load checkpoint
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        
        # Import model creators
        if model_type == "resnet50":
            from models.resnet50_face import create_resnet50_face
            # Dummy num_classes, will be replaced by checkpoint
            self.model = create_resnet50_face(num_identities=1000, pretrained=False)
        elif model_type == "vit_b16":
            from models.vit_b16_face import create_vit_b16_face
            self.model = create_vit_b16_face(num_identities=1000, pretrained=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded {model_type} model from {model_checkpoint}")
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract embedding vector for single image.
        
        Args:
            image: PIL Image
        
        Returns:
            Embedding vector (512-dim for ResNet50, 768-dim for ViT-B/16)
        """
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass (get backbone output before head)
        with torch.no_grad():
            if self.model_type == "resnet50":
                # For ResNet50: extract from backbone
                embedding = self.model.backbone(image_tensor)  # [1, 2048, 1, 1]
                embedding = F.adaptive_avg_pool2d(embedding, 1)  # [1, 2048]
                embedding = embedding.view(embedding.size(0), -1)
            else:
                # For ViT: extract from backbone (before head)
                embedding = self.model.backbone(image_tensor)  # [1, 768]
        
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalization
        return embedding.squeeze(0).cpu().numpy()
    
    def extract_embeddings_batch(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for multiple images efficiently.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
        
        Returns:
            Embeddings array [N, embedding_dim]
        """
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i+batch_size]
            batch_embeddings = []
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    embedding = self.extract_embedding(image)
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"  ⚠️  Failed to process {img_path}: {e}")
                    continue
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class VerificationEvaluator:
    """
    Evaluate face verification performance (EER, accuracy, robustness).
    """
    
    def __init__(
        self,
        embedder: FaceEmbedder,
        output_dir: str | Path = "eval/results",
    ):
        """
        Initialize evaluator.
        
        Args:
            embedder: FaceEmbedder instance
            output_dir: Directory for saving results
        """
        self.embedder = embedder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_test_set(
        self,
        test_dir: str | Path,
        num_pairs: Optional[int] = None,
        same_person_pairs: float = 0.5,
    ) -> dict:
        """
        Evaluate on test set with same/different person pairs.
        
        Args:
            test_dir: Directory with identity folders
            num_pairs: Number of pairs to evaluate (None = all)
            same_person_pairs: Fraction of same-person pairs (0.5 = balanced)
        
        Returns:
            Results dict with EER, AUC, etc.
        """
        test_dir = Path(test_dir)
        
        # Get all identity folders
        identity_folders = sorted([d for d in test_dir.iterdir() if d.is_dir()])
        
        print(f"\nPreparing verification pairs...")
        print(f"Found {len(identity_folders)} identities")
        
        # Collect all image paths by identity
        identity_images = {}
        for identity_folder in identity_folders:
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            images = sorted([
                f for f in identity_folder.iterdir()
                if f.suffix.lower() in image_extensions
            ])
            if images:
                identity_images[identity_folder.name] = images
        
        # Generate same/different pairs
        pairs = []  # List of (path1, path2, label)
        
        # Same person pairs (label=1)
        for identity_id, images in identity_images.items():
            if len(images) >= 2:
                n_same = max(1, len(images) // 5)  # Use ~20% of images for same pairs
                for i in range(n_same):
                    idx1 = np.random.randint(0, len(images))
                    idx2 = np.random.randint(0, len(images))
                    if idx1 != idx2:
                        pairs.append((images[idx1], images[idx2], 1))
        
        # Different person pairs (label=0)
        identity_ids = list(identity_images.keys())
        n_different = int(len(pairs) / same_person_pairs) - len(pairs)
        for _ in range(n_different):
            id1, id2 = np.random.choice(identity_ids, 2, replace=False)
            img1 = np.random.choice(identity_images[id1])
            img2 = np.random.choice(identity_images[id2])
            pairs.append((img1, img2, 0))
        
        if num_pairs:
            pairs = pairs[:num_pairs]
        
        print(f"Generated {len(pairs)} verification pairs")
        print(f"  Same person: {sum(1 for _, _, l in pairs if l == 1)}")
        print(f"  Different: {sum(1 for _, _, l in pairs if l == 0)}")
        
        # Extract embeddings and compute similarities
        print(f"\nExtracting embeddings...")
        similarities = []
        labels = []
        
        for img_path1, img_path2, label in tqdm(pairs, desc="Computing similarities"):
            try:
                img1 = Image.open(img_path1).convert('RGB')
                img2 = Image.open(img_path2).convert('RGB')
                
                emb1 = self.embedder.extract_embedding(img1)
                emb2 = self.embedder.extract_embedding(img2)
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)
                labels.append(label)
            except Exception:
                continue
        
        similarities = np.array(similarities)
        labels = np.array(labels)
        
        # Compute metrics
        eer, eer_threshold = compute_eer(labels, similarities)
        fpr, tpr, auc, _ = compute_roc_curve(labels, similarities)
        
        results = {
            "model_type": self.embedder.model_type,
            "num_pairs": len(similarities),
            "num_same": sum(labels),
            "num_different": len(labels) - sum(labels),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "auc": float(auc),
            "similarities_mean_same": float(similarities[labels == 1].mean()),
            "similarities_mean_different": float(similarities[labels == 0].mean()),
        }
        
        # Print results
        print_verification_results(labels, similarities, f"{self.embedder.model_type} Verification")
        
        # Plot and save ROC
        roc_path = self.output_dir / f"roc_{self.embedder.model_type}.png"
        plot_roc_curve(labels, similarities, str(roc_path), title=f"ROC Curve - {self.embedder.model_type}")
        
        return results, similarities, labels
    
    def save_results(self, results: dict, perturbation_type: str = "clean"):
        """Save results to JSON."""
        output_file = self.output_dir / f"results_{perturbation_type}_{self.embedder.model_type}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face verification on test set")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--model-type", type=str, choices=["resnet50", "vit_b16"], required=True,
                        help="Model architecture")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test directory")
    parser.add_argument("--output-dir", type=str, default="eval/results",
                        help="Output directory for results")
    parser.add_argument("--num-pairs", type=int, default=None,
                        help="Number of verification pairs (None = all)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Initialize embedder
    embedder = FaceEmbedder(
        model_checkpoint=args.model_checkpoint,
        model_type=args.model_type,
        device=args.device,
    )
    
    # Initialize evaluator
    evaluator = VerificationEvaluator(embedder, args.output_dir)
    
    # Run evaluation
    results, similarities, labels = evaluator.evaluate_test_set(
        test_dir=args.test_dir,
        num_pairs=args.num_pairs,
    )
    
    # Save results
    evaluator.save_results(results, "clean")


if __name__ == "__main__":
    main()
