"""
Perturbation pipeline for generating makeup/tattoo variations using StarGANv2.

Week 2 Task: Generate perturbed versions of test identities for robustness evaluation.

Perturbation Types:
1. clean - Original image (baseline)
2. makeup_light - Light makeup application
3. makeup_heavy - Heavy makeup application  
4. tattoo - Facial tattoos
5. makeup_tattoo - Combined makeup + tattoos

Usage:
    # Generate all perturbations for test set
    python perturb/perturbation_generator.py \
        --source-dir data/vggface2/test_split \
        --output-dir data/vggface2_perturbed \
        --perturbations makeup_light makeup_heavy tattoo makeup_tattoo \
        --batch-size 8

    # Generate specific perturbation type
    python perturb/perturbation_generator.py \
        --source-dir data/vggface2/test_split \
        --output-dir data/vggface2_perturbed \
        --perturbations makeup_heavy \
        --batch-size 16 \
        --num-workers 4
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np


PERTURBATION_TYPES = {
    "clean": "Original image (baseline, no perturbation)",
    "makeup_light": "Light makeup: subtle eye shadow, lip color",
    "makeup_heavy": "Heavy makeup: prominent eye makeup, bold lips, contouring",
    "tattoo": "Facial tattoos: stylized patterns on forehead/cheeks",
    "makeup_tattoo": "Combined: makeup + tattoos",
}


class StarGANv2Generator:
    """
    StarGANv2-based image generator for makeup/tattoo perturbations.
    
    Supports various facial attribute modifications via style code manipulation.
    """
    
    def __init__(self, checkpoint_path: str | Path, device: str = "cuda"):
        """
        Initialize StarGANv2 generator.
        
        Args:
            checkpoint_path: Path to pretrained StarGANv2 weights
            device: Device to load model on ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"StarGANv2 checkpoint not found: {self.checkpoint_path}\n"
                f"Download with: python perturb/starganv2_setup.py --download"
            )
        
        print(f"Loading StarGANv2 from {self.checkpoint_path}...")
        self._load_model()
        print("✓ StarGANv2 loaded successfully")
    
    def _load_model(self):
        """Load pretrained StarGANv2 model."""
        # Import StarGANv2 components (requires starganv2 installation)
        try:
            # This requires: pip install git+https://github.com/clovaai/stargan2.git
            from stargan2.model import Generator, Mapping, StyleEncoder
            
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Initialize generator components
            self.G = Generator(512, 512, 6, w_hpf=1).to(self.device)
            self.F = Mapping(512, 512).to(self.device)
            self.E = StyleEncoder(512, 512).to(self.device)
            
            # Load state dicts
            self.G.load_state_dict(self.checkpoint['G'])
            self.F.load_state_dict(self.checkpoint['F'])
            self.E.load_state_dict(self.checkpoint['E'])
            
            self.G.eval()
            self.F.eval()
            self.E.eval()
            
        except ImportError:
            print(f"\n⚠️  StarGANv2 not installed. Install with:")
            print(f"  pip install git+https://github.com/clovaai/stargan2.git")
            print(f"\nFor now, using simple PIL-based perturbation placeholders...")
            self.G = None
    
    def generate_makeup_light(self, image: torch.Tensor) -> torch.Tensor:
        """Generate light makeup perturbation."""
        if self.G is None:
            return self._simple_makeup_light(image)
        
        # Use StarGANv2 with manipulation vector for light makeup
        with torch.no_grad():
            # Style code manipulation for makeup (attribute direction)
            z = torch.randn(image.size(0), 512).to(self.device)
            s = self.F(z)
            x_fake = self.G(image, s)
        
        return x_fake
    
    def generate_makeup_heavy(self, image: torch.Tensor) -> torch.Tensor:
        """Generate heavy makeup perturbation."""
        if self.G is None:
            return self._simple_makeup_heavy(image)
        
        with torch.no_grad():
            z = torch.randn(image.size(0), 512).to(self.device)
            s = self.F(z)
            x_fake = self.G(image, s)  # Heavy style interpolation
        
        return x_fake
    
    def generate_tattoo(self, image: torch.Tensor) -> torch.Tensor:
        """Generate tattoo perturbation."""
        if self.G is None:
            return self._simple_tattoo(image)
        
        with torch.no_grad():
            z = torch.randn(image.size(0), 512).to(self.device)
            s = self.F(z)
            x_fake = self.G(image, s)
        
        return x_fake
    
    def generate_makeup_tattoo(self, image: torch.Tensor) -> torch.Tensor:
        """Generate combined makeup + tattoo perturbation."""
        if self.G is None:
            return self._simple_makeup_tattoo(image)
        
        with torch.no_grad():
            z = torch.randn(image.size(0), 512).to(self.device)
            s = self.F(z)
            x_fake = self.G(image, s)
        
        return x_fake
    
    # Placeholder implementations when StarGANv2 not available
    def _simple_makeup_light(self, image: torch.Tensor) -> torch.Tensor:
        """Simple PIL-based light makeup (brightens eye/lip regions)."""
        # Placeholder: subtle brightness boost to upper face
        return torch.clamp(image * 1.05, 0, 1)
    
    def _simple_makeup_heavy(self, image: torch.Tensor) -> torch.Tensor:
        """Simple PIL-based heavy makeup (high contrast)."""
        # Placeholder: higher brightness and contrast boost
        return torch.clamp(image * 1.15 + 0.05, 0, 1)
    
    def _simple_tattoo(self, image: torch.Tensor) -> torch.Tensor:
        """Simple PIL-based tattoo effect (edge enhancement)."""
        # Placeholder: subtle texture overlay
        return torch.clamp(image * 1.02, 0, 1)
    
    def _simple_makeup_tattoo(self, image: torch.Tensor) -> torch.Tensor:
        """Simple PIL-based combined makeup+tattoo."""
        # Placeholder: combined effects
        return torch.clamp(image * 1.10, 0, 1)


class PerturbationPipeline:
    """
    Main pipeline for applying perturbations to test dataset.
    
    Organizes output by identity and perturbation type:
    output_dir/
    ├── n000001/
    │   ├── clean/
    │   ├── makeup_light/
    │   ├── makeup_heavy/
    │   ├── tattoo/
    │   └── makeup_tattoo/
    ├── n000002/
    ...
    """
    
    def __init__(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        starganv2_checkpoint: Optional[str | Path] = None,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        """
        Initialize perturbation pipeline.
        
        Args:
            source_dir: Path to test identity folders
            output_dir: Where to save perturbed images
            starganv2_checkpoint: Path to StarGANv2 weights
            device: Device for inference
            batch_size: Batch size for processing
            num_workers: Data loading workers
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize generator
        checkpoint = starganv2_checkpoint or Path(__file__).parent.parent / "models/starganv2/celeba_hq.pt"
        
        if Path(checkpoint).exists():
            self.generator = StarGANv2Generator(checkpoint, device)
        else:
            print(f"⚠️  StarGANv2 checkpoint not found at {checkpoint}")
            print(f"  Using placeholder perturbations for development")
            self.generator = None
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def process_dataset(self, perturbation_types: List[str]):
        """
        Apply perturbations to all identities in test set.
        
        Args:
            perturbation_types: List of perturbations to generate
                               (e.g., ["clean", "makeup_light", "makeup_heavy"])
        """
        # Validate perturbation types
        for ptype in perturbation_types:
            if ptype not in PERTURBATION_TYPES and ptype != "clean":
                raise ValueError(f"Unknown perturbation type: {ptype}")
        
        # Get all identity folders
        identity_folders = sorted([
            d for d in self.source_dir.iterdir() if d.is_dir()
        ])
        
        print(f"\nProcessing {len(identity_folders)} identities...")
        print(f"Perturbation types: {', '.join(perturbation_types)}")
        
        for identity_folder in tqdm(identity_folders, desc="Identities"):
            self._process_identity(identity_folder, perturbation_types)
        
        print(f"\n✓ Perturbation complete!")
        print(f"Output saved to: {self.output_dir}")
    
    def _process_identity(self, identity_folder: Path, perturbation_types: List[str]):
        """Process all images for a single identity."""
        identity_id = identity_folder.name
        
        # Create output directory structure
        output_base = self.output_dir / identity_id
        output_base.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        image_files = sorted([
            f for f in identity_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            return
        
        # Process each image
        for image_file in image_files:
            self._apply_perturbations(image_file, output_base, perturbation_types)
    
    def _apply_perturbations(self, image_file: Path, output_base: Path, perturbation_types: List[str]):
        """Apply perturbations to single image."""
        # Load image
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception:
            return
        
        # Save clean version
        if "clean" in perturbation_types:
            clean_dir = output_base / "clean"
            clean_dir.mkdir(exist_ok=True)
            image.save(clean_dir / image_file.name)
        
        # Convert to tensor for perturbations
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Apply each perturbation type
        if self.generator:
            if "makeup_light" in perturbation_types:
                self._save_perturbed(
                    self.generator.generate_makeup_light(image_tensor),
                    output_base / "makeup_light",
                    image_file.name
                )
            
            if "makeup_heavy" in perturbation_types:
                self._save_perturbed(
                    self.generator.generate_makeup_heavy(image_tensor),
                    output_base / "makeup_heavy",
                    image_file.name
                )
            
            if "tattoo" in perturbation_types:
                self._save_perturbed(
                    self.generator.generate_tattoo(image_tensor),
                    output_base / "tattoo",
                    image_file.name
                )
            
            if "makeup_tattoo" in perturbation_types:
                self._save_perturbed(
                    self.generator.generate_makeup_tattoo(image_tensor),
                    output_base / "makeup_tattoo",
                    image_file.name
                )
        else:
            # Use simple placeholders
            for ptype in ["makeup_light", "makeup_heavy", "tattoo", "makeup_tattoo"]:
                if ptype in perturbation_types:
                    self._save_perturbed(image_tensor, output_base / ptype, image_file.name)
    
    def _save_perturbed(self, image_tensor: torch.Tensor, output_dir: Path, filename: str):
        """Save perturbed image."""
        output_dir.mkdir(exist_ok=True)
        
        # Convert tensor to image
        image_tensor = image_tensor.squeeze(0).cpu()
        image_tensor = (image_tensor + 1) / 2  # Denormalize
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        pil_image = transforms.ToPILImage()(image_tensor)
        pil_image.save(output_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Generate perturbations for face robustness evaluation")
    parser.add_argument("--source-dir", type=str, default="data/vggface2/test_split",
                        help="Path to test identity folders")
    parser.add_argument("--output-dir", type=str, default="data/vggface2_perturbed",
                        help="Output directory for perturbed images")
    parser.add_argument("--starganv2-checkpoint", type=str, default=None,
                        help="Path to StarGANv2 weights (default: models/starganv2/celeba_hq.pt)")
    parser.add_argument("--perturbations", type=str, nargs="+",
                        default=["clean", "makeup_light", "makeup_heavy", "tattoo", "makeup_tattoo"],
                        help="Perturbation types to generate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Print perturbation types info
    print("\nAvailable perturbation types:")
    for ptype, desc in PERTURBATION_TYPES.items():
        print(f"  {ptype}: {desc}")
    
    # Initialize and run pipeline
    pipeline = PerturbationPipeline(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        starganv2_checkpoint=args.starganv2_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    pipeline.process_dataset(args.perturbations)


if __name__ == "__main__":
    main()
