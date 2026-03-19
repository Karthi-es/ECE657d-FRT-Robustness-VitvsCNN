"""
StarGANv2 setup and model management.

Downloads and initializes StarGANv2 pretrained weights for makeup/tattoo generation.

Usage:
    python perturb/starganv2_setup.py --download
"""

import os
import gdown
from pathlib import Path


STARGANV2_MODEL_URL = "https://drive.google.com/uc?id=1cNf7BB4xZLagdsJUmxgcpF-NlWR04p5l"  # celeba_hq checkpoint
MODELS_DIR = Path(__file__).parent.parent / "models" / "starganv2"


def download_starganv2_weights(model_dir=MODELS_DIR, force=False):
    """
    Download pretrained StarGANv2 weights.
    
    Models from: https://github.com/clovaai/stargan2
    
    Args:
        model_dir: Directory to save model weights
        force: Redownload even if file exists
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = model_dir / "celeba_hq.pt"
    
    if checkpoint_path.exists() and not force:
        print(f"✓ StarGANv2 model already exists: {checkpoint_path}")
        return checkpoint_path
    
    print(f"Downloading StarGANv2 pretrained weights...")
    print(f"  Source: Google Drive (CelebA-HQ checkpoint)")
    print(f"  Destination: {checkpoint_path}")
    print(f"  Size: ~200 MB")
    
    try:
        gdown.download(STARGANV2_MODEL_URL, str(checkpoint_path), quiet=False)
        print(f"✓ Download complete!")
        return checkpoint_path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"\nAlternative: Download manually from:")
        print(f"  https://github.com/clovaai/stargan2/releases")
        print(f"And place at: {checkpoint_path}")
        raise


def verify_starganv2_installation():
    """Verify StarGANv2 dependencies are installed."""
    try:
        import torch
        import torchvision
        from PIL import Image
        import numpy as np
        print("✓ All StarGANv2 dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup StarGANv2 for perturbations")
    parser.add_argument("--download", action="store_true", help="Download StarGANv2 weights")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR),
                        help="Directory for model weights")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_starganv2_installation()
    
    if args.download:
        download_starganv2_weights(args.model_dir)
    
    if not args.verify and not args.download:
        print("Usage: python starganv2_setup.py --download [--model-dir PATH]")
        print("       python starganv2_setup.py --verify")
