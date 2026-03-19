"""
Validation script to verify all project components work correctly.

Runs before full training to catch configuration/dependency issues.

Usage:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --verbose
"""

import sys
import importlib
from pathlib import Path
from typing import Tuple, List


class ValidationCheck:
    """Track validation results."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def check(self, name: str, test_fn, description: str = ""):
        """Run a validation check."""
        try:
            result = test_fn()
            self.results.append({
                'name': name,
                'status': 'PASS' if result else 'FAIL',
                'description': description,
            })
            if result:
                self.passed += 1
                print(f"  ✓ {name}")
            else:
                self.failed += 1
                print(f"  ✗ {name}")
        except Exception as e:
            self.results.append({
                'name': name,
                'status': 'ERROR',
                'error': str(e),
                'description': description,
            })
            self.failed += 1
            print(f"  ✗ {name}: {e}")
    
    def print_summary(self):
        """Print validation summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Validation Summary: {self.passed}/{total} passed")
        print(f"{'='*60}")
        
        if self.failed > 0:
            print("\n❌ Some validation checks failed. See details above.")
            return False
        else:
            print("\n✓ All validation checks passed!")
            return True


def validate_dependencies():
    """Check all required dependencies."""
    print("\n1. Checking dependencies...")
    checker = ValidationCheck()
    
    deps = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'timm (Vision models)',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
    }
    
    for module, name in deps.items():
        checker.check(
            f"Import {module}",
            lambda m=module: importlib.import_module(m) is not None,
            f"Required: {name}"
        )
    
    return checker.passed, checker.failed


def validate_project_structure():
    """Check project folder structure."""
    print("\n2. Checking project structure...")
    checker = ValidationCheck()
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'data',
        'models',
        'training',
        'perturb',
        'eval',
        'scripts',
        'configs',
    ]
    
    for dirname in required_dirs:
        path = project_root / dirname
        checker.check(
            f"Directory: {dirname}/",
            lambda p=path: p.exists() and p.is_dir(),
            f"Required folder"
        )
    
    required_files = [
        'requirements.txt',
        '.gitignore',
        'README.md',
        'training/train_id.py',
        'training/data_loaders.py',
        'models/resnet50_face.py',
        'models/vit_b16_face.py',
        'perturb/perturbation_generator.py',
        'eval/evaluator.py',
        'eval/metrics.py',
    ]
    
    for filepath in required_files:
        path = project_root / filepath
        checker.check(
            f"File: {filepath}",
            lambda p=path: p.exists() and p.is_file(),
            f"Required file"
        )
    
    return checker.passed, checker.failed


def validate_model_initialization():
    """Test model loading and initialization."""
    print("\n3. Testing model initialization...")
    checker = ValidationCheck()
    
    try:
        from models.resnet50_face import create_resnet50_face
        checker.check(
            "ResNet50 initialization",
            lambda: create_resnet50_face(num_identities=100) is not None,
            "Create ResNet50 model"
        )
    except Exception as e:
        checker.results.append({
            'name': 'ResNet50 initialization',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ ResNet50 initialization: {e}")
    
    try:
        from models.vit_b16_face import create_vit_b16_face
        checker.check(
            "ViT-B/16 initialization",
            lambda: create_vit_b16_face(num_identities=100) is not None,
            "Create ViT-B/16 model"
        )
    except Exception as e:
        checker.results.append({
            'name': 'ViT-B/16 initialization',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ ViT-B/16 initialization: {e}")
    
    return checker.passed, checker.failed


def validate_data_loaders():
    """Test data loader functionality."""
    print("\n4. Testing data loaders...")
    checker = ValidationCheck()
    
    try:
        from training.data_loaders import VGGFace2FolderDataset, get_vggface2_transforms
        
        # Test transforms
        checker.check(
            "VGGFace2 transforms",
            lambda: get_vggface2_transforms('train') is not None,
            "Create preprocessing pipeline"
        )
        
        print(f"  ℹ️  Note: Full data loader test skipped (requires data/)")
        print(f"     Will validate during actual training")
        
    except Exception as e:
        checker.results.append({
            'name': 'Data loader imports',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ Data loader imports: {e}")
    
    return checker.passed, checker.failed


def validate_evaluation_pipeline():
    """Test evaluation pipeline imports."""
    print("\n5. Testing evaluation pipeline...")
    checker = ValidationCheck()
    
    try:
        from eval.metrics import compute_eer, compute_roc_curve
        checker.check(
            "Metrics module",
            lambda: True,
            "Import evaluation metrics"
        )
    except Exception as e:
        checker.results.append({
            'name': 'Metrics module',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ Metrics module: {e}")
    
    try:
        from eval.evaluator import FaceEmbedder, VerificationEvaluator
        checker.check(
            "Evaluator module",
            lambda: True,
            "Import evaluator classes"
        )
    except Exception as e:
        checker.results.append({
            'name': 'Evaluator module',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ Evaluator module: {e}")
    
    return checker.passed, checker.failed


def validate_training_script():
    """Test training script argument parsing."""
    print("\n6. Testing training script...")
    checker = ValidationCheck()
    
    try:
        import argparse
        from training.train_id import main
        checker.check(
            "Training script import",
            lambda: True,
            "train_id.py imports successfully"
        )
    except Exception as e:
        checker.results.append({
            'name': 'Training script import',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ Training script import: {e}")
    
    return checker.passed, checker.failed


def validate_perturbation_pipeline():
    """Test perturbation pipeline imports."""
    print("\n7. Testing perturbation pipeline...")
    checker = ValidationCheck()
    
    try:
        from perturb.perturbation_generator import PerturbationPipeline
        checker.check(
            "Perturbation pipeline",
            lambda: True,
            "Import perturbation module"
        )
    except Exception as e:
        checker.results.append({
            'name': 'Perturbation pipeline',
            'status': 'ERROR',
            'error': str(e),
        })
        checker.failed += 1
        print(f"  ✗ Perturbation pipeline: {e}")
    
    return checker.passed, checker.failed


def main():
    print("="*60)
    print("ECE657D Face Recognition Robustness - Validation Check")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    
    # Run all validations
    p, f = validate_dependencies()
    total_passed += p
    total_failed += f
    
    p, f = validate_project_structure()
    total_passed += p
    total_failed += f
    
    p, f = validate_model_initialization()
    total_passed += p
    total_failed += f
    
    p, f = validate_data_loaders()
    total_passed += p
    total_failed += f
    
    p, f = validate_evaluation_pipeline()
    total_passed += p
    total_failed += f
    
    p, f = validate_training_script()
    total_passed += p
    total_failed += f
    
    p, f = validate_perturbation_pipeline()
    total_passed += p
    total_failed += f
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {total_passed} passed, {total_failed} failed")
    print(f"{'='*60}")
    
    if total_failed == 0:
        print("\n✅ ALL VALIDATION CHECKS PASSED!")
        print("Your project is ready for training.\n")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED!")
        print("Please fix the issues above before starting training.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
