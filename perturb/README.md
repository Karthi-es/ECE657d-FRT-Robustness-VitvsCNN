# Perturbation Pipeline (Week 2)

Generate makeup/tattoo perturbations for face robustness evaluation.

## Quick Start

### Step 1: Verify StarGANv2 Setup
```bash
python perturb/starganv2_setup.py --verify
```

### Step 2: Download StarGANv2 Weights (Optional)
```bash
python perturb/starganv2_setup.py --download
```

**Note:** If gdown fails, manually download from:
- [StarGAN2 GitHub Releases](https://github.com/clovaai/stargan2/releases)
- Place at: `models/starganv2/celeba_hq.pt`

### Step 3: Generate Perturbations
After dataset splitting (Week 1 complete):

```bash
# Generate all perturbation types
python perturb/perturbation_generator.py \
    --source-dir data/vggface2/test_split \
    --output-dir data/vggface2_perturbed \
    --perturbations clean makeup_light makeup_heavy tattoo makeup_tattoo \
    --batch-size 8

# Or specific types only
python perturb/perturbation_generator.py \
    --source-dir data/vggface2/test_split \
    --output-dir data/vggface2_perturbed \
    --perturbations makeup_heavy tattoo
```

## Perturbation Types

| Type | Description |
|------|-------------|
| `clean` | Original image (baseline) |
| `makeup_light` | Subtle eye shadow + lip color |
| `makeup_heavy` | Bold eye makeup + contouring |
| `tattoo` | Facial tattoos (forehead/cheeks) |
| `makeup_tattoo` | Combined makeup + tattoos |

## Output Structure

```
data/vggface2_perturbed/
├── n000001/
│   ├── clean/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── makeup_light/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── makeup_heavy/
│   └── tattoo/
│   └── makeup_tattoo/
├── n000002/
└── ...
```

## StarGANv2 Installation (Optional for Full Version)

For full StarGANv2 functionality (not just placeholders):

```bash
pip install git+https://github.com/clovaai/stargan2.git

# Or clone manually:
git clone https://github.com/clovaai/stargan2.git
cd stargan2
pip install -e .
```

## Without StarGANv2

Pipeline includes **placeholder perturbations** for development/testing:
- Works without StarGANv2 installation
- Generates deterministic variations
- Useful for testing evaluation pipeline
- Real perturbations require full StarGANv2+ setup

## Workflow

1. **Week 1**: Train baseline models (ResNet50 + ViT-B/16) on clean data
2. **Week 2**: Generate perturbations on 150-identity test set
3. **Week 3**: Evaluate model robustness on perturbed images
4. **Compare**: Calculate EER drop (ViT vs CNN) across perturbations

## Expected Output Size

- **Test set**: ~150 identities × 200-500 images = 30k-75k images
- **Per perturbation**: ~300-1500 MB (5 types)
- **Total perturbed output**: ~1.5-7.5 GB

Store perturbed images on ecetesla1 external storage if needed.

## Next: Evaluation Pipeline

See `eval/` for robustness evaluation and EER calculation.
