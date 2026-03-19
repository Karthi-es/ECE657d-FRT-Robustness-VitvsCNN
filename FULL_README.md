# Face Recognition Robustness: ViT vs CNN

**ECE657D Research Project** - Comparing Vision Transformer robustness against makeup/tattoo perturbations vs CNNs

## 📋 Quick Links

- **📖 [Full Project Proposal](PROJECT_SETUP.md)**
- **📊 [Perturbation Pipeline](perturb/README.md)**
- **📈 [Evaluation Pipeline](eval/README.md)**
- **🏋️ [Training Guide](#training-workflow)**

## 🎯 Research Objective

**Question**: Are Vision Transformers more robust to facial perturbations (makeup/tattoos) than CNNs?

**Success Criteria**:
- Week 1: Achieve >95% accuracy on clean VGGFace2 (both models)
- Week 2: Generate makeup/tattoo perturbations using StarGANv2
- Week 3: Verify ViT EER degradation ≤ CNN degradation on ≥3/5 perturbations

## 📁 Project Structure

```
ECE657d-FRT-Robustness-VitvsCNN/
├── data/
│   ├── metadata/                    # CSV metadata files
│   ├── vggface2/
│   │   ├── test/                   # Original test set (500 identities)
│   │   ├── train/                  # 350 identities (after split)
│   │   └── test_split/             # 150 identities (after split)
│   └── vggface2_perturbed/         # Generated perturbations
├── models/
│   ├── resnet50_face.py            # ResNet50 with ID head (baseline)
│   ├── vit_b16_face.py             # ViT-B/16 with ID head
│   ├── checkpoints/                # Saved model weights
│   └── starganv2/                  # StarGANv2 pretrained weights
├── training/
│   ├── train_id.py                 # Main training script
│   ├── data_loaders.py             # VGGFace2 + LFW data loading
│   └── train_id.py                 # Trainer class
├── perturb/
│   ├── perturbation_generator.py   # StarGANv2-based perturbations
│   ├── starganv2_setup.py          # Download StarGANv2 weights
│   └── README.md                   # Perturbation documentation
├── eval/
│   ├── evaluator.py                # Verification evaluator
│   ├── metrics.py                  # EER, ROC, AUC computation
│   ├── robustness_comparison.py    # ViT vs CNN comparison
│   ├── results/                    # Evaluation results (JSON)
│   ├── analysis/                   # Robustness visualization
│   └── README.md                   # Evaluation documentation
├── scripts/
│   ├── split_vggface2_dataset.py   # Split 500 → 350 train, 150 test
│   ├── generate_split_csv.py       # Generate CSV metadata
│   └── validate_setup.py           # Validate project setup
├── configs/
│   └── training_config.yaml        # Hyperparameters
├── notebooks/
│   └── PROJECT_PROGRESS_ASSESSMENT.md
├── requirements.txt                # All dependencies
├── .gitignore                      # Exclude data/models
├── README.md                       # Project overview
└── PROJECT_SETUP.md                # Detailed proposal
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python scripts/validate_setup.py
```

✅ This checks all dependencies and project structure before starting.

### 3. Prepare Dataset

After downloading VGGFace2 (500 identities):

```bash
# Split into 350 train + 150 test
python scripts/split_vggface2_dataset.py \
    --source data/vggface2/test \
    --output_dir data/vggface2 \
    --train_ratio 0.7 \
    --seed 42

# Generate split CSV for GitHub
python scripts/generate_split_csv.py \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --output-csv data/metadata/vggface2_split.csv
```

Output:
- `data/vggface2/train/` - 350 identities for training
- `data/vggface2/test_split/` - 150 identities for testing
- `data/metadata/vggface2_split.csv` - Split metadata (commit to GitHub)

## 🏋️ Training Workflow

### Week 1: Baseline Training (Head-Only Fine-Tuning)

**Goal**: Achieve >95% accuracy on clean VGGFace2

**Train ResNet50**:
```bash
python training/train_id.py \
    --model resnet50 \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --checkpoint-dir models/checkpoints
```

**Train ViT-B/16**:
```bash
python training/train_id.py \
    --model vit_b16 \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --checkpoint-dir models/checkpoints
```

**Expected Results**:
- ResNet50: ~95-98% validation accuracy (head-only)
- ViT-B/16: ~96-99% validation accuracy (head-only)

### Week 2: Generate Perturbations

**Download StarGANv2** (optional, has placeholder fallback):
```bash
python perturb/starganv2_setup.py --download
```

**Generate Perturbations**:
```bash
python perturb/perturbation_generator.py \
    --source-dir data/vggface2/test_split \
    --output-dir data/vggface2_perturbed \
    --perturbations clean makeup_light makeup_heavy tattoo makeup_tattoo \
    --batch-size 8
```

Output structure:
```
data/vggface2_perturbed/
├── n000001/
│   ├── clean/
│   ├── makeup_light/
│   ├── makeup_heavy/
│   ├── tattoo/
│   └── makeup_tattoo/
└── ...
```

### Week 3: Evaluate Robustness

**Baseline Evaluation** (clean images):
```bash
# ResNet50
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/resnet50_best.pth \
    --model-type resnet50 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results

# ViT-B/16
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/vit_b16_best.pth \
    --model-type vit_b16 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results
```

**Robustness Evaluation** (perturbed images):
```bash
for PERTURB in makeup_light makeup_heavy tattoo makeup_tattoo; do
    python eval/evaluator.py \
        --model-checkpoint models/checkpoints/resnet50_best.pth \
        --model-type resnet50 \
        --test-dir data/vggface2_perturbed/$PERTURB \
        --output-dir eval/results
    
    python eval/evaluator.py \
        --model-checkpoint models/checkpoints/vit_b16_best.pth \
        --model-type vit_b16 \
        --test-dir data/vggface2_perturbed/$PERTURB \
        --output-dir eval/results
done
```

**Compare Robustness**:
```bash
python eval/robustness_comparison.py \
    --results-dir eval/results \
    --output-dir eval/analysis
```

Outputs:
- `eval/analysis/robustness_comparison.png` - Main comparison plot
- `eval/analysis/robustness_analysis.json` - Detailed metrics

## 📊 Key Metrics

### Training Metrics
- **Accuracy**: Classification accuracy on clean faces
- **Loss**: CrossEntropy loss
- **Learning Rate**: Step scheduler (1e-3 → 5e-4 → 2.5e-4)

### Verification Metrics (from `eval/metrics.py`)
- **EER (Equal Error Rate)**: Point where FAR = FRR (primary metric)
- **ROC Curve**: True Positive vs False Positive rates
- **AUC**: Area under ROC curve
- **EER Degradation**: `(perturbed_EER - clean_EER) / clean_EER × 100%`

### Success Criteria
- ✅ Clean EER < 5% (both models)
- ✅ ViT EER degradation ≤ CNN degradation on ≥3/5 perturbations
- ✅ Detailed visualization + analysis

## 🛠 Tools & Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Models | PyTorch + timm | ResNet50 & ViT-B/16 |
| Data Loading | PyTorch DataLoader | Efficient batch processing |
| Training | PyTorch Lightning (optional) | Training orchestration |
| Perturbations | StarGANv2 | Realistic makeup/tattoo generation |
| Evaluation | SciPy + scikit-learn | EER, ROC, metrics |
| Logging | Weights & Biases (optional) | Experiment tracking |
| Visualization | Matplotlib | Results visualization |

## 📝 Training Configuration

See `configs/training_config.yaml`:
```yaml
training:
  epochs: 20
  batch_size: 64
  learning_rate: 1.0e-03
  weight_decay: 1.0e-04
  optimizer: SGD
  scheduler: StepLR
  scheduler_decay_rate: 0.5
  scheduler_decay_step: 10
  num_workers: 4

model:
  backbone_freeze: true
  input_size: 224
  pretrained: true

data:
  train_split: 0.7
  val_split: 0.2
  seed: 42
```

## 🔍 Troubleshooting

### "No module named 'torch'"
```bash
source .venv/bin/activate
pip install torch torchvision
```

### "No identity folders found"
Ensure you've run the split script and data is in correct location:
```bash
ls -la data/vggface2/train/  # Should show identity folders
```

### "Model checkpoint not found"
Ensure training completed and checkpoint saved:
```bash
ls -la models/checkpoints/
```

### Out of memory (OOM)
Reduce batch size:
```bash
python training/train_id.py --batch-size 32 ...
```

## 📚 Additional Resources

- [timm Model Zoo](https://github.com/rwightman/pytorch-image-models)
- [VGGFace2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [StarGAN2 GitHub](https://github.com/clovaai/stargan2)
- [Face Recognition Metrics](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

## 📖 Project Documentation

- **[PROJECT_SETUP.md](PROJECT_SETUP.md)** - Detailed proposal & methodology
- **[perturb/README.md](perturb/README.md)** - Perturbation pipeline guide
- **[eval/README.md](eval/README.md)** - Evaluation pipeline guide
- **[Training Script Help](training/train_id.py)** - `python training/train_id.py --help`

## ✅ Validation Checklist

Before starting training, verify:
- [ ] All dependencies installed: `python scripts/validate_setup.py`
- [ ] Dataset split into train/test folders
- [ ] Checkpoint directory exists: `mkdir -p models/checkpoints`
- [ ] GPU available: Python has torch with CUDA support
- [ ] Enough disk space: ~20GB for 350 identities + perturbations

## 🎓 Author

ECE657D - Face Recognition Robustness Analysis
University of Waterloo

## 📄 License

This project is academic research. Dataset usage follows VGGFace2 terms.

---

**Start with**: `python scripts/validate_setup.py` ✓
