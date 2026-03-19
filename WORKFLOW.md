# Complete Workflow Guide

Step-by-step guide for the entire project from dataset to results.

## Phase 1: Setup (Day 1)

### 1.1 Environment Setup
```bash
# Clone/download project
cd ECE657d-FRT-Robustness-VitvsCNN

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate setup
python scripts/validate_setup.py
```

**Expected Output**:
```
✓ All validation checks passed!
Your project is ready for training.
```

### 1.2 Prepare SSH Access to Data Server
```bash
# Test SSH connection
ssh kelayape@ecetesla1.uwaterloo.ca

# Note: Have local VGGFace2 data ready for transfer
```

## Phase 2: Data Transfer (Day 1-2, ~2-4 hours)

### 2.1 Transfer Dataset to Server
From Windows local system (Git Bash):
```bash
# Use SCP with keep-alive to handle disconnections
scp -r -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 \
    "D:\VGG-Face2\data\vggface2_test\test" \
    kelayape@ecetesla1.uwaterloo.ca:/home/kelayape/ECE657d-FRT-Robustness-VitvsCNN/data/
```

**Status**: Currently at 10% (1.2 hours estimated remaining)

### 2.2 Verify Transfer Complete
On ecetesla1 server:
```bash
# Check transfer size
du -sh ~/ECE657d-FRT-Robustness-VitvsCNN/data/

# Count identity folders
ls -1 ~/ECE657d-FRT-Robustness-VitvsCNN/data/vggface2/test/ | wc -l
# Should show: 500
```

## Phase 3: Dataset Preparation (Day 2, ~30 min)

### 3.1 Split Dataset
On ecetesla1 server:
```bash
cd ~/ECE657d-FRT-Robustness-VitvsCNN

python scripts/split_vggface2_dataset.py \
    --source data/vggface2/test \
    --output_dir data/vggface2 \
    --train_ratio 0.7 \
    --seed 42
```

**Expected Output**:
```
✓ Split complete!
  Train: 350 identities (~70k-175k images)
  Test: 150 identities (~30k-75k images)
```

### 3.2 Generate Split CSV
```bash
python scripts/generate_split_csv.py \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --output-csv data/metadata/vggface2_split.csv
```

**Output**: `data/metadata/vggface2_split.csv` (commit to GitHub for reproducibility)

## Phase 4: Week 1 - Baseline Training (Day 3-4, ~4-8 hours)

### 4.1 Train ResNet50 (Head-Only Fine-Tuning)
```bash
python training/train_id.py \
    --model resnet50 \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --checkpoint-dir models/checkpoints \
    --use-wandb
```

**Expected Time**: ~20-30 min per epoch on RTX 3070 = ~400-600 min total

**Success Criteria**:
- Validation accuracy > 95%
- Training loss decreasing
- Model saved to: `models/checkpoints/resnet50_best.pth`

### 4.2 Train ViT-B/16 (Head-Only Fine-Tuning)
```bash
python training/train_id.py \
    --model vit_b16 \
    --train-dir data/vggface2/train \
    --test-dir data/vggface2/test_split \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --checkpoint-dir models/checkpoints \
    --use-wandb
```

**Expected Time**: ~20-30 min per epoch = ~400-600 min total

**Success Criteria**:
- Validation accuracy > 95%
- Model saved to: `models/checkpoints/vit_b16_best.pth`

### 4.3 Review Training Results
Check wandb dashboard or local logs for:
- Training vs validation curves
- Both models achieving >95% accuracy
- Stable loss progression

## Phase 5: Week 2 - Generate Perturbations (Day 5, ~2-4 hours)

### 5.1 Setup StarGANv2 (Optional)
```bash
# Verify dependencies
python perturb/starganv2_setup.py --verify

# Download pretrained weights (if available)
python perturb/starganv2_setup.py --download

# Or install from GitHub for full functionality
pip install git+https://github.com/clovaai/stargan2.git
```

**Note**: Pipeline works with or without StarGANv2 (uses placeholders if unavailable)

### 5.2 Generate Perturbations
```bash
python perturb/perturbation_generator.py \
    --source-dir data/vggface2/test_split \
    --output-dir data/vggface2_perturbed \
    --perturbations clean makeup_light makeup_heavy tattoo makeup_tattoo \
    --batch-size 8
```

**Expected Time**: 2-4 hours depending on StarGANv2 efficiency

**Output Structure**:
```
data/vggface2_perturbed/
├── n000001/
│   ├── clean/ (baseline)
│   ├── makeup_light/
│   ├── makeup_heavy/
│   ├── tattoo/
│   └── makeup_tattoo/
└── ... (150 identities)
```

**Disk Space**: ~1.5-7.5 GB total (7 versions × ~200 MB each)

## Phase 6: Week 3 - Robustness Evaluation (Day 6-7, ~6-8 hours)

### 6.1 Baseline Evaluation (Clean Images)
```bash
# ResNet50 on clean
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/resnet50_best.pth \
    --model-type resnet50 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results

# ViT-B/16 on clean
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/vit_b16_best.pth \
    --model-type vit_b16 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results
```

**Expected Output**:
```
eval/results/
├── results_clean_resnet50.json
├── results_clean_vit_b16.json
├── roc_resnet50.png
└── roc_vit_b16.png
```

**Expected EER**: <5% (if training succeeded)

### 6.2 Perturbed Image Evaluation
```bash
# Evaluate both models on each perturbation type
for PERTURB in makeup_light makeup_heavy tattoo makeup_tattoo; do
    echo "Evaluating on $PERTURB..."
    
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

**Time per perturbation**: ~30-60 min (2 models × 150 identities × ~200 images each)

### 6.3 Robustness Comparison
```bash
python eval/robustness_comparison.py \
    --results-dir eval/results \
    --output-dir eval/analysis
```

**Expected Output**:
```
eval/analysis/
├── robustness_comparison.png  # Main comparison plot
└── robustness_analysis.json   # Detailed metrics
```

**Output includes**:
- EER for each model & perturbation type
- EER degradation from baseline (%)
- Visual comparison of robustness
- JSON with all metrics

## Phase 7: Final Results & Submission (Day 8)

### 7.1 Prepare Results Summary
```bash
# Package all results
mkdir -p results_summary
cp eval/analysis/* results_summary/
cp eval/results/roc_*.png results_summary/
cp notebooks/PROJECT_PROGRESS_ASSESSMENT.md results_summary/
```

### 7.2 Commit Code to GitHub
```bash
git add .
git commit -m "Final robustness evaluation results"
git push origin main
```

**Important**: Do NOT commit:
- `data/vggface2/train/` and `data/vggface2/test_split/` (too large)
- `data/vggface2_perturbed/` (too large)
- `models/checkpoints/` (model weights)

**DO commit**:
- `data/metadata/vggface2_split.csv` (split metadata)
- `eval/results/*.json` (small results files)
- `eval/analysis/*.json` (analysis results)

### 7.3 Generate Final Report
```bash
# Copy all analysis
results_summary/
├── robustness_comparison.png
├── robustness_analysis.json
├── roc_resnet50.png
├── roc_vit_b16.png
└── training_summary.md
```

## Timeline Estimate

| Phase | Task | Duration | Cumulative |
|-------|------|----------|-----------|
| 1 | Environment Setup | 30 min | 30 min |
| 2 | Data Transfer (SCP) | 2-4 hrs | 3-4.5 hrs |
| 3 | Dataset Split | 30 min | 4-5 hrs |
| 4 | ResNet50 Training | 6-10 hrs | 10-15 hrs |
| 4 | ViT-B/16 Training | 6-10 hrs | 16-25 hrs |
| 5 | Generate Perturbations | 2-4 hrs | 18-29 hrs |
| 6 | Evaluate All | 6-8 hrs | 24-37 hrs |
| 7 | Final Results | 1 hr | 25-38 hrs |

**Total: ~3-5 days of actual work (parallel tasks where possible)**

## Key Checkpoints

- [ ] **Day 1**: Validation passes, data transfer starts
- [ ] **Day 2**: Dataset split complete, split CSV generated
- [ ] **Day 4**: ResNet50 training complete, >95% accuracy
- [ ] **Day 4**: ViT-B/16 training complete, >95% accuracy
- [ ] **Day 5**: Perturbations generated (~150 identities, 5 types)
- [ ] **Day 7**: All models evaluated on clean + perturbed
- [ ] **Day 7**: Robustness comparison generated
- [ ] **Day 8**: Results committed to GitHub

## Troubleshooting Quick Guide

**Training stuck?**
- Check GPU memory: `nvidia-smi`
- Reduce batch size: `--batch-size 32`
- Check data loading: Verify folders exist and have images

**Evaluation slow?**
- Use subset: `--num-pairs 5000`
- Increase batch size: `--batch-size 64`
- Use CPU: `--device cpu` (if GPU memory issue)

**Out of space?**
- Remove old checkpoints: `rm models/checkpoints/*.pth` (keep best)
- Compress perturbations: Archive to external storage
- Check disk: `df -h /home/kelayape/`

**Results don't look right?**
- Verify training accuracy >95% (should be)
- Check embedding extraction works
- Ensure correct model checkpoint loaded

## Next Steps

1. **Immediately**: Wait for SCP transfer to complete (monitor progress)
2. **When ready**: Run `validate_setup.py` to confirm everything
3. **Then**: Execute Phase 3 (dataset split) as first action
4. **Follow**: The workflow phases in order

---

**Current Status**: Data transfer ~10% complete (2-4 hours remaining)

**Next Phase**: Dataset split (when transfer finishes)
