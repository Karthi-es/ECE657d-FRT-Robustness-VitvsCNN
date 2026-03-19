# Evaluation Pipeline (Week 3)

Face verification evaluation and robustness analysis against makeup/tattoo perturbations.

## Quick Start

### Step 1: Evaluate Baseline (Clean Images)
```bash
# ResNet50 on clean test set
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/resnet50_best.pth \
    --model-type resnet50 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results

# ViT-B/16 on clean test set
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/vit_b16_best.pth \
    --model-type vit_b16 \
    --test-dir data/vggface2/test_split \
    --output-dir eval/results
```

### Step 2: Evaluate on Perturbed Images
```bash
# ResNet50 on makeup_heavy
python eval/evaluator.py \
    --model-checkpoint models/checkpoints/resnet50_best.pth \
    --model-type resnet50 \
    --test-dir data/vggface2_perturbed/makeup_heavy \
    --output-dir eval/results

# Repeat for all perturbations: makeup_light, makeup_heavy, tattoo, makeup_tattoo
```

### Step 3: Compare Robustness
```bash
python eval/robustness_comparison.py \
    --results-dir eval/results \
    --output-dir eval/analysis
```

## What Each Script Does

### `metrics.py` - Verification Metrics
Computes face verification metrics:
- **EER (Equal Error Rate)**: Where False Accept Rate = False Reject Rate
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **AUC**: Area Under ROC Curve
- **Accuracy**: Verification accuracy at various thresholds

Functions:
- `compute_eer()` - Calculate EER and threshold
- `compute_roc_curve()` - Generate ROC curve
- `plot_roc_curve()` - Visualize ROC
- `print_verification_results()` - Print comprehensive report

### `evaluator.py` - Face Verification Evaluator

**FaceEmbedder**:
- Loads trained identification model
- Extracts embedding vectors (512-dim for ResNet, 768-dim for ViT)
- L2-normalizes embeddings

**VerificationEvaluator**:
- Generates same/different person pairs from test set
- Computes pairwise cosine similarities
- Calculates EER and other metrics
- Saves results to JSON + ROC plots

### `robustness_comparison.py` - Robustness Analysis
- Loads all evaluation results
- Computes EER degradation from baseline
- Compares ViT vs CNN robustness
- Generates visualization + analysis report

## Output Structure

```
eval/
├── results/
│   ├── results_clean_resnet50.json      # Clean baseline (ResNet50)
│   ├── results_clean_vit_b16.json       # Clean baseline (ViT)
│   ├── results_makeup_heavy_resnet50.json
│   ├── results_makeup_heavy_vit_b16.json
│   ├── roc_resnet50.png
│   └── roc_vit_b16.png
└── analysis/
    ├── robustness_comparison.png         # Main comparison plot
    └── robustness_analysis.json          # Detailed analysis
```

## Interpretation

### Key Metrics

**EER (Equal Error Rate)**: 
- Lower is better
- Ideal: <5% for high-quality face recognition

**EER Degradation**:
- Measures robustness to perturbations
- Formula: `(perturbed_EER - clean_EER) / clean_EER × 100%`
- Lower degradation = more robust model

**Example**:
- ResNet50: Clean EER = 0.05, Makeup EER = 0.08 → Degradation = 60%
- ViT-B/16: Clean EER = 0.04, Makeup EER = 0.07 → Degradation = 75%
- ResNet50 is more robust to makeup in this example

## Success Criteria (from Proposal)

✓ **Week 3 Success**: 
- EER < 5% on clean VGGFace2 test pairs
- ViT EER degradation ≤ CNN degradation on ≥3/5 perturbations
- Both metrics achieved across all 5 perturbation types

## Example Workflow

```bash
# After Week 1: Train and save both models
python training/train_id.py --model resnet50 --epochs 20
python training/train_id.py --model vit_b16 --epochs 20

# After Week 2: Generate perturbations
python perturb/perturbation_generator.py \
    --source-dir data/vggface2/test_split \
    --output-dir data/vggface2_perturbed

# Week 3: Evaluate robustness
# Baseline
python eval/evaluator.py --model-checkpoint models/checkpoints/resnet50_best.pth \
    --model-type resnet50 --test-dir data/vggface2/test_split \
    --output-dir eval/results
python eval/evaluator.py --model-checkpoint models/checkpoints/vit_b16_best.pth \
    --model-type vit_b16 --test-dir data/vggface2/test_split \
    --output-dir eval/results

# Each perturbation
for PERTURB in makeup_light makeup_heavy tattoo makeup_tattoo; do
    python eval/evaluator.py --model-checkpoint models/checkpoints/resnet50_best.pth \
        --model-type resnet50 --test-dir data/vggface2_perturbed/$PERTURB \
        --output-dir eval/results
    
    python eval/evaluator.py --model-checkpoint models/checkpoints/vit_b16_best.pth \
        --model-type vit_b16 --test-dir data/vggface2_perturbed/$PERTURB \
        --output-dir eval/results
done

# Final comparison
python eval/robustness_comparison.py --results-dir eval/results --output-dir eval/analysis
```

## Dependencies

```
torch>=2.0
torchvision
PIL
numpy
scipy
matplotlib
pandas
```

All included in `requirements.txt`.

## Notes

- Evaluation works immediately after training (Week 1)
- Can test on clean test set first (baseline)
- Perturbations not required for initial evaluation
- Each evaluation saves ROC curve for visualization
- Results are JSON for easy parsing/comparison
