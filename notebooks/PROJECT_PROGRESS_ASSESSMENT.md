# ECE657d Project Progress Assessment
## ViT vs CNN Face Recognition Robustness under Makeup/Tattoos

**Student**: Karthi Elayaperumal Sampath (21182921)  
**Date**: March 17, 2026  
**Current Week**: 3 (of 5) — **Status: BEHIND SCHEDULE**

---

## 📊 Executive Summary

| Metric | Status | Score |
|--------|--------|-------|
| **Project Foundation** | ✅ Correct | 95% |
| **Timeline Adherence** | ⚠️ Behind | ~30% |
| **Overall Completion** | ⚠️ Early Stage | ~20-25% |
| **Path to Success** | ⚠️ Feasible but Urgent | Needs acceleration |

---

## ✅ What's Been Done Correctly

### 1. **Model Architecture** ✅ CORRECT
- [x] ResNet-50 and ViT-B/16 properly implemented using `timm`
- [x] ImageNet pretraining correctly configured
- [x] Two forward modes: embedding extraction + classification head
- [x] L2 normalization of embeddings for face verification
- [x] Proper separation: backbone (feature extraction) + head (identity classification)
- [x] Both models support `return_embedding=True` for verification pipeline

**Alignment with Proposal**: ✅ Matches Q6 resource specifications exactly
- Uses `timm` library: ✓
- ViT-B/16 and ResNet50: ✓
- ImageNet-pretrained: ✓

### 2. **Data Infrastructure** ✅ CORRECT
- [x] VGGFace2 subset CSV builder implemented
- [x] Correct train/val split (80/20 by identity)
- [x] Metadata CSV with columns: `image_path`, `identity_id`, `split`
- [x] Handles multiple image extensions (.jpg, .jpeg, .png)
- [x] Proper path handling and relative paths

**Alignment with Proposal**: ✅ Matches Q6 resources
- VGGFace2 dataset setup: ✓
- Metadata organization: ✓

### 3. **Testing Infrastructure** ✅ CORRECT
- [x] Unit tests for ResNet-50 model
- [x] Unit tests for ViT-B/16 model
- [x] Shape verification (batch × embedding_dim)
- [x] Embedding normalization verification
- [x] Multiple forward pass configurations tested
- [x] Tests use `pytest` framework

**Alignment with Proposal**: ✅ Good practice (PROJECT_SETUP.md requirement)

### 4. **Project Setup & Documentation** ✅ CORRECT
- [x] Clean directory structure
- [x] requirements.txt with all dependencies (torch, torchvision, timm, wandb, etc.)
- [x] Comprehensive README.md
- [x] PROJECT_SETUP.md with clear steps
- [x] Git repository initialized properly

**Alignment with Proposal**: ✅ Well-organized

---

## ❌ What's Missing / Behind Schedule

### 2. **WEEK 1 (Mar 2-7): OVERDUE - Not Started**

**Deliverable**: Baseline training on clean VGGFace2
- Identity classification accuracy > 95%
- Both ViT and ResNet trained
- Head-only fine-tuning

**Status**: ❌ **NOT STARTED**

**Missing Components**:
```
training/
├── data_loaders.py          ❌ MISSING
├── train_id.py              ❌ MISSING
├── train_utils.py           ❌ MISSING
configs/
├── baseline_config.yaml     ❌ MISSING
├── training_params.json     ❌ MISSING
```

**Critical Code Needed**:
1. **PyTorch DataLoader setup**
   - Load images from CSV
   - Normalize (ImageNet stats)
   - Resize to 224×224
   - Batch size 64
   - Augmentation pipeline

2. **Training Loop**
   - Cross-entropy loss for identity classification
   - SGD or Adam optimizer
   - Learning rate scheduling
   - Validation after each epoch
   - Checkpoint saving
   - wandb logging

3. **Expected Output**:
   - Trained ResNet50 weights: `models/checkpoints/resnet50_clean_epoch20.pth`
   - Trained ViT weights: `models/checkpoints/vit_b16_clean_epoch20.pth`
   - Training curves plot
   - Final accuracy metrics > 95%

---

### 3. **WEEK 2 (Mar 9-14): IN PROGRESS - Not Started**

**Deliverable**: Perturbation pipeline + partial fine-tuning
- Generate 5+ perturbation types (makeup/tattoos)
- Train models on perturbed data
- Partial fine-tuning strategy

**Status**: ❌ **NOT STARTED** (Should be 50% done now)

**Missing Components**:
```
perturb/
├── starganv2_setup.py       ❌ MISSING
├── makeup_generator.py      ❌ MISSING
├── tattoo_generator.py      ❌ MISSING
├── perturbation_pipeline.py ❌ MISSING

training/
├── train_id_partial_ft.py   ❌ MISSING (partial fine-tuning)

data/
├── perturbed/               ❌ MISSING
│   ├── makeup_light/
│   ├── makeup_heavy/
│   ├── tattoo_face/
│   ├── tattoo_arms/
│   └── tattoo_neck/
```

**Critical Code Needed**:
1. **StarGANv2 Integration**
   - Download/setup StarGANv2 weights
   - Makeup generation (domain translation)
   
2. **Tattoo Generation**
   - OpenCV-based overlay creation
   - Natural contour following
   - 5+ types as per proposal

3. **Partial Fine-tuning Training**
   - Freeze backbone, train head only (done in Week 1)
   - Unfreeze last N layers for partial fine-tuning
   - Compare results
   - Track EER metrics

---

### 4. **WEEK 3 (Mar 16-21): CURRENT WEEK - Not Started**

**Deliverable**: Evaluation on perturbed pairs
- EER < 15% on clean LFW pairs (baseline)
- EER measured across all perturbations
- Robustness comparison begins

**Status**: ❌ **NOT STARTED** (Should be 25% done now)

**Missing Components**:
```
eval/
├── lfw_loader.py            ❌ MISSING
├── extract_embeddings.py    ❌ MISSING
├── verify_pairs.py          ❌ MISSING
├── compute_metrics.py       ❌ MISSING
├── robustness_eval.py       ❌ MISSING

data/metadata/
├── lfw_pairs.csv            ❌ MISSING
```

**Critical Code Needed**:
1. **LFW Data Loading**
   - Parse pairs.txt file
   - Load same/different pair images
   - Create evaluation set (1k pairs per proposal)

2. **Embedding Extraction**
   - Batch forward pass
   - Extract embeddings from trained models
   - Save to disk for analysis

3. **Verification Metrics**
   - Cosine similarity scores
   - EER computation (Equal Error Rate)
   - ROC curves
   - Threshold tuning

4. **Robustness Evaluation**
   - Compare clean vs perturbed EER
   - Calculate EER drop (%)
   - Rank perturbations by impact

---

### 5. **WEEK 4 (Mar 23-28): NOT YET - Not Started**

**Deliverable**: Ablation studies & analysis
- Fine-tuning depth ablation
- Complete all perturbed evaluations
- Results analysis + plots

**Status**: ❌ **NOT STARTED** (Blocked by weeks 1-3)

---

## 📋 Mapping: Proposal vs Implementation

| Proposal Section | Expected | Status |
|------------------|----------|--------|
| Q1: Research Question | Testing ViT vs CNN robustness | ✅ Clear |
| Q2-4: Motivation | Understand which architecture is better | ✅ Clear |
| Q5: Assumptions | Pre-training sufficient, consistency | ✅ Understood |
| Q6: Resources | VGGFace2, LFW, ViT-B/16, ResNet50, timm | ✅ 70% Ready |
| Q7: Weekly Plan | 5-week timeline | ❌ BEHIND |
| Q8: Success Criteria | Primary: ViT EER drop ≤ CNN drop (≥3/5) | ⚠️ Not measurable yet |
| Technical Details | 224×224, BS64, 20 epochs, StarGANv2, wandb | ⚠️ Partial |

---

## 🚨 Critical Path to Success

### **IMMEDIATE ACTIONS (This Week - Mar 17-21)**

**Priority 1 - URGENT** (Must complete by end of week):
1. [ ] Implement `training/data_loaders.py`
   - VGGFace2 DataLoader
   - LFW DataLoader (verification)
   - Image normalization + augmentation
   
2. [ ] Implement `training/train_id.py`
   - Basic head-only fine-tuning training loop
   - wandb integration
   - Model checkpointing

3. [ ] Create baseline config in `configs/`
   - Learning rate, batch size, epochs, etc.

**Goal**: Get **Week 1 baseline training working** (Current day is March 17, Week 1 ended March 7!)

### **Next Week Priority (Mar 22-28)**

**Priority 2** (Catch up on Week 2-3):
1. [ ] Perturbation pipeline (makeup/tattoos)
2. [ ] Evaluation modules (EER, verification)
3. [ ] Run complete evaluation

---

## ✅ Success Criteria Status

| Criteria | Target | Current | Status |
|----------|--------|---------|--------|
| Week 1 milestone | Clean ID acc >95% | Not measured | ❌ |
| Week 1 milestone | EER <15% on LFW | Not measured | ❌ |
| Week 2 milestone | 5+ perturbations generated | 0 generated | ❌ |
| Week 3 milestone | Perturbed evaluation pipeline | Not implemented | ❌ |
| **Primary** | ViT EER drop ≤ CNN drop (≥3/5 perturbations) | Not measurable | ❌ |
| **Secondary** | ViT partial FT ≤5% relative EER drop | Not measurable | ❌ |

---

## 📈 What's Been Done Right

Your foundation is **solid and correct**:
- ✅ Models are architecturally sound
- ✅ Data pipeline logic is correct
- ✅ Testing practices are good
- ✅ Project structure is clean

**This is 15-20% of the project**, but it's the RIGHT 15-20%.

---

## ⚠️ What Needs Urgent Attention

You are **1.5 weeks behind schedule**. To recover:

1. **THIS WEEK**: Complete Week 1 training (should've been done Mar 7)
2. **NEXT WEEK**: Complete Week 2 perturbations + Week 3 evaluation (overlap both)
3. **THEN**: Week 4 analysis (ablation studies)
4. **FINAL**: Week 5 write-up

---

## 🎯 Recommended Next Steps

### **Immediate (Next 3 days)**
1. Create `training/data_loaders.py` with VGGFace2 loader
2. Create `training/train_id.py` with basic training loop
3. Run training on clean data for one model

### **This Week (Days 4-7)**
1. Complete training both models (ResNet50 + ViT-B/16)
2. Verify > 95% accuracy on clean data
3. Test basic LFW evaluation (compute EER)

### **Next Week (Days 8-14)**
1. Implement perturbation pipeline (StarGANv2 + tattoos)
2. Implement full evaluation pipeline
3. Run complete evaluation on all perturbations

---

## 📝 Files to Create (Priority Order)

1. `training/data_loaders.py` - DataLoader classes
2. `training/train_id.py` - Main training script
3. `configs/training_config.yaml` - Configuration
4. `eval/extract_embeddings.py` - Embedding extraction
5. `eval/verify_pairs.py` - Verification metrics
6. `perturb/makeup_generator.py` - Makeup synthesis
7. `perturb/tattoo_generator.py` - Tattoo synthesis
8. `eval/robustness_eval.py` - Robustness metrics

---

## ✅ VERDICT: Are Your Steps Correct?

**YES, but INCOMPLETE and BEHIND SCHEDULE**

| Aspect | Verdict |
|--------|---------|
| **Approach** | ✅ Correct |
| **Architecture** | ✅ Correct |
| **Data setup** | ✅ Correct |
| **Testing** | ✅ Good practices |
| **Timeline** | ❌ **1.5 weeks behind** |
| **Completion** | ❌ ~20% done (should be ~60%) |
| **Feasibility** | ⚠️ Tight but doable if you accelerate |

**To be back on track**: Need to complete training, perturbations, AND evaluation by end of next week (Mar 28).

Week 1 (Mar 2-7) - IDENTITY CLASSIFICATION:
1. Load VGGFace2 images into DataLoader
2. Train ResNet-50 head-only (→ checkpoint: resnet_clean.pth)
3. Train ViT-B/16 head-only (→ checkpoint: vit_clean.pth)
4. Achieve >95% accuracy on val set ✓

Week 3 (Mar 16-21) - FACE VERIFICATION:
5. Load LFW test pairs
6. Extract embeddings from trained models
7. Compute similarity scores for all pairs
8. Calculate EER metric
9. Achieve <15% EER ✓

Week 2 (Mar 9-14) - PERTURBATIONS (between Week 1 & 3):
10. Generate makeup/tattoo perturbations
11. Create perturbed training data
12. Train models on perturbed data
13. Compare: Clean EER vs Perturbed EER


VGGFace2 CSV (~8,181 images)
         ↓
   Split by identity (80/20)
         ↓
   Train set (6,532 images, 400 identities)
   Val set   (1,648 images, 100 identities)
         ↓
   DataLoader (batch_size=64)
         ↓
   Image preprocessing:
   - Resize to 224×224
   - Normalize with ImageNet stats
   - Light augmentation (train only)
         ↓
   Model forward pass:
   - ResNet50/ViT backbone (frozen)
   - Classification head (trainable)
   ↓
   CrossEntropy loss
   ↓
   Update head weights only