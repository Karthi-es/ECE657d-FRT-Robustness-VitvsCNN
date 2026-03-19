# Project Progress Checklist

Track completion of all project tasks and milestones.

## ✅ Infrastructure & Planning

- [x] Research proposal written
- [x] Success criteria defined
- [x] Project structure created
- [x] Models initialized (ResNet50, ViT-B/16)
- [x] Data loaders implemented
- [x] Training script created
- [x] Perturbation pipeline built
- [x] Evaluation pipeline implemented
- [x] Validation script created
- [x] Full documentation written

## 🔄 Data Management

- [ ] **Phase 2**: SCP transfer initiated (10% complete)
  - [ ] File transfer completes
  - [ ] Verify 500 identities transferred
  - [ ] Check disk space (~2GB used)
  
- [ ] **Phase 3**: Dataset splitting
  - [ ] Run `split_vggface2_dataset.py`
  - [ ] Verify: 350 train + 150 test folders
  - [ ] Generate split CSV
  - [ ] Commit CSV to GitHub

## 🏋️ Week 1: Baseline Training

- [ ] **RSNet50 Training**
  - [ ] Run: `python training/train_id.py --model resnet50`
  - [ ] Target: >95% validation accuracy
  - [ ] Save checkpoint: `resnet50_best.pth`
  - [ ] Training time: ~6-10 hours
  - [ ] Achieved accuracy: ____%
  
- [ ] **ViT-B/16 Training**
  - [ ] Run: `python training/train_id.py --model vit_b16`
  - [ ] Target: >95% validation accuracy
  - [ ] Save checkpoint: `vit_b16_best.pth`
  - [ ] Training time: ~6-10 hours
  - [ ] Achieved accuracy: ____%

- [ ] **Baseline Evaluation**
  - [ ] Evaluate ResNet50 on clean test set
  - [ ] Evaluate ViT-B/16 on clean test set
  - [ ] Record EER values:
    - ResNet50 clean EER: ____%
    - ViT-B/16 clean EER: ____%

## 🎨 Week 2: Perturbations

- [ ] **StarGANv2 Setup**
  - [ ] Verify installation: `python perturb/starganv2_setup.py --verify`
  - [ ] Download weights (optional): `--download`
  - [ ] Backup weights location: `models/starganv2/celeba_hq.pt`

- [ ] **Generate Perturbations**
  - [ ] Run: `python perturb/perturbation_generator.py`
  - [ ] Perturbation types: makeup_light, makeup_heavy, tattoo, makeup_tattoo
  - [ ] Test set: 150 identities
  - [ ] Total generated images: ~_____
  - [ ] Disk space used: ~_____  GB

- [ ] **Perturbation Verification**
  - [ ] Check folder structure: `data/vggface2_perturbed/`
  - [ ] Spot-check images (visual quality)
  - [ ] All 5 perturbation types present

## 📊 Week 3: Robustness Evaluation

- [ ] **Evaluate on Perturbed Images**

| Perturbation | ResNet50 | ViT-B/16 | Degradation |
|--------------|----------|----------|-------------|
| makeup_light | ___% | ___% | ResNet: __% / ViT: __% |
| makeup_heavy | ___% | ___% | ResNet: __% / ViT: __% |
| tattoo | ___% | ___% | ResNet: __% / ViT: __% |
| makeup_tattoo | ___% | ___% | ResNet: __% / ViT: __% |

- [ ] **Generate Comparison**
  - [ ] Run: `python eval/robustness_comparison.py`
  - [ ] Output: `eval/analysis/robustness_comparison.png`
  - [ ] Output: `eval/analysis/robustness_analysis.json`

- [ ] **Analyze Results**
  - [ ] Review robustness comparison plot
  - [ ] Count perturbations where ViT ≤ CNN degradation
  - [ ] Number of ≥3/5 criterion met: ___/5
  - [ ] Success criterion achieved: [ ] Yes [ ] No

## 📝 Documentation & Submission

- [ ] **Results Summary**
  - [ ] Copy all results to results_summary/
  - [ ] Generate final report
  - [ ] Document findings & observations

- [ ] **GitHub Commit**
  - [ ] Review .gitignore (data not committed)
  - [ ] Add: `data/metadata/vggface2_split.csv`
  - [ ] Add: `eval/results/` (JSON files)
  - [ ] Add: `eval/analysis/` (PNG + JSON)
  - [ ] Commit with message: "Final robustness evaluation"
  - [ ] Push to remote

- [ ] **Final Review**
  - [ ] All success criteria documented
  - [ ] Metrics tables filled in
  - [ ] Visualizations saved
  - [ ] Code repository clean
  - [ ] README updated with actual results

## 🎯 Success Criteria Status

- [ ] **Criterion 1**: Clean accuracy >95% (both models)
  - ResNet50: [ ] Met
  - ViT-B/16: [ ] Met

- [ ] **Criterion 2**: EER <5% on clean VGGFace2
  - ResNet50: [ ] Met (actual: ___%)
  - ViT-B/16: [ ] Met (actual: ___%)

- [ ] **Criterion 3**: ViT degradation ≤ CNN on ≥3/5 perturbations
  - Perturbations where ViT ≤ CNN: ___/5
  - Status: [ ] Met

## 📈 Key Metrics Log

### Training Metrics
- ResNet50 best validation accuracy: ____%
- ViT-B/16 best validation accuracy: ____%
- Training time ResNet50: ___ hours
- Training time ViT-B/16: ___ hours

### Verification Metrics (Clean Baseline)
- ResNet50 clean EER: ____%
- ViT-B/16 clean EER: ____%

### Robustness Degradation
- ResNet50 max degradation: ___% (on perturbation: _______)
- ViT-B/16 max degradation: ___% (on perturbation: _______)

### Comparison Results
- Perturbations where ViT more robust: ___/5
- Perturbations where CNN more robust: ___/5
- Perturbations tied: ___/5

## 📋 File Generated Checklist

### Code Files
- [x] `training/train_id.py` - Training orchestration
- [x] `training/data_loaders.py` - Data loading
- [x] `models/resnet50_face.py` - ResNet50 model
- [x] `models/vit_b16_face.py` - ViT-B/16 model
- [x] `perturb/perturbation_generator.py` - Perturbation pipeline
- [x] `perturb/starganv2_setup.py` - StarGANv2 setup
- [x] `eval/evaluator.py` - Face verification
- [x] `eval/metrics.py` - Verification metrics
- [x] `eval/robustness_comparison.py` - Robustness analysis
- [x] `scripts/split_vggface2_dataset.py` - Dataset splitting
- [x] `scripts/generate_split_csv.py` - CSV generation
- [x] `scripts/validate_setup.py` - Setup validation

### Documentation Files
- [x] `FULL_README.md` - Comprehensive guide
- [x] `WORKFLOW.md` - Step-by-step workflow
- [x] `CHECKLIST.md` - This file
- [x] `perturb/README.md` - Perturbation guide
- [x] `eval/README.md` - Evaluation guide
- [ ] `results_summary/` - Final results (pending)

### Configuration Files
- [x] `.gitignore` - Updated with data/models exclusions
- [x] `requirements.txt` - All dependencies

## 🚀 Getting Started

1. **NOW**: Read this checklist first ✓
2. **TODAY**: Wait for SCP transfer (currently 10% - ~2-4 hours)
3. **TOMORROW**: 
   - [ ] Run `python scripts/validate_setup.py`
   - [ ] Split dataset: `python scripts/split_vggface2_dataset.py`
   - [ ] Generate CSV: `python scripts/generate_split_csv.py`
4. **NEXT 2 DAYS**: Train models (ResNet50 then ViT-B/16)
5. **DAY 5**: Generate perturbations
6. **DAY 6-7**: Evaluate robustness
7. **DAY 8**: Commit & submit

## Notes & Observations

```
Add your notes here as you progress:

- [Day X]: 
- [Day X]:
```

---

**Last Updated**: Today (March 18, 2026)
**Transfer Status**: 10% (2-4 hours remaining)
