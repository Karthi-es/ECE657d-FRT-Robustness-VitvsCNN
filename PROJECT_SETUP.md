# Project setup: downloads and steps

Use this as the single checklist for what to download and in what order.

---

## 1. Main downloadable files

| What | Where | Size (approx) | Purpose |
|------|--------|----------------|---------|
| **VGGFace2 test set** | [Academic Torrents – VGGFace2](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b) — download only `data/vggface2_test.tar.gz` | ~2 GB | Train/val IDs (500 identities). Extract so you have identity subfolders (e.g. `data/vggface2/test/n000001/`, …). |
| **LFW (aligned)** | [LFW – UMass](http://vis-www.cs.umass.edu/lfw/) — **lfw-deepfunneled.tgz** (or lfw funneled) | ~100–200 MB | Verification pairs for EER/ROC. |
| **LFW pairs list** | Same LFW page — **pairs.txt** or “View 2” evaluation pairs | Tiny | Same/different pair indices for verification protocol. |
| **ResNet-50 / ViT-B/16 weights** | **No manual download.** Fetched by `timm` on first run (ImageNet pretrained). | ~100 MB + ~350 MB | Backbone weights; cached under `~/.cache/torch/hub/` (or similar). |
| **StarGANv2 (later)** | Optional; for makeup/tattoo perturbations. Repo + pretrained weights. | ~1–2 GB | Phase 3 perturbation pipeline. |

---

## 2. One-time environment setup

1. **Python env** (from repo root):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # On ecetesla (tcsh): source .venv/bin/activate.csh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Run tests** (after adding features, run often):
   ```bash
   pytest tests/ -v
   ```

3. **GPU** (optional): confirm PyTorch sees CUDA:
   ```python
   import torch; print(torch.cuda.is_available())
   ```

---

## 3. Data steps (order)

1. **VGGFace2**
   - Download `vggface2_test.tar.gz` (e.g. from Academic Torrents).
   - Extract so identity folders live under a single root (e.g. `data/vggface2/test/`).
   - From repo root:
     ```bash
     python scripts/build_vggface2_subset.py --root data/vggface2/test
     ```
   - This creates `data/metadata/vggface2_subset.csv` (train/val by identity).

2. **LFW**
   - Download and extract **lfw-deepfunneled.tgz** into e.g. `data/lfw/lfw-deepfunneled/`.
   - Download the **pairs** file (e.g. `pairs.txt` or View 2) into `data/lfw/` (or `data/metadata/`).
   - Later we’ll add a script to build `data/metadata/lfw_pairs.csv` from that.

---

## 4. What you don’t download by hand

- **ResNet-50 / ViT-B/16**: `timm` downloads weights on first `create_model(..., pretrained=True)`.
- **VGGFace2 train set** (optional): only if you later want the full 38 GB train split; not required for the 500-identity test-subset pipeline.

---

## 5. Quick reference: repo layout after setup

```
data/
  vggface2/test/          # extracted VGGFace2 test identities
  metadata/
    vggface2_subset.csv   # from build_vggface2_subset.py
    lfw_pairs.csv         # (to be added) from LFW pairs
  lfw/
    lfw-deepfunneled/     # LFW aligned images
models/                   # ResNet50 + ViT-B/16 face wrappers (done)
training/                 # train_id.py (to add)
eval/                     # extract_embeddings, verify_pairs, robustness_eval (to add)
perturb/                  # StarGANv2 + overlays (Phase 3)
configs/
scripts/
tests/                    # unit tests for each feature
```

---

## 6. Going forward: tests for every feature

- Every new script or module (data loaders, training, eval, perturb) should have a corresponding `tests/test_<module_or_script>.py` with at least one smoke test or unit test.
- Run `pytest tests/ -v` before committing.
