# ECE657d-FRT-Robustness-ViTvsCNN
Comparing the robustness of Vision Transformers and CNNs for face recognition under heavy makeup and tattoo perturbations using public face datasets.

## Data (VGGFace2 test set only)

If you downloaded only the **VGGFace2 test set** (~2 GB, 500 identities) from [Academic Torrents](https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b):

1. Extract `vggface2_test.tar.gz` so that you have identity subfolders, e.g.:
   - `data/vggface2/test/n000001/`, `data/vggface2/test/n000002/`, …

2. From the repo root, build the subset CSV (splits 80% train / 20% val by identity):
   ```bash
   python scripts/build_vggface2_subset.py --root data/vggface2/test
   ```
   This writes `data/metadata/vggface2_subset.csv` (columns: `image_path`, `identity_id`, `split`). Use this CSV in your data loaders for training and validation.

## Testing

We add unit tests for every new feature. Run from repo root:

```bash
pytest tests/ -v
```

## Setup checklist and downloads

See **[PROJECT_SETUP.md](PROJECT_SETUP.md)** for the full list of downloadable files (VGGFace2, LFW, etc.), environment steps, and data layout.
