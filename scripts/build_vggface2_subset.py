#!/usr/bin/env python3
"""Build data/metadata/vggface2_subset.csv from extracted VGGFace2 test set.

Usage:
  python scripts/build_vggface2_subset.py --root data/vggface2/test
Expects: root/ contains identity folders (e.g. n000001, n000002) with images inside.
Output: data/metadata/vggface2_subset.csv with columns image_path, identity_id, split.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path


def build_vggface2_subset_csv(
    root: Path,
    out_path: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    rel_to: Path | None = None,
) -> list[dict]:
    """
    Scan root for identity subfolders, assign train/val by identity, write CSV.
    Returns list of row dicts (image_path, identity_id, split).
    rel_to: base path for relative image_path (default Path.cwd()).
    """
    root = Path(root)
    out_path = Path(out_path)
    rel_to = Path(rel_to) if rel_to is not None else Path.cwd()

    identity_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not identity_dirs:
        return []

    rng = random.Random(seed)
    rng.shuffle(identity_dirs)
    n_val = max(1, int(len(identity_dirs) * val_ratio))
    val_ids = {d.name for d in identity_dirs[:n_val]}

    rows = []
    for ident_dir in identity_dirs:
        identity_id = ident_dir.name
        split = "val" if identity_id in val_ids else "train"
        for f in ident_dir.iterdir():
            if f.is_file() and f.suffix.lower() in extensions:
                rel = os.path.relpath(f, rel_to)
                rows.append({"image_path": rel, "identity_id": identity_id, "split": split})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "identity_id", "split"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build VGGFace2 subset CSV from extracted test set.")
    parser.add_argument("--root", type=str, default="data/vggface2/test",
                        help="Root directory containing identity subfolders (e.g. data/vggface2/test)")
    parser.add_argument("--out", type=str, default="data/metadata/vggface2_subset.csv",
                        help="Output CSV path")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of identities used for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"],
                        help="Image file extensions to include")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    identity_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not identity_dirs:
        raise SystemExit(f"No identity subfolders found under {root}")

    ext_tuple = tuple(e if e.startswith(".") else f".{e}" for e in args.extensions)
    rows = build_vggface2_subset_csv(
        root,
        Path(args.out),
        val_ratio=args.val_ratio,
        seed=args.seed,
        extensions=ext_tuple,
    )
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val_imgs = sum(1 for r in rows if r["split"] == "val")
    val_ids = {r["identity_id"] for r in rows if r["split"] == "val"}
    train_ids = {r["identity_id"] for r in rows if r["split"] == "train"}
    print(f"Wrote {len(rows)} rows to {args.out}")
    print(f"  Identities: {len(identity_dirs)} (train: {len(train_ids)}, val: {len(val_ids)})")
    print(f"  Images: train={n_train}, val={n_val_imgs}")


if __name__ == "__main__":
    main()
