"""Unit tests for scripts/build_vggface2_subset.py (build_vggface2_subset_csv)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

# Import the testable function (run from repo root so scripts/ is on path, or use sys.path)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_vggface2_subset import build_vggface2_subset_csv


@pytest.fixture
def fake_vgg_root(tmp_path: Path) -> Path:
    """Create a minimal VGGFace2-like tree: root/id1/img1.jpg, id2/img2.png."""
    (tmp_path / "n000001").mkdir()
    (tmp_path / "n000001" / "img1.jpg").write_text("x")
    (tmp_path / "n000001" / "img2.jpeg").write_text("x")
    (tmp_path / "n000002").mkdir()
    (tmp_path / "n000002" / "photo.png").write_text("x")
    (tmp_path / "n000003").mkdir()
    (tmp_path / "n000003" / "a.jpg").write_text("x")
    return tmp_path


def test_build_vggface2_subset_csv_writes_csv_with_required_columns(
    fake_vgg_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.csv"
    rows = build_vggface2_subset_csv(fake_vgg_root, out, val_ratio=0.33, seed=42, rel_to=tmp_path)
    assert out.exists()
    assert len(rows) == 4  # 2 + 1 + 1 images
    with open(out, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        assert fieldnames == ["image_path", "identity_id", "split"]
        all_rows = list(r)
    assert len(all_rows) == 4
    for row in all_rows:
        assert "image_path" in row and "identity_id" in row and "split" in row
        assert row["split"] in ("train", "val")


def test_build_vggface2_subset_csv_train_val_split_by_identity(
    fake_vgg_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.csv"
    build_vggface2_subset_csv(fake_vgg_root, out, val_ratio=0.33, seed=42, rel_to=tmp_path)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    ids_by_split: dict[str, set[str]] = {"train": set(), "val": set()}
    for r in rows:
        ids_by_split[r["split"]].add(r["identity_id"])
    assert len(ids_by_split["val"]) >= 1
    assert ids_by_split["train"] | ids_by_split["val"] == {"n000001", "n000002", "n000003"}
    assert ids_by_split["train"] & ids_by_split["val"] == set()


def test_build_vggface2_subset_csv_respects_extensions(
    tmp_path: Path,
) -> None:
    (tmp_path / "id1").mkdir()
    (tmp_path / "id1" / "a.jpg").write_text("x")
    (tmp_path / "id1" / "b.txt").write_text("x")
    (tmp_path / "id1" / "c.png").write_text("x")
    out = tmp_path / "out.csv"
    rows = build_vggface2_subset_csv(
        tmp_path, out, extensions=(".jpg", ".png"), seed=42, rel_to=tmp_path
    )
    paths = [r["image_path"] for r in rows]
    assert any("a.jpg" in p for p in paths)
    assert any("c.png" in p for p in paths)
    assert not any("b.txt" in p for p in paths)


def test_build_vggface2_subset_csv_empty_root_returns_empty_list(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out.csv"
    rows = build_vggface2_subset_csv(tmp_path, out, rel_to=tmp_path)
    assert rows == []
    assert not out.exists()
