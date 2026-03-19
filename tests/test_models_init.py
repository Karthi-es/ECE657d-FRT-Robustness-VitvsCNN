"""Unit tests for models/__init__.py exports and factory functions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ResNet50Face,
    create_resnet50_face,
    resnet50_embed_dim,
    ViT_B16_Face,
    create_vit_b16_face,
    vit_b16_embed_dim,
)


def test_resnet50_export() -> None:
    assert resnet50_embed_dim() == 2048
    m = create_resnet50_face(num_identities=10, pretrained=False)
    assert isinstance(m, ResNet50Face)


def test_vit_b16_export() -> None:
    assert vit_b16_embed_dim() == 768
    m = create_vit_b16_face(num_identities=10, pretrained=False)
    assert isinstance(m, ViT_B16_Face)
