"""Unit tests for models/resnet50_face.py."""
from __future__ import annotations

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.resnet50_face import (
    ResNet50Face,
    create_resnet50_face,
    get_default_embed_dim,
)


def test_get_default_embed_dim() -> None:
    assert get_default_embed_dim() == 2048


def test_create_resnet50_face_shape_no_pretrained() -> None:
    model = create_resnet50_face(num_identities=100, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 100)
    emb = model.get_embedding(x)
    assert emb.shape == (2, 2048)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2))


def test_forward_return_embedding() -> None:
    model = create_resnet50_face(num_identities=10, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out_emb = model(x, return_embedding=True)
    out_head = model(x, return_embedding=False)
    assert out_emb.shape == (1, 2048)
    assert out_head.shape == (1, 10)


def test_classifier_forward() -> None:
    model = create_resnet50_face(num_identities=5, pretrained=False)
    emb = torch.randn(3, 2048)
    logits = model.classifier_forward(emb)
    assert logits.shape == (3, 5)


def test_resnet50_face_embed_dim_matches_backbone() -> None:
    model = create_resnet50_face(num_identities=1, pretrained=False)
    assert model.embed_dim == 2048
    assert model.backbone.num_features == 2048
