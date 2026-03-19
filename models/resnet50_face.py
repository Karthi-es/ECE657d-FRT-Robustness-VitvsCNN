"""
ResNet-50 backbone + classification head for face recognition.
Uses timm (ImageNet pretrained); head is trained on VGGFace2 identities.
"""

import torch
import torch.nn as nn
import timm


def get_default_embed_dim():
    return 2048  # ResNet-50 penultimate dimension in timm


def create_resnet50_face(
    num_identities: int,
    pretrained: bool = True,
    embed_dim: int | None = None,
    drop_rate: float = 0.0,
) -> "ResNet50Face":
    """Create ResNet-50 face model with identity classification head."""
    return ResNet50Face(
        num_identities=num_identities,
        pretrained=pretrained,
        embed_dim=embed_dim or get_default_embed_dim(),
        drop_rate=drop_rate,
    )


class ResNet50Face(nn.Module):
    """ResNet-50 with optional embedding output and configurable identity head."""

    def __init__(
        self,
        num_identities: int,
        pretrained: bool = True,
        embed_dim: int = 2048,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_identities = num_identities
        self.embed_dim = embed_dim

        # Backbone: no classification head (num_classes=0) so forward returns features
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_rate=drop_rate,
        )
        assert getattr(self.backbone, "num_features", None) == embed_dim, (
            f"Expected backbone num_features {embed_dim}, got {getattr(self.backbone, 'num_features', None)}"
        )

        self.head = nn.Linear(embed_dim, num_identities)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding (for verification / cosine similarity)."""
        h = self.backbone(x)
        return nn.functional.normalize(h, p=2, dim=1)

    def classifier_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Logits from precomputed embedding."""
        return self.head(embedding)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        if return_embedding:
            return self.get_embedding(x)
        h = self.backbone(x)
        return self.head(h)
