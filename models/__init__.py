"""Face recognition models: ResNet-50 and ViT-B/16 with identity heads."""

from .resnet50_face import ResNet50Face, create_resnet50_face, get_default_embed_dim as resnet50_embed_dim
from .vit_b16_face import ViT_B16_Face, create_vit_b16_face, get_default_embed_dim as vit_b16_embed_dim

__all__ = [
    "ResNet50Face",
    "create_resnet50_face",
    "resnet50_embed_dim",
    "ViT_B16_Face",
    "create_vit_b16_face",
    "vit_b16_embed_dim",
]
