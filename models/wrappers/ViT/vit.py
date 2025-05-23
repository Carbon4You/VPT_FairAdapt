import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

__all__ = [
    "vit_base",
]

def vit_base(in_config, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model

