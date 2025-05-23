import torch
import timm
from functools import partial
import math
import torch
import torch.nn as nn
import torchvision as tv
from torch.nn import Dropout
from timm.models.vision_transformer import VisionTransformer, _cfg
from torchvision import transforms

from functools import partial, reduce
from operator import mul
from PIL import Image, ImageFile
import PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pathlib
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from typing import List, Type

from models.wrappers.core_model import CoreModel
from dataset.dataset_interface import EnumDatasetTaskType

import utility.logger as logger

logger_handle = logger.get_logger("vpt_demographic_adaptation")

__all__ = [
    "vpt_vit_small",
    "vpt_vit_base",
    "vpt_vit_large",
    "vpt_vit_huge",
]


class PromptedVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, in_config, **kwargs):
        super().__init__(**kwargs)

        # self._m_is_global_pool = True if "global_pool" in kwargs else False
        self._m_config = in_config
        self._m_num_tokens = self._m_config.MODEL.PROMPT.DEEP
        self.prompt_dropout = Dropout(self._m_config.MODEL.PROMPT.DROPOUT)

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self._m_num_tokens, self.embed_dim)
        )

        if self._m_config.MODEL.PROMPT.DEEP:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(len(self.blocks) - 1, self._m_num_tokens, self.embed_dim)
            )

        val = math.sqrt(
            6.0
            / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim)
        )
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self._m_config.MODEL.PROMPT.DEEP:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(len(self.blocks) - 1, self._m_num_tokens, self.embed_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    # def load_pretrained(self, checkpoint_path, prefix=""):
    #     if self._m_is_global_pool:
    #         self.norm, self.fc_norm = self.fc_norm, self.norm
    #     super().load_pretrained(checkpoint_path, prefix)
    #     if self._m_is_global_pool:
    #         self.norm, self.fc_norm = self.fc_norm, self.norm
    #         del self.norm  

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)),
                x[:, 1:, :],
            ),
            dim=1,
        )
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.incorporate_prompt(x)

        B = x.shape[0]
        num_layers = len(self.blocks)

        if self._m_config.MODEL.PROMPT.DEEP:
            x = self.blocks[0](x)
            for i in range(1, num_layers):
                x = torch.cat(
                    (
                        x[:, :1, :],
                        self.prompt_dropout(
                            self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                        ),
                        x[:, (1 + self._m_num_tokens) :, :],
                    ),
                    dim=1,
                )
                x = self.blocks[i](x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x



from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath


# based on timm Attention implementation
class GatedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temp=1.0):
        """ 
        temp = 1.0 by default or learnable scalar
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        attn = (attn / temp).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
            
        return x


class GatedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = GatedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, temp=1.0):
        """ 
        temp = 1.0 by default or learnable scalar
        """
        x = x + self.drop_path(self.attn(self.norm1(x), temp=temp))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GatedVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(GatedVisionTransformer, self).__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            GatedBlock(
                dim=embed_dim, num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'], qkv_bias=kwargs['qkv_bias'],
                drop_path=dpr[i], norm_layer=kwargs['norm_layer'])
            for i in range(kwargs['depth'])])

class GatedPromptedVisionTransformer(GatedVisionTransformer):
    def __init__(self, in_config, **kwargs):
        super().__init__(**kwargs)
        self._m_config = in_config

        num_tokens = self._m_config.MODEL.GATED.PROMPT.NUM_TOKENS

        self.num_tokens = num_tokens
        self.prompt_dropout = Dropout(self._m_config.MODEL.GATED.PROMPT.DROPOUT)
        
        # define temperature for attention shaping
        self.prompt_temp = self._m_config.MODEL.GATED.PROMPT.TEMP
        self.temp_learn = self._m_config.MODEL.GATED.PROMPT.TEMP_LEARN
        if self.temp_learn:
            self.prompt_temp = nn.Parameter(torch.ones(self._m_config.MODEL.GATED.PROMPT.TEMP_NUM))

        # initiate prompt:
        if self._m_config.MODEL.GATED.PROMPT.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            raise ValueError("Other initiation scheme is not supported")
        
        # define block-wise learnable gate scalar
        if self._m_config.MODEL.GATED.PROMPT.GATE_PRIOR:       
            gate_logit = (-torch.ones(self._m_config.MODEL.GATED.PROMPT.GATE_NUM) * self._m_config.MODEL.GATED.PROMPT.GATE_INIT)        
            self.prompt_gate_logit = nn.Parameter(gate_logit)
            print(self.prompt_gate_logit)
       
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(
                    self.prompt_embeddings.expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)
                
    def reinit_temp(self):
        assert self.temp_learn, "reinit_temp() could be run only when config.TEMP_LEARN == True"
        self.prompt_temp.data.copy_(self.prompt_temp.data.clamp(min=self._m_config.MODEL.GATED.PROMPT.TEMP_MIN, max=self._m_config.MODEL.GATED.PROMPT.TEMP_MAX))

    def forward_features(self, x):
        x = self.incorporate_prompt(x)
            
        # clamp temperatures not to be too small or too large
        if self.temp_learn:
            self.reinit_temp()

        for i, blk in enumerate(self.blocks):
            # current block's input prompt representation
            if self._m_config.MODEL.GATED.PROMPT.GATE_PRIOR and i < self.prompt_gate_logit.shape[0]:
                gate = self.prompt_gate_logit[i].sigmoid()
                prompt_in = x[:, 1: 1+self._m_config.MODEL.GATED.PROMPT.NUM_TOKENS, :]

            # block-wise learnable temperature
            prompt_temp = self.prompt_temp if not isinstance(self.prompt_temp, nn.Parameter) else self.prompt_temp[i]
            
            x = blk(x, temp=prompt_temp)
            if self._m_config.MODEL.GATED.PROMPT.GATE_PRIOR and i < self.prompt_gate_logit.shape[0]:
                # current block's output prompt representation
                prompt_out = x[:, 1: 1+self._m_config.MODEL.GATED.PROMPT.NUM_TOKENS, :]
                # convex combinate input and output prompt representations of current block via learnalbe gate
                x = torch.cat([
                    x[:, 0:1, :], 
                    gate * prompt_out + (1 - gate) * prompt_in, 
                    x[:, 1+self._m_config.MODEL.GATED.PROMPT.NUM_TOKENS:, :]
                ], dim=1)
        x = self.norm(x)
        return x


def gated_vpt_vit_base(in_config, **kwargs):
    model = GatedPromptedVisionTransformer(
        in_config=in_config,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # block_fn=GatedBlock,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vpt_vit_small(in_config, **kwargs):
    model = PromptedVisionTransformer(
        in_config=in_config,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        # global_pool="avg",
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vpt_vit_base(in_config, **kwargs):
    model = PromptedVisionTransformer(
        in_config=in_config,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        # global_pool="avg",
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vpt_vit_large(in_config, **kwargs):
    model = PromptedVisionTransformer(
        in_config=in_config,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        # global_pool="avg",
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vpt_vit_huge(in_config, **kwargs):
    model = PromptedVisionTransformer(
        in_config=in_config,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        # global_pool="avg",
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


__archs__ = {
    "gated_vpt_vit_base": gated_vpt_vit_base,
    "vpt_vit_small": vpt_vit_small,
    "vpt_vit_base": vpt_vit_base,
    "vpt_vit_large": vpt_vit_large,
    "vpt_vit_huge": vpt_vit_huge,
}


class VPT_ViT(CoreModel):

    def __init__(
        self,
        in_config,
        in_distribution=None,
        in_device=None,
        in_gpu=None,
        in_global_rank=0,
        in_global_world_size=1,
    ) -> None:
        super().__init__(
            in_config,
            in_distribution,
            in_device,
            in_gpu,
            in_global_rank,
            in_global_world_size,
        )

    def initialize(
        self, in_num_of_classes: int, in_dataset_task_type: EnumDatasetTaskType, **kwargs
    ):
        super().initialize(in_num_of_classes, in_dataset_task_type, **kwargs)
        self._m_model = self.setup_model(in_num_of_classes)

        if self._m_config.SOLVER.FIRST_EPOCH == 0:
            # Backbone
            self.load_backbone(self._m_model)

        # Freeze all but the head
        for _, p in self._m_model.named_parameters():
            p.requires_grad = False
        for _, p in self._m_model.head.named_parameters():
            p.requires_grad = True
        for k, p in self._m_model.named_parameters():
            if "prompt" in k:
                p.requires_grad = True
        self._m_optimizer = self.setup_optimizer()
        self._m_scheduler = self.setup_scheduler()
        self.distribute()
        self.setup_criterion(in_dataset_task_type)
        self.load_checkpoint()

    def setup_model(self, in_num_of_classes):
        model = __archs__[self._m_config.MODEL.ARCH](
            in_config=self._m_config,
            img_size=self._m_config.DATASET.IMAGE_SIZE,
            num_classes=in_num_of_classes,
            drop_rate=self._m_config.MODEL.DROP_RATE,
            drop_path_rate=self._m_config.MODEL.DROP_RATE_PATH,
        )
        return model

    def setup_optimizer(self, in_parameters=None):
        params = []
        for _, value in self._m_model.named_parameters():
            if value.requires_grad:
                params.append({"params": value})
        return super().setup_optimizer(params)

    def setup_scheduler(self):
        scheduler = CosineLRScheduler(
            self._m_optimizer,
            self._m_config.SOLVER.LAST_EPOCH,
            warmup_t=self._m_config.SOLVER.WARMUP_EPOCHS,
        )
        return scheduler
