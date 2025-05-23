import torch
import timm
from functools import partial
import math
import torch
import torch.nn as nn
from torch.nn import Dropout
from functools import partial, reduce
from operator import mul

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

__all__ = [
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_huge",
    "prompted_vit_small",
    "prompted_vit_base",
    "prompted_vit_large",
    "prompted_vit_huge",
]


class VisionTransformerMAE(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, in_config=None, **kwargs):
        super(VisionTransformerMAE, self).__init__(**kwargs)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x


class PromptedVisionTransformerMAE(VisionTransformerMAE):
    def __init__(self, in_config, **kwargs):
        super().__init__(in_config, **kwargs)

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

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1)),
                x[:, 1:, :],
            ),
            dim=1,
        )
        return x

    def embeddings(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.incorporate_prompt(x)
        if self._m_config.MODEL.PROMPT.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)
            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
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
            # Not deep
            x = self.blocks(x)

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


class GatedVisionTransformer(VisionTransformerMAE):
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


class GatedPromptedVisionTransformerMAE(GatedVisionTransformer):
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
    model = GatedPromptedVisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
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
    return model

def vit_small(in_config, **kwargs):
    model = VisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans = in_config.DATASET.NUM_CHANNELS,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base(in_config, **kwargs):
    model = VisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans = in_config.DATASET.NUM_CHANNELS,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large(in_config, **kwargs):
    model = VisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans = in_config.DATASET.NUM_CHANNELS,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge(in_config, **kwargs):
    model = VisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        in_chans = in_config.DATASET.NUM_CHANNELS,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def prompted_vit_small(in_config, **kwargs):
    model = PromptedVisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def prompted_vit_base(in_config, **kwargs):
    model = PromptedVisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def prompted_vit_large(in_config, **kwargs):
    model = PromptedVisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def prompted_vit_huge(in_config, **kwargs):
    model = PromptedVisionTransformerMAE(
        in_config=in_config,
        global_pool="avg",
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
