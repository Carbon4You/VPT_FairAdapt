import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
from torch.nn import Dropout
import math
import torch
import torch.nn as nn

from functools import partial, reduce
from operator import mul
from timm.models.vision_transformer import VisionTransformer, _cfg

from timm.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

__all__ = [
    "vit_small",
    "vit_base",
    "prompted_vit_small",
    "prompted_vit_base",
    "gated_vpt_vit_base",
    "vit_conv_small",
    "vit_conv_base",
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        # assert self.num_tokens == 1, "Assuming one and only one token, [cls]"
        assert self.num_prefix_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class PromptedVisionTransformerMoCo(VisionTransformerMoCo):

    def __init__(self, in_config, **kwargs):
        super().__init__(**kwargs)

        self._m_config = in_config
        self._m_num_tokens = self._m_config.MODEL.PROMPT.NUM_TOKENS
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.incorporate_prompt(x)
        # deep
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

        x = self.norm(x)
        return x

        # x = self.norm(x)
        # if self.dist_token is None:
        #     logger_handle.info("not dist")
        #     return self.pre_logits(x[:, 0])
        # else:
        #     logger_handle.info("is dist")
        #     return x[:, 0], x[:, 1]


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        assert patch_size == 16, "ConvStem only supports patch size of 16"
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
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


class GatedVisionTransformer(VisionTransformerMoCo):
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

    # def forward_features(self, x):
    #     B = x.shape[0]
    #     x = self.patch_embed(x)

    #     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = x + self.pos_embed
    #     x = self.pos_drop(x)

    #     for blk in self.blocks:
    #         x, attn = blk(x)
    #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #     outcome = self.norm(x)
    #     return outcome

class GatedPromptedVisionTransformerMoCo(GatedVisionTransformer):
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


def prompted_vit_small(in_config, **kwargs):
    model = PromptedVisionTransformerMoCo(
        in_config=in_config,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def prompted_vit_base(in_config, **kwargs):
    model = PromptedVisionTransformerMoCo(
        in_config=in_config,
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

def gated_vpt_vit_base(in_config, **kwargs):
    model = GatedPromptedVisionTransformerMoCo(
        in_config=in_config,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=0.1,
        # block_fn=GatedBlock,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_small(in_config, **kwargs):
    model = VisionTransformerMoCo(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def vit_base(in_config, **kwargs):
    model = VisionTransformerMoCo(
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

