# https://github.com/ChengHan111/E2VPT
from models.wrappers.core_model import CoreModel
from dataset.dataset_interface import EnumDatasetTaskType
from models.wrappers.architecture.vision_transformer_changeVK import VisionTransformer_changeVK
import utility.logger as logger
logger_handle = logger.get_logger("vpt_demographic_adaptation")

import math
import torch
from torch.nn import Dropout
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul
import os
from timm.models.layers import PatchEmbed
import json


class EEVisionTransformer(VisionTransformer_changeVK):
    def __init__(self, in_config, stop_grad_conv1=False, **kwargs):
        super(EEVisionTransformer, self).__init__(**kwargs)

        self._m_config = in_config
        self.build_2d_sincos_position_embedding()
        del self.blocks
        
        self.blocks = nn.Sequential(*[
            Block_VK(
                in_config=in_config, dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop_rate,
                attn_drop=self.attn_drop_rate, drop_path=self.dpr[i], norm_layer=self.norm_layer, act_layer=self.act_layer)
            for i in range(self.depth)])
        
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
        
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

class EE_PromptedVisionTransformer(EEVisionTransformer):
    def __init__(self, in_config, **kwargs):
        super().__init__(in_config, **kwargs)
        self._m_config = in_config

        num_tokens_P = self._m_config.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS_P

        self.num_tokens_P = num_tokens_P
        self.prompt_dropout = Dropout(self._m_config.MODEL.E2VPT.KV_PROMPT.DROPOUT)

        # initiate prompt:
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
            )
        )  # noqa

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_tokens_P, self.embed_dim)
        )
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

        if self._m_config.MODEL.E2VPT.KV_PROMPT.DEEP:
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(len(self.blocks) - 1, num_tokens_P, self.embed_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)


        # print('self.embed_dim', self.embed_dim)
        # new added for cls_token masked
        if self._m_config.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN is True:
            if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK == True:
                self.prompt_soft_tokens_mask_cls_token = nn.Parameter(
                    torch.ones(self.num_tokens_P), requires_grad=True
                )

            if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PIECES == True:
                self.prompt_soft_tokens_pieces_mask_cls_token = nn.Parameter(
                    torch.ones(
                        self.num_tokens_P, self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_P_PIECES_NUM
                    ),
                    requires_grad=True,
                )
                self.soft_token_chunks_num_cls_token = int(
                    self.embed_dim / self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_P_PIECES_NUM
                )

            # Rewind status mark here.
            if self._m_config.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN and self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_STATUS:

                soft_token_mask_dir = os.path.join(
                    self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_OUTPUT_DIR, "mask_tokens"
                )
                assert soft_token_mask_dir is not None

                soft_token_mask_file = os.path.join(
                    soft_token_mask_dir,
                    "{}_soft_tokens_to_mask.json".format(
                        self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_MASK_CLS_TOKEN_NUM
                    ),
                )
                soft_token_to_mask = self.load_soft_token_mask_file(
                    soft_token_mask_file
                )
                self.mask_soft_tokens(soft_token_to_mask)

            if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PIECES and self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_STATUS:
                soft_tokens_pieces_mask_dir = os.path.join(
                    self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_OUTPUT_DIR, "mask_tokens_pieces"
                )
                soft_tokens_pieces_mask_file = os.path.join(
                    soft_tokens_pieces_mask_dir,
                    "{}_soft_tokens_pieces_to_mask.json".format(
                        self._m_config.MODEL.E2VPT.KV_PROMPT.REWIND_MASK_CLS_TOKEN_PIECE_NUM
                    ),
                )  # rewind_mask_token_pieces_number
                soft_tokens_pieces_to_mask = self.load_soft_tokens_pieces_mask_file(
                    soft_tokens_pieces_mask_file
                )
                self.mask_soft_tokens_pieces(soft_tokens_pieces_to_mask)

        # add drop-out or not
        self.prompt_dropout = Dropout(self._m_config.MODEL.E2VPT.KV_PROMPT.DROPOUT_P)

    def load_soft_token_mask_file(self, path):
        with open(path) as f:
            t = json.load(f)

        soft_token_to_mask = set()
        for mask_number, soft_token in t.items():
            for soft_token_i in soft_token:
                soft_token_to_mask.add(soft_token_i)

        return soft_token_to_mask

    def load_soft_tokens_pieces_mask_file(self, path):
        with open(path) as f:
            t = json.load(f)
        soft_tokens_pieces_to_mask = {}
        for soft_token_idx, soft_token_pieces in t.items():
            soft_tokens_pieces_to_mask[int(soft_token_idx)] = set(soft_token_pieces)
        return soft_tokens_pieces_to_mask

    def mask_soft_tokens(self, soft_tokens_to_mask):
        self.soft_tokens_to_mask = list(soft_tokens_to_mask)
        for soft_token_idx in self.soft_tokens_to_mask:
            # print('soft_token_idx',soft_token_idx)
            self.prompt_soft_tokens_mask_cls_token.data[soft_token_idx] = 0
        # Self added no grad during rewind
        self.prompt_soft_tokens_mask_cls_token.requires_grad_(False)

    def mask_soft_tokens_pieces(self, soft_tokens_pieces_to_mask):
        for soft_token_id, soft_token_pieces in soft_tokens_pieces_to_mask.items():
            for soft_token_piece in soft_token_pieces:
                self.prompt_soft_tokens_pieces_mask_cls_token.data[soft_token_id][
                    soft_token_piece
                ] = 0
        # Self added no grad during rewind
        self.prompt_soft_tokens_pieces_mask_cls_token.requires_grad_(False)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        if self._m_config.MODEL.E2VPT.KV_PROMPT.LOCATION == "prepend":
            # after CLS token, all before image patches

            prompt_embeddings = self.prompt_dropout(
                self.prompt_embeddings.expand(B, -1, -1)
            )
            if self._m_config.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN is True:
                if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PIECES is True:
                    # print('222', self.soft_tokens_pieces_mask_cls_token.shape) torch.Size([32, 16])
                    prompt_embeddings = (
                        prompt_embeddings
                        * self.prompt_soft_tokens_pieces_mask_cls_token.repeat(
                            (1, self.soft_token_chunks_num_cls_token)
                        ).repeat(B, 1, 1)
                    )
                    # print('mark1', self.prompt_soft_tokens_pieces_mask_cls_token)
                    # print('prompt_embeddings', prompt_embeddings)
                if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK == True:
                    prompt_embeddings = (
                        prompt_embeddings
                        * self.prompt_soft_tokens_mask_cls_token.view(-1, 1)
                        .repeat(1, prompt_embeddings.shape[2])
                        .repeat(B, 1, 1)
                    )
                    # print('mark2', self.prompt_soft_tokens_mask_cls_token)
                    # print('prompt_embeddings', prompt_embeddings)

            x = torch.cat((x[:, :1, :], prompt_embeddings, x[:, 1:, :]), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def embeddings(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
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
        x = self.incorporate_prompt(x)

        if self._m_config.MODEL.E2VPT.KV_PROMPT.DEEP:
            B = x.shape[0]
            num_layers = len(self.blocks)

            for i in range(num_layers):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    # prepend
                    deep_prompt_emb = self.prompt_dropout(
                        self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                    )

                    if self._m_config.MODEL.E2VPT.KV_PROMPT.MASK_CLS_TOKEN is True:
                        if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK_PIECES is True:
                            # print(self.soft_tokens_pieces_mask_cls_token.repeat((1,self.soft_token_chunks_num_cls_token)).repeat(B, 1, 1).shape)
                            deep_prompt_emb = (
                                deep_prompt_emb
                                * self.prompt_soft_tokens_pieces_mask_cls_token.repeat(
                                    (1, self.soft_token_chunks_num_cls_token)
                                ).repeat(B, 1, 1)
                            )
                        if self._m_config.MODEL.E2VPT.KV_PROMPT.CLS_TOKEN_MASK == True:
                            # print(self.soft_tokens_mask_cls_token.view(-1, 1).repeat(1, self.deep_prompt_embeddings.shape[2]).repeat(B, 1, 1))
                            deep_prompt_emb = (
                                deep_prompt_emb
                                * self.prompt_soft_tokens_mask_cls_token.view(-1, 1)
                                .repeat(1, self.deep_prompt_embeddings.shape[2])
                                .repeat(B, 1, 1)
                            )

                    x = torch.cat(
                        (
                            x[:, :1, :],
                            deep_prompt_emb,
                            x[:, (1 + self.num_tokens_P) :, :],
                        ),
                        dim=1,
                    )
                    x = self.blocks[i](x)
        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        
        # Default pooling behavior from MoCoV3.
        return self.pre_logits(x[:, 0])

class Block_VK(nn.Module):

    def __init__(self, in_config, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_VK(in_config, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Attention_VK(nn.Module):
    def __init__(self, in_config, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._m_config = in_config
        num_tokens = self._m_config.MODEL.E2VPT.KV_PROMPT.NUM_TOKENS
        if self._m_config.MODEL.E2VPT.KV_PROMPT.SHARE_PARAM_KV == True:
            head_fixed, num_patches_QKV, head_size_fixed = self.num_heads, num_tokens, head_dim
            self.deep_QKV_embeddings = nn.Parameter(torch.zeros(
                        head_fixed, num_patches_QKV, head_size_fixed))
            torch.nn.init.kaiming_uniform_(self.deep_QKV_embeddings, a=0, mode='fan_in', nonlinearity='leaky_relu')
        else:
            raise ValueError("Not supported for unshare VK in MAE setting! Under construction")

        self.QKV_proj = nn.Identity()
        self.QKV_dropout = Dropout(self._m_config.MODEL.E2VPT.KV_PROMPT.DROPOUT) # should add config here
        
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        B = q.shape[0] # should be the batch size
        if self._m_config.MODEL.E2VPT.KV_PROMPT.SHARE_PARAM_KV == True:
            if self._m_config.MODEL.E2VPT.KV_PROMPT.LAYER_BEHIND == False:
                k = torch.cat((k, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
                v = torch.cat((v, self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1))), dim=2)
            else:
                k = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), k), dim=2)
                v = torch.cat((self.QKV_dropout(self.QKV_proj(self.deep_QKV_embeddings).expand(B, -1, -1, -1)), v), dim=2)
        else:
            raise ValueError("Not supported for unshare VK in MAE setting! Under construction")
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def vit_base(in_config, **kwargs):
    model = EE_PromptedVisionTransformer(
        in_config,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

__archs__ = {
    "vit_base": vit_base,
}


class E2VPT_MoCoV3(CoreModel):

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
            if "QKV" in k:
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
