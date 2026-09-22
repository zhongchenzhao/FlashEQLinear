#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A minimal, self-contained original Vision Transformer.

training / evaluation scripts can continue to import it, but the model itself
is now a plain ViT:
  - Conv2d patch embedding
  - standard nn.Linear QKV / MLP / classifier head
  - standard nn.LayerNorm
  - standard learnable absolute positional embedding

All C4-equivariant modules from the previous eq_vit.py have been removed.
"""

from itertools import repeat
import collections.abc
from typing import Optional, Tuple, Union

import torch
from torch import nn


# -----------------------------
# small utilities
# -----------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False):
    """Stochastic depth per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


# -----------------------------
# plain ViT building blocks
# -----------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding with a standard Conv2d."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
        norm_layer: Optional[type] = None,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def feat_ratio(self):
        return max(self.patch_size)

    def forward(self, x: torch.Tensor):
        _, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Input image size {(H, W)} must match model size {self.img_size}.")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: type = nn.LayerNorm,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Plain ViT MLP implemented with nn.Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type = nn.GELU,
        norm_layer: Optional[type] = None,
        bias: bool = True,
        drop: float = 0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    """Original ViT pre-norm transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
        mlp_layer: type = Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# -----------------------------
# pooling / init
# -----------------------------

def global_pool_nlc(
    x: torch.Tensor,
    pool_type: str = 'token',
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x
    if pool_type == 'token':
        return x[:, 0]

    x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
    if pool_type == 'avg':
        return x.mean(dim=1)
    if pool_type == 'max':
        return x.amax(dim=1)
    if pool_type == 'avgmax':
        return 0.5 * (x.mean(dim=1) + x.amax(dim=1))
    raise ValueError(f"Unsupported pool_type: {pool_type}")


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# -----------------------------
# main model
# -----------------------------

class VisionTransformer(nn.Module):
    """Plain original Vision Transformer with the old class name kept.

    Despite the name, this class is no longer equivariant. It is intentionally
    converted to a standard ViT so old scripts that instantiate
    `VisionTransformer(...)` do not need import-name changes.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = 'token',
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        scale_attn_norm: bool = False,
        scale_mlp_norm: bool = False,
        proj_bias: bool = True,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: Union[str, bool] = 'learn',
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        final_norm: bool = True,
        fc_norm: Optional[bool] = None,
        pool_include_prefix: bool = False,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        patch_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        embed_layer: type = PatchEmbed,
        norm_layer: type = nn.LayerNorm,
        act_layer: type = nn.GELU,
        block_fn: type = Block,
        mlp_layer: type = Mlp,
        **unused_kwargs,
    ):
        super().__init__()
        if unused_kwargs:
            # Keep compatibility with larger timm-style call sites without silently changing behavior.
            unsupported = ', '.join(sorted(unused_kwargs.keys()))
            raise TypeError(f"Unsupported arguments in plain VisionTransformer: {unsupported}")
        if global_pool not in ('', 'avg', 'avgmax', 'max', 'token'):
            raise ValueError("global_pool must be one of: '', 'avg', 'avgmax', 'max', 'token'.")
        if global_pool == 'token' and not class_token:
            raise ValueError("global_pool='token' requires class_token=True.")
        if pos_embed not in (True, False, '', 'none', 'learn'):
            raise ValueError("pos_embed must be True, False, '', 'none', or 'learn'.")

        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = (1 if class_token else 0) + reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.pool_include_prefix = pool_include_prefix

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if pos_embed in (False, '', 'none'):
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_len, embed_dim))

        self.pos_drop = nn.Dropout(pos_drop_rate)
        # Kept as Identity to stay close to original ViT unless user explicitly implements patch dropout.
        self.patch_drop = nn.Dropout(patch_drop_rate) if patch_drop_rate > 0 else nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                scale_attn_norm=scale_attn_norm,
                scale_mlp_norm=scale_mlp_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)
        ])
        self.feature_info = [dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.reg_token is not None:
            nn.init.normal_(self.reg_token, std=1e-6)
        self.apply(init_weights_vit_timm)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'reg_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            if global_pool not in ('', 'avg', 'avgmax', 'max', 'token'):
                raise ValueError("Unsupported global_pool")
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor):
        if self.pos_embed is None:
            x = x.view(x.shape[0], -1, x.shape[-1])
            to_cat = []
            if self.cls_token is not None:
                to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
            if self.reg_token is not None:
                to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            return self.pos_drop(x)

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            x = x + self.pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        else:
            x = self.blocks(x)

        return self.norm(x)

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None):
        pool_type = self.global_pool if pool_type is None else pool_type
        return global_pool_nlc(
            x,
            pool_type=pool_type,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x



if __name__ == "__main__":
    if __package__:
        from .test_throughput import get_vit_config, run_throughput_cli
    else:
        from test_throughput import get_vit_config, run_throughput_cli

    def build_model(model_name, img_size):
        cfg = get_vit_config(model_name)
        return VisionTransformer(img_size=img_size, **cfg)

    run_throughput_cli(
        build_model=build_model,
        target_classes=nn.Linear,
        tran_num=1,
        model_family="vit",
        title="Original ViT",
    )
    print("\n\n")

