#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A minimal, self-contained equivariant Vision Transformer."""

import math
from itertools import repeat
import collections.abc
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Small local replacement for torch.nn.init.trunc_normal_ compatibility."""
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0:
        mask.div_(keep_prob)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class EQLinearInter(nn.Module):
    """
    输入: (..., inNum * tranNum)
    输出: (..., outNum * tranNum)
    存储基权重 (outNum, 1, inNum, tranNum),前向时按块循环展开为
    (outNum*tranNum, inNum*tranNum) 的等价线性矩阵。
    """

    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super().__init__()
        self.inNum = inNum
        self.outNum = outNum
        self.tranNum = tranNum
        self.use_bias = bias
        self.iniScale = iniScale  # 保留 API 兼容,不使用

        # 保留原 4D 形状以兼容旧 checkpoint
        self.weights = nn.Parameter(torch.empty(outNum, 1, inNum, tranNum))
        if bias:
            self.c = nn.Parameter(torch.empty(outNum, 1))
        else:
            self.register_parameter("c", None)

        # 预计算循环移位索引: shift_idx[s, t] = (t - s) mod T
        idx = torch.arange(tranNum)
        shift_idx = (idx.view(1, tranNum) - idx.view(tranNum, 1)) % tranNum
        self.register_buffer("shift_idx", shift_idx, persistent=False)

        # eval 缓存槽:不写入 state_dict
        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.reset_parameters()

    def _expand_weight(self) -> torch.Tensor:
        # (O,1,I,T) -> (O,I,T) -> gather 成 (O,I,T_s,T_t) -> (O*T, I*T)
        w = self.weights.squeeze(1)
        rolled = w[:, :, self.shift_idx]                     # (O, I, T_s, T_t)
        return rolled.permute(0, 2, 1, 3).reshape(self.outNum * self.tranNum, self.inNum * self.tranNum)

    def _expand_bias(self):
        if not self.use_bias:
            return None
        return self.c.squeeze(-1).repeat_interleave(self.tranNum)


    def forward(self, x):
        if self.training:
            weight = self._expand_weight()
            bias = self._expand_bias()
        else:
            if self._cached_weight is None:
                self._cached_weight = self._expand_weight().detach()
                if self.use_bias:
                    self._cached_bias = self._expand_bias().detach()
            weight = self._cached_weight
            bias = self._cached_bias if self.use_bias else None
        return F.linear(x, weight, bias)

    def train(self, mode: bool = True):
        if mode:
            self._cached_weight = None
            self._cached_bias = None
        return super().train(mode)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.use_bias:
            fan_in = self.inNum * self.tranNum
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.c, -bound, bound)

    def extra_repr(self):
        return (f"inNum={self.inNum}, outNum={self.outNum}, "
                f"tranNum={self.tranNum}, bias={self.use_bias}")



class EQLinearOutput(nn.Module):
    """
    输入: (..., in_num * tran_num)
    输出: (..., out_num),对群作用不变。

    数学等价: y = (x.view(..,I,T).sum(-1)) @ W^T + b
    通过把"沿 T 求和"提前到 GEMM 之前,GEMM 工作量从 O*(I*T) 降到 O*I (~ T 倍加速)。
    train/eval 路径完全一致,无需缓存。
    """

    def __init__(self, in_num: int, out_num: int, tran_num: int = 4, bias: bool = True):
        super().__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.tran_num = tran_num
        self.use_bias = bias

        self.weight = nn.Parameter(torch.empty(out_num, in_num))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_num))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # 修正:等效线性层的 fan_in 是 I*T,kaiming 默认按 I 算偏大 √T
        with torch.no_grad():
            self.weight.mul_(1.0 / math.sqrt(self.tran_num))
        if self.use_bias:
            bound = 1.0 / math.sqrt(self.in_num * self.tran_num)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        *lead, last = x.shape
        assert last == self.in_num * self.tran_num, f"expected last dim = {self.in_num * self.tran_num}, got {last}"
        x_pooled = x.view(*lead, self.in_num, self.tran_num).sum(dim=-1)
        return F.linear(x_pooled, self.weight, self.bias)



def _mask_c(size_p: int, tran_num: int):
    p = (size_p - 1) / 2
    x = np.arange(-p, p + 1) / p
    X, Y = np.meshgrid(x, x)
    C = X ** 2 + Y ** 2
    if tran_num == 4:
        mask = np.ones((size_p, size_p))
    else:
        mask = np.exp(-np.maximum(C - 1, 0) / (0.2 if size_p > 4 else 2))
    return X, Y, mask


def _bicubic_ini(x):
    ax = np.abs(x)
    ax2, ax3 = ax ** 2, ax ** 3
    return ((ax <= 1) * (1.5 * ax3 - 2.5 * ax2 + 1) +
            ((ax > 1) & (ax <= 2)) * (-0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2))


def get_basis_pca_even(size_p: int, tran_num: int = 4, in_p: Optional[int] = None):
    """Bicubic interpolation basis for even-size C4 filters."""
    in_p = size_p if in_p is None else in_p
    inp = (in_p - 1) / 2
    in_x, in_y, mask = _mask_c(size_p, tran_num)

    X0 = np.expand_dims(in_x, 2)
    Y0 = np.expand_dims(in_y, 2)
    mask = np.expand_dims(np.expand_dims(mask, 2), 3)
    theta = np.expand_dims(np.expand_dims(np.arange(tran_num) / tran_num * 2 * np.pi, 0), 0)

    X = (np.cos(theta) * X0 - np.sin(theta) * Y0) * inp
    Y = (np.cos(theta) * Y0 + np.sin(theta) * X0) * inp
    X = np.expand_dims(np.expand_dims(X, 3), 4)
    Y = np.expand_dims(np.expand_dims(Y, 3), 4)

    k = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, in_p, 1))
    l = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, 1, in_p))
    basis = _bicubic_ini(X - k) * _bicubic_ini(Y - l)
    rank = in_p * in_p
    basis = basis.reshape(size_p, size_p, tran_num, rank) * mask
    return torch.FloatTensor(basis)


class FConvPCA(nn.Module):
    """
    存储基权重 weight: (out_num, in_num, expand, basis_dim) 与 PCA 基 basis,
    前向时通过 einsum 投影回空间核 + 沿 expand 维循环移位,
    再 reshape 成标准 Conv2d 核 (out_num*T, in_num*expand, k, k)。
    """

    def __init__(self, size_p: int, in_num: int, out_num: int, tran_num: int = 4,
                 stride: int = 1, padding: int = 0, first_layer: bool = True, bias: bool = True):
        super().__init__()
        if size_p % 2 != 0:
            raise ValueError("This slim version keeps only the even-kernel PCA basis path.")
        self.size_p = size_p
        self.in_num = in_num
        self.out_num = out_num
        self.tran_num = tran_num
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.expand = 1 if first_layer else tran_num

        basis = get_basis_pca_even(size_p, tran_num)
        self.register_buffer("basis", basis)  # 保留 persistent=True 兼容旧 ckpt
        self.weight = nn.Parameter(torch.empty(out_num, in_num, self.expand, basis.size(3)))
        if bias:
            # 简化为标准 1D bias (注意:破坏与原版 (1,O,1,1) 形状的 ckpt 兼容)
            self.bias = nn.Parameter(torch.empty(out_num))
        else:
            self.register_parameter("bias", None)

        # eval 缓存槽:不写入 state_dict
        self.register_buffer("_cached_filter", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def _expand_filter(self) -> torch.Tensor:
        t, e = self.tran_num, self.expand
        # einsum 投影: (i,j,o,k) ⊗ (m,n,a,k) -> (m,o,n,a,i,j)
        # 即 (out, tran, in, expand, kH, kW)
        tempW = torch.einsum('ijok,mnak->monaij', self.basis, self.weight)
        num = t // e
        # 沿 expand 轴 (axis=3) 用 torch.roll 替代 cat([..,-i:], [..,:-i]),
        # 数学等价但更简洁,且 i=0 边界自动正确
        parts = [
            torch.roll(tempW[:, i * num:(i + 1) * num], shifts=i, dims=3)
            for i in range(e)
        ]
        tempW = torch.cat(parts, dim=1)
        return tempW.reshape(self.out_num * t, self.in_num * e, self.size_p, self.size_p)

    def _expand_bias(self):
        if not self.use_bias:
            return None
        # (out_num,) -> (out_num*tran_num,),每个原值连续重复 T 次
        return self.bias.repeat_interleave(self.tran_num)

    def forward(self, x):
        if self.training:
            weight = self._expand_filter()
            bias = self._expand_bias()
        else:
            if self._cached_filter is None:
                self._cached_filter = self._expand_filter().detach()
                if self.use_bias:
                    self._cached_bias = self._expand_bias().detach()
            weight = self._cached_filter
            bias = self._cached_bias if self.use_bias else None
        return F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding)

    def train(self, mode: bool = True):
        if mode:
            self._cached_filter = None
            self._cached_bias = None
        return super().train(mode)

    def extra_repr(self):
        return (f"size_p={self.size_p}, in_num={self.in_num}, out_num={self.out_num}, "
                f"tran_num={self.tran_num}, expand={self.expand}, "
                f"stride={self.stride}, padding={self.padding}, bias={self.use_bias}")



class EQLayerNorm(nn.LayerNorm):
    """
    群等变 LayerNorm:在 dim//tran_num 个 group 上共享 affine 参数,
    每个 group 的参数被复制 tran_num 份施加到全 dim 维上。
    """

    def __init__(self, dim: int, eps: float = 1e-6, tran_num: int = 4):
        if dim % tran_num != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by tran_num ({tran_num})"
            )
        super().__init__(dim // tran_num, eps=eps)
        self.tran_num = tran_num
        self.full_dim = dim

        # eval 缓存槽位:persistent=False 不写入 state_dict,避免 ckpt 膨胀/陈旧
        self.register_buffer("_cached_weight", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

    def forward(self, x):
        if self.training:
            weight = self.weight.repeat_interleave(self.tran_num)
            bias = self.bias.repeat_interleave(self.tran_num)
        else:
            # 推理态:懒加载缓存,首次 eval 调用时构建,后续直接复用
            if self._cached_weight is None:
                # detach 避免缓存持有训练计算图
                self._cached_weight = self.weight.detach().repeat_interleave(self.tran_num)
                self._cached_bias = self.bias.detach().repeat_interleave(self.tran_num)
            weight = self._cached_weight
            bias = self._cached_bias

        return F.layer_norm(x, (self.full_dim,), weight, bias, self.eps)

    def train(self, mode: bool = True):
        # 切回训练态时清空缓存,避免下次 eval 拿到参数更新前的陈旧值
        if mode:
            self._cached_weight = None
            self._cached_bias = None
        return super().train(mode)



class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = EQLinearInter(dim // 4, dim * 3 // 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = EQLinearInter(dim // 4, dim // 4, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.):
        super().__init__()
        self.fc1 = EQLinearInter(dim // 4, hidden_dim // 4, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = EQLinearInter(hidden_dim // 4, dim // 4, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))



class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = True,
                 proj_drop: float = 0., attn_drop: float = 0., drop_path_rate: float = 0.):
        super().__init__()
        self.norm1 = EQLayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = EQLayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), proj_drop)
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x



class PatchEmbed(nn.Module):
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 768, bias: bool = True):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = FConvPCA(patch_size, in_chans, embed_dim // 4, tran_num=4, stride=patch_size,
                             first_layer=True, bias=bias)

    def feat_ratio(self):
        return max(self.patch_size)

    def forward(self, x):
        _, _, H, W = x.shape
        if (H, W) != self.img_size:
            raise ValueError(f"Input image size {(H, W)} must match model size {self.img_size}.")
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)



class DropoutEq(nn.Module):
    def __init__(self, p: float = 0., tran_num: int = 4):
        super().__init__()
        self.tran_num = tran_num
        self.drop = nn.Dropout1d(p)

    def forward(self, x):
        shape = x.shape
        return self.drop(x.reshape(-1, self.tran_num)).view(shape)



def global_pool_nlc(x, pool_type: str = 'token', num_prefix_tokens: int = 1):
    if pool_type == 'token':
        return x[:, 0]
    x = x[:, num_prefix_tokens:]
    if pool_type == 'avg':
        return x.mean(dim=1)
    if pool_type == 'max':
        return x.max(dim=1).values
    if pool_type == 'avgmax':
        return 0.5 * (x.mean(dim=1) + x.max(dim=1).values)
    if pool_type == '':
        return x
    raise ValueError(f"Unsupported pool_type: {pool_type}")



class VisionTransformer_eq(nn.Module):
    """Minimal C4-equivariant Vision Transformer.

    Kept arguments are the common ones needed for normal classification training/inference.
    Unsupported timm compatibility branches were removed intentionally.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4., qkv_bias: bool = True,
                 global_pool: str = 'token', drop_rate: float = 0., pos_drop_rate: float = 0.,
                 proj_drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.,
                 class_token: bool = True, pos_embed: bool = True):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("embed_dim must be divisible by 4 for C4 equivariant channels.")
        if global_pool == 'token' and not class_token:
            raise ValueError("global_pool='token' requires class_token=True.")
        if global_pool not in ('token', 'avg', 'max', 'avgmax', ''):
            raise ValueError("global_pool must be one of: 'token', 'avg', 'max', 'avgmax', ''.")

        self.tran_num = 4
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.global_pool = global_pool
        self.num_prefix_tokens = 1 if class_token else 0
        self.has_class_token = class_token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, bias=True)
        grid_h, grid_w = self.patch_embed.grid_size
        if pos_embed and (grid_h != grid_w or grid_h % 2 != 0):
            raise ValueError("The compact equivariant positional embedding requires an even square patch grid.")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim // 4)) if class_token else None
        if pos_embed:
            self.pos_embed_weights = nn.Parameter(torch.zeros(1, grid_h // 2, grid_w // 2, embed_dim // 4))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim // 4)) if class_token else None
            trunc_normal_(self.pos_embed_weights, std=.02)
            if self.cls_pos_embed is not None:
                trunc_normal_(self.cls_pos_embed, std=.02)
        else:
            self.pos_embed_weights = None
            self.cls_pos_embed = None

        self.pos_drop = nn.Dropout(pos_drop_rate)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, proj_drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = EQLayerNorm(embed_dim)
        self.head_drop = DropoutEq(drop_rate, 4)
        self.head = EQLinearOutput(embed_dim // 4, num_classes, 4) if num_classes > 0 else nn.Identity()

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def _pos_embed(self, x):
        B = x.shape[0]
        if self.cls_token is not None:
            cls_token = self.cls_token.repeat_interleave(4, dim=2).expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.pos_embed_weights is None:
            return self.pos_drop(x)

        gh, gw = self.patch_embed.grid_size
        half_h, half_w = gh // 2, gw // 2
        pos = torch.zeros(1, gh, gw, self.embed_dim // 4, device=x.device, dtype=x.dtype)
        w = self.pos_embed_weights.to(dtype=x.dtype)
        pos[:, :half_h, :half_w] = w
        pos[:, :half_h, half_w:] = torch.rot90(w, -1, [1, 2])
        pos[:, half_h:, half_w:] = torch.rot90(w, -2, [1, 2])
        pos[:, half_h:, :half_w] = torch.rot90(w, -3, [1, 2])
        pos = pos.reshape(1, gh * gw, self.embed_dim // 4).repeat_interleave(4, dim=2)

        if self.cls_token is not None:
            cls_pos = self.cls_pos_embed.to(dtype=x.dtype).repeat_interleave(4, dim=2)
            pos = torch.cat((cls_pos, pos), dim=1)
        return self.pos_drop(x + pos)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        return self.norm(x)

    def forward_head(self, x, pre_logits: bool = False):
        x = global_pool_nlc(x, self.global_pool, self.num_prefix_tokens)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

# =====================================================

def rotate_and_shift(x, rotate_times=0, rotate_dims=[-2, -1], shift_times=0, shift_dim=1):
    """
    对输入张量进行逆时针旋转90度和通道轮换

    参数:
        x: 输入张量, 形状为 [batch, 4, d_inner, height, width]

    返回:
        处理后的张量, 形状与输入相同
    """
    # 逆时针旋转90度 (对最后两个维度)
    x_rot = torch.rot90(x, k=rotate_times, dims=rotate_dims)

    # 在通道维度(维度1)进行轮换，最后一个放到第一个
    x_shifted = torch.roll(x_rot, shifts=shift_times, dims=shift_dim)

    return x_shifted


def print_error(x_triton, x_torch, epsilon=1e-5):
    diff_abs = torch.abs(x_triton - x_torch)
    diff = torch.abs(x_triton - x_torch) / (torch.abs(x_torch) + epsilon)
    print(f"Error mean (abs): {diff_abs.mean()}, Error max (abs): {diff_abs.max()}, "
          f"Error mean (relative): {diff.mean()}, Error max (relative): {diff.max()}, ")



if __name__ == "__main__":
    if __package__:
        from .test_throughput import get_vit_config, run_throughput_cli
    else:
        from test_throughput import get_vit_config, run_throughput_cli

    def build_model(model_name, img_size):
        cfg = get_vit_config(model_name)
        return VisionTransformer_eq(img_size=img_size, **cfg)

    run_throughput_cli(
        build_model=build_model,
        target_classes=(EQLinearInter, EQLinearOutput),
        tran_num=4,
        model_family="vit",
        title="Naive EQ-ViT",
    )
    print("\n\n")
