#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C4-equivariant Swin Transformer and throughput benchmark."""

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import math
import torch.nn as nn
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F

try:
    if __package__:
        from .window_process.window_process import WindowProcess, WindowProcessReverse
    else:
        from window_process.window_process import WindowProcess, WindowProcessReverse
except ModuleNotFoundError as exc:
    if exc.name != "swin_window_process":
        raise
    WindowProcess = None
    WindowProcessReverse = None



# ================================================================

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
            weight, bias = self._expand_weight()
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

        self.weights = nn.Parameter(torch.empty(out_num, in_num))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_num))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # 修正:等效线性层的 fan_in 是 I*T,kaiming 默认按 I 算偏大 √T
        with torch.no_grad():
            self.weights.mul_(1.0 / math.sqrt(self.tran_num))
        if self.use_bias:
            bound = 1.0 / math.sqrt(self.in_num * self.tran_num)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        *lead, last = x.shape
        assert last == self.in_num * self.tran_num, f"expected last dim = {self.in_num * self.tran_num}, got {last}"
        x_pooled = x.view(*lead, self.in_num, self.tran_num).sum(dim=-1)
        return F.linear(x_pooled, self.weights, self.bias)



class EQDropout(nn.Module):
    def __init__(self, p: float = 0., tran_num: int = 4):
        super().__init__()
        self.tran_num = tran_num
        self.drop = nn.Dropout1d(p)

    def forward(self, x):
        shape = x.shape
        return self.drop(x.reshape(-1, self.tran_num)).view(shape)




def _mask_c(size_p: int, tran_num: int):
    p = (size_p - 1) / 2
    x = np.arange(-p, p + 1) / p
    x_grid, y_grid = np.meshgrid(x, x)

    radius_square = x_grid ** 2 + y_grid ** 2

    if tran_num == 4:
        mask = np.ones((size_p, size_p))
    else:
        decay = 0.2 if size_p > 4 else 2
        mask = np.exp(-np.maximum(radius_square - 1, 0) / decay)

    return x_grid, y_grid, mask


def _bicubic_ini(x):
    abs_x = np.abs(x)
    abs_x2 = abs_x ** 2
    abs_x3 = abs_x ** 3

    inner = abs_x <= 1
    outer = (abs_x > 1) & (abs_x <= 2)

    return (
        inner * (1.5 * abs_x3 - 2.5 * abs_x2 + 1)
        + outer * (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2)
    )


def get_basis_pca(size_p: int, tran_num: int = 8, in_p = None, smooth: bool = True):
    if in_p is None:
        in_p = size_p

    inp = in_p // 2

    in_x, in_y, mask = _mask_c(size_p, tran_num)

    x0 = np.expand_dims(in_x, 2)
    y0 = np.expand_dims(in_y, 2)
    mask = np.expand_dims(np.expand_dims(mask, 2), 3)

    theta = np.arange(tran_num) / tran_num * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)

    x = np.cos(theta) * x0 - np.sin(theta) * y0
    y = np.cos(theta) * y0 + np.sin(theta) * x0

    x = x * inp
    y = y * inp

    x = np.expand_dims(np.expand_dims(x, 3), 4)
    y = np.expand_dims(np.expand_dims(y, 3), 4)

    k = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, in_p, 1))
    l = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, 1, in_p))

    basis = _bicubic_ini(x - k) * _bicubic_ini(y - l)

    rank = in_p * in_p
    basis = basis.reshape(size_p, size_p, tran_num, rank) * mask

    basis_weight = 1

    return torch.FloatTensor(basis), rank, basis_weight


def get_basis_pca_even(size_p: int, tran_num: int = 8, in_p = None, smooth: bool = True):
    if in_p is None:
        in_p = size_p

    inp = (in_p - 1) / 2

    in_x, in_y, mask = _mask_c(size_p, tran_num)

    x0 = np.expand_dims(in_x, 2)
    y0 = np.expand_dims(in_y, 2)
    mask = np.expand_dims(np.expand_dims(mask, 2), 3)

    theta = np.arange(tran_num) / tran_num * 2 * np.pi
    theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)

    x = np.cos(theta) * x0 - np.sin(theta) * y0
    y = np.cos(theta) * y0 + np.sin(theta) * x0

    x = x * inp
    y = y * inp

    x = np.expand_dims(np.expand_dims(x, 3), 4)
    y = np.expand_dims(np.expand_dims(y, 3), 4)

    k = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, in_p, 1))
    l = np.reshape(np.arange(-inp, inp + 1), (1, 1, 1, 1, in_p))

    basis = _bicubic_ini(x - k) * _bicubic_ini(y - l)

    rank = in_p * in_p
    basis = basis.reshape(size_p, size_p, tran_num, rank) * mask

    basis_weight = 1

    return torch.FloatTensor(basis), rank, basis_weight



class FConvPCA(nn.Module):
    """
    PCA-basis steerable convolution.

    weight: (out_num, in_num, expand, basis_dim)
    basis : (size_p, size_p, tran_num, basis_dim)

    Forward:
        1. project PCA coefficients back to spatial filters by einsum
        2. circularly shift filters along the expand dimension
        3. reshape to Conv2d weight:
           (out_num * tran_num, in_num * expand, size_p, size_p)
    """

    def __init__(
        self,
        size_p: int,
        in_num: int,
        out_num: int,
        tran_num: int = 8,
        in_p = None,
        stride: int = 1,
        padding = None,
        first_layer: bool = False,
        bias: bool = True,
        smooth: bool = True,
        ini_scale: float = 1.0,
    ):
        super().__init__()

        in_p = size_p if in_p is None else in_p

        self.size_p = size_p
        self.in_num = in_num
        self.out_num = out_num
        self.tran_num = tran_num
        self.stride = stride
        self.padding = 0 if padding is None else padding
        self.use_bias = bias
        self.expand = 1 if first_layer else tran_num
        self.ini_scale = ini_scale

        if tran_num % self.expand != 0:
            raise ValueError(
                f"tran_num must be divisible by expand, got tran_num={tran_num}, expand={self.expand}"
            )

        if size_p % 2 == 0:
            basis, rank, basis_weight = get_basis_pca_even(
                size_p, tran_num, in_p, smooth=smooth
            )
        else:
            basis, rank, basis_weight = get_basis_pca(
                size_p, tran_num, in_p, smooth=smooth
            )

        self.register_buffer("basis", basis)

        self.weight = nn.Parameter(
            torch.empty(out_num, in_num, self.expand, basis.size(3))
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_num))
        else:
            self.register_parameter("bias", None)

        # eval 模式缓存展开后的卷积核，不写入 state_dict
        self.register_buffer("_cached_filter", None, persistent=False)
        self.register_buffer("_cached_bias", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        if self.ini_scale != 1.0:
            with torch.no_grad():
                self.weight.mul_(self.ini_scale)

    def _expand_filter(self) -> torch.Tensor:
        """
        Expand PCA coefficients into regular Conv2d kernels.

        basis : (kH, kW, T, rank)
        weight: (O, I, E, rank)

        output filter:
            (O * T, I * E, kH, kW)
        """
        t = self.tran_num
        e = self.expand

        # (i, j, o, k) x (m, n, a, k) -> (m, o, n, a, i, j)
        # m: out_num, o: tran_num, n: in_num, a: expand
        temp_w = torch.einsum("ijok,mnak->monaij", self.basis, self.weight)

        num = t // e

        # 原代码:
        # torch.cat([tempW[..., -i:, :, :], tempW[..., :-i, :, :]], dim=3)
        # 可用 torch.roll 简化；i=0 时也安全
        parts = [
            torch.roll(
                temp_w[:, i * num:(i + 1) * num],
                shifts=i,
                dims=3,
            )
            for i in range(e)
        ]

        temp_w = torch.cat(parts, dim=1)

        return temp_w.reshape(
            self.out_num * t,
            self.in_num * e,
            self.size_p,
            self.size_p,
        )

    def _expand_bias(self):
        if self.bias is None:
            return None

        # (out_num,) -> (out_num * tran_num,)
        return self.bias.repeat_interleave(self.tran_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self._expand_filter()
            bias = self._expand_bias()
        else:
            if self._cached_filter is None:
                self._cached_filter = self._expand_filter().detach()
                self._cached_bias = (
                    self._expand_bias().detach()
                    if self.bias is not None
                    else None
                )

            weight = self._cached_filter
            bias = self._cached_bias

        return F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            groups=1,
        )

    def train(self, mode: bool = True):
        # 从 eval 切回 train 时清空缓存，避免参数更新后仍使用旧 filter
        if mode:
            self._cached_filter = None
            self._cached_bias = None

        return super().train(mode)

    def extra_repr(self) -> str:
        return (
            f"size_p={self.size_p}, "
            f"in_num={self.in_num}, "
            f"out_num={self.out_num}, "
            f"tran_num={self.tran_num}, "
            f"expand={self.expand}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}"
        )

# ================================================================



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., tranNum = 4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = EQLinearInter(in_features, hidden_features, tranNum)
        self.act = act_layer()
        self.fc2 = EQLinearInter(hidden_features, out_features, tranNum)
        self.drop = EQDropout(drop, tranNum)
    def forward(self, x):
        # sizeX = x.shape
        # x = x.reshape([-1, sizeX[-1]])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # return x.reshape(sizeX)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., tranNum = 4 ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.tranNum = tranNum

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))


        coords = torch.stack([torch.rot90(coords, -i, [1,2]) for i in range(tranNum)], dim=3) # 2, Wh, Ww, 4 要注意转的方向
        coords_flatten = coords.reshape([2, -1, tranNum])  # 2, Wh*Ww, 4
        relative_coords = coords_flatten[:, :, None,:] - coords_flatten[:, None, :,:]  # 2, Wh*Ww, Wh*Ww, 4 算出了相对的x差与相对的y差
        relative_coords = relative_coords.permute(1, 2, 3, 0).contiguous()  # Wh*Ww, Wh*Ww, 4, 0
        relative_coords[:, :, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww, 4
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = EQLinearInter(dim//tranNum, dim//tranNum*3, tranNum)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj = EQLinearInter(dim//tranNum, dim//tranNum, tranNum)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1,  self.num_heads)  # Wh*Ww, Wh*Ww, 4, nH
        relative_position_bias = relative_position_bias.mean(2)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) #B, nH, Wh*Ww, Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, tranNum = 4,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, tranNum = tranNum)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim//tranNum, hidden_features=mlp_hidden_dim//tranNum, act_layer=act_layer, drop=drop, tranNum=tranNum)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        if fused_window_process and WindowProcess is None:
            raise ImportError(
                "Fused window processing requires the swin_window_process extension. "
                "Install it with: python -m pip install ./flash_eq_network/window_process"
            )
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class EQPatchMergingConv(FConvPCA):
    """
    PatchMerging 中的等变降采样卷积 (2x2, stride=2)。

    与 FConvPCA 完全相同(参数、buffer、state_dict key 均不变,旧 checkpoint 可直接加载),
    仅多一个类标记 count_as_linear_like,供 test_throughput.UnifiedLinearLikeProfiler
    把它计入 "linear-like" 统计口径。

    原因: 原版 Swin 的 PatchMerging 是 nn.Linear(4C -> 2C),会被计入 linear FLOPs/latency;
    EQ-Swin 用 2x2 stride-2 卷积实现同一步骤,FLOPs 与 Linear(4C -> 2C) 严格相等
    (每图 2 * (H/2) * (W/2) * 4C * 2C)。若不计入,Plain 与 EQ/Flash 的 linear 口径不一致,
    会出现 Flash/Plain < 3/8 以及 "总 FLOPs 降幅 != linear FLOPs 降幅" 的假象。
    注意: 该层不经过 FlashEQLinear,Flash 模型中它仍是稠密卷积,不被加速。
    """

    count_as_linear_like = True


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.reduction = en.EQLinearInter(4 * dim//4, 2*dim//4, 4, bias=False)
        # 用带 linear-like 标记的子类,保证与原版 Swin 的 linear 统计口径一致
        self.reduction = EQPatchMergingConv(2, dim//4, 2*dim//4, 4, stride=2)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # x = self.norm(x)
        # x = self.reduction(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, C*2)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, tranNum = 4,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 tranNum=tranNum,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, tranNum=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj = FConvPCA(patch_size[0], in_chans, embed_dim//tranNum, tranNum, stride=patch_size[0], first_layer=True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer_EQ_PM_Conv(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, tranNum=4, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, tranNum=tranNum)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               tranNum=tranNum,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head = EQLinearOutput(self.num_features//tranNum, self.num_classes, tranNum)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, EQLinearInter) or isinstance(m, EQLinearOutput):
            trunc_normal_(m.weights, std=.005)
            if isinstance(m, EQLinearInter) and m.c is not None:
                nn.init.constant_(m.c, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops



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
        from .test_throughput import get_swin_config, run_throughput_cli
    else:
        from test_throughput import get_swin_config, run_throughput_cli

    def build_model(model_name, img_size):
        cfg = get_swin_config(model_name)
        return SwinTransformer_EQ_PM_Conv(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            tranNum=4,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path_rate=0.0,
            **cfg,
        )

    run_throughput_cli(
        build_model=build_model,
        # EQPatchMergingConv 对应原版 Swin PatchMerging 中的 nn.Linear,一并计入 linear 口径
        target_classes=(EQLinearInter, EQLinearOutput, EQPatchMergingConv),
        tran_num=4,
        model_family="swin",
        title="Naive EQ-SwinTransformer",
    )
    print("\n\n")
