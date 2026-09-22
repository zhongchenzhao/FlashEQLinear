#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch implementations of cyclic equivariant linear layers."""

import math

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F



class EQ_linear_inter(nn.Module):
    def __init__(self, inNum, outNum, tranNum=4, bias=True, iniScale=1.0):
        super(EQ_linear_inter, self).__init__()
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.bias = bias
        self.weights = nn.Parameter(torch.Tensor(outNum, 1, inNum, tranNum), requires_grad=True)

        if bias:
            self.c = nn.Parameter(torch.Tensor(outNum, 1))
        else:
            self.register_parameter('c', None)

        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.c is not None:
            bound = 1 / math.sqrt(inNum * tranNum)
            nn.init.uniform_(self.c, -bound, bound)

    def forward(self, input):
        """
        input: (B, L, C*T)
        """
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        tempW = self.weights.repeat([1, tranNum, 1, 1])

        tempWList = [torch.cat([tempW[:, i:i + 1, :, -i:], tempW[:, i:i + 1, :, :-i]], dim=3) for i in range(tranNum)]
        tempW = torch.cat(tempWList, dim=1)

        weight = tempW.reshape([outNum * tranNum, inNum * tranNum])
        if self.bias:
            bias = self.c.repeat([1, tranNum]).reshape([1, outNum * tranNum])  # .cuda()
        else:
            bias = self.c
        return F.linear(input, weight, bias=bias)





# ==========================  ==========================

class Flash_EQ_linear_Torch(nn.Module):
    def __init__(self, inNum, outNum, eqlinear_weight=None, tranNum=4, bias=False, device="cuda"):
        super(Flash_EQ_linear_Torch, self).__init__()
        if tranNum != 4:
            raise ValueError(f"Only tranNum=4 is supported, got {tranNum}")
        self.tranNum = tranNum
        self.bias = bias

        self.register_buffer("DFT_transform_light", torch.tensor([
            [1, 1, 1, 1],
            [1, 0, -1, 0],
            [1, -1, 1, -1],
            [0, -1, 0, 1],
        ], device=device, dtype=torch.float32), persistent=False)

        self.register_buffer("iDFT_transform_light", torch.tensor([
            [0.25, 0.5, 0.25, 0],
            [0.25, 0, -0.25, -0.5],
            [0.25, -0.5, 0.25, 0],
            [0.25, 0, -0.25, 0.5],
        ], device=device, dtype=torch.float32), persistent=False)

        if eqlinear_weight is not None:
            eqlinear_weight_reverse = torch.roll(torch.flip(eqlinear_weight, dims=(-1,)), shifts=1, dims=(-1,))
            self.W = nn.Parameter(torch.einsum('kg,dcg->dck', self.DFT_transform_light,
                                                eqlinear_weight_reverse.to(self.DFT_transform_light)))
        else:
            self.W = nn.Parameter(torch.empty(outNum, inNum, tranNum, device=device))
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if bias:
            self.c = nn.Parameter(torch.zeros(outNum, 1, device=device))
        else:
            self.register_parameter("c", None)


    def forward(self, x):
        """
        input: (B, L, C, T)
        """

        x = torch.einsum('kg,blcg->blck', self.DFT_transform_light, x)  # O(bckg)=O(24xNxC1)
        y = torch.einsum('blck,dck->bldk', x[:, :, :, :3], self.W[:, :, :3])  # O(bdck)=O(3xNxC1xC2)
        y[:, :, :, 1] -= torch.einsum('blc,dc->bld', x[:, :, :, 3], self.W[:, :, 3])  # O(bdck)=O(NxC1xC2)
        y = torch.cat([y, torch.einsum('blck,dck->bld', x[:, :, :, [1, 3]], self.W[:, :, [3, 1]]).unsqueeze(-1)], dim=-1)

        y = torch.einsum('tk,bldk->bldt', self.iDFT_transform_light, y)  # O(bdtk)=O(32xNxC2)
        return y if self.c is None else y + self.c




def print_error(x_test, x_ref, name='', epsilon=1e-12):
    diff_abs = (x_test - x_ref).abs()
    diff_rel = diff_abs / (x_ref.abs() + epsilon)
    print(f"[{name}] | Abs Error: Max {diff_abs.max().item():.2e}, "
          f"Mean {diff_abs.mean().item():.2e} | "
          f"Rel Error: Mean {diff_rel.mean().item():.2e}")
    diff = (x_test - x_ref).abs()
    rel = diff / (x_ref.abs() + epsilon)
    print(f"  rel max: {rel.max().item():.4e}", f" | rel mean: {rel.mean().item():.4e}",
          f" | rel p50: {torch.quantile(rel.float(), 0.50).item():.4e}",
          f" | rel p95: {torch.quantile(rel.float(), 0.95).item():.4e}",
          f" | rel p99: {torch.quantile(rel.float(), 0.99).item():.4e}",
          f"  ||  ref max: {x_ref.abs().max().item():.4e}", f" | ref mean: {x_ref.abs().mean().item():.4e}")


if __name__ == "__main__":
    device = torch.device("cuda" )

    batch = 1
    seqlen = 32
    group = 4
    dim_in_each_group = 32
    dim_out_each_group = 32
    x = torch.rand([batch, seqlen, dim_in_each_group, group], device=device)

    x = rearrange(x, 'b l c t -> b l (c t)').contiguous()

    eq_linear = EQ_linear_inter(dim_in_each_group, dim_out_each_group, bias=False).to(device)

    print("==== Naive EQ-Linear, MACs: O(16xNxLxC1xC2) ====")
    y_naive = eq_linear(x)
    y_naive = rearrange(y_naive, 'b l (c t) -> b l c t', t=4).contiguous()
    print("y_naive", y_naive.shape, y_naive.max(), y_naive.mean())

    x = rearrange(x, 'b l (c t) -> b l c t', t=4).contiguous()

    print("==== Flash EQ-Linear (pytorch) ====")
    eqlinear_weight_ = eq_linear.weights[:, 0].contiguous()

    flash_eq_linear_pytorch = Flash_EQ_linear_TorchV2(dim_in_each_group, dim_out_each_group,
                                                    eqlinear_weight=eqlinear_weight_.clone(), tranNum=4, bias=False,
                                                    device=device).to(device)
    y_flash_pytorch = flash_eq_linear_pytorch(x)
    print("y_flash_pytorch", y_flash_pytorch.shape, y_flash_pytorch.max(), y_flash_pytorch.mean())

    print_error(x_test=y_naive, x_ref=y_flash_pytorch)

