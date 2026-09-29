#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared network throughput and linear-layer profiling utilities."""

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def apply_precision(model: nn.Module, x: torch.Tensor, precision: str):
    if precision == "fp32":
        return model.float(), x.float()
    if precision == "fp16":
        if x.device.type != "cuda":
            raise ValueError("FP16 throughput benchmarking is only supported on CUDA.")
        return model.half(), x.half()
    raise ValueError(f"Unsupported precision: {precision}")


def autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        if hasattr(torch, "amp"):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def _class_name(module: nn.Module) -> str:
    return module.__class__.__name__


def _class_name_norm(module: nn.Module) -> str:
    return _class_name(module).replace("_", "").lower()


def _module_path_norm(module: nn.Module) -> str:
    return getattr(module.__class__, "__module__", "").replace("_", "").lower()


def _get_attr(module: nn.Module, names, default=None):
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


def _is_flash_eqlinear(module: nn.Module) -> bool:
    """Detect FlashEQLinear without importing the custom CUDA class."""
    name = _class_name_norm(module)
    path = _module_path_norm(module)
    return (
        ("flash" in name and "eqlinear" in name)
        or ("flash" in path and "eqlinear" in path)
    )


def _is_opaque_cuda_flash_eqlinear(module: nn.Module) -> bool:
    """
    Custom CUDA FlashEQLinear is invisible to torch.profiler FLOPs.
    PyTorch reference modules should not be added to model FLOPs, otherwise their
    internal einsum/matmul ops may be double counted by torch.profiler.
    """
    if not _is_flash_eqlinear(module):
        return False
    selected_backend_is_opaque = getattr(
        module, "uses_opaque_cuda_kernel", None
    )
    if selected_backend_is_opaque is not None:
        return bool(selected_backend_is_opaque)
    name = _class_name_norm(module)
    path = _module_path_norm(module)
    return (
        "cuda" in name
        or "gemm" in name
        or "kernel" in path
        or "kernels" in path
    )


def _is_eqlinear_output(module: nn.Module) -> bool:
    name = _class_name_norm(module)
    return name == "eqlinearoutput" or name.endswith("eqlinearoutput")


def _prod(values) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def _has_tensor_bias(module: nn.Module) -> bool:
    bias = getattr(module, "bias", None)
    return torch.is_tensor(bias)


def _is_linear_like_conv(module: nn.Module) -> bool:
    """
    数学上等价于 Linear 的卷积 (如 EQ-Swin PatchMerging 的 2x2 stride-2 卷积)。
    由模型侧通过类属性 count_as_linear_like = True 显式标记,避免误把 PatchEmbed 等普通卷积计入。
    """
    return bool(getattr(module, "count_as_linear_like", False))


def _linear_like_conv_macs(module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> int:
    """
    通用卷积 MACs = 输出元素数 * (C_in / groups) * kH * kW,输入为 NCHW。
    对 2x2 stride-2 的 PatchMerging 卷积,结果与 Linear(4C -> 2C) 的 MACs 严格相等。
    """
    k = _get_attr(module, ["size_p", "kernel_size"], 1)
    if isinstance(k, (tuple, list)):
        kh, kw = int(k[0]), int(k[1])
    else:
        kh = kw = int(k)
    groups = int(getattr(module, "groups", 1))
    c_in = int(x.shape[1])
    return int(output.numel()) * (c_in // groups) * kh * kw


@dataclass
class LinearLikeRecord:
    name: str
    layer_type: str
    calls: int = 0
    macs: int = 0
    flops: int = 0
    extra_flops: int = 0
    bias_flops: int = 0
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    tran_num: int = 1
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None


class LinearLikeFLOPsProfiler:
    """
    Count FLOPs of linear-like layers by forward hooks.

    Supported layers:
      1. nn.Linear
      2. FlashEQLinear custom CUDA kernels
      3. EQLinearOutput classifier head used by Flash EQ-ViT
      4. Linear-like convolutions marked with count_as_linear_like = True
         (EQ-Swin PatchMerging), only when include_linear_like_conv=True

    For FlashEQLinear, the FLOPs follow the provided PyTorch reference
    Flash_EQ_linear_TorchV2:
      - input transform: 8 * B * L * C add FLOPs
      - 6 GEMMs of shape C -> D: 6 * B * L * C * D MACs
      - y1/y3 combine: 2 * B * L * D add FLOPs
      - iDFT scalings: 4 * B * L * D mul FLOPs
      - iDFT combines: 8 * B * L * D add FLOPs

    Therefore, with 1 MAC = 2 FLOPs:
      FlashEQLinear FLOPs = 2 * 6 * B * L * C * D + 8 * B * L * C + 14 * B * L * D
    """

    def __init__(
        self,
        model: nn.Module,
        mac_as_flops: bool = True,
        count_bias: bool = True,
        include_nn_linear: bool = True,
        include_flash_eqlinear: bool = True,
        include_eqlinear_output: bool = True,
        only_opaque_cuda_flash_eqlinear: bool = False,
        default_tran_num: int = 4,
        include_linear_like_conv: bool = False,
    ):
        # include_linear_like_conv 默认关闭: measure_model_flops_per_image 中卷积已由
        # torch.profiler 统计,若在此再计一次会重复计数;只在统计 linear-like 口径时打开。
        self.model = model
        self.mac_as_flops = mac_as_flops
        self.count_bias = count_bias
        self.include_nn_linear = include_nn_linear
        self.include_flash_eqlinear = include_flash_eqlinear
        self.include_eqlinear_output = include_eqlinear_output
        self.include_linear_like_conv = include_linear_like_conv
        self.only_opaque_cuda_flash_eqlinear = only_opaque_cuda_flash_eqlinear
        self.default_tran_num = int(default_tran_num)
        self.handles: List[Any] = []
        self.records: Dict[str, LinearLikeRecord] = {}
        self.name_map = {module: name for name, module in self.model.named_modules()}

    def _should_hook(self, module: nn.Module) -> bool:
        if self.include_nn_linear and isinstance(module, nn.Linear):
            return True
        if self.include_flash_eqlinear and _is_flash_eqlinear(module):
            if self.only_opaque_cuda_flash_eqlinear:
                return _is_opaque_cuda_flash_eqlinear(module)
            return True
        if self.include_eqlinear_output and _is_eqlinear_output(module):
            return True
        if self.include_linear_like_conv and _is_linear_like_conv(module):
            return True
        return False

    def _get_record(self, module: nn.Module, layer_type: str) -> LinearLikeRecord:
        name = self.name_map.get(module, "unknown")
        rec = self.records.get(name)
        if rec is None:
            rec = LinearLikeRecord(name=name, layer_type=layer_type)
            self.records[name] = rec
        return rec

    def _add_record(
        self,
        module: nn.Module,
        layer_type: str,
        macs: int,
        flops: int,
        extra_flops: int,
        bias_flops: int,
        in_features: int,
        out_features: int,
        tran_num: int,
        x: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        rec = self._get_record(module, layer_type)
        rec.calls += 1
        rec.macs += int(macs)
        rec.flops += int(flops)
        rec.extra_flops += int(extra_flops)
        rec.bias_flops += int(bias_flops)
        rec.in_features = int(in_features)
        rec.out_features = int(out_features)
        rec.tran_num = int(tran_num)
        rec.input_shape = tuple(x.shape)
        rec.output_shape = tuple(output.shape)

    def _profile_nn_linear(self, module: nn.Linear, x: torch.Tensor, output: torch.Tensor) -> None:
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        macs = int(output.numel()) * in_features
        flops = 2 * macs if self.mac_as_flops else macs
        bias_flops = 0
        if self.count_bias and module.bias is not None:
            bias_flops = int(output.numel())
            flops += bias_flops
        self._add_record(
            module=module,
            layer_type="nn.Linear",
            macs=macs,
            flops=flops,
            extra_flops=0,
            bias_flops=bias_flops,
            in_features=in_features,
            out_features=out_features,
            tran_num=1,
            x=x,
            output=output,
        )

    def _infer_flash_eq_shape(self, module: nn.Module, x: torch.Tensor, output: torch.Tensor):
        t = int(_get_attr(module, ["tran_num", "tranNum", "t"], self.default_tran_num))

        if x.ndim >= 4 and int(x.shape[-1]) == t:
            # PyTorch reference layout: (..., C, T) -> (..., D, T)
            batch_items = _prod(x.shape[:-2])
            in_num = int(x.shape[-2])
            if output.ndim >= 4 and int(output.shape[-1]) == t:
                out_num = int(output.shape[-2])
            else:
                out_num = int(output.shape[-1]) // t
        else:
            # Custom CUDA layer in Flash EQ-ViT is called with flattened C*T.
            batch_items = int(x.numel()) // int(x.shape[-1])
            in_num = int(x.shape[-1]) // t
            out_num = int(output.shape[-1]) // t

        # Prefer explicit base dimensions when the CUDA module exposes them.
        explicit_in = _get_attr(
            module,
            ["in_num", "inNum", "in_features_base", "base_in_features"],
            None,
        )
        explicit_out = _get_attr(
            module,
            ["out_num", "outNum", "out_features_base", "base_out_features"],
            None,
        )
        if explicit_in is not None:
            in_num = int(explicit_in)
        if explicit_out is not None:
            out_num = int(explicit_out)

        return batch_items, in_num, out_num, t

    def _profile_flash_eqlinear(self, module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
        batch_items, in_num, out_num, t = self._infer_flash_eq_shape(module, x, output)

        # Flash_EQ_linear_TorchV2 has 6 C->D GEMMs.
        macs = 6 * batch_items * in_num * out_num

        # Input DFT adds + y1/y3 combines + iDFT multiplications + iDFT adds.
        input_dft_flops = 8 * batch_items * in_num
        y_combine_flops = 2 * batch_items * out_num
        idft_scale_flops = 4 * batch_items * out_num
        idft_add_flops = 8 * batch_items * out_num
        extra_flops = input_dft_flops + y_combine_flops + idft_scale_flops + idft_add_flops

        flops = (2 * macs if self.mac_as_flops else macs) + extra_flops

        # The provided PyTorch reference keeps `bias` but does not add it in forward.
        # Count bias only when the actual CUDA module exposes a Tensor/Parameter bias.
        bias_flops = 0
        if self.count_bias and _has_tensor_bias(module):
            bias_flops = int(output.numel())
            flops += bias_flops

        self._add_record(
            module=module,
            layer_type="FlashEQLinear",
            macs=macs,
            flops=flops,
            extra_flops=extra_flops,
            bias_flops=bias_flops,
            in_features=in_num,
            out_features=out_num,
            tran_num=t,
            x=x,
            output=output,
        )

    def _profile_eqlinear_output(self, module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
        t = int(_get_attr(module, ["tran_num", "tranNum", "t"], self.default_tran_num))
        in_num = int(_get_attr(module, ["in_num", "inNum"], int(x.shape[-1]) // t))
        out_num = int(_get_attr(module, ["out_num", "outNum"], int(output.shape[-1])))
        batch_items = int(x.numel()) // int(x.shape[-1])

        reduce_add_flops = batch_items * in_num * max(t - 1, 0)
        macs = batch_items * in_num * out_num
        flops = (2 * macs if self.mac_as_flops else macs) + reduce_add_flops

        bias_flops = 0
        if self.count_bias and _has_tensor_bias(module):
            bias_flops = int(output.numel())
            flops += bias_flops

        self._add_record(
            module=module,
            layer_type="EQLinearOutput",
            macs=macs,
            flops=flops,
            extra_flops=reduce_add_flops,
            bias_flops=bias_flops,
            in_features=in_num,
            out_features=out_num,
            tran_num=t,
            x=x,
            output=output,
        )

    def _profile_linear_like_conv(self, module: nn.Module, x: torch.Tensor, output: torch.Tensor) -> None:
        macs = _linear_like_conv_macs(module, x, output)
        flops = 2 * macs if self.mac_as_flops else macs
        bias_flops = 0
        if self.count_bias and _has_tensor_bias(module):
            bias_flops = int(output.numel())
            flops += bias_flops
        self._add_record(
            module=module,
            layer_type="LinearLikeConv",
            macs=macs,
            flops=flops,
            extra_flops=0,
            bias_flops=bias_flops,
            in_features=int(x.shape[1]),
            out_features=int(output.shape[1]),
            tran_num=int(_get_attr(module, ["tran_num", "tranNum", "t"], self.default_tran_num)),
            x=x,
            output=output,
        )

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(output):
            return

        if isinstance(module, nn.Linear):
            self._profile_nn_linear(module, x, output)
        elif _is_flash_eqlinear(module):
            self._profile_flash_eqlinear(module, x, output)
        elif _is_eqlinear_output(module):
            self._profile_eqlinear_output(module, x, output)
        elif _is_linear_like_conv(module):
            self._profile_linear_like_conv(module, x, output)

    def start(self) -> None:
        self.records = {}
        self.handles = []
        for module in self.model.modules():
            if self._should_hook(module):
                self.handles.append(module.register_forward_hook(self._hook))

    def stop(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.no_grad()
    def profile(self, x: torch.Tensor) -> float:
        self.model.eval()
        batch_size = int(x.shape[0])

        self.start()
        try:
            _ = self.model(x)
            _sync_if_cuda(x.device)
        finally:
            self.stop()

        return sum(r.flops for r in self.records.values()) / batch_size

    def total_flops_batch(self) -> int:
        return int(sum(r.flops for r in self.records.values()))


@torch.no_grad()
def measure_model_flops_per_image(
    model: nn.Module,
    x: torch.Tensor,
    mac_as_flops: bool = True,
    count_bias: bool = True,
    warmup_cache: bool = True,
) -> float:
    """
    Return model FLOPs / image.

    This uses torch.profiler(with_flops=True) for regular PyTorch ops and
    manually adds opaque CUDA FlashEQLinear FLOPs because torch.profiler usually
    cannot infer FLOPs for custom CUDA kernels.

    No printing is performed inside this function.
    """
    model.eval()
    device = x.device
    batch_size = int(x.shape[0])

    if warmup_cache:
        _ = model(x)
        _sync_if_cuda(device)

    custom_profiler = LinearLikeFLOPsProfiler(
        model=model,
        mac_as_flops=mac_as_flops,
        count_bias=count_bias,
        include_nn_linear=False,
        include_flash_eqlinear=True,
        include_eqlinear_output=False,
        only_opaque_cuda_flash_eqlinear=True,
    )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    custom_profiler.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = model(x)
        _sync_if_cuda(device)
    except Exception:
        custom_profiler.stop()
        return -1.0
    finally:
        # stop() is idempotent for our use here because handles are cleared.
        if custom_profiler.handles:
            custom_profiler.stop()

    profiler_flops = 0
    for evt in prof.key_averages():
        profiler_flops += int(evt.flops or 0)

    manual_custom_flops = custom_profiler.total_flops_batch()
    return float(profiler_flops + manual_custom_flops) / batch_size


@torch.no_grad()
def measure_all_linear_flops_per_image(
    model: nn.Module,
    x: torch.Tensor,
    mac_as_flops: bool = True,
    count_bias: bool = True,
) -> float:
    """
    Return FLOPs / image for all linear-like layers.

    For plain ViT this counts all nn.Linear modules.
    For Flash EQ-ViT this additionally counts FlashEQLinear and EQLinearOutput.

    No printing is performed inside this function.
    """
    profiler = LinearLikeFLOPsProfiler(
        model=model,
        mac_as_flops=mac_as_flops,
        count_bias=count_bias,
        include_nn_linear=True,
        include_flash_eqlinear=True,
        include_eqlinear_output=True,
        only_opaque_cuda_flash_eqlinear=False,
        include_linear_like_conv=True,
    )
    return float(profiler.profile(x))


# Backward-compatible alias for old scripts.
measure_flops = measure_model_flops_per_image

# ================ ================


@torch.no_grad()
def measure_latency_and_throughput(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int = 20,
    runs: int = 100,
    use_amp: bool = False,
):
    model.eval()
    device = x.device

    for _ in range(warmup):
        with autocast_context(device, use_amp):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(runs):
            with autocast_context(device, use_amp):
                _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
        avg_ms = total_ms / runs
    else:
        start = time.perf_counter()
        for _ in range(runs):
            with autocast_context(device, use_amp):
                _ = model(x)
        end = time.perf_counter()
        avg_ms = (end - start) * 1000.0 / runs

    throughput = x.shape[0] * 1000.0 / avg_ms
    return avg_ms, throughput



class UnifiedLinearLikeProfiler:
    """
    Profile nn.Linear / EQLinearInter / FlashEQLinear 等 Linear-like 层的：
    1. MACs
    2. FLOPs
    3. forward latency
    4. 占整个模型 forward latency 的比例

    默认 FLOPs 只统计主矩阵乘法:
        MACs = batch_tokens * effective_in * effective_out

    latency 统计整个 module.forward()，所以：
        nn.Linear:        主要是 Linear GEMM
        EQLinearInter:    repeat/cat/reshape + F.linear
        FlashEQLinear:    custom CUDA kernel forward
        LinearLikeConv:   带 count_as_linear_like 标记的卷积 (EQ-Swin PatchMerging),
                          对应原版 Swin PatchMerging 的 nn.Linear(4C -> 2C),
                          MACs = 输出元素数 * C_in * kH * kW
    """

    def __init__(
        self,
        model,
        target_classes,
        tran_num=4,
        use_cuda_event=True,
        mac_as_flops=True,
        count_bias=True,
    ):
        """
        Args:
            model: 待 profile 的模型
            target_classes:
                一个类或多个类，例如：
                    nn.Linear
                    EQLinearInter
                    FlashEQLinear
                    (nn.Linear, EQLinearInter, FlashEQLinear)
            tran_num: EQ/FlashEQ 默认 C4，所以是 4
            use_cuda_event: CUDA 上建议 True
            mac_as_flops:
                True  -> 1 MAC = 2 FLOPs
                False -> 1 MAC = 1 FLOP/MAC
            count_bias: 是否统计 bias add FLOPs
        """
        self.model = model

        if not isinstance(target_classes, tuple):
            target_classes = (target_classes,)
        self.target_classes = target_classes

        self.tran_num = tran_num
        self.use_cuda_event = use_cuda_event and torch.cuda.is_available()
        self.mac_as_flops = mac_as_flops
        self.count_bias = count_bias

        self.records = {}
        self.handles = []

        self.name_map = {
            module: name
            for name, module in self.model.named_modules()
        }

    def _get_module_name(self, module):
        return self.name_map.get(module, "unknown")

    @staticmethod
    def _get_attr(module, names, default=None):
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        return default

    def _is_plain_linear(self, module):
        return isinstance(module, nn.Linear)

    def _has_eq_attrs(self, module):
        has_in = any(hasattr(module, n) for n in ["in_num", "inNum"])
        has_out = any(hasattr(module, n) for n in ["out_num", "outNum"])
        has_t = any(hasattr(module, n) for n in ["tran_num", "tranNum"])
        return has_in and has_out and has_t

    def _infer_layer_type(self, module):
        if isinstance(module, nn.Linear):
            return "nn.Linear"

        # 必须在 _has_eq_attrs 之前判断: FConvPCA 同样带 in_num/out_num/tran_num 属性
        if _is_linear_like_conv(module):
            return "LinearLikeConv"

        cls_name = module.__class__.__name__.lower()
        if "flash" in cls_name and "eq" in cls_name:
            return "FlashEQLinear"

        if cls_name.replace("_", "").endswith("eqlinearoutput"):
            return "EQLinearOutput"

        if self._has_eq_attrs(module):
            return "EQLinear"

        return module.__class__.__name__

    def _infer_effective_shape(self, module, x, output):
        """
        返回:
            layer_type, in_num, out_num, tran_num, effective_in, effective_out

        对三类层统一处理：

        1. nn.Linear:
            effective_in = module.in_features
            effective_out = module.out_features

        2. EQLinearInter:
            in_num = module.in_num
            out_num = module.out_num
            t = module.tran_num
            effective_in = in_num * t
            effective_out = out_num * t

        3. FlashEQLinear:
            优先读 module 属性
            如果没有属性，则从 input/output shape 反推
        """
        layer_type = self._infer_layer_type(module)

        # -------- 普通 nn.Linear --------
        if isinstance(module, nn.Linear):
            effective_in = int(module.in_features)
            effective_out = int(module.out_features)
            t = 1
            in_num = effective_in
            out_num = effective_out
            return layer_type, in_num, out_num, t, effective_in, effective_out

        # -------- EQ / FlashEQ Linear-like --------
        t = self._get_attr(
            module,
            ["tran_num", "tranNum", "t"],
            self.tran_num,
        )
        t = int(t)

        in_num = self._get_attr(
            module,
            ["in_num", "inNum", "in_features_base", "base_in_features"],
            None,
        )
        out_num = self._get_attr(
            module,
            ["out_num", "outNum", "out_features_base", "base_out_features"],
            None,
        )

        # 有些 FlashEQLinear 可能把 effective dim 存成 in_features/out_features
        in_features = self._get_attr(module, ["in_features"], None)
        out_features = self._get_attr(module, ["out_features"], None)

        # EQ-VMamba calls EQLinear through its NCHW/1x1-convolution path,
        # whereas EQ-ViT calls it with the packed channel on the last axis.
        # Prefer axis 1 only when it matches the declared packed dimensions.
        input_effective_in = int(x.shape[-1])
        output_effective_out = int(output.shape[-1])
        if x.ndim == 4 and output.ndim == 4 and in_num is not None and out_num is not None:
            declared_in = int(in_num) * t
            declared_out = int(out_num) * t
            if int(x.shape[1]) == declared_in and int(output.shape[1]) == declared_out:
                input_effective_in = int(x.shape[1])
                output_effective_out = int(output.shape[1])

        # 优先使用 base in/out
        if in_num is not None:
            in_num = int(in_num)
            effective_in = in_num * t
        elif in_features is not None:
            in_features = int(in_features)
            if in_features == input_effective_in:
                effective_in = in_features
                in_num = effective_in // t
            elif in_features * t == input_effective_in:
                in_num = in_features
                effective_in = input_effective_in
            else:
                effective_in = input_effective_in
                in_num = effective_in // t
        else:
            effective_in = input_effective_in
            in_num = effective_in // t

        if out_num is not None:
            out_num = int(out_num)
            effective_out = out_num * t
        elif out_features is not None:
            out_features = int(out_features)
            if out_features == output_effective_out:
                effective_out = out_features
                out_num = effective_out // t
            elif out_features * t == output_effective_out:
                out_num = out_features
                effective_out = output_effective_out
            else:
                effective_out = output_effective_out
                out_num = effective_out // t
        else:
            effective_out = output_effective_out
            out_num = effective_out // t

        # 最终以真实 input/output shape 为准，防止属性名不一致
        effective_in = input_effective_in
        effective_out = output_effective_out

        return layer_type, int(in_num), int(out_num), int(t), int(effective_in), int(effective_out)

    def _has_bias(self, module):
        bias = self._get_attr(module, ["bias", "b"], None)
        if isinstance(bias, bool):
            return bias
        if bias is not None:
            return True
        return getattr(module, "c", None) is not None

    def _pre_hook(self, module, inputs):
        if self.use_cuda_event:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            module.__unified_profiler_start_event__ = start_event
            module.__unified_profiler_end_event__ = end_event
        else:
            module.__unified_profiler_start_time__ = time.perf_counter()

    def _post_hook(self, module, inputs, output):
        name = self._get_module_name(module)
        x = inputs[0]

        (
            layer_type,
            in_num,
            out_num,
            t,
            effective_in,
            effective_out,
        ) = self._infer_effective_shape(module, x, output)

        if layer_type == "FlashEQLinear":
            # C4 real-frequency factorization: six C->D GEMMs plus the fused
            # input/output transforms. Counting it as a dense 4C->4D layer was
            # why the VMamba Flash rows previously reported impossible FLOPs.
            batch_tokens = x.numel() // (in_num * t)
            macs = 6 * batch_tokens * in_num * out_num
            flops = 2 * macs if self.mac_as_flops else macs
            flops += 8 * batch_tokens * in_num
            flops += 14 * batch_tokens * out_num
            if self.count_bias and self._has_bias(module):
                flops += batch_tokens * out_num * t
        elif layer_type == "EQLinearOutput":
            batch_tokens = x.numel() // (in_num * t)
            macs = batch_tokens * in_num * out_num
            flops = 2 * macs if self.mac_as_flops else macs
            flops += batch_tokens * in_num * max(t - 1, 0)
            if self.count_bias and self._has_bias(module):
                flops += batch_tokens * out_num
        elif layer_type == "LinearLikeConv":
            # 按卷积的真实输出尺寸计数 (stride-2 下输出 token 数为输入的 1/4)
            macs = _linear_like_conv_macs(module, x, output)
            flops = 2 * macs if self.mac_as_flops else macs
            if self.count_bias and self._has_bias(module):
                flops += int(output.numel())
        else:
            # nn.Linear and Naive EQLinear execute the dense effective map.
            batch_tokens = x.numel() // effective_in
            macs = batch_tokens * effective_in * effective_out
            flops = 2 * macs if self.mac_as_flops else macs
            if self.count_bias and self._has_bias(module):
                flops += batch_tokens * effective_out

        if self.use_cuda_event:
            end_event = module.__unified_profiler_end_event__
            end_event.record()

            # 注意：这里为了拿到每层准确 event 时间，会同步。
            # 这会引入 profiling overhead，但适合做逐层占比分析。
            torch.cuda.synchronize()

            latency_ms = module.__unified_profiler_start_event__.elapsed_time(end_event)
        else:
            latency_ms = (
                time.perf_counter() - module.__unified_profiler_start_time__
            ) * 1000.0

        if name not in self.records:
            self.records[name] = {
                "layer_type": layer_type,
                "calls": 0,
                "macs": 0,
                "flops": 0,
                "latency_ms": 0.0,
                "in_num": in_num,
                "out_num": out_num,
                "tran_num": t,
                "effective_in": effective_in,
                "effective_out": effective_out,
                "input_shape": tuple(x.shape),
                "output_shape": tuple(output.shape),
            }

        self.records[name]["calls"] += 1
        self.records[name]["macs"] += macs
        self.records[name]["flops"] += flops
        self.records[name]["latency_ms"] += latency_ms

    def start(self):
        self.records = {}
        self.handles = []

        for module in self.model.modules():
            if isinstance(module, self.target_classes):
                self.handles.append(module.register_forward_pre_hook(self._pre_hook))
                self.handles.append(module.register_forward_hook(self._post_hook))

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summary(
        self,
        total_forward_latency_ms=None,
        topk=None,
        batch_size=1,
        sort_by="latency_ms",
        print_target_ratio=True,
    ):
        items = list(self.records.items())

        if sort_by == "flops":
            items = sorted(items, key=lambda x: x[1]["flops"], reverse=True)
        elif sort_by == "macs":
            items = sorted(items, key=lambda x: x[1]["macs"], reverse=True)
        else:
            items = sorted(items, key=lambda x: x[1]["latency_ms"], reverse=True)

        show_items = items if topk is None else items[:topk]

        total_macs = sum(v["macs"] for v in self.records.values())
        total_flops = sum(v["flops"] for v in self.records.values())
        total_latency = sum(v["latency_ms"] for v in self.records.values())

        # Keep the detailed per-layer table available for debugging.
        # It is intentionally disabled by default because the paper table only
        # needs the aggregate linear-like cost and latency.
        if topk is not None and topk > 0:
            print("=" * 140)
            print(
                f"{'name':28s} "
                f"{'type':12s} "
                f"{'calls':>6s} "
                f"{'MACs(G)':>12s} "
                f"{'FLOPs(G)':>12s} "
                f"{'latency(ms)':>12s} "
                f"{'eff shape':>12s} "
                f"{'input':>20s} "
                f"{'output':>20s}"
            )
            print("=" * 140)

            for name, v in show_items:
                effective_shape = f"{v['effective_in']}->{v['effective_out']}"
                print(
                    f"{name:28s} "
                    f"{v['layer_type']:12s} "
                    f"{v['calls']:6d} "
                    f"{v['macs'] / 1e9:12.4f} "
                    f"{v['flops'] / 1e9:12.4f} "
                    f"{v['latency_ms']:12.4f} "
                    f"{effective_shape:>12s} "
                    f"{str(v['input_shape']):>20s} "
                    f"{str(v['output_shape']):>20s}"
                )

        total_latency_per_image = total_latency / batch_size
        total_macs_per_image = total_macs / batch_size
        total_flops_per_image = total_flops / batch_size

        print("=" * 70)
        print(f"Total Linear MACs:       {total_macs_per_image / 1e9:.4f} GMACs/image")
        print(f"Total Linear FLOPs:      {total_flops_per_image / 1e9:.4f} GFLOPs/image")
        print(f"Total Linear latency:    {total_latency_per_image:.4f} ms/image")
        print()

        result = {
            "total_macs": total_macs_per_image,
            "total_flops": total_flops_per_image,
            "total_latency_ms": total_latency_per_image,
            "latency_ratio": None,
            "records": self.records,
        }

        if print_target_ratio and total_forward_latency_ms is not None:
            total_forward_latency_per_image = total_forward_latency_ms / batch_size
            ratio = total_latency_per_image / total_forward_latency_per_image
            result["latency_ratio"] = ratio
            result["ratio_forward_latency_ms_per_image"] = total_forward_latency_per_image
            print(f"Target latency ratio:    {ratio * 100:.2f}%")

        print("=" * 70)

        return result



@torch.no_grad()
def profile_linear_like_average(
    model,
    x,
    target_classes,
    tran_num=4,
    warmup=50,
    runs=100,
    use_amp=False,
    mac_as_flops=True,
    count_bias=True,
    topk=None,
    sort_by="latency_ms",
    total_forward_latency_ms_for_ratio=None,
    print_target_ratio=True,
):
    """
    Profile linear-like layers and print the aggregate block used by the paper table.

    Important:
        The forward pass inside this function contains hook/event profiling overhead.
        Therefore, do NOT use its profiled forward latency as the main model
        throughput unless you explicitly want a profiling-instrumented number.

        Pass `total_forward_latency_ms_for_ratio` from a clean model benchmark if
        you want the target latency ratio to be computed against the real
        end-to-end forward latency.
    """
    batch_size = x.shape[0]
    model.eval()
    device = x.device

    for _ in range(warmup):
        with autocast_context(device, use_amp):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    profiler = UnifiedLinearLikeProfiler(
        model=model,
        target_classes=target_classes,
        tran_num=tran_num,
        use_cuda_event=device.type == "cuda",
        mac_as_flops=mac_as_flops,
        count_bias=count_bias,
    )

    profiler.start()

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(runs):
            with autocast_context(device, use_amp):
                _ = model(x)
        end.record()

        torch.cuda.synchronize()
        total_latency_ms = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(runs):
            with autocast_context(device, use_amp):
                _ = model(x)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

    profiler.stop()

    profiled_avg_forward_latency_ms = total_latency_ms / runs

    # profiler.records is accumulated over `runs`; convert it to one forward pass.
    for v in profiler.records.values():
        v["macs"] /= runs
        v["flops"] /= runs
        v["latency_ms"] /= runs

    ratio_latency_ms = (
        total_forward_latency_ms_for_ratio
        if total_forward_latency_ms_for_ratio is not None
        else profiled_avg_forward_latency_ms
    )

    result = profiler.summary(
        total_forward_latency_ms=ratio_latency_ms,
        topk=topk,
        batch_size=batch_size,
        sort_by=sort_by,
        print_target_ratio=print_target_ratio,
    )

    result["profiled_avg_forward_latency_ms"] = profiled_avg_forward_latency_ms
    result["profiled_avg_forward_latency_ms_per_image"] = (
        profiled_avg_forward_latency_ms / batch_size
    )
    return result



def get_vit_config(model_name: str):
    configs = {
        # "tiny": dict(patch_size=16, embed_dim=240, depth=12, num_heads=3),
        "tiny": dict(patch_size=16, embed_dim=384, depth=12, num_heads=3),
        "small": dict(patch_size=16, embed_dim=480, depth=12, num_heads=6),
        "base": dict(patch_size=16, embed_dim=768, depth=12, num_heads=6),
        # "base": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        "large": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        "huge": dict(patch_size=16, embed_dim=1280, depth=32, num_heads=16),
    }
    return dict(configs[model_name])


def get_swin_config(model_name: str):
    configs = {
        "tiny": dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
        "small": dict(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
        "base": dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
        "large": dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
        "huge": dict(embed_dim=352, depths=[2, 2, 18, 2], num_heads=[11, 22, 44, 88]),
    }
    return dict(configs[model_name])


def get_vmamba_config(model_name: str):
    """Return the shared Plain/EQ-VMamba scale used by throughput scripts."""
    configs = {
        "tiny": dict(embed_dim=96, depths=[2, 2, 8, 2], drop_path_rate=0.2),
        "small": dict(embed_dim=96, depths=[2, 2, 20, 2], drop_path_rate=0.3),
        "base": dict(embed_dim=128, depths=[2, 2, 20, 2], drop_path_rate=0.5),
        "large": dict(embed_dim=192, depths=[2, 2, 20, 2], drop_path_rate=0.6),
        "huge": dict(embed_dim=352, depths=[2, 2, 20, 2], drop_path_rate=0.7),
    }
    config = dict(configs[model_name])
    config["depths"] = list(config["depths"])
    return config


@torch.no_grad()
def benchmark_model_throughput(
    model: nn.Module,
    x: torch.Tensor,
    target_classes,
    tran_num: int,
    precision: str = "fp32",
    warmup: int = 20,
    runs: int = 100,
    linear_warmup: int = 50,
    linear_runs: int = 100,
    ratio_source: str = "clean",
):
    """
    Benchmark one model and print one consistent throughput number.

    ratio_source:
        "clean"    -> Target latency ratio uses the clean end-to-end forward
                      latency measured without profiling hooks. This is the
                      recommended/default setting.
        "profiled" -> Target latency ratio uses the hook-instrumented forward
                      latency from profile_linear_like_average. This reproduces
                      the old ratio style, but its throughput is not the clean
                      model throughput.
    """
    if ratio_source not in ("clean", "profiled"):
        raise ValueError("ratio_source must be either 'clean' or 'profiled'.")

    model.eval()
    model, x = apply_precision(model, x, precision)
    batch_size = int(x.shape[0])
    device = x.device

    y = model(x)
    _sync_if_cuda(device)

    params = count_params(model)
    model_flops = measure_model_flops_per_image(model=model, x=x)
    avg_ms, throughput = measure_latency_and_throughput(
        model,
        x,
        warmup=warmup,
        runs=runs,
        use_amp=False,
    )

    print(f"device: {device}")
    print(f"input: {tuple(x.shape)},   y: {tuple(y.shape)}")
    print(f"params: {params / 1e6:.3f} M")

    ratio_latency_ms = avg_ms if ratio_source == "clean" else None
    result_linear = profile_linear_like_average(
        model=model,
        x=x,
        target_classes=target_classes,
        tran_num=tran_num,
        warmup=linear_warmup,
        runs=linear_runs,
        use_amp=False,
        mac_as_flops=True,
        count_bias=True,
        total_forward_latency_ms_for_ratio=ratio_latency_ms,
        print_target_ratio=True,
    )

    print()
    if model_flops >= 0:
        print(f"Model FLOPs / image: {model_flops / 1e9:.3f} GFLOPs")
    else:
        print("Model FLOPs / image: unavailable")
    print(f"Average forward latency: {avg_ms / batch_size:.4f} ms / image")
    print(f"Throughput: {throughput:.2f} images/s")

    result_linear["model_flops"] = model_flops
    result_linear["avg_forward_latency_ms"] = avg_ms
    result_linear["avg_forward_latency_ms_per_image"] = avg_ms / batch_size
    result_linear["throughput"] = throughput
    result_linear["ratio_source"] = ratio_source
    return result_linear



def run_throughput_cli(
    build_model,
    target_classes,
    tran_num: int,
    model_family: str,
    title: str,
    default_batch_size: int = 128,
    build_accepts_precision: bool = False,
    default_img_size: int = 224,
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "--scale", dest="model",
        choices=("tiny", "small", "base", "large", "huge"),
        default="tiny",
        help="Model scale. --scale is an alias of --model.",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--img-size", type=int, default=default_img_size)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--linear-warmup", type=int, default=50)
    parser.add_argument("--linear-runs", type=int, default=100)
    parser.add_argument(
        "--ratio-source",
        choices=("clean", "profiled"),
        default="clean",
        help=(
            "clean: compute Target latency ratio against the normal end-to-end "
            "forward latency; profiled: use hook-instrumented latency, matching "
            "the old output style."
        ),
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if build_accepts_precision:
        model = build_model(args.model, args.img_size, args.precision).to(device).eval()
    else:
        model = build_model(args.model, args.img_size).to(device).eval()
    resolved_target_classes = (
        target_classes()
        if callable(target_classes) and not isinstance(target_classes, type)
        else target_classes
    )
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

    print(f"================ {title}-{args.model} ({args.precision}) ================")
    return benchmark_model_throughput(
        model=model,
        x=x,
        target_classes=resolved_target_classes,
        tran_num=tran_num,
        precision=args.precision,
        warmup=args.warmup,
        runs=args.runs,
        linear_warmup=args.linear_warmup,
        linear_runs=args.linear_runs,
        ratio_source=args.ratio_source,
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "--scale", dest="model",
        choices=("tiny", "small", "base", "large", "huge"),
        default="small",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--linear-warmup", type=int, default=50)
    parser.add_argument("--linear-runs", type=int, default=100)
    parser.add_argument("--ratio-source", choices=("clean", "profiled"), default="clean")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    if __package__:
        from .eq_vit import EQLinearInter, EQLinearOutput, VisionTransformer_eq
    else:
        from eq_vit import EQLinearInter, EQLinearOutput, VisionTransformer_eq

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cfg = get_vit_config(args.model)

    vit_base = VisionTransformer_eq(
        img_size=args.img_size,
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        num_classes=args.num_classes,
    ).to(device).eval()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    benchmark_model_throughput(
        model=vit_base,
        x=x,
        target_classes=(EQLinearInter, EQLinearOutput),
        tran_num=4,
        precision=args.precision,
        warmup=args.warmup,
        runs=args.runs,
        linear_warmup=args.linear_warmup,
        linear_runs=args.linear_runs,
        ratio_source=args.ratio_source,
    )


if __name__ == "__main__":
    main()
