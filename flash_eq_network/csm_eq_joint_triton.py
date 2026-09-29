"""C4 joint cross-scan/cross-merge with an optional Triton CUDA path.

The public ``*_fn`` functions operate on the packed channel-first layout used
by EQ-VMamba: ``(B, C * 4, H, W)``.  The Torch reference functions retain the
older explicit-group layout so they can also be used for parity checks.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cross_scan_eq_joint_torch(xs: torch.Tensor) -> torch.Tensor:
    """Reference joint scan: ``(B, 4, C, H, W) -> (B, 4, 4C, HW)``."""
    if xs.ndim != 5:
        raise ValueError(f"expected a 5-D tensor, got shape {tuple(xs.shape)}")
    batch, tran_num, dim, height, width = xs.shape
    if tran_num != 4:
        raise ValueError(f"joint C4 scan requires tranNum=4, got {tran_num}")

    seqlen = height * width
    ys = xs.new_empty((batch, 4, 4 * dim, seqlen))
    for path in range(4):
        path_x = torch.roll(xs, shifts=-path, dims=1).reshape(
            batch, 4 * dim, height, width
        )
        if path == 1:
            path_x = torch.rot90(path_x, k=1, dims=(-2, -1))
        elif path == 2:
            path_x = path_x.flatten(-2).flip(-1)
        elif path == 3:
            path_x = torch.rot90(path_x, k=-1, dims=(-2, -1))
        ys[:, path] = path_x.reshape(batch, 4 * dim, seqlen)
    return ys


def cross_merge_eq_joint_torch(
    ys: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Reference adjoint merge: ``(B, 4, 4C, HW) -> (B, 4, C, H, W)``."""
    if ys.ndim != 4:
        raise ValueError(f"expected a 4-D tensor, got shape {tuple(ys.shape)}")
    batch, tran_num, dim_total, seqlen = ys.shape
    height, width = int(height), int(width)
    if tran_num != 4 or dim_total % 4:
        raise ValueError(
            f"joint C4 merge expects shape (B, 4, 4C, L), got {tuple(ys.shape)}"
        )
    if seqlen != height * width:
        raise ValueError(f"sequence length {seqlen} != {height} * {width}")

    dim = dim_total // 4
    y0 = ys[:, 0].reshape(batch, 4, dim, height, width)

    y1 = ys[:, 1].reshape(batch, dim_total, width, height)
    y1 = torch.rot90(y1, k=-1, dims=(-2, -1)).reshape(
        batch, 4, dim, height, width
    )
    y1 = torch.roll(y1, shifts=1, dims=1)

    y2 = ys[:, 2].flip(-1).reshape(batch, 4, dim, height, width)
    y2 = torch.roll(y2, shifts=2, dims=1)

    y3 = ys[:, 3].reshape(batch, dim_total, width, height)
    y3 = torch.rot90(y3, k=1, dims=(-2, -1)).reshape(
        batch, 4, dim, height, width
    )
    y3 = torch.roll(y3, shifts=3, dims=1)
    return y0 + y1 + y2 + y3


def _cross_scan_eq_joint_packed_torch(
    x: torch.Tensor, tran_num: int = 4
) -> torch.Tensor:
    batch, packed_channels, height, width = x.shape
    if tran_num != 4 or packed_channels % tran_num:
        raise ValueError(
            f"joint C4 scan expects channels divisible by 4, got {packed_channels}"
        )
    channels = packed_channels // tran_num
    grouped = x.reshape(batch, channels, tran_num, height, width).permute(
        0, 2, 1, 3, 4
    )
    return cross_scan_eq_joint_torch(grouped)


def _cross_merge_eq_joint_packed_torch(
    y: torch.Tensor, height: int, width: int, tran_num: int = 4
) -> torch.Tensor:
    if tran_num != 4:
        raise ValueError(f"joint C4 merge requires tranNum=4, got {tran_num}")
    merged = cross_merge_eq_joint_torch(y, height, width)
    batch, _, channels, _, _ = merged.shape
    return merged.permute(0, 2, 1, 3, 4).reshape(
        batch, channels * tran_num, height, width
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _cross_scan_eq_joint_kernel(
        x_ptr,
        y_ptr,
        n_elements: tl.constexpr,
        channels: tl.constexpr,
        packed_channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        seqlen: tl.constexpr,
        stride_xb: tl.constexpr,
        stride_xc: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_xw: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Packed (B, C*4, H, W) -> (B, 4, C*4, H*W)."""
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        l_idx = offsets % seqlen
        packed_idx = (offsets // seqlen) % packed_channels
        path_idx = (offsets // (seqlen * packed_channels)) % 4
        batch_idx = offsets // (seqlen * packed_channels * 4)

        group_idx = packed_idx // channels
        channel_idx = packed_idx % channels
        source_group = (group_idx + path_idx) & 3
        source_channel = channel_idx * 4 + source_group

        h0 = l_idx // width
        w0 = l_idx % width
        h1 = l_idx % height
        w1 = width - 1 - (l_idx // height)
        reverse_l = seqlen - 1 - l_idx
        h2 = reverse_l // width
        w2 = reverse_l % width
        h3 = height - 1 - (l_idx % height)
        w3 = l_idx // height

        source_h = tl.where(
            path_idx == 0,
            h0,
            tl.where(path_idx == 1, h1, tl.where(path_idx == 2, h2, h3)),
        )
        source_w = tl.where(
            path_idx == 0,
            w0,
            tl.where(path_idx == 1, w1, tl.where(path_idx == 2, w2, w3)),
        )

        x_offset = (
            batch_idx * stride_xb
            + source_channel * stride_xc
            + source_h * stride_xh
            + source_w * stride_xw
        )
        value = tl.load(x_ptr + x_offset, mask=mask)
        tl.store(y_ptr + offsets, value, mask=mask)


    @triton.jit
    def _cross_merge_eq_joint_kernel(
        y_ptr,
        x_ptr,
        n_elements: tl.constexpr,
        channels: tl.constexpr,
        packed_channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        seqlen: tl.constexpr,
        stride_yb: tl.constexpr,
        stride_yk: tl.constexpr,
        stride_yc: tl.constexpr,
        stride_yl: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Adjoint merge: (B, 4, C*4, HW) -> packed (B, C*4, H, W)."""
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        w_idx = offsets % width
        h_idx = (offsets // width) % height
        packed_idx = (offsets // seqlen) % packed_channels
        batch_idx = offsets // (seqlen * packed_channels)

        target_group = packed_idx & 3
        channel_idx = packed_idx // 4

        group0 = target_group
        group1 = (target_group - 1) & 3
        group2 = (target_group - 2) & 3
        group3 = (target_group - 3) & 3
        packed0 = group0 * channels + channel_idx
        packed1 = group1 * channels + channel_idx
        packed2 = group2 * channels + channel_idx
        packed3 = group3 * channels + channel_idx

        l0 = h_idx * width + w_idx
        l1 = (width - 1 - w_idx) * height + h_idx
        l2 = seqlen - 1 - l0
        l3 = w_idx * height + (height - 1 - h_idx)

        base = batch_idx * stride_yb
        value0 = tl.load(
            y_ptr + base + packed0 * stride_yc + l0 * stride_yl, mask=mask
        ).to(tl.float32)
        value1 = tl.load(
            y_ptr
            + base
            + stride_yk
            + packed1 * stride_yc
            + l1 * stride_yl,
            mask=mask,
        ).to(tl.float32)
        value2 = tl.load(
            y_ptr
            + base
            + 2 * stride_yk
            + packed2 * stride_yc
            + l2 * stride_yl,
            mask=mask,
        ).to(tl.float32)
        value3 = tl.load(
            y_ptr
            + base
            + 3 * stride_yk
            + packed3 * stride_yc
            + l3 * stride_yl,
            mask=mask,
        ).to(tl.float32)
        tl.store(x_ptr + offsets, value0 + value1 + value2 + value3, mask=mask)


    def _launch_cross_scan_eq_joint(x: torch.Tensor) -> torch.Tensor:
        batch, packed_channels, height, width = map(int, x.shape)
        if packed_channels % 4:
            raise ValueError(f"channels must be divisible by 4, got {packed_channels}")
        channels = packed_channels // 4
        seqlen = height * width
        y = torch.empty(
            (batch, 4, packed_channels, seqlen), device=x.device, dtype=x.dtype
        )
        if y.numel() == 0:
            return y
        block_size = 256
        n_elements = int(y.numel())
        strides = tuple(int(stride) for stride in x.stride())
        grid = (triton.cdiv(n_elements, block_size),)
        with torch.cuda.device(x.device):
            _cross_scan_eq_joint_kernel[grid](
                x,
                y,
                n_elements=n_elements,
                channels=channels,
                packed_channels=packed_channels,
                height=height,
                width=width,
                seqlen=seqlen,
                stride_xb=strides[0],
                stride_xc=strides[1],
                stride_xh=strides[2],
                stride_xw=strides[3],
                block_size=block_size,
            )
        return y


    def _launch_cross_merge_eq_joint(
        y: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        batch, tran_num, packed_channels, seqlen = map(int, y.shape)
        height, width = int(height), int(width)
        if tran_num != 4 or packed_channels % 4:
            raise ValueError(
                f"expected shape (B, 4, 4C, L), got {tuple(y.shape)}"
            )
        if seqlen != height * width:
            raise ValueError(f"sequence length {seqlen} != {height} * {width}")
        channels = packed_channels // 4
        x = torch.empty(
            (batch, packed_channels, height, width), device=y.device, dtype=y.dtype
        )
        if x.numel() == 0:
            return x
        block_size = 256
        n_elements = int(x.numel())
        strides = tuple(int(stride) for stride in y.stride())
        grid = (triton.cdiv(n_elements, block_size),)
        with torch.cuda.device(y.device):
            _cross_merge_eq_joint_kernel[grid](
                y,
                x,
                n_elements=n_elements,
                channels=channels,
                packed_channels=packed_channels,
                height=height,
                width=width,
                seqlen=seqlen,
                stride_yb=strides[0],
                stride_yk=strides[1],
                stride_yc=strides[2],
                stride_yl=strides[3],
                block_size=block_size,
            )
        return x


    class _CrossScanEQJointTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            ctx.height = int(x.shape[-2])
            ctx.width = int(x.shape[-1])
            return _launch_cross_scan_eq_joint(x)

        @staticmethod
        def backward(ctx, grad_y: torch.Tensor):
            return _launch_cross_merge_eq_joint(grad_y, ctx.height, ctx.width)


    class _CrossMergeEQJointTriton(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, y: torch.Tensor, height: int, width: int
        ) -> torch.Tensor:
            ctx.height = int(height)
            ctx.width = int(width)
            return _launch_cross_merge_eq_joint(y, height, width)

        @staticmethod
        def backward(ctx, grad_x: torch.Tensor):
            return _launch_cross_scan_eq_joint(grad_x), None, None


def _triton_compatible(x: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.layout == torch.strided
        and x.dtype in _SUPPORTED_DTYPES
        and all(stride >= 0 for stride in x.stride())
    )


def cross_scan_eq_joint_fn(
    x: torch.Tensor, tran_num: int = 4, force_torch: bool = False
) -> torch.Tensor:
    """Fast packed joint scan with an autograd-enabled Triton path."""
    if x.ndim != 4:
        raise ValueError(f"expected (B, C*4, H, W), got {tuple(x.shape)}")
    if tran_num != 4:
        raise ValueError(f"joint C4 scan requires tranNum=4, got {tran_num}")
    if _triton_compatible(x) and not force_torch:
        return _CrossScanEQJointTriton.apply(x)
    return _cross_scan_eq_joint_packed_torch(x, tran_num)


def cross_merge_eq_joint_fn(
    y: torch.Tensor,
    height: int,
    width: int,
    tran_num: int = 4,
    force_torch: bool = False,
) -> torch.Tensor:
    """Fast packed adjoint merge with an autograd-enabled Triton path."""
    if y.ndim != 4:
        raise ValueError(f"expected (B, 4, C*4, L), got {tuple(y.shape)}")
    if tran_num != 4:
        raise ValueError(f"joint C4 merge requires tranNum=4, got {tran_num}")
    if _triton_compatible(y) and not force_torch:
        return _CrossMergeEQJointTriton.apply(y, int(height), int(width))
    return _cross_merge_eq_joint_packed_torch(y, height, width, tran_num)


__all__ = [
    "cross_scan_eq_joint_fn",
    "cross_merge_eq_joint_fn",
    "cross_scan_eq_joint_torch",
    "cross_merge_eq_joint_torch",
]
