import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


TRAN_NUM = 4

DFT_TRANSFORM_LIGHT = torch.tensor(
    [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, -1.0, 0.0],
        [1.0, -1.0, 1.0, -1.0],
        [0.0, -1.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

IDFT_TRANSFORM_LIGHT = torch.tensor(
    [
        [0.25, 0.5, 0.25, 0.0],
        [0.25, 0.0, -0.25, -0.5],
        [0.25, -0.5, 0.25, 0.0],
        [0.25, 0.0, -0.25, 0.5],
    ],
    dtype=torch.float32,
)


def _check_last_dim_is_4(name: str, tensor: torch.Tensor) -> None:
    if tensor.shape[-1] != TRAN_NUM:
        raise ValueError(f"{name}.shape[-1] must be {TRAN_NUM}, got {tensor.shape}")


def _like(constant: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return constant.to(device=ref.device, dtype=ref.dtype)


def spatial_to_freq_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply the same 4-point DFT used in the CUDA forward kernel.

    x shape: (..., in_channels, 4)
    return shape: (..., in_channels, 4)
    """
    _check_last_dim_is_4("x", x)
    x0, x1, x2, x3 = x.unbind(dim=-1)
    return torch.stack(
        [
            x0 + x1 + x2 + x3,
            x0 - x2,
            x0 - x1 + x2 - x3,
            -x1 + x3,
        ],
        dim=-1,
    )


def freq_to_spatial_activation(x_freq: torch.Tensor) -> torch.Tensor:
    """Apply the same 4-point inverse DFT used in the CUDA forward kernel."""
    _check_last_dim_is_4("x_freq", x_freq)
    f0, f1, f2, f3 = x_freq.unbind(dim=-1)
    t0 = 0.25 * (f0 + f2)
    t1 = 0.25 * (f0 - f2)
    t2 = 0.5 * f1
    t3 = 0.5 * f3
    return torch.stack(
        [
            t0 + t2,
            t1 - t3,
            t0 - t2,
            t1 + t3,
        ],
        dim=-1,
    )


def output_grad_to_freq(grad_y: torch.Tensor) -> torch.Tensor:
    """Map dY from spatial domain to the frequency-domain representation used by CUDA."""
    _check_last_dim_is_4("grad_y", grad_y)
    g0, g1, g2, g3 = grad_y.unbind(dim=-1)
    s0 = g0 + g2
    s1 = g0 - g2
    s2 = g1 + g3
    s3 = g3 - g1
    return torch.stack(
        [
            0.25 * (s0 + s2),
            0.5 * s1,
            0.25 * (s0 - s2),
            0.5 * s3,
        ],
        dim=-1,
    )


def freq_grad_to_input_spatial(grad_x_freq: torch.Tensor) -> torch.Tensor:
    """Map dX from frequency-domain accumulators back to the original 4-value layout."""
    _check_last_dim_is_4("grad_x_freq", grad_x_freq)
    g0, g1, g2, g3 = grad_x_freq.unbind(dim=-1)
    t0 = g0 + g2
    t1 = g0 - g2
    return torch.stack(
        [
            t0 + g1,
            t1 - g3,
            t0 - g1,
            t1 + g3,
        ],
        dim=-1,
    )


def spatial_weight_to_freq_weight(spatial_weight: torch.Tensor) -> torch.Tensor:
    """Match the weight preprocessing in `flash_EQLinear_fp32.py`.

    spatial_weight shape: (out_channels, in_channels, 4)
    return shape:         (out_channels, in_channels, 4)
    """
    _check_last_dim_is_4("spatial_weight", spatial_weight)
    spatial_weight_reversed = torch.roll(
        torch.flip(spatial_weight, dims=(-1,)),
        shifts=1,
        dims=(-1,),
    )
    dft = _like(DFT_TRANSFORM_LIGHT, spatial_weight)
    return torch.einsum("kg,dcg->dck", dft, spatial_weight_reversed).contiguous()


def freq_grad_to_spatial_weight(freq_grad: torch.Tensor) -> torch.Tensor:
    """Map the gradient of stored frequency weights back to spatial weights."""
    _check_last_dim_is_4("freq_grad", freq_grad)
    dft = _like(DFT_TRANSFORM_LIGHT, freq_grad)
    grad_reversed = torch.einsum("kg,dck->dcg", dft, freq_grad)
    return torch.roll(torch.flip(grad_reversed, dims=(-1,)), shifts=1, dims=(-1,))


def _pack_linear_core_input(x_freq: torch.Tensor) -> torch.Tensor:
    x0, x1, x2, x3 = x_freq.unbind(dim=-1)
    return torch.cat([x0, x1, x2, x1, x3, x3], dim=-1)


def _build_linear_core_weight(weight: torch.Tensor) -> torch.Tensor:
    w0, w1, w2, w3 = weight.unbind(dim=-1)
    return torch.block_diag(w0, w1, w2, w3, -w3, w1).contiguous()


def _linear_core_forward_from_packed(x_linear: torch.Tensor, weight_linear: torch.Tensor) -> torch.Tensor:
    return F.linear(x_linear, weight_linear)


def _unpack_linear_core_output(y_linear: torch.Tensor, out_num: int) -> torch.Tensor:
    y_parts = y_linear.reshape(*y_linear.shape[:-1], 6, out_num)
    y0 = y_parts[..., 0, :]
    y1 = y_parts[..., 1, :] + y_parts[..., 4, :]
    y2 = y_parts[..., 2, :]
    y3 = y_parts[..., 3, :] + y_parts[..., 5, :]
    return torch.stack([y0, y1, y2, y3], dim=-1)


def _pack_linear_core_grad_output(grad_y_freq: torch.Tensor) -> torch.Tensor:
    gy0, gy1, gy2, gy3 = grad_y_freq.unbind(dim=-1)
    return torch.cat([gy0, gy1, gy2, gy3, gy1, gy3], dim=-1)


def flash_eq_linear_forward_python(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Readable forward reference for `flash_EQLinear_cuda_fp32.cu`.

    x shape:      (..., in_channels, 4)
    weight shape: (out_channels, in_channels, 4) in frequency domain
    return shape: (..., out_channels, 4)
    """
    _check_last_dim_is_4("x", x)
    _check_last_dim_is_4("weight", weight)

    x_freq = spatial_to_freq_activation(x)
    x_linear = _pack_linear_core_input(x_freq)
    weight_linear = _build_linear_core_weight(weight)
    y_linear = _linear_core_forward_from_packed(x_linear, weight_linear)
    y_freq = _unpack_linear_core_output(y_linear, weight.size(0))
    return freq_to_spatial_activation(y_freq)


def flash_eq_linear_backward_dx_python(
    grad_y: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Reference dX for the CUDA backward kernel.

    grad_y shape: (..., out_channels, 4)
    weight shape: (out_channels, in_channels, 4) in frequency domain
    return shape: (..., in_channels, 4)
    """
    _check_last_dim_is_4("grad_y", grad_y)
    _check_last_dim_is_4("weight", weight)
    x_dummy = torch.zeros(*grad_y.shape[:-2], weight.shape[1], TRAN_NUM, device=grad_y.device, dtype=grad_y.dtype)
    grad_x, _ = flash_eq_linear_backward_python(grad_y, x_dummy, weight)
    return grad_x


def flash_eq_linear_backward_dw_python(
    grad_y: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Reference dW for the CUDA backward kernel.

    grad_y shape: (..., out_channels, 4)
    x shape:      (..., in_channels, 4)
    return shape: (out_channels, in_channels, 4) in frequency domain
    """
    _check_last_dim_is_4("grad_y", grad_y)
    _check_last_dim_is_4("x", x)
    weight_dummy = torch.zeros(grad_y.shape[-2], x.shape[-2], TRAN_NUM, device=x.device, dtype=x.dtype)
    _, grad_weight = flash_eq_linear_backward_python(grad_y, x, weight_dummy)
    return grad_weight


def flash_eq_linear_backward_python(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference backward that matches the CUDA extension outputs."""
    _check_last_dim_is_4("grad_y", grad_y)
    _check_last_dim_is_4("x", x)
    _check_last_dim_is_4("weight", weight)

    with torch.enable_grad():
        x_leaf = x.detach().requires_grad_(True)
        weight_leaf = weight.detach().requires_grad_(True)
        y = flash_eq_linear_forward_python(x_leaf, weight_leaf)
        grad_x, grad_weight = torch.autograd.grad(y, (x_leaf, weight_leaf), grad_outputs=grad_y)
    return grad_x, grad_weight


class FlashEQLinearPythonRef(nn.Module):
    """Readable pure-PyTorch reference for the fp32 CUDA kernel.

    By default the parameter is stored in the same frequency-domain layout as the
    CUDA kernel. If you already have a spatial-domain EQ weight, pass it through
    `eqlinear_weight`.
    """

    def __init__(
        self,
        in_num: int,
        out_num: int,
        tran_num: int = TRAN_NUM,
        bias: bool = False,
        eqlinear_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if tran_num != TRAN_NUM:
            raise ValueError(f"This reference only supports tran_num={TRAN_NUM}, got {tran_num}")

        self.in_num = in_num
        self.out_num = out_num
        self.tran_num = tran_num
        self.use_bias = bias

        if eqlinear_weight is not None:
            if eqlinear_weight.shape != (out_num, in_num, TRAN_NUM):
                raise ValueError(
                    "eqlinear_weight must have shape "
                    f"({out_num}, {in_num}, {TRAN_NUM}), got {tuple(eqlinear_weight.shape)}"
                )
            weight = spatial_weight_to_freq_weight(eqlinear_weight)
        else:
            weight = torch.empty(out_num, in_num, TRAN_NUM)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_num, 1))
            bound = 1.0 / math.sqrt(in_num * tran_num)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Support both packed and unpacked inputs.

        unpacked input: (..., in_channels, 4) -> (..., out_channels, 4)
        packed input:   (..., in_channels * 4) -> (..., out_channels * 4)
        """
        if x.dim() >= 2 and x.shape[-1] == self.tran_num and x.shape[-2] == self.in_num:
            y = flash_eq_linear_forward_python(
                x,
                self.weight.to(device=x.device, dtype=x.dtype),
            )
            if self.bias is not None:
                y = y + self.bias.to(device=y.device, dtype=y.dtype).transpose(0, 1).unsqueeze(-1)
            return y

        if x.shape[-1] == self.in_num * self.tran_num:
            x_unpacked = x.reshape(*x.shape[:-1], self.in_num, self.tran_num)
            y = flash_eq_linear_forward_python(
                x_unpacked,
                self.weight.to(device=x.device, dtype=x.dtype),
            )
            if self.bias is not None:
                y = y + self.bias.to(device=y.device, dtype=y.dtype).transpose(0, 1).unsqueeze(-1)
            return y.reshape(*x.shape[:-1], self.out_num * self.tran_num)

        raise ValueError(
            "Expected input shape (..., in_channels, 4) or (..., in_channels * 4), "
            f"got {tuple(x.shape)}"
        )
