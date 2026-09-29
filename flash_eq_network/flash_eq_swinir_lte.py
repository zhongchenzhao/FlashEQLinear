"""FlashEQLinear-accelerated C4-equivariant SwinIR + LTE-EQ.

The architecture follows Equivariant-ASISR with embed_dim=288:
  - EQ-SwinIR: dim=288, depths=(6,)*6, heads=(6,)*6, window=8,
    MLP ratio=2, C4 group, 3x3 B-Convs, no positional embedding.
  - LTE-EQ: 3x3 equivariant coefficient/frequency heads, hidden_dim=256,
    coordinate scale=0.1, local ensemble, and bilinear LR residual.

Unlike the released swinir_eq.py, every LayerNorm affine parameter is tied
over the four orientation channels. Intermediate equivariant linear maps use
the repository's FlashEQLinear CUDA kernels.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_ROTATIONS = 4
WINDOW_SIZE = 8
# EMBED_DIM = 192       # params: 3.3M
EMBED_DIM = 288         # params: 7.7M

# Direct execution also needs the repository root for the kernels package.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from kernels.flash_eqlinear_fp32.flash_EQLinear_fp32_direct_gemm_ada4090 import (
        CudaFlashEQLinearDirectGemmAda4090 as FlashEQLinearFp32,
    )
except ImportError as exc:
    FlashEQLinearFp32 = None
    _flash_eqlinear_fp32_import_error = exc

FlashEQLinear = FlashEQLinearFp32

try:
    from kernels.flash_eqlinear_fp16.flash_EQLinear_fp16_direct_gemm_release_ada4090 import (
        FlashEQLinearFp16DirectGemmAda4090 as FlashEQLinearFp16,
    )
except ImportError as exc:
    FlashEQLinearFp16 = None
    _flash_eqlinear_fp16_import_error = exc



def _trunc_normal_(tensor, std=0.02):
    if tensor.device.type == "cpu" and tensor.dtype == torch.float16:
        temporary = torch.empty_like(tensor, dtype=torch.float32)
        nn.init.trunc_normal_(temporary, std=std)
        with torch.no_grad():
            tensor.copy_(temporary)
        return tensor
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, std=std)
    with torch.no_grad():
        return tensor.normal_(0.0, std).clamp_(-2.0 * std, 2.0 * std)


def _meshgrid(*axes):
    try:
        return torch.meshgrid(*axes, indexing="ij")
    except TypeError:  # PyTorch < 1.10
        return torch.meshgrid(*axes)


def _bicubic_basis(x):
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax2 * ax
    inside_one = 1.5 * ax3 - 2.5 * ax2 + 1.0
    inside_two = -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0
    return torch.where(
        ax <= 1.0,
        inside_one,
        torch.where(ax <= 2.0, inside_two, torch.zeros_like(x)),
    )


def _rotated_filter_basis(kernel_size, num_rotations=NUM_ROTATIONS):
    if kernel_size % 2 != 1:
        raise ValueError("B-Conv kernel_size must be odd")
    radius = kernel_size // 2
    axis = torch.linspace(-1.0, 1.0, kernel_size)
    grid_y, grid_x = _meshgrid(axis, axis)
    grid_x = grid_x.unsqueeze(-1)
    grid_y = grid_y.unsqueeze(-1)

    theta = torch.arange(num_rotations, dtype=torch.float32)
    theta = theta.mul(2.0 * math.pi / num_rotations).view(1, 1, num_rotations)
    rot_x = (theta.cos() * grid_x - theta.sin() * grid_y) * radius
    rot_y = (theta.cos() * grid_y + theta.sin() * grid_x) * radius

    samples = torch.arange(-radius, radius + 1, dtype=torch.float32)
    sample_x = samples.view(1, 1, 1, kernel_size, 1)
    sample_y = samples.view(1, 1, 1, 1, kernel_size)
    basis = _bicubic_basis(rot_x[..., None, None] - sample_x)
    basis = basis * _bicubic_basis(rot_y[..., None, None] - sample_y)
    return basis.reshape(
        kernel_size, kernel_size, num_rotations, kernel_size * kernel_size
    )


class EQConv2d(nn.Module):
    """B-Conv lifting/group convolution with base-major C4 channel order."""

    def __init__(
        self,
        kernel_size,
        in_base,
        out_base,
        first_layer=False,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_base = in_base
        self.out_base = out_base
        self.expand = 1 if first_layer else NUM_ROTATIONS
        self.padding = kernel_size // 2
        basis = _rotated_filter_basis(kernel_size)
        self.register_buffer("basis", basis)
        self.weight = nn.Parameter(
            torch.empty(out_base, in_base, self.expand, basis.shape[-1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_base))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # [K,K,T,R] x [O,I,E,R] -> [O,T,I,E,K,K]
        weight = torch.einsum("ijok,mnak->monaij", self.basis, self.weight)
        step = NUM_ROTATIONS // self.expand
        blocks = []
        for shift in range(self.expand):
            block = weight[:, shift * step : (shift + 1) * step]
            block = torch.cat(
                (block[:, :, :, -shift:], block[:, :, :, :-shift]), dim=3
            )
            blocks.append(block)
        weight = torch.cat(blocks, dim=1).reshape(
            self.out_base * NUM_ROTATIONS,
            self.in_base * self.expand,
            self.kernel_size,
            self.kernel_size,
        )
        bias = None
        if self.bias is not None:
            bias = self.bias.repeat_interleave(NUM_ROTATIONS)
        return F.conv2d(x, weight, bias, padding=self.padding)


class FlashEQLinearAdapter(nn.Module):
    """FlashEQLinear with support for the decoder's flattened 2-D tokens."""

    def __init__(self, in_base, out_base, bias=True):
        super().__init__()
        self.linear = FlashEQLinear(
            in_base,
            out_base,
            tranNum=NUM_ROTATIONS,
            bias=bias,
        )

    def forward(self, x):
        if x.ndim == 2:
            return self.linear(x.unsqueeze(0)).squeeze(0)
        return self.linear(x)


class EQLinearInput(nn.Module):
    """Append the relative coordinate rotated into every C4 frame."""

    def __init__(self, in_base, out_base, coord_scale=0.1):
        super().__init__()
        self.in_base = in_base
        self.coord_scale = coord_scale
        self.linear = FlashEQLinearAdapter(in_base + 2, out_base)
        theta = -torch.arange(NUM_ROTATIONS, dtype=torch.float32)
        theta = theta.mul(2.0 * math.pi / NUM_ROTATIONS).view(
            1, 1, NUM_ROTATIONS
        )
        self.register_buffer("cos_theta", theta.cos())
        self.register_buffer("sin_theta", theta.sin())

    def forward(self, x):
        coord = x[:, -2:] * self.coord_scale
        feature = x[:, :-2].reshape(-1, self.in_base, NUM_ROTATIONS)
        coord_y = coord[:, 0].view(-1, 1, 1)
        coord_x = coord[:, 1].view(-1, 1, 1)
        rot_y = self.cos_theta * coord_y - self.sin_theta * coord_x
        rot_x = self.cos_theta * coord_x + self.sin_theta * coord_y
        lifted = torch.cat((feature, rot_y, rot_x), dim=1).reshape(x.shape[0], -1)
        return self.linear(lifted)


class EQLinearOutput(nn.Module):
    """C4 regular representation -> rotation-invariant vector."""

    def __init__(self, in_base, out_features, bias=True):
        super().__init__()
        self.in_base = in_base
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_base))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_base)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight = self.weight[:, :, None].expand(-1, -1, NUM_ROTATIONS)
        return F.linear(x, weight.reshape(self.out_features, -1), self.bias)


class EQLayerNorm(nn.Module):
    """LayerNorm with affine parameters shared over each C4 orientation orbit."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        if dim % NUM_ROTATIONS:
            raise ValueError("LayerNorm dimension must be divisible by 4")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim // NUM_ROTATIONS))
        self.bias = nn.Parameter(torch.zeros(dim // NUM_ROTATIONS))

    def forward(self, x):
        weight = self.weight.repeat_interleave(NUM_ROTATIONS)
        bias = self.bias.repeat_interleave(NUM_ROTATIONS)
        return F.layer_norm(x, (self.dim,), weight, bias, self.eps)


class DropPath(nn.Module):
    """Per-sample stochastic depth; one scalar mask preserves C4 channels."""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


def window_partition(x, window_size=WINDOW_SIZE):
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, channels
    )


def window_reverse(windows, height, width, window_size=WINDOW_SIZE):
    batch = int(windows.shape[0] / (height * width / window_size**2))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        batch, height, width, -1
    )


class EQWindowAttention(nn.Module):
    """Window MSA without absolute/relative positional embeddings."""

    def __init__(self, dim=EMBED_DIM, num_heads=6):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % NUM_ROTATIONS:
            raise ValueError("Each attention head must contain complete C4 orbits")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = FlashEQLinearAdapter(
            dim // NUM_ROTATIONS, 3 * dim // NUM_ROTATIONS
        )
        self.proj = FlashEQLinearAdapter(
            dim // NUM_ROTATIONS, dim // NUM_ROTATIONS
        )

    def forward(self, x, mask=None):
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_windows, tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + mask[None, :, None, :, :]
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(
            batch_windows, tokens, channels
        )
        return self.proj(output)


class EQMLP(nn.Module):
    def __init__(self, dim=EMBED_DIM, ratio=2.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = FlashEQLinearAdapter(
            dim // NUM_ROTATIONS, hidden // NUM_ROTATIONS
        )
        self.fc2 = FlashEQLinearAdapter(
            hidden // NUM_ROTATIONS, dim // NUM_ROTATIONS
        )

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EQSwinBlock(nn.Module):
    def __init__(self, shift_size, drop_path=0.0):
        super().__init__()
        self.shift_size = shift_size
        self.norm1 = EQLayerNorm(EMBED_DIM)
        self.attention = EQWindowAttention(EMBED_DIM, num_heads=6)
        self.norm2 = EQLayerNorm(EMBED_DIM)
        self.mlp = EQMLP(EMBED_DIM, ratio=2.0)
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def _attention_mask(height, width, shift_size, device, dtype):
        mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        slices_h = (
            slice(0, -WINDOW_SIZE),
            slice(-WINDOW_SIZE, -shift_size),
            slice(-shift_size, None),
        )
        slices_w = slices_h
        count = 0
        for region_h in slices_h:
            for region_w in slices_w:
                mask[:, region_h, region_w, :] = count
                count += 1
        mask = window_partition(mask).view(-1, WINDOW_SIZE * WINDOW_SIZE)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x, spatial_size):
        height, width = spatial_size
        batch, _, channels = x.shape
        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)
        if self.shift_size:
            x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        windows = window_partition(x).view(
            -1, WINDOW_SIZE * WINDOW_SIZE, channels
        )
        mask = None
        if self.shift_size:
            mask = self._attention_mask(
                height, width, self.shift_size, x.device, x.dtype
            )
        windows = self.attention(windows, mask)
        windows = windows.view(-1, WINDOW_SIZE, WINDOW_SIZE, channels)
        x = window_reverse(windows, height, width)
        if self.shift_size:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        x = x.reshape(batch, height * width, channels)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class EQRSTB(nn.Module):
    def __init__(self, depth, drop_paths):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EQSwinBlock(
                    shift_size=0 if index % 2 == 0 else WINDOW_SIZE // 2,
                    drop_path=drop_paths[index],
                )
                for index in range(depth)
            ]
        )
        self.conv = EQConv2d(3, EMBED_DIM // NUM_ROTATIONS, EMBED_DIM // NUM_ROTATIONS)

    def forward(self, x, spatial_size):
        residual = x
        for block in self.blocks:
            x = block(x, spatial_size)
        height, width = spatial_size
        batch = x.shape[0]
        x = x.transpose(1, 2).reshape(batch, EMBED_DIM, height, width)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return residual + x


class EQSwinIR(nn.Module):
    """Paper/released EQ-SwinIR encoder with corrected equivariant norms."""

    def __init__(self):
        super().__init__()
        depths = (6, 6, 6, 6, 6, 6)
        total_depth = sum(depths)
        drop_paths = torch.linspace(0.0, 0.1, total_depth).tolist()

        self.conv_first = EQConv2d(
            3, 3, EMBED_DIM // NUM_ROTATIONS, first_layer=True
        )
        self.patch_norm = EQLayerNorm(EMBED_DIM)
        self.layers = nn.ModuleList()
        offset = 0
        for depth in depths:
            self.layers.append(EQRSTB(depth, drop_paths[offset : offset + depth]))
            offset += depth
        self.norm = EQLayerNorm(EMBED_DIM)
        self.conv_after_body = EQConv2d(
            3, EMBED_DIM // NUM_ROTATIONS, EMBED_DIM // NUM_ROTATIONS
        )
        self.conv_out = EQConv2d(
            3, EMBED_DIM // NUM_ROTATIONS, 64 // NUM_ROTATIONS
        )
        self.activation = nn.LeakyReLU(inplace=True)
        self.out_channels = 64
        self.apply(self._init_transformer_weights)

    @staticmethod
    def _init_transformer_weights(module):
        if isinstance(module, FlashEQLinear):
            _trunc_normal_(module.weights, std=0.002)
            if module.c is not None:
                nn.init.zeros_(module.c)
        elif isinstance(module, EQLayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_size(image):
        height, width = image.shape[-2:]
        if height != width or height % WINDOW_SIZE:
            raise ValueError(
                "Exact C4 mode requires a square input whose side is divisible by 8; "
                "the paper training size is 48x48."
            )
        return height, width

    def forward_features(self, x):
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.patch_norm(x)
        for layer in self.layers:
            x = layer(x, (height, width))
        x = self.norm(x)
        return x.transpose(1, 2).reshape(-1, EMBED_DIM, height, width)

    def forward(self, image):
        self._validate_size(image)
        shallow = self.conv_first(image)
        deep = self.conv_after_body(self.forward_features(shallow)) + shallow
        return self.activation(self.conv_out(deep))


class EQLTEInput(nn.Module):
    """Build orientation-aligned LTE Fourier features without parameters."""

    def __init__(self, coord_scale=0.1):
        super().__init__()
        self.coord_scale = coord_scale
        theta = -torch.arange(NUM_ROTATIONS, dtype=torch.float32)
        theta = theta.mul(2.0 * math.pi / NUM_ROTATIONS).view(
            1, 1, NUM_ROTATIONS
        )
        self.register_buffer("cos_theta", theta.cos())
        self.register_buffer("sin_theta", theta.sin())

    def forward(self, frequency, coefficient, phase, coord):
        coord = coord * self.coord_scale
        coord_y = coord[:, 0].view(-1, 1, 1)
        coord_x = coord[:, 1].view(-1, 1, 1)
        rot_y = self.cos_theta * coord_y - self.sin_theta * coord_x
        rot_x = self.cos_theta * coord_x + self.sin_theta * coord_y
        rotated_coord = torch.cat((rot_y, rot_x), dim=1)

        samples, channels = frequency.shape
        frequency = frequency.reshape(
            samples,
            2,
            channels // (2 * NUM_ROTATIONS),
            NUM_ROTATIONS,
        )
        angle = torch.einsum("bckt,bct->bkt", frequency, rotated_coord)
        angle = angle.reshape(samples, channels // 2) + phase
        basis = torch.cat(
            (torch.cos(math.pi * angle), torch.sin(math.pi * angle)), dim=-1
        )
        return coefficient * basis


class LTEEQDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.eq_output = EQLinearOutput(64, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_rgb = nn.Linear(256, 3)

    def forward(self, x):
        x = self.eq_output(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_rgb(x)


def make_coord(shape, device=None, dtype=torch.float32):
    axes = []
    for length in shape:
        radius = 1.0 / length
        axes.append(
            torch.linspace(
                -1.0 + radius,
                1.0 - radius,
                length,
                device=device,
                dtype=dtype,
            )
        )
    return torch.stack(_meshgrid(*axes), dim=-1).reshape(-1, len(shape))


class EQSwinIRLTE(nn.Module):
    """Complete EQ-SwinIR + LTE-EQ model; instantiate with no arguments."""

    def __init__(self):
        super().__init__()
        self.encoder = EQSwinIR()
        self.coefficient_head = EQConv2d(3, 16, 64)
        self.frequency_head = EQConv2d(3, 16, 64)
        self.phase = FlashEQLinearAdapter(2, 32, bias=False)
        self.fourier_features = EQLTEInput(coord_scale=0.1)
        self.decoder = LTEEQDecoder()

    def query_rgb(self, image, coefficient, frequency, coord, cell):
        height, width = coefficient.shape[-2:]
        feature_coord = make_coord(
            (height, width), coefficient.device, coefficient.dtype
        ).transpose(0, 1).reshape(1, 2, height, width)
        feature_coord = feature_coord.expand(coefficient.shape[0], -1, -1, -1)

        predictions, areas = [], []
        for shift_y in (-1, 1):
            for shift_x in (-1, 1):
                shifted = coord.clone()
                shifted[..., 0] += shift_y / height + 1e-6
                shifted[..., 1] += shift_x / width + 1e-6
                shifted.clamp_(-1.0 + 1e-6, 1.0 - 1e-6)
                grid = shifted.flip(-1).unsqueeze(1)
                query_coefficient = F.grid_sample(
                    coefficient, grid, mode="nearest", align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                query_frequency = F.grid_sample(
                    frequency, grid, mode="nearest", align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)
                query_coord = F.grid_sample(
                    feature_coord, grid, mode="nearest", align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)

                relative_coord = coord - query_coord
                relative_coord[..., 0] *= height
                relative_coord[..., 1] *= width
                relative_cell = cell.clone()
                relative_cell[..., 0] *= height
                relative_cell[..., 1] *= width
                cell_y = relative_cell[..., 0:1].expand(
                    -1, -1, NUM_ROTATIONS
                )
                cell_x = relative_cell[..., 1:2].expand(
                    -1, -1, NUM_ROTATIONS
                )
                batch, queries = coord.shape[:2]
                flat_count = batch * queries
                cell_group = torch.cat((cell_y, cell_x), dim=-1)
                phase = self.phase(cell_group.reshape(flat_count, -1))
                decoder_input = self.fourier_features(
                    query_frequency.reshape(flat_count, -1),
                    query_coefficient.reshape(flat_count, -1),
                    phase,
                    relative_coord.reshape(flat_count, 2),
                )
                prediction = self.decoder(decoder_input).reshape(
                    batch, queries, 3
                )
                predictions.append(prediction)
                areas.append(
                    relative_coord[..., 0].abs()
                    * relative_coord[..., 1].abs()
                    + 1e-9
                )

        total_area = torch.stack(areas).sum(0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]
        output = 0.0
        for prediction, area in zip(predictions, areas):
            output = output + prediction * (area / total_area).unsqueeze(-1)
        base = F.grid_sample(
            image,
            coord.flip(-1).unsqueeze(1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        return output + base

    def forward(self, image, coord, cell):
        feature = self.encoder(image)
        coefficient = self.coefficient_head(feature)
        frequency = self.frequency_head(feature)
        return self.query_rgb(image, coefficient, frequency, coord, cell)


class EQSwinIRLTEThroughput(nn.Module):
    """Adapt the three-input ASISR model to the image-only benchmark API."""

    def __init__(self, image_size=48, scale=2):
        super().__init__()
        if image_size % WINDOW_SIZE:
            raise ValueError("image_size must be divisible by window_size=8")
        self.image_size = int(image_size)
        self.output_size = round(self.image_size * scale)
        self.model = EQSwinIRLTE()
        coord = make_coord((self.output_size, self.output_size)).unsqueeze(0)
        cell = torch.ones_like(coord) * (2.0 / self.output_size)
        self.register_buffer("coord", coord, persistent=False)
        self.register_buffer("cell", cell, persistent=False)

    def forward(self, image):
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} input, "
                f"got {tuple(image.shape[-2:])}"
            )
        batch = image.shape[0]
        coord = self.coord.expand(batch, -1, -1)
        cell = self.cell.expand(batch, -1, -1)
        return self.model(image, coord, cell)


if __name__ == "__main__":
    if __package__:
        from .test_throughput import get_vmamba_config, run_throughput_cli
    else:
        from test_throughput import get_vmamba_config, run_throughput_cli


    def build_model(model_name, image_size, precision):
        global FlashEQLinear
        if precision == "fp16":
            if FlashEQLinearFp16 is None:
                raise ImportError("FP16 FlashEQLinear kernel is unavailable.") from _flash_eqlinear_fp16_import_error
            FlashEQLinear = FlashEQLinearFp16
        else:
            if FlashEQLinearFp32 is None:
                raise ImportError("FP32 FlashEQLinear kernel is unavailable.") from _flash_eqlinear_fp32_import_error
            FlashEQLinear = FlashEQLinearFp32
        # Keep the common benchmark scale selector, while preserving the
        # paper/released EQ-SwinIR architecture fixed above.
        get_vmamba_config(model_name)
        return EQSwinIRLTEThroughput(image_size=image_size, scale=2)


    def target_classes():
        return (FlashEQLinear, EQLinearOutput, nn.Linear)


    run_throughput_cli(
        build_model=build_model,
        target_classes=target_classes,
        tran_num=NUM_ROTATIONS,
        model_family="eq-swinir-lte",
        title="EQ-SwinIR + LTE-EQ",
        default_batch_size=1,
        default_img_size=48,
        build_accepts_precision=True,
    )
