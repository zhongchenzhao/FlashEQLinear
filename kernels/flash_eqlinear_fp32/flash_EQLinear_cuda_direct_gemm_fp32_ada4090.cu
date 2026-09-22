#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ==========================================
// 前向与 dX 优化参数 (2D Register Tiling)
// ==========================================
#define FWD_BLOCK_N 128
#define FWD_BLOCK_D 32
#define FWD_BLOCK_C 16
#define TILE_N 8
#define TILE_D 2

#define DX_BLOCK_N 128
#define DX_BLOCK_C 32
#define DX_BLOCK_D 16

// ==========================================
// dW 反向优化参数 (2D Split-K Reduction)
// ==========================================
#define DW_BLOCK_D 32
#define DW_BLOCK_C 32
#define DW_BLOCK_N 32

struct DirectShapeKey {
    int64_t N;
    int64_t inNum;
    int64_t outNum;

    bool operator==(const DirectShapeKey& other) const {
        return N == other.N && inNum == other.inNum && outNum == other.outNum;
    }
};

struct DirectShapeKeyHash {
    size_t operator()(const DirectShapeKey& key) const noexcept {
        size_t h1 = std::hash<int64_t>{}(key.N);
        size_t h2 = std::hash<int64_t>{}(key.inNum);
        size_t h3 = std::hash<int64_t>{}(key.outNum);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::mutex g_direct_runtime_mutex;
std::unordered_map<DirectShapeKey, int, DirectShapeKeyHash> g_direct_best_splitk_cache;
int g_direct_dw_mode_override = -1;

// ---------------------------------------------------------
// Helper: float4 Atomic Add for Split-K Optimization
// ---------------------------------------------------------
__device__ __forceinline__ void atomicAddFloat4(float4* address, float4 val) {
    float* f_addr = reinterpret_cast<float*>(address);
    atomicAdd(f_addr + 0, val.x);
    atomicAdd(f_addr + 1, val.y);
    atomicAdd(f_addr + 2, val.z);
    atomicAdd(f_addr + 3, val.w);
}

__global__ void flash_eq_linear_empty_kernel() {}

__device__ __forceinline__ float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

int default_split_k_for_shape(int N) {
    int split_k = 64;
    if (N <= 1024) split_k = 1;
    else if (N <= 4096) split_k = 8;
    else if (N <= 16384) split_k = 32;
    return split_k;
}

bool use_two_pass_dw_for_shape(int N, int split_k) {
    if (g_direct_dw_mode_override == 0) {
        return false;
    }
    if (g_direct_dw_mode_override == 1) {
        return true;
    }
    return split_k > 1 && N > 4096;
}

int get_cached_split_k(int N, int inNum, int outNum) {
    std::lock_guard<std::mutex> lock(g_direct_runtime_mutex);
    auto it = g_direct_best_splitk_cache.find(DirectShapeKey{N, inNum, outNum});
    if (it != g_direct_best_splitk_cache.end()) {
        return it->second;
    }
    return default_split_k_for_shape(N);
}

void set_cached_split_k(int N, int inNum, int outNum, int split_k) {
    std::lock_guard<std::mutex> lock(g_direct_runtime_mutex);
    g_direct_best_splitk_cache[DirectShapeKey{N, inNum, outNum}] = split_k;
}

void clear_cached_split_k() {
    std::lock_guard<std::mutex> lock(g_direct_runtime_mutex);
    g_direct_best_splitk_cache.clear();
}

py::dict get_direct_cache_stats() {
    std::lock_guard<std::mutex> lock(g_direct_runtime_mutex);
    py::dict result;
    result["splitk_cache_size"] = static_cast<int64_t>(g_direct_best_splitk_cache.size());
    result["dw_mode_override"] = g_direct_dw_mode_override;
    return result;
}

void set_direct_dw_mode_override(int mode) {
    std::lock_guard<std::mutex> lock(g_direct_runtime_mutex);
    g_direct_dw_mode_override = mode;
}

__global__ void flash_eq_linear_forward_legacy_kernel(
    const float4* __restrict__ X,
    const float4* __restrict__ W,
    float4* __restrict__ Y,
    int N, int inNum, int outNum
) {
    __shared__ float4 s_X[FWD_BLOCK_N][FWD_BLOCK_C + 1];
    __shared__ float4 s_W[FWD_BLOCK_C][FWD_BLOCK_D + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int n_base = by * FWD_BLOCK_N;
    int d_base = bx * FWD_BLOCK_D;

    float acc0[TILE_N][TILE_D] = {0}, acc1[TILE_N][TILE_D] = {0};
    float acc2[TILE_N][TILE_D] = {0}, acc3[TILE_N][TILE_D] = {0};

    for (int c_start = 0; c_start < inNum; c_start += FWD_BLOCK_C) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / FWD_BLOCK_C;
            int col_c = linear_idx % FWD_BLOCK_C;
            int load_n = n_base + row_n;
            int load_c = c_start + col_c;

            float4 x_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_c < inNum) {
                x_val = __ldg(&X[load_n * inNum + load_c]);
            }

            float vx = x_val.x, vy = x_val.y, vz = x_val.z, vw = x_val.w;
            float s0 = vx + vz;
            float s1 = vx - vz;
            float s2 = vy + vw;
            float s3 = vw - vy;

            float4 x_f;
            x_f.x = s0 + s2;
            x_f.y = s1;
            x_f.z = s0 - s2;
            x_f.w = s3;

            s_X[row_n][col_c] = x_f;
        }

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int linear_idx = tid + i * 256;
            int row_d = linear_idx / FWD_BLOCK_C;
            int col_c = linear_idx % FWD_BLOCK_C;
            int load_d = d_base + row_d;
            int load_c = c_start + col_c;

            float4 w_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_d < outNum && load_c < inNum) {
                w_val = __ldg(&W[load_d * inNum + load_c]);
            }
            s_W[col_c][row_d] = w_val;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < FWD_BLOCK_C; ++k) {
            float4 w_frag[TILE_D];
            w_frag[0] = s_W[k][tx * 2 + 0];
            w_frag[1] = s_W[k][tx * 2 + 1];

            float4 x_frag[TILE_N];
            #pragma unroll
            for (int i = 0; i < TILE_N; ++i) x_frag[i] = s_X[ty * 8 + i][k];

            #pragma unroll
            for (int j = 0; j < TILE_D; ++j) {
                #pragma unroll
                for (int i = 0; i < TILE_N; ++i) {
                    acc0[i][j] += x_frag[i].x * w_frag[j].x;
                    acc1[i][j] += x_frag[i].y * w_frag[j].y - x_frag[i].w * w_frag[j].w;
                    acc2[i][j] += x_frag[i].z * w_frag[j].z;
                    acc3[i][j] += x_frag[i].y * w_frag[j].w + x_frag[i].w * w_frag[j].y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        #pragma unroll
        for (int i = 0; i < TILE_N; ++i) {
            int out_n = n_base + ty * 8 + i;
            int out_d = d_base + tx * 2 + j;
            if (out_n < N && out_d < outNum) {
                float a0 = acc0[i][j], a1 = acc1[i][j], a2 = acc2[i][j], a3 = acc3[i][j];
                float t0 = 0.25f * (a0 + a2);
                float t1 = 0.25f * (a0 - a2);
                float t2 = 0.5f * a1;
                float t3 = 0.5f * a3;

                float4 y_out;
                y_out.x = t0 + t2;
                y_out.y = t1 - t3;
                y_out.z = t0 - t2;
                y_out.w = t1 + t3;

                Y[out_n * outNum + out_d] = y_out;
            }
        }
    }
}

// ---------------------------------------------------------
// 1. 极致 2D Tiling 前向传播
// ---------------------------------------------------------
__global__ __launch_bounds__(256, 2) void flash_eq_linear_forward_pack6_kernel(
    const float4* __restrict__ X,
    const float4* __restrict__ W,
    const float* __restrict__ bias,
    float4* __restrict__ Y,
    int N, int inNum, int outNum, bool fuse_bias
) {
    __shared__ float4 s_X[FWD_BLOCK_N][FWD_BLOCK_C + 1];
    __shared__ float4 s_W[FWD_BLOCK_C][FWD_BLOCK_D + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int n_base = by * FWD_BLOCK_N;
    int d_base = bx * FWD_BLOCK_D;

    float acc0[TILE_N][TILE_D] = {0}, acc1[TILE_N][TILE_D] = {0};
    float acc2[TILE_N][TILE_D] = {0}, acc3[TILE_N][TILE_D] = {0};

    for (int c_start = 0; c_start < inNum; c_start += FWD_BLOCK_C) {
        // --- 协同加载 X ---
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / FWD_BLOCK_C;
            int col_c = linear_idx % FWD_BLOCK_C;
            int load_n = n_base + row_n;
            int load_c = c_start + col_c;

            float4 x_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_c < inNum) {
                x_val = __ldg(&X[load_n * inNum + load_c]);
            }

            // ALU 优化版的频域转换
            float vx = x_val.x, vy = x_val.y, vz = x_val.z, vw = x_val.w;
            float s0 = vx + vz;
            float s1 = vx - vz;
            float s2 = vy + vw;
            float s3 = vw - vy;

            float4 x_f;
            x_f.x = s0 + s2;
            x_f.y = s1;
            x_f.z = s0 - s2;
            x_f.w = s3;

            s_X[row_n][col_c] = x_f;
        }

        // --- 协同加载 W ---
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int linear_idx = tid + i * 256;
            int row_d = linear_idx / FWD_BLOCK_C;
            int col_c = linear_idx % FWD_BLOCK_C;
            int load_d = d_base + row_d;
            int load_c = c_start + col_c;

            float4 w_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_d < outNum && load_c < inNum) {
                w_val = __ldg(&W[load_d * inNum + load_c]);
            }
            s_W[col_c][row_d] = w_val;
        }
        __syncthreads();

        // --- 2D 寄存器块计算 ---
        #pragma unroll
        for (int k = 0; k < FWD_BLOCK_C; ++k) {
            float4 w_frag[TILE_D];
            w_frag[0] = s_W[k][tx * 2 + 0];
            w_frag[1] = s_W[k][tx * 2 + 1];

            float4 x_frag[TILE_N];
            #pragma unroll
            for (int i = 0; i < TILE_N; ++i) x_frag[i] = s_X[ty * 8 + i][k];

            #pragma unroll
            for (int j = 0; j < TILE_D; ++j) {
                #pragma unroll
                for (int i = 0; i < TILE_N; ++i) {
                    acc0[i][j] += x_frag[i].x * w_frag[j].x;
                    acc1[i][j] += x_frag[i].y * w_frag[j].y - x_frag[i].w * w_frag[j].w;
                    acc2[i][j] += x_frag[i].z * w_frag[j].z;
                    acc3[i][j] += x_frag[i].y * w_frag[j].w + x_frag[i].w * w_frag[j].y;
                }
            }
        }
        __syncthreads();
    }

    // --- iDFT 优化并写回 ---
    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        #pragma unroll
        for (int i = 0; i < TILE_N; ++i) {
            int out_n = n_base + ty * 8 + i;
            int out_d = d_base + tx * 2 + j;
            if (out_n < N && out_d < outNum) {
                float a0 = acc0[i][j], a1 = acc1[i][j], a2 = acc2[i][j], a3 = acc3[i][j];
                float t0 = 0.25f * (a0 + a2);
                float t1 = 0.25f * (a0 - a2);
                float t2 = 0.5f * a1;
                float t3 = 0.5f * a3;

                float4 y_out;
                y_out.x = t0 + t2;
                y_out.y = t1 - t3;
                y_out.z = t0 - t2;
                y_out.w = t1 + t3;

                if (fuse_bias) {
                    float b = bias[out_d];
                    y_out.x += b;
                    y_out.y += b;
                    y_out.z += b;
                    y_out.w += b;
                }

                Y[out_n * outNum + out_d] = y_out;
            }
        }
    }
}

// ---------------------------------------------------------
// 2. 极致 2D Tiling dX 反向传播
// ---------------------------------------------------------
__global__ __launch_bounds__(256, 2) void flash_eq_linear_backward_dx_kernel_v3(
    const float4* __restrict__ dY,
    const float4* __restrict__ W,
    float4* __restrict__ dX,
    int N, int inNum, int outNum
) {
    __shared__ float4 s_dY[DX_BLOCK_N][DX_BLOCK_D + 1];
    __shared__ float4 s_W[DX_BLOCK_D][DX_BLOCK_C + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int n_base = by * DX_BLOCK_N;
    int c_base = bx * DX_BLOCK_C;

    float acc0[TILE_N][TILE_D]={0}, acc1[TILE_N][TILE_D]={0};
    float acc2[TILE_N][TILE_D]={0}, acc3[TILE_N][TILE_D]={0};

    for (int d_start = 0; d_start < outNum; d_start += DX_BLOCK_D) {
        // --- 协同加载 dY ---
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / DX_BLOCK_D;
            int col_d = linear_idx % DX_BLOCK_D;
            int load_n = n_base + row_n;
            int load_d = d_start + col_d;

            float4 dy_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_d < outNum) {
                dy_val = __ldg(&dY[load_n * outNum + load_d]);
            }

            // ALU 优化
            float dyx = dy_val.x, dyy = dy_val.y, dyz = dy_val.z, dyw = dy_val.w;
            float s0 = dyx + dyz;
            float s1 = dyx - dyz;
            float s2 = dyy + dyw;
            float s3 = dyw - dyy;

            float4 dy_f;
            dy_f.x = 0.25f * (s0 + s2);
            dy_f.y = 0.5f  * s1;
            dy_f.z = 0.25f * (s0 - s2);
            dy_f.w = 0.5f  * s3;
            s_dY[row_n][col_d] = dy_f;
        }

        // --- 协同加载 W ---
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int linear_idx = tid + i * 256;
            int row_d = linear_idx / DX_BLOCK_C;
            int col_c = linear_idx % DX_BLOCK_C;
            int load_d = d_start + row_d;
            int load_c = c_base + col_c;

            float4 w_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_d < outNum && load_c < inNum) {
                w_val = __ldg(&W[load_d * inNum + load_c]);
            }
            s_W[row_d][col_c] = w_val;
        }
        __syncthreads();

        // --- 计算 ---
        #pragma unroll
        for (int k = 0; k < DX_BLOCK_D; ++k) {
            float4 w_frag[TILE_D];
            w_frag[0] = s_W[k][tx * 2 + 0];
            w_frag[1] = s_W[k][tx * 2 + 1];

            float4 dy_frag[TILE_N];
            #pragma unroll
            for (int i = 0; i < TILE_N; ++i) dy_frag[i] = s_dY[ty * 8 + i][k];

            #pragma unroll
            for (int j = 0; j < TILE_D; ++j) {
                #pragma unroll
                for (int i = 0; i < TILE_N; ++i) {
                    acc0[i][j] += dy_frag[i].x * w_frag[j].x;
                    acc1[i][j] += dy_frag[i].y * w_frag[j].y + dy_frag[i].w * w_frag[j].w;
                    acc2[i][j] += dy_frag[i].z * w_frag[j].z;
                    acc3[i][j] += -dy_frag[i].y * w_frag[j].w + dy_frag[i].w * w_frag[j].y;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < TILE_D; ++j) {
        #pragma unroll
        for (int i = 0; i < TILE_N; ++i) {
            int out_n = n_base + ty * 8 + i;
            int out_c = c_base + tx * 2 + j;
            if (out_n < N && out_c < inNum) {
                float a0 = acc0[i][j], a1 = acc1[i][j], a2 = acc2[i][j], a3 = acc3[i][j];
                float t0 = a0 + a2;
                float t1 = a0 - a2;

                float4 dx_out;
                dx_out.x = t0 + a1;
                dx_out.y = t1 - a3;
                dx_out.z = t0 - a1;
                dx_out.w = t1 + a3;

                dX[out_n * inNum + out_c] = dx_out;
            }
        }
    }
}

// ---------------------------------------------------------
// 3. 终极 2D Tiling dW 算子 (Split-K Atomic In-Place)
// ---------------------------------------------------------
__global__ __launch_bounds__(256, 2) void flash_eq_linear_backward_dw_kernel_splitk_v3(
    const float4* __restrict__ dY,
    const float4* __restrict__ X,
    float4* __restrict__ dW_final,  // 现直接汇集至全局内存，省去 PyTorch Out-of-Core Reduction
    int N, int inNum, int outNum, int chunk_size
) {
    __shared__ float4 s_dY[DW_BLOCK_D][DW_BLOCK_N + 1];
    __shared__ float4 s_X[DW_BLOCK_C][DW_BLOCK_N + 1];

    int c_base = blockIdx.x * DW_BLOCK_C;
    int d_base = blockIdx.y * DW_BLOCK_D;
    int k_idx = blockIdx.z;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int n_start = k_idx * chunk_size;
    int n_end = min(n_start + chunk_size, N);

    float acc0[2][2]={0}, acc1[2][2]={0}, acc2[2][2]={0}, acc3[2][2]={0};

    for (int n_base = n_start; n_base < n_end; n_base += DW_BLOCK_N) {
        // --- 协同加载 dY ---
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / DW_BLOCK_D;
            int col_d = linear_idx % DW_BLOCK_D;
            int load_n = n_base + row_n;
            int load_d = d_base + col_d;

            float4 dy_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_n < n_end && load_d < outNum) {
                dy_val = __ldg(&dY[load_n * outNum + load_d]);
            }

            float dyx = dy_val.x, dyy = dy_val.y, dyz = dy_val.z, dyw = dy_val.w;
            float s0 = dyx + dyz, s1 = dyx - dyz, s2 = dyy + dyw, s3 = dyw - dyy;

            float4 dy_f;
            dy_f.x = 0.25f * (s0 + s2);
            dy_f.y = 0.5f  * s1;
            dy_f.z = 0.25f * (s0 - s2);
            dy_f.w = 0.5f  * s3;
            s_dY[col_d][row_n] = dy_f;
        }

        // --- 协同加载 X ---
        #pragma unroll
        for(int i = 0; i < 4; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / DW_BLOCK_C;
            int col_c = linear_idx % DW_BLOCK_C;
            int load_n = n_base + row_n;
            int load_c = c_base + col_c;

            float4 x_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_n < n_end && load_c < inNum) {
                x_val = __ldg(&X[load_n * inNum + load_c]);
            }

            float vx = x_val.x, vy = x_val.y, vz = x_val.z, vw = x_val.w;
            float s0 = vx + vz, s1 = vx - vz, s2 = vy + vw, s3 = vw - vy;

            float4 x_f;
            x_f.x = s0 + s2;
            x_f.y = s1;
            x_f.z = s0 - s2;
            x_f.w = s3;
            s_X[col_c][row_n] = x_f;
        }
        __syncthreads();

        int valid_k = min(DW_BLOCK_N, n_end - n_base);
        // 主干加速计算（对齐块做完全展开）
        if (valid_k == DW_BLOCK_N) {
            #pragma unroll
            for (int k = 0; k < DW_BLOCK_N; ++k) {
                float4 dy_frag[2], x_frag[2];
                dy_frag[0] = s_dY[ty * 2 + 0][k]; dy_frag[1] = s_dY[ty * 2 + 1][k];
                x_frag[0] = s_X[tx * 2 + 0][k];  x_frag[1] = s_X[tx * 2 + 1][k];

                #pragma unroll
                for(int j = 0; j < 2; ++j) {
                    #pragma unroll
                    for(int i = 0; i < 2; ++i) {
                        acc0[j][i] += dy_frag[j].x * x_frag[i].x;
                        acc1[j][i] += dy_frag[j].y * x_frag[i].y + dy_frag[j].w * x_frag[i].w;
                        acc2[j][i] += dy_frag[j].z * x_frag[i].z;
                        acc3[j][i] += -dy_frag[j].y * x_frag[i].w + dy_frag[j].w * x_frag[i].y;
                    }
                }
            }
        } else {
            // 处理不能整除的尾部
            for (int k = 0; k < valid_k; ++k) {
                float4 dy_frag[2], x_frag[2];
                dy_frag[0] = s_dY[ty * 2 + 0][k]; dy_frag[1] = s_dY[ty * 2 + 1][k];
                x_frag[0] = s_X[tx * 2 + 0][k];  x_frag[1] = s_X[tx * 2 + 1][k];

                for(int j = 0; j < 2; ++j) {
                    for(int i = 0; i < 2; ++i) {
                        acc0[j][i] += dy_frag[j].x * x_frag[i].x;
                        acc1[j][i] += dy_frag[j].y * x_frag[i].y + dy_frag[j].w * x_frag[i].w;
                        acc2[j][i] += dy_frag[j].z * x_frag[i].z;
                        acc3[j][i] += -dy_frag[j].y * x_frag[i].w + dy_frag[j].w * x_frag[i].y;
                    }
                }
            }
        }
        __syncthreads();
    }

    // 原子加直接写入 Split-K 结果至全局显存
    #pragma unroll
    for(int j = 0; j < 2; ++j) {
        #pragma unroll
        for(int i = 0; i < 2; ++i) {
            int out_d = d_base + ty * 2 + j;
            int out_c = c_base + tx * 2 + i;
            if (out_d < outNum && out_c < inNum) {
                float4 dw_out;
                dw_out.x = acc0[j][i]; dw_out.y = acc1[j][i];
                dw_out.z = acc2[j][i]; dw_out.w = acc3[j][i];

                // Index is exactly aligned with global [outNum, inNum, 4] tensor
                atomicAddFloat4(&dW_final[out_d * inNum + out_c], dw_out);
            }
        }
    }
}

// ==========================================
// C++ 包装函数 (已适配全新 Block 分配策略)
// ==========================================
torch::Tensor flash_eq_linear_forward_impl(
    torch::Tensor X,
    torch::Tensor W,
    const float* bias_ptr,
    bool fuse_bias
) {
    const c10::cuda::CUDAGuard device_guard(X.device());
    int N = X.size(0) * X.size(1);
    int inNum = X.size(2);
    int outNum = W.size(0);

    auto Y = torch::empty({X.size(0), X.size(1), outNum, 4}, X.options());

    dim3 block(16, 16);
    dim3 grid((outNum + 31) / 32, (N + 127) / 128);

    flash_eq_linear_forward_pack6_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(X.data_ptr<float>()),
        reinterpret_cast<const float4*>(W.data_ptr<float>()),
        bias_ptr,
        reinterpret_cast<float4*>(Y.data_ptr<float>()),
        N, inNum, outNum, fuse_bias
    );
    return Y;
}

__global__ __launch_bounds__(256, 2) void flash_eq_linear_backward_dw_kernel_partial_v1(
    const float4* __restrict__ dY,
    const float4* __restrict__ X,
    float4* __restrict__ dW_partial,
    int N, int inNum, int outNum, int chunk_size
) {
    __shared__ float4 s_dY[DW_BLOCK_D][DW_BLOCK_N + 1];
    __shared__ float4 s_X[DW_BLOCK_C][DW_BLOCK_N + 1];

    int c_base = blockIdx.x * DW_BLOCK_C;
    int d_base = blockIdx.y * DW_BLOCK_D;
    int split_idx = blockIdx.z;

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int n_start = split_idx * chunk_size;
    int n_end = min(n_start + chunk_size, N);

    float acc0[2][2] = {0}, acc1[2][2] = {0}, acc2[2][2] = {0}, acc3[2][2] = {0};

    for (int n_base = n_start; n_base < n_end; n_base += DW_BLOCK_N) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / DW_BLOCK_D;
            int col_d = linear_idx % DW_BLOCK_D;
            int load_n = n_base + row_n;
            int load_d = d_base + col_d;

            float4 dy_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_n < n_end && load_d < outNum) {
                dy_val = __ldg(&dY[load_n * outNum + load_d]);
            }

            float dyx = dy_val.x, dyy = dy_val.y, dyz = dy_val.z, dyw = dy_val.w;
            float s0 = dyx + dyz, s1 = dyx - dyz, s2 = dyy + dyw, s3 = dyw - dyy;

            float4 dy_f;
            dy_f.x = 0.25f * (s0 + s2);
            dy_f.y = 0.5f * s1;
            dy_f.z = 0.25f * (s0 - s2);
            dy_f.w = 0.5f * s3;
            s_dY[col_d][row_n] = dy_f;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx = tid + i * 256;
            int row_n = linear_idx / DW_BLOCK_C;
            int col_c = linear_idx % DW_BLOCK_C;
            int load_n = n_base + row_n;
            int load_c = c_base + col_c;

            float4 x_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (load_n < N && load_n < n_end && load_c < inNum) {
                x_val = __ldg(&X[load_n * inNum + load_c]);
            }

            float vx = x_val.x, vy = x_val.y, vz = x_val.z, vw = x_val.w;
            float s0 = vx + vz, s1 = vx - vz, s2 = vy + vw, s3 = vw - vy;

            float4 x_f;
            x_f.x = s0 + s2;
            x_f.y = s1;
            x_f.z = s0 - s2;
            x_f.w = s3;
            s_X[col_c][row_n] = x_f;
        }
        __syncthreads();

        int valid_k = min(DW_BLOCK_N, n_end - n_base);
        for (int k = 0; k < valid_k; ++k) {
            float4 dy_frag[2], x_frag[2];
            dy_frag[0] = s_dY[ty * 2 + 0][k];
            dy_frag[1] = s_dY[ty * 2 + 1][k];
            x_frag[0] = s_X[tx * 2 + 0][k];
            x_frag[1] = s_X[tx * 2 + 1][k];

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    acc0[j][i] += dy_frag[j].x * x_frag[i].x;
                    acc1[j][i] += dy_frag[j].y * x_frag[i].y + dy_frag[j].w * x_frag[i].w;
                    acc2[j][i] += dy_frag[j].z * x_frag[i].z;
                    acc3[j][i] += -dy_frag[j].y * x_frag[i].w + dy_frag[j].w * x_frag[i].y;
                }
            }
        }
        __syncthreads();
    }

    float4* partial_base = dW_partial + static_cast<int64_t>(split_idx) * outNum * inNum;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int out_d = d_base + ty * 2 + j;
            int out_c = c_base + tx * 2 + i;
            if (out_d < outNum && out_c < inNum) {
                float4 dw_out;
                dw_out.x = acc0[j][i];
                dw_out.y = acc1[j][i];
                dw_out.z = acc2[j][i];
                dw_out.w = acc3[j][i];
                partial_base[out_d * inNum + out_c] = dw_out;
            }
        }
    }
}

__global__ __launch_bounds__(256, 2) void flash_eq_linear_backward_dw_reduce_kernel_v1(
    const float4* __restrict__ dW_partial,
    float4* __restrict__ dW_final,
    int split_k,
    int inNum,
    int outNum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= inNum || d >= outNum) {
        return;
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int64_t base_offset = static_cast<int64_t>(d) * inNum + c;
    int64_t plane_stride = static_cast<int64_t>(outNum) * inNum;
    for (int split_idx = 0; split_idx < split_k; ++split_idx) {
        float4 val = dW_partial[static_cast<int64_t>(split_idx) * plane_stride + base_offset];
        acc.x += val.x;
        acc.y += val.y;
        acc.z += val.z;
        acc.w += val.w;
    }
    dW_final[base_offset] = acc;
}

torch::Tensor flash_eq_linear_forward(torch::Tensor X, torch::Tensor W) {
    return flash_eq_linear_forward_impl(X, W, nullptr, false);
}

torch::Tensor flash_eq_linear_forward_fused_bias(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias
) {
    return flash_eq_linear_forward_impl(X, W, bias.data_ptr<float>(), true);
}


__global__ __launch_bounds__(256, 2) void flash_eq_linear_backward_bias_kernel_v1(
    const float4* __restrict__ dY,
    float* __restrict__ dBias,
    int N,
    int outNum
) {
    int out_d = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float4 dy = __ldg(&dY[static_cast<int64_t>(n) * outNum + out_d]);
        sum += dy.x + dy.y + dy.z + dy.w;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dBias[out_d] = s_sum[0];
    }
}

torch::Tensor flash_eq_linear_backward_bias(torch::Tensor dY) {
    const c10::cuda::CUDAGuard device_guard(dY.device());
    int N = dY.size(0) * dY.size(1);
    int outNum = dY.size(2);
    auto dBias = torch::empty({outNum, 1}, dY.options());
    flash_eq_linear_backward_bias_kernel_v1<<<outNum, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(dY.data_ptr<float>()),
        dBias.data_ptr<float>(),
        N,
        outNum
    );
    return dBias;
}

torch::Tensor flash_eq_linear_forward_legacy(torch::Tensor X, torch::Tensor W) {
    const c10::cuda::CUDAGuard device_guard(X.device());
    int N = X.size(0) * X.size(1);
    int inNum = X.size(2);
    int outNum = W.size(0);

    auto Y = torch::empty({X.size(0), X.size(1), outNum, 4}, X.options());

    dim3 block(16, 16);
    dim3 grid((outNum + 31) / 32, (N + 127) / 128);

    flash_eq_linear_forward_legacy_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(X.data_ptr<float>()),
        reinterpret_cast<const float4*>(W.data_ptr<float>()),
        reinterpret_cast<float4*>(Y.data_ptr<float>()),
        N, inNum, outNum
    );
    return Y;
}

void flash_eq_linear_launch_empty() {
    flash_eq_linear_empty_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>();
}

std::vector<torch::Tensor> flash_eq_linear_backward(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W
) {
    const c10::cuda::CUDAGuard device_guard(X.device());
    int N = X.size(0) * X.size(1);
    int inNum = X.size(2);
    int outNum = W.size(0);

    auto dX = torch::empty_like(X);

    dim3 block_dx(16, 16);
    dim3 grid_dx((inNum + 31) / 32, (N + 127) / 128);
    flash_eq_linear_backward_dx_kernel_v3<<<grid_dx, block_dx, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float4*>(dY.data_ptr<float>()),
        reinterpret_cast<const float4*>(W.data_ptr<float>()),
        reinterpret_cast<float4*>(dX.data_ptr<float>()),
        N, inNum, outNum
    );

    int SPLIT_K = get_cached_split_k(N, inNum, outNum);
    int chunk_size = (N + SPLIT_K - 1) / SPLIT_K;

    auto dW = torch::zeros_like(W);
    bool use_two_pass_dw = use_two_pass_dw_for_shape(N, SPLIT_K);
    torch::Tensor dW_partial;

    dim3 block_dw(16, 16);
    dim3 grid_dw((inNum + 31) / 32, (outNum + 31) / 32, SPLIT_K);
    dim3 reduce_block(16, 16);
    dim3 reduce_grid((inNum + reduce_block.x - 1) / reduce_block.x, (outNum + reduce_block.y - 1) / reduce_block.y);

    if (use_two_pass_dw) {
        dW_partial = torch::empty({SPLIT_K, outNum, inNum, 4}, W.options());
        flash_eq_linear_backward_dw_kernel_partial_v1<<<grid_dw, block_dw, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(dY.data_ptr<float>()),
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<float4*>(dW_partial.data_ptr<float>()),
            N, inNum, outNum, chunk_size
        );
        flash_eq_linear_backward_dw_reduce_kernel_v1<<<reduce_grid, reduce_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(dW_partial.data_ptr<float>()),
            reinterpret_cast<float4*>(dW.data_ptr<float>()),
            SPLIT_K, inNum, outNum
        );
    } else {
        flash_eq_linear_backward_dw_kernel_splitk_v3<<<grid_dw, block_dw, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(dY.data_ptr<float>()),
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<float4*>(dW.data_ptr<float>()),
            N, inNum, outNum, chunk_size
        );
    }

    return {dX, dW};
}

template <typename Fn>
float measure_cuda_ms(Fn&& fn, int repeats, int warmup = 20) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    cudaDeviceSynchronize();

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
    for (int i = 0; i < repeats; ++i) {
        fn();
    }
    cudaEventRecord(end, c10::cuda::getCurrentCUDAStream());
    cudaEventSynchronize(end);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return elapsed_ms / static_cast<float>(repeats);
}

torch::Tensor flash_eq_linear_forward_direct_gemm_ada4090(torch::Tensor X, torch::Tensor W) {
    return flash_eq_linear_forward(X, W);
}


torch::Tensor flash_eq_linear_forward_direct_gemm_bias_ada4090(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor bias
) {
    return flash_eq_linear_forward_fused_bias(X, W, bias);
}

std::vector<torch::Tensor> flash_eq_linear_backward_direct_gemm_ada4090(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W
) {
    return flash_eq_linear_backward(dY, X, W);
}


std::vector<torch::Tensor> flash_eq_linear_backward_direct_gemm_bias_ada4090(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W
) {
    auto grads = flash_eq_linear_backward(dY, X, W);
    auto dBias = flash_eq_linear_backward_bias(dY);
    return {grads[0], grads[1], dBias};
}

py::dict flash_eq_linear_profile_direct_gemm_ada4090(
    torch::Tensor X,
    torch::Tensor dY,
    torch::Tensor W,
    int repeats,
    int warmup
) {
    const c10::cuda::CUDAGuard device_guard(X.device());
    int N = X.size(0) * X.size(1);
    int inNum = X.size(2);
    int outNum = W.size(0);

    auto Y = torch::empty({X.size(0), X.size(1), outNum, 4}, X.options());
    auto dX = torch::empty_like(X);
    auto dW = torch::zeros_like(W);

    dim3 block_fwd(16, 16);
    dim3 grid_fwd((outNum + 31) / 32, (N + 127) / 128);
    dim3 block_dx(16, 16);
    dim3 grid_dx((inNum + 31) / 32, (N + 127) / 128);

    int SPLIT_K = get_cached_split_k(N, inNum, outNum);
    int chunk_size = (N + SPLIT_K - 1) / SPLIT_K;
    bool use_two_pass_dw = use_two_pass_dw_for_shape(N, SPLIT_K);
    dim3 block_dw(16, 16);
    dim3 grid_dw((inNum + 31) / 32, (outNum + 31) / 32, SPLIT_K);
    dim3 reduce_block(16, 16);
    dim3 reduce_grid((inNum + reduce_block.x - 1) / reduce_block.x, (outNum + reduce_block.y - 1) / reduce_block.y);
    auto dW_partial = use_two_pass_dw ? torch::empty({SPLIT_K, outNum, inNum, 4}, W.options()) : torch::Tensor();

    auto run_forward_kernel = [&] {
        flash_eq_linear_forward_pack6_kernel<<<grid_fwd, block_fwd, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(X.data_ptr<float>()),
            reinterpret_cast<const float4*>(W.data_ptr<float>()),
            nullptr,
            reinterpret_cast<float4*>(Y.data_ptr<float>()),
            N, inNum, outNum, false
        );
    };
    auto run_dx_kernel = [&] {
        flash_eq_linear_backward_dx_kernel_v3<<<grid_dx, block_dx, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float4*>(dY.data_ptr<float>()),
            reinterpret_cast<const float4*>(W.data_ptr<float>()),
            reinterpret_cast<float4*>(dX.data_ptr<float>()),
            N, inNum, outNum
        );
    };
    auto run_dw_stage1 = [&] {
        if (use_two_pass_dw) {
            flash_eq_linear_backward_dw_kernel_partial_v1<<<grid_dw, block_dw, 0, c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const float4*>(dY.data_ptr<float>()),
                reinterpret_cast<const float4*>(X.data_ptr<float>()),
                reinterpret_cast<float4*>(dW_partial.data_ptr<float>()),
                N, inNum, outNum, chunk_size
            );
        } else {
            dW.zero_();
            flash_eq_linear_backward_dw_kernel_splitk_v3<<<grid_dw, block_dw, 0, c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const float4*>(dY.data_ptr<float>()),
                reinterpret_cast<const float4*>(X.data_ptr<float>()),
                reinterpret_cast<float4*>(dW.data_ptr<float>()),
                N, inNum, outNum, chunk_size
            );
        }
    };
    auto run_dw_stage2 = [&] {
        if (use_two_pass_dw) {
            flash_eq_linear_backward_dw_reduce_kernel_v1<<<reduce_grid, reduce_block, 0, c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const float4*>(dW_partial.data_ptr<float>()),
                reinterpret_cast<float4*>(dW.data_ptr<float>()),
                SPLIT_K, inNum, outNum
            );
        } else {
            flash_eq_linear_launch_empty();
        }
    };
    auto run_full_backward = [&] {
        auto out = flash_eq_linear_backward(dY, X, W);
        (void)out;
    };

    float forward_kernel_ms = measure_cuda_ms(run_forward_kernel, repeats, warmup);
    float dx_kernel_ms = measure_cuda_ms(run_dx_kernel, repeats, warmup);
    float dw_stage1_ms = measure_cuda_ms(run_dw_stage1, repeats, warmup);
    float dw_stage2_ms = measure_cuda_ms(run_dw_stage2, repeats, warmup);
    float backward_total_ms = measure_cuda_ms(run_full_backward, repeats, warmup);

    py::dict result;
    result["forward_kernel_ms"] = forward_kernel_ms;
    result["dx_kernel_ms"] = dx_kernel_ms;
    result["dw_stage1_ms"] = dw_stage1_ms;
    result["dw_stage2_ms"] = dw_stage2_ms;
    result["backward_total_ms"] = backward_total_ms;
    result["split_k"] = SPLIT_K;
    result["chunk_size"] = chunk_size;
    result["dw_mode"] = use_two_pass_dw ? "two_pass" : "atomic";
    return result;
}

py::dict flash_eq_linear_autotune_direct_gemm_ada4090(
    torch::Tensor X,
    torch::Tensor dY,
    torch::Tensor W,
    std::vector<int> candidate_split_ks,
    int repeats,
    int warmup
) {
    const c10::cuda::CUDAGuard device_guard(X.device());
    int N = X.size(0) * X.size(1);
    int inNum = X.size(2);
    int outNum = W.size(0);

    if (candidate_split_ks.empty()) {
        candidate_split_ks = {1, 2, 4, 8, 16, 32, 64};
    }

    std::sort(candidate_split_ks.begin(), candidate_split_ks.end());
    candidate_split_ks.erase(
        std::remove_if(
            candidate_split_ks.begin(),
            candidate_split_ks.end(),
            [](int v) { return v <= 0; }
        ),
        candidate_split_ks.end()
    );
    candidate_split_ks.erase(std::unique(candidate_split_ks.begin(), candidate_split_ks.end()), candidate_split_ks.end());

    dim3 block_dw(16, 16);
    std::vector<int> valid_candidates;
    std::vector<float> timings_ms;
    float best_ms = std::numeric_limits<float>::max();
    int best_split_k = default_split_k_for_shape(N);

    for (int split_k : candidate_split_ks) {
        split_k = std::min(split_k, std::max(1, N));
        int chunk_size = (N + split_k - 1) / split_k;
        auto dW = torch::zeros_like(W);
        dim3 grid_dw((inNum + 31) / 32, (outNum + 31) / 32, split_k);

        auto run_dw = [&] {
            dW.zero_();
            flash_eq_linear_backward_dw_kernel_splitk_v3<<<grid_dw, block_dw, 0, c10::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<const float4*>(dY.data_ptr<float>()),
                reinterpret_cast<const float4*>(X.data_ptr<float>()),
                reinterpret_cast<float4*>(dW.data_ptr<float>()),
                N, inNum, outNum, chunk_size
            );
        };

        float elapsed_ms = measure_cuda_ms(run_dw, repeats, warmup);
        valid_candidates.push_back(split_k);
        timings_ms.push_back(elapsed_ms);
        if (elapsed_ms < best_ms) {
            best_ms = elapsed_ms;
            best_split_k = split_k;
        }
    }

    set_cached_split_k(N, inNum, outNum, best_split_k);

    py::dict result;
    result["N"] = N;
    result["inNum"] = inNum;
    result["outNum"] = outNum;
    result["best_split_k"] = best_split_k;
    result["best_ms"] = best_ms;
    result["split_ks"] = valid_candidates;
    result["timings_ms"] = timings_ms;
    result["cache_stats"] = get_direct_cache_stats();
    return result;
}

// ====================================================================================
// Register
// ====================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_eq_linear_forward", &flash_eq_linear_forward, "Flash EQ Linear Forward");
    m.def(
        "flash_eq_linear_forward_fused_bias",
        &flash_eq_linear_forward_fused_bias,
        "Flash EQ Linear Forward With Fused Bias"
    );
    m.def(
        "flash_eq_linear_forward_legacy",
        &flash_eq_linear_forward_legacy,
        "Flash EQ Linear Forward Legacy"
    );
    m.def("flash_eq_linear_launch_empty", &flash_eq_linear_launch_empty, "Launch an empty CUDA kernel");
    m.def("flash_eq_linear_backward", &flash_eq_linear_backward, "Flash EQ Linear Backward");
    m.def(
        "flash_eq_linear_backward_bias",
        &flash_eq_linear_backward_bias,
        "Flash EQ Linear Bias Backward"
    );
    m.def(
        "flash_eq_linear_forward_direct_gemm_ada4090",
        &flash_eq_linear_forward_direct_gemm_ada4090,
        "Direct structured GEMM forward"
    );
    m.def(
        "flash_eq_linear_forward_direct_gemm_bias_ada4090",
        &flash_eq_linear_forward_direct_gemm_bias_ada4090,
        "Direct structured GEMM forward with fused bias"
    );
    m.def(
        "flash_eq_linear_backward_direct_gemm_ada4090",
        &flash_eq_linear_backward_direct_gemm_ada4090,
        "Direct structured GEMM backward"
    );
    m.def(
        "flash_eq_linear_backward_direct_gemm_bias_ada4090",
        &flash_eq_linear_backward_direct_gemm_bias_ada4090,
        "Direct structured GEMM backward with bias grad"
    );
    m.def(
        "flash_eq_linear_profile_direct_gemm_ada4090",
        &flash_eq_linear_profile_direct_gemm_ada4090,
        py::arg("X"),
        py::arg("dY"),
        py::arg("W"),
        py::arg("repeats") = 80,
        py::arg("warmup") = 20,
        "Direct structured GEMM profile"
    );
    m.def(
        "flash_eq_linear_autotune_direct_gemm_ada4090",
        &flash_eq_linear_autotune_direct_gemm_ada4090,
        py::arg("X"),
        py::arg("dY"),
        py::arg("W"),
        py::arg("candidate_split_ks") = std::vector<int>{},
        py::arg("repeats") = 80,
        py::arg("warmup") = 20,
        "Autotune direct GEMM split-k and cache best result"
    );
    m.def(
        "flash_eq_linear_direct_cache_stats_ada4090",
        &get_direct_cache_stats,
        "Get direct GEMM runtime cache stats"
    );
    m.def(
        "flash_eq_linear_clear_direct_cache_ada4090",
        &clear_cached_split_k,
        "Clear direct GEMM runtime cache"
    );
    m.def(
        "flash_eq_linear_set_direct_dw_mode_ada4090",
        &set_direct_dw_mode_override,
        py::arg("mode"),
        "Set dW mode override (-1 auto, 0 atomic, 1 two_pass)"
    );
}
