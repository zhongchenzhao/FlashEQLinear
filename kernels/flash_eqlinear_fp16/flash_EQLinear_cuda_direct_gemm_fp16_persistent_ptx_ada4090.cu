#include "flash_EQLinear_cuda_direct_gemm_fp16_pipeline_common_ada4090.cuh"

#include <algorithm>
#include <climits>
#include <string>

namespace py = pybind11;

namespace {

using namespace flash_eq_fp16_pipeline;

constexpr int kPersistentPtxM32N64TileM = 32;
constexpr int kPersistentPtxM32N64TileN = 64;
constexpr int kPersistentPtxM32N64WarpsM = 2;
constexpr int kPersistentPtxM32N64WarpsN = 4;

constexpr int kPersistentPtxM64N64TileM = 64;
constexpr int kPersistentPtxM64N64TileN = 64;
constexpr int kPersistentPtxM64N64WarpsM = 4;
constexpr int kPersistentPtxM64N64WarpsN = 2;

constexpr int kPersistentPtxM128N64TileM = 128;
constexpr int kPersistentPtxM128N64TileN = 64;
constexpr int kPersistentPtxM128N64WarpsM = 8;
constexpr int kPersistentPtxM128N64WarpsN = 2;

constexpr int kPersistentPtxM64N128TileM = 64;
constexpr int kPersistentPtxM64N128TileN = 128;
constexpr int kPersistentPtxM64N128WarpsM = 4;
constexpr int kPersistentPtxM64N128WarpsN = 4;

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_persistent_mma_pipe_f16_cpasync_w_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int packedInNum,
    int outNum,
    int total_tiles,
    int d_tiles) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");

  __shared__ half sA[4][WarpsM][4][8][8];
  __shared__ half sB[5][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
    int d_tile = tile_id % d_tiles;
    int n_tile = tile_id / d_tiles;
    int n_base = n_tile * TileM;
    int d_base = d_tile * TileN;

    uint32_t acc[4][NFrags][2];
#pragma unroll
    for (int f = 0; f < 4; ++f) {
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        acc[f][nf][0] = 0;
        acc[f][nf][1] = 0;
      }
    }

    for (int c_start = 0; c_start < packedInNum; c_start += kMmaAtomK) {
      int plane_size = outNum * packedInNum;
      constexpr int GroupsK = 2;
      constexpr int TotalGroups = kPackedTranNum * TileN * GroupsK;
      for (int idx = threadIdx.x; idx < TotalGroups; idx += Threads) {
        int plane_id = idx / (TileN * GroupsK);
        int rem = idx - plane_id * TileN * GroupsK;
        int col = rem >> 1;
        int group = rem & 1;
        int kk = group * 8;
        int d = d_base + col;
        int c = c_start + kk;
        int wn = col / (NFrags * kMmaAtomN);
        int nfrag = (col / kMmaAtomN) - wn * NFrags;
        int sr = col & 7;
        half* dst = &sB[plane_id][wn][nfrag][group][sr][0];

        if (d < outNum && c + 7 < packedInNum) {
          const half* src = Wp + plane_id * plane_size + d * packedInNum + c;
          cp_async_cg_16(dst, src);
        } else {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            dst[i] = hzero();
          }
        }
      }
      cp_async_commit_group();

      for (int idx = threadIdx.x; idx < TileM * kMmaAtomK; idx += Threads) {
        int row = idx / kMmaAtomK;
        int kk = idx - row * kMmaAtomK;
        int n = n_base + row;
        int c = c_start + kk;
        half x0 = hzero();
        half x1 = hzero();
        half x2 = hzero();
        half x3 = hzero();
        if (n < N && c < inNum) {
          load_x_freq_to_half<half>(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
        }
        int wm = row >> 4;
        int sub_row = row & 15;
        int mat = ((kk >> 3) << 1) + (sub_row >> 3);
        int off = ldmatrix_swizzled_offset(sub_row & 7, kk & 7);
        int sr = off >> 3;
        int sc = off & 7;
        sA[0][wm][mat][sr][sc] = x0;
        sA[1][wm][mat][sr][sc] = x1;
        sA[2][wm][mat][sr][sc] = x2;
        sA[3][wm][mat][sr][sc] = x3;
      }
      cp_async_wait_all();
      __syncthreads();

      uint32_t a0[4], a1[4], a2[4], a3[4];
      ldmatrix_x4(&sA[0][warp_m][0][0][0], lane_id, a0);
      ldmatrix_x4(&sA[1][warp_m][0][0][0], lane_id, a1);
      ldmatrix_x4(&sA[2][warp_m][0][0][0], lane_id, a2);
      ldmatrix_x4(&sA[3][warp_m][0][0][0], lane_id, a3);
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        uint32_t b0[2], b1[2], b2[2], b3[2], b3neg[2];
        ldmatrix_x2(&sB[0][warp_n][nf][0][0][0], lane_id, b0);
        ldmatrix_x2(&sB[1][warp_n][nf][0][0][0], lane_id, b1);
        ldmatrix_x2(&sB[2][warp_n][nf][0][0][0], lane_id, b2);
        ldmatrix_x2(&sB[3][warp_n][nf][0][0][0], lane_id, b3);
        ldmatrix_x2(&sB[4][warp_n][nf][0][0][0], lane_id, b3neg);
        mma_m16n8k16_f16(acc[0][nf], a0, b0);
        mma_m16n8k16_f16(acc[1][nf], a1, b1);
        mma_m16n8k16_f16(acc[1][nf], a3, b3neg);
        mma_m16n8k16_f16(acc[2][nf], a2, b2);
        mma_m16n8k16_f16(acc[3][nf], a1, b3);
        mma_m16n8k16_f16(acc[3][nf], a3, b1);
      }
      __syncthreads();
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
      int offset = mma_c_layout_offset(lane_id, v);
      int row = offset & 15;
      int col = offset >> 4;
      int n = n_base + warp_row + row;
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        int d = d_base + warp_col + nf * kMmaAtomN + col;
        half a0v = unpack_half_u32(acc[0][nf][v >> 1], v & 1);
        half a1v = unpack_half_u32(acc[1][nf][v >> 1], v & 1);
        half a2v = unpack_half_u32(acc[2][nf][v >> 1], v & 1);
        half a3v = unpack_half_u32(acc[3][nf][v >> 1], v & 1);
        if (n < N && d < outNum) {
          write_spatial_output<half>(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
        }
      }
    }
    __syncthreads();
  }
}

template <int TileN, int WarpsN, int NFrags, int Threads, bool BoundsCheck>
__device__ __forceinline__ void load_w_tile_padded_cpasync(
    const half* __restrict__ Wp,
    half (&sB)[4][WarpsN][NFrags][2][8][8],
    int d_base,
    int c_start,
    int packedInNum,
    int outNum) {
  int plane_size = outNum * packedInNum;
  constexpr int GroupsK = 2;
  constexpr int ActivePlanes = 4;
  constexpr int TotalGroups = ActivePlanes * TileN * GroupsK;
  for (int idx = threadIdx.x; idx < TotalGroups; idx += Threads) {
    int plane_id = idx / (TileN * GroupsK);
    int rem = idx - plane_id * TileN * GroupsK;
    int col = rem >> 1;
    int group = rem & 1;
    int kk = group * 8;
    int d = d_base + col;
    int c = c_start + kk;
    int wn = col / (NFrags * kMmaAtomN);
    int nfrag = (col / kMmaAtomN) - wn * NFrags;
    half* dst = &sB[plane_id][wn][nfrag][group][col & 7][0];

    if constexpr (!BoundsCheck) {
      const half* src = Wp + plane_id * plane_size + d * packedInNum + c;
      cp_async_cg_16(dst, src);
    } else {
      if (d < outNum) {
        const half* src = Wp + plane_id * plane_size + d * packedInNum + c;
        cp_async_cg_16(dst, src);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          dst[i] = hzero();
        }
      }
    }
  }
}

template <int TileM, int WarpsM, int Threads, bool BoundsCheck, bool VectorLoadA = false>
__device__ __forceinline__ void load_a_tile_padded_half(
    const half* __restrict__ X,
    half (&sA)[4][WarpsM][4][8][8],
    int n_base,
    int c_start,
    int N,
    int inNum) {
  for (int idx = threadIdx.x; idx < TileM * kMmaAtomK; idx += Threads) {
    int row = idx >> 4;
    int kk = idx & 15;
    int n = n_base + row;
    int c = c_start + kk;
    half x0 = hzero();
    half x1 = hzero();
    half x2 = hzero();
    half x3 = hzero();
    if constexpr (!BoundsCheck) {
      if constexpr (VectorLoadA) {
        load_x_freq_half_vec4(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
      } else {
        load_x_freq_to_half<half>(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
      }
    } else if (n < N && c < inNum) {
      if constexpr (VectorLoadA) {
        load_x_freq_half_vec4(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
      } else {
        load_x_freq_to_half<half>(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
      }
    }
    int wm = row >> 4;
    int sub_row = row & 15;
    int mat = ((kk >> 3) << 1) + (sub_row >> 3);
    sA[0][wm][mat][sub_row & 7][kk & 7] = x0;
    sA[1][wm][mat][sub_row & 7][kk & 7] = x1;
    sA[2][wm][mat][sub_row & 7][kk & 7] = x2;
    sA[3][wm][mat][sub_row & 7][kk & 7] = x3;
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN, bool BoundsCheck, bool VectorLoadA = false>
__global__ void forward_persistent_mma_pipe_f16_dbvec_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int packedInNum,
    int outNum,
    int total_tiles,
    int d_tiles) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");

  __shared__ half sA[2][4][WarpsM][4][8][8];
  __shared__ half sB[2][4][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
    int d_tile = tile_id % d_tiles;
    int n_tile = tile_id / d_tiles;
    int n_base = n_tile * TileM;
    int d_base = d_tile * TileN;

    uint32_t acc[4][NFrags][2];
#pragma unroll
    for (int f = 0; f < 4; ++f) {
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        acc[f][nf][0] = 0;
        acc[f][nf][1] = 0;
      }
    }

    load_w_tile_padded_cpasync<TileN, WarpsN, NFrags, Threads, BoundsCheck>(
        Wp, sB[0], d_base, 0, packedInNum, outNum);
    cp_async_commit_group();
    load_a_tile_padded_half<TileM, WarpsM, Threads, BoundsCheck, VectorLoadA>(X, sA[0], n_base, 0, N, inNum);
    cp_async_wait_all();
    __syncthreads();

    for (int c_start = 0, tile = 0; c_start < packedInNum; c_start += kMmaAtomK, ++tile) {
      int buf = tile & 1;
      int next_buf = buf ^ 1;
      int next_c = c_start + kMmaAtomK;
      bool has_next = next_c < packedInNum;

      if (has_next) {
        load_w_tile_padded_cpasync<TileN, WarpsN, NFrags, Threads, BoundsCheck>(
            Wp, sB[next_buf], d_base, next_c, packedInNum, outNum);
        cp_async_commit_group();
      }

      uint32_t a0[4], a1[4], a2[4], a3[4];
      ldmatrix_x4(&sA[buf][0][warp_m][0][0][0], lane_id, a0);
      ldmatrix_x4(&sA[buf][1][warp_m][0][0][0], lane_id, a1);
      ldmatrix_x4(&sA[buf][2][warp_m][0][0][0], lane_id, a2);
      ldmatrix_x4(&sA[buf][3][warp_m][0][0][0], lane_id, a3);

      if (has_next) {
        load_a_tile_padded_half<TileM, WarpsM, Threads, BoundsCheck, VectorLoadA>(X, sA[next_buf], n_base, next_c, N, inNum);
      }

#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        uint32_t b0[2], b1[2], b2[2], b3[2], b3neg[2];
        ldmatrix_x2(&sB[buf][0][warp_n][nf][0][0][0], lane_id, b0);
        ldmatrix_x2(&sB[buf][1][warp_n][nf][0][0][0], lane_id, b1);
        ldmatrix_x2(&sB[buf][2][warp_n][nf][0][0][0], lane_id, b2);
        ldmatrix_x2(&sB[buf][3][warp_n][nf][0][0][0], lane_id, b3);
        b3neg[0] = b3[0] ^ 0x80008000u;
        b3neg[1] = b3[1] ^ 0x80008000u;
        mma_m16n8k16_f16(acc[0][nf], a0, b0);
        mma_m16n8k16_f16(acc[1][nf], a1, b1);
        mma_m16n8k16_f16(acc[1][nf], a3, b3neg);
        mma_m16n8k16_f16(acc[2][nf], a2, b2);
        mma_m16n8k16_f16(acc[3][nf], a1, b3);
        mma_m16n8k16_f16(acc[3][nf], a3, b1);
      }

      if (has_next) {
        cp_async_wait_all();
        __syncthreads();
      }
    }

#pragma unroll
    for (int v = 0; v < 4; ++v) {
      int offset = mma_c_layout_offset(lane_id, v);
      int row = offset & 15;
      int col = offset >> 4;
      int n = n_base + warp_row + row;
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        int d = d_base + warp_col + nf * kMmaAtomN + col;
        half a0v = unpack_half_u32(acc[0][nf][v >> 1], v & 1);
        half a1v = unpack_half_u32(acc[1][nf][v >> 1], v & 1);
        half a2v = unpack_half_u32(acc[2][nf][v >> 1], v & 1);
        half a3v = unpack_half_u32(acc[3][nf][v >> 1], v & 1);
        if constexpr (!BoundsCheck) {
          write_spatial_output_half4(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
        } else if (n < N && d < outNum) {
          write_spatial_output_half4(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
        }
      }
    }
    __syncthreads();
  }
}

inline int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

int ctas_for_variant(const std::string& variant) {
  if (variant.find("_cta288") != std::string::npos) {
    return 288;
  }
  if (variant.find("_cta128") != std::string::npos) {
    return 128;
  }
  if (variant.find("_cta192") != std::string::npos) {
    return 192;
  }
  if (variant.find("_cta224") != std::string::npos) {
    return 224;
  }
  if (variant.find("_cta256") != std::string::npos) {
    return 256;
  }
  if (variant.find("_cta384") != std::string::npos) {
    return 384;
  }
  if (variant.find("_cta320") != std::string::npos) {
    return 320;
  }
  if (variant.find("_cta512") != std::string::npos) {
    return 512;
  }
  if (variant.find("_cta576") != std::string::npos) {
    return 576;
  }
  if (variant.find("_cta640") != std::string::npos) {
    return 640;
  }
  if (variant.find("_cta768") != std::string::npos) {
    return 768;
  }
  if (variant.find("_cta4096") != std::string::npos) {
    return 4096;
  }
  if (variant.find("_cta2048") != std::string::npos) {
    return 2048;
  }
  if (variant.find("_cta1024") != std::string::npos) {
    return 1024;
  }
  if (variant.find("_large") != std::string::npos) {
    return 2048;
  }
  if (variant.find("_mid") != std::string::npos) {
    return 1024;
  }
  return 4096;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
void launch_persistent_ptx(torch::Tensor X, torch::Tensor Wp, torch::Tensor Y, int ctas) {
  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int packedInNum = static_cast<int>(Wp.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int d_tiles = ceil_div_int(outNum, TileN);
  int n_tiles = ceil_div_int(N, TileM);
  int total_tiles = d_tiles * n_tiles;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid(std::max(1, std::min(ctas, total_tiles)));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_persistent_mma_pipe_f16_cpasync_w_kernel<TileM, TileN, WarpsM, WarpsN>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      packedInNum,
      outNum,
      total_tiles,
      d_tiles);
}

template <int TileM, int TileN, int WarpsM, int WarpsN, bool BoundsCheck = true, bool VectorLoadA = false>
void launch_persistent_ptx_dbvec(torch::Tensor X, torch::Tensor Wp, torch::Tensor Y, int ctas) {
  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int packedInNum = static_cast<int>(Wp.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int d_tiles = ceil_div_int(outNum, TileN);
  int n_tiles = ceil_div_int(N, TileM);
  int total_tiles = d_tiles * n_tiles;
  if constexpr (!BoundsCheck) {
    TORCH_CHECK(N % TileM == 0, "nobounds persistent PTX requires N divisible by TileM");
    TORCH_CHECK(outNum % TileN == 0, "nobounds persistent PTX requires outNum divisible by TileN");
    TORCH_CHECK(inNum == packedInNum && inNum % kMmaAtomK == 0,
        "nobounds persistent PTX requires unpadded C divisible by 16");
  }
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid(std::max(1, std::min(ctas, total_tiles)));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_persistent_mma_pipe_f16_dbvec_kernel<TileM, TileN, WarpsM, WarpsN, BoundsCheck, VectorLoadA>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      packedInNum,
      outNum,
      total_tiles,
      d_tiles);
}

__global__ void pack_weight_padded_kernel(
    const half* __restrict__ W,
    half* __restrict__ Wp,
    int inNum,
    int packedInNum,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outNum * packedInNum;
  if (idx >= total) {
    return;
  }
  int d = idx / packedInNum;
  int c = idx - d * packedInNum;
  int plane = outNum * packedInNum;
  half w0 = hzero();
  half w1 = hzero();
  half w2 = hzero();
  half w3 = hzero();
  if (c < inNum) {
    const half* w = W + ((d * inNum + c) * kTranNum);
    w0 = w[0];
    w1 = w[1];
    w2 = w[2];
    w3 = w[3];
  }
  Wp[0 * plane + idx] = w0;
  Wp[1 * plane + idx] = w1;
  Wp[2 * plane + idx] = w2;
  Wp[3 * plane + idx] = w3;
  Wp[4 * plane + idx] = __hneg(w3);
}

torch::Tensor pack_weight(torch::Tensor W) {
  check_weight_3d_half_cuda(W, "W");
  c10::cuda::CUDAGuard device_guard(W.device());
  int outNum = static_cast<int>(W.size(0));
  int inNum = static_cast<int>(W.size(1));
  int packedInNum = ((inNum + 7) / 8) * 8;
  auto Wp = torch::empty({kPackedTranNum, outNum, packedInNum}, W.options());
  int threads = 256;
  int total = outNum * packedInNum;
  int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  pack_weight_padded_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Wp.data_ptr<at::Half>()),
      inNum,
      packedInNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Wp;
}

torch::Tensor forward(torch::Tensor X, torch::Tensor Wp, const std::string& variant) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  TORCH_CHECK(Wp.size(2) >= X.size(2), "packed W channel dimension must cover X inNum");
  TORCH_CHECK(X.size(0) * X.size(1) <= INT_MAX, "flattened sequence is too large");

  auto Y = torch::empty({X.size(0), X.size(1), Wp.size(1), kTranNum}, X.options());
  int ctas = ctas_for_variant(variant);
  bool dbvec = variant.find("dbvec") != std::string::npos;
  bool nobounds = variant.find("nobounds") != std::string::npos;
  bool vecld = variant.find("vecld") != std::string::npos;
  bool use_m128n64 = variant.find("m128n64") != std::string::npos ||
      variant.find("_mid") != std::string::npos ||
      variant.find("_large") != std::string::npos;
  if (dbvec && vecld && nobounds && variant.find("m32n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM32N64TileM,
        kPersistentPtxM32N64TileN,
        kPersistentPtxM32N64WarpsM,
        kPersistentPtxM32N64WarpsN,
        false,
        true>(X, Wp, Y, ctas);
  } else if (dbvec && nobounds && variant.find("m32n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM32N64TileM,
        kPersistentPtxM32N64TileN,
        kPersistentPtxM32N64WarpsM,
        kPersistentPtxM32N64WarpsN,
        false>(X, Wp, Y, ctas);
  } else if (dbvec && vecld && nobounds && variant.find("m64n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM64N64TileM,
        kPersistentPtxM64N64TileN,
        kPersistentPtxM64N64WarpsM,
        kPersistentPtxM64N64WarpsN,
        false,
        true>(X, Wp, Y, ctas);
  } else if (dbvec && nobounds && variant.find("m64n128") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM64N128TileM,
        kPersistentPtxM64N128TileN,
        kPersistentPtxM64N128WarpsM,
        kPersistentPtxM64N128WarpsN,
        false>(X, Wp, Y, ctas);
  } else if (dbvec && nobounds && variant.find("m64n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM64N64TileM,
        kPersistentPtxM64N64TileN,
        kPersistentPtxM64N64WarpsM,
        kPersistentPtxM64N64WarpsN,
        false>(X, Wp, Y, ctas);
  } else if (dbvec && nobounds && use_m128n64) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM128N64TileM,
        kPersistentPtxM128N64TileN,
        kPersistentPtxM128N64WarpsM,
        kPersistentPtxM128N64WarpsN,
        false>(X, Wp, Y, ctas);
  } else if (dbvec && variant.find("m64n128") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM64N128TileM,
        kPersistentPtxM64N128TileN,
        kPersistentPtxM64N128WarpsM,
        kPersistentPtxM64N128WarpsN>(X, Wp, Y, ctas);
  } else if (dbvec && use_m128n64) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM128N64TileM,
        kPersistentPtxM128N64TileN,
        kPersistentPtxM128N64WarpsM,
        kPersistentPtxM128N64WarpsN>(X, Wp, Y, ctas);
  } else if (dbvec && variant.find("m64n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM64N64TileM,
        kPersistentPtxM64N64TileN,
        kPersistentPtxM64N64WarpsM,
        kPersistentPtxM64N64WarpsN>(X, Wp, Y, ctas);
  } else if (dbvec && variant.find("m32n64") != std::string::npos) {
    launch_persistent_ptx_dbvec<
        kPersistentPtxM32N64TileM,
        kPersistentPtxM32N64TileN,
        kPersistentPtxM32N64WarpsM,
        kPersistentPtxM32N64WarpsN>(X, Wp, Y, ctas);
  } else if (variant.find("m128n64") != std::string::npos) {
    launch_persistent_ptx<
        kPersistentPtxM128N64TileM,
        kPersistentPtxM128N64TileN,
        kPersistentPtxM128N64WarpsM,
        kPersistentPtxM128N64WarpsN>(X, Wp, Y, ctas);
  } else if (variant.find("m64n64") != std::string::npos) {
    launch_persistent_ptx<
        kPersistentPtxM64N64TileM,
        kPersistentPtxM64N64TileN,
        kPersistentPtxM64N64WarpsM,
        kPersistentPtxM64N64WarpsN>(X, Wp, Y, ctas);
  } else {
    TORCH_CHECK(false, "unknown persistent PTX fp16 variant: ", variant);
  }
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN, bool VectorLoadA = false>
float profile_variant(torch::Tensor X, torch::Tensor Wp, torch::Tensor Y, int ctas, int repeats, int warmup, bool dbvec, bool bounds_check) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    if (dbvec && !bounds_check) {
      launch_persistent_ptx_dbvec<TileM, TileN, WarpsM, WarpsN, false, VectorLoadA>(X, Wp, Y, ctas);
    } else if (dbvec) {
      launch_persistent_ptx_dbvec<TileM, TileN, WarpsM, WarpsN, true, VectorLoadA>(X, Wp, Y, ctas);
    } else {
      launch_persistent_ptx<TileM, TileN, WarpsM, WarpsN>(X, Wp, Y, ctas);
    }
  };
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    launch();
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(repeats);
}

py::dict profile_forward(torch::Tensor X, torch::Tensor Wp, const std::string& variant, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(repeats > 0, "repeats must be positive");
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");

  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int outNum = static_cast<int>(Wp.size(1));
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int ctas = ctas_for_variant(variant);
  bool dbvec = variant.find("dbvec") != std::string::npos;
  bool nobounds = variant.find("nobounds") != std::string::npos;
  bool vecld = variant.find("vecld") != std::string::npos;
  bool use_m128n64 = variant.find("m128n64") != std::string::npos ||
      variant.find("_mid") != std::string::npos ||
      variant.find("_large") != std::string::npos;
  bool bounds_check = !nobounds;
  float kernel_ms = 0.0f;
  int tile_m = 0;
  int tile_n = 0;
  int warps = 0;
  if (dbvec && variant.find("m64n128") != std::string::npos) {
    tile_m = kPersistentPtxM64N128TileM;
    tile_n = kPersistentPtxM64N128TileN;
    warps = kPersistentPtxM64N128WarpsM * kPersistentPtxM64N128WarpsN;
    kernel_ms = profile_variant<
        kPersistentPtxM64N128TileM,
        kPersistentPtxM64N128TileN,
        kPersistentPtxM64N128WarpsM,
        kPersistentPtxM64N128WarpsN>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
  } else if (use_m128n64) {
    tile_m = kPersistentPtxM128N64TileM;
    tile_n = kPersistentPtxM128N64TileN;
    warps = kPersistentPtxM128N64WarpsM * kPersistentPtxM128N64WarpsN;
    kernel_ms = profile_variant<
        kPersistentPtxM128N64TileM,
        kPersistentPtxM128N64TileN,
        kPersistentPtxM128N64WarpsM,
        kPersistentPtxM128N64WarpsN>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
  } else if (variant.find("m32n64") != std::string::npos) {
    tile_m = kPersistentPtxM32N64TileM;
    tile_n = kPersistentPtxM32N64TileN;
    warps = kPersistentPtxM32N64WarpsM * kPersistentPtxM32N64WarpsN;
    if (dbvec && vecld) {
      kernel_ms = profile_variant<
          kPersistentPtxM32N64TileM,
          kPersistentPtxM32N64TileN,
          kPersistentPtxM32N64WarpsM,
          kPersistentPtxM32N64WarpsN,
          true>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
    } else {
      kernel_ms = profile_variant<
          kPersistentPtxM32N64TileM,
          kPersistentPtxM32N64TileN,
          kPersistentPtxM32N64WarpsM,
          kPersistentPtxM32N64WarpsN>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
    }
  } else if (variant.find("m64n64") != std::string::npos) {
    tile_m = kPersistentPtxM64N64TileM;
    tile_n = kPersistentPtxM64N64TileN;
    warps = kPersistentPtxM64N64WarpsM * kPersistentPtxM64N64WarpsN;
    if (dbvec && vecld) {
      kernel_ms = profile_variant<
          kPersistentPtxM64N64TileM,
          kPersistentPtxM64N64TileN,
          kPersistentPtxM64N64WarpsM,
          kPersistentPtxM64N64WarpsN,
          true>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
    } else {
      kernel_ms = profile_variant<
          kPersistentPtxM64N64TileM,
          kPersistentPtxM64N64TileN,
          kPersistentPtxM64N64WarpsM,
          kPersistentPtxM64N64WarpsN>(X, Wp, Y, ctas, repeats, warmup, dbvec, bounds_check);
    }
  } else {
    TORCH_CHECK(false, "unknown persistent PTX fp16 profile variant: ", variant);
  }

  py::dict result;
  result["forward_kernel_ms"] = kernel_ms;
  result["variant"] = variant;
  result["tile_m"] = tile_m;
  result["tile_n"] = tile_n;
  result["warps"] = warps;
  result["persistent_ctas"] = ctas;
  result["single_kernel"] = true;
  result["uses_ldmatrix"] = true;
  result["uses_cp_async"] = true;
  result["materializes_x_freq"] = false;
  result["materializes_y_freq"] = false;
  result["double_buffer_ab"] = dbvec;
  result["vectorized_spatial_store"] = dbvec;
  result["register_negates_w3"] = dbvec;
  result["bounds_check"] = bounds_check;
  result["vectorized_a_load"] = vecld;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_weight", &pack_weight, py::arg("W"));
  m.def("forward", &forward, py::arg("X"), py::arg("Wp"), py::arg("variant"));
  m.def(
      "profile_forward",
      &profile_forward,
      py::arg("X"),
      py::arg("Wp"),
      py::arg("variant"),
      py::arg("repeats") = 80,
      py::arg("warmup") = 20);
}
