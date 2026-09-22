#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace py = pybind11;
using namespace nvcuda;

namespace {

constexpr int kTranNum = 4;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kFwdTileM = 64;
constexpr int kFwdTileN = 32;
constexpr int kFwdWarpsM = 4;
constexpr int kFwdWarpsN = 2;
constexpr int kFwdWarps = kFwdWarpsM * kFwdWarpsN;
constexpr int kFwdThreads = kFwdWarps * 32;

constexpr int kFwdSmallDTileM = 32;
constexpr int kFwdSmallDTileN = 32;
constexpr int kFwdSmallDWarpsM = 2;
constexpr int kFwdSmallDWarpsN = 2;
constexpr int kFwdSmallDM32N64TileM = 32;
constexpr int kFwdSmallDM32N64TileN = 64;
constexpr int kFwdSmallDM32N64WarpsM = 2;
constexpr int kFwdSmallDM32N64WarpsN = 4;
constexpr int kFwdSmallDM16N64TileM = 16;
constexpr int kFwdSmallDM16N64TileN = 64;
constexpr int kFwdSmallDM16N64WarpsM = 1;
constexpr int kFwdSmallDM16N64WarpsN = 4;
constexpr int kFwdSmallDM16N32TileM = 16;
constexpr int kFwdSmallDM16N32TileN = 32;
constexpr int kFwdSmallDM16N32WarpsM = 1;
constexpr int kFwdSmallDM16N32WarpsN = 2;
constexpr int kFwdSmallDM16N16TileM = 16;
constexpr int kFwdSmallDM16N16TileN = 16;
constexpr int kFwdSmallDM16N16WarpsM = 1;
constexpr int kFwdSmallDM16N16WarpsN = 1;
constexpr int kFwdSmallDM64N64TileM = 64;
constexpr int kFwdSmallDM64N64TileN = 64;
constexpr int kFwdSmallDM64N64WarpsM = 4;
constexpr int kFwdSmallDM64N64WarpsN = 4;
constexpr int kFwdSmallDM64N16TileM = 64;
constexpr int kFwdSmallDM64N16TileN = 16;
constexpr int kFwdSmallDM64N16WarpsM = 4;
constexpr int kFwdSmallDM64N16WarpsN = 1;
constexpr int kFwdSmallDM32N128TileM = 32;
constexpr int kFwdSmallDM32N128TileN = 128;
constexpr int kFwdSmallDM32N128WarpsM = 2;
constexpr int kFwdSmallDM32N128WarpsN = 8;
constexpr int kDxTileM = 64;
constexpr int kDxTileN = 32;
constexpr int kDxWarpsM = 4;
constexpr int kDxWarpsN = 2;
constexpr int kDxWarps = kDxWarpsM * kDxWarpsN;
constexpr int kDxThreads = kDxWarps * 32;

constexpr int kDwTileM = 64;
constexpr int kDwTileN = 64;
constexpr int kDwWarpsM = 4;
constexpr int kDwWarpsN = 4;
constexpr int kDwWarps = kDwWarpsM * kDwWarpsN;
constexpr int kDwThreads = kDwWarps * 32;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be float16")

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err__ = (call); \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__)); \
  } while (0)

inline void check_tensor_4d_half_cuda(const torch::Tensor& t, const char* name) {
  CHECK_CUDA(t);
  CHECK_CONTIGUOUS(t);
  CHECK_HALF(t);
  TORCH_CHECK(t.dim() == 4, name, " must be 4D");
  TORCH_CHECK(t.size(-1) == kTranNum, name, " last dim must be 4");
}

inline void check_weight_3d_half_cuda(const torch::Tensor& t, const char* name) {
  CHECK_CUDA(t);
  CHECK_CONTIGUOUS(t);
  CHECK_HALF(t);
  TORCH_CHECK(t.dim() == 3, name, " must be 3D");
  TORCH_CHECK(t.size(-1) == kTranNum, name, " last dim must be 4");
}

__device__ __forceinline__ half hzero() {
  return __float2half(0.0f);
}

__device__ __forceinline__ half hquarter() {
  return __float2half(0.25f);
}

__device__ __forceinline__ half hhalf() {
  return __float2half(0.5f);
}

__device__ __forceinline__ void load_x_freq(
    const half* __restrict__ x_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3) {
  half x0 = x_ptr[0];
  half x1 = x_ptr[1];
  half x2 = x_ptr[2];
  half x3 = x_ptr[3];
  f0 = __hadd(__hadd(x0, x1), __hadd(x2, x3));
  f1 = __hsub(x0, x2);
  f2 = __hsub(__hadd(x0, x2), __hadd(x1, x3));
  f3 = __hsub(x3, x1);
}

__device__ __forceinline__ void load_dy_freq(
    const half* __restrict__ gy_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3) {
  half g0 = gy_ptr[0];
  half g1 = gy_ptr[1];
  half g2 = gy_ptr[2];
  half g3 = gy_ptr[3];
  half s0 = __hadd(g0, g2);
  half s1 = __hsub(g0, g2);
  half s2 = __hadd(g1, g3);
  half s3 = __hsub(g3, g1);
  f0 = __hmul(hquarter(), __hadd(s0, s2));
  f1 = __hmul(hhalf(), s1);
  f2 = __hmul(hquarter(), __hsub(s0, s2));
  f3 = __hmul(hhalf(), s3);
}

template <int TileM, int TileN, int WarpsM, int WarpsN, bool UseGauss = false>
__global__ void flash_eq_linear_forward_direct_gemm_fp16_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int APlanes = UseGauss ? 5 : 4;
  __shared__ half sA[APlanes][TileM][kWmmaK];
  __shared__ half sB[5][TileN][kWmmaK];
  __shared__ half sAcc[Warps][4][kWmmaM][kWmmaN];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id % WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc0;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc1;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc2;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc3;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc4;
  wmma::fill_fragment(acc0, hzero());
  wmma::fill_fragment(acc1, hzero());
  wmma::fill_fragment(acc2, hzero());
  wmma::fill_fragment(acc3, hzero());
  wmma::fill_fragment(acc4, hzero());

  for (int c_start = 0; c_start < inNum; c_start += kWmmaK) {
    for (int idx = threadIdx.x; idx < TileM * kWmmaK; idx += Threads) {
      int row = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int n = n_base + row;
      int c = c_start + kk;
      half out0 = hzero();
      half out1 = hzero();
      half out2 = hzero();
      half out3 = hzero();
      if (n < N && c < inNum) {
        const half* x_ptr = X + ((n * inNum + c) * kTranNum);
        load_x_freq(x_ptr, out0, out1, out2, out3);
      }
      if constexpr (UseGauss) {
        sA[0][row][kk] = out0;
        sA[1][row][kk] = out2;
        sA[2][row][kk] = __hadd(out1, out3);
        sA[3][row][kk] = out1;
        sA[4][row][kk] = out3;
      } else {
        sA[0][row][kk] = out0;
        sA[1][row][kk] = out1;
        sA[2][row][kk] = out2;
        sA[3][row][kk] = out3;
      }
    }

    for (int idx = threadIdx.x; idx < TileN * kWmmaK; idx += Threads) {
      int col = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      if (d < outNum && c < inNum) {
        const half* w_ptr = W + ((d * inNum + c) * kTranNum);
        w0 = w_ptr[0];
        w1 = w_ptr[1];
        w2 = w_ptr[2];
        w3 = w_ptr[3];
      }
      if constexpr (UseGauss) {
        sB[0][col][kk] = w0;
        sB[1][col][kk] = w2;
        sB[2][col][kk] = w1;
        sB[3][col][kk] = __hsub(w3, w1);
        sB[4][col][kk] = __hadd(w1, w3);
      } else {
        sB[0][col][kk] = w0;
        sB[1][col][kk] = w1;
        sB[2][col][kk] = w2;
        sB[3][col][kk] = w3;
        sB[4][col][kk] = __hneg(w3);
      }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a2;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a3;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a4;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b0;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b1;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b2;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b3;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b4;

    int warp_row = warp_m * kWmmaM;
    int warp_col = warp_n * kWmmaN;

    wmma::load_matrix_sync(a0, &sA[0][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a1, &sA[1][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a2, &sA[2][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a3, &sA[3][warp_row][0], kWmmaK);
    if constexpr (UseGauss) {
      wmma::load_matrix_sync(a4, &sA[4][warp_row][0], kWmmaK);
    }
    wmma::load_matrix_sync(b0, &sB[0][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b1, &sB[1][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b2, &sB[2][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b3, &sB[3][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b4, &sB[4][warp_col][0], kWmmaK);

    if constexpr (UseGauss) {
      wmma::mma_sync(acc0, a0, b0, acc0);
      wmma::mma_sync(acc2, a1, b1, acc2);
      wmma::mma_sync(acc1, a2, b2, acc1);
      wmma::mma_sync(acc3, a3, b3, acc3);
      wmma::mma_sync(acc4, a4, b4, acc4);
    } else {
      wmma::mma_sync(acc0, a0, b0, acc0);
      wmma::mma_sync(acc1, a1, b1, acc1);
      wmma::mma_sync(acc1, a3, b4, acc1);
      wmma::mma_sync(acc2, a2, b2, acc2);
      wmma::mma_sync(acc3, a1, b3, acc3);
      wmma::mma_sync(acc3, a3, b1, acc3);
    }
    __syncthreads();
  }

  if constexpr (UseGauss) {
    for (int i = 0; i < acc1.num_elements; ++i) {
      half p = acc1.x[i];
      acc1.x[i] = __hsub(p, acc4.x[i]);
      acc3.x[i] = __hadd(p, acc3.x[i]);
    }
  }

  wmma::store_matrix_sync(&sAcc[warp_id][0][0][0], acc0, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][1][0][0], acc1, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][2][0][0], acc2, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][3][0][0], acc3, kWmmaN, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane_id; idx < kWmmaM * kWmmaN; idx += 32) {
    int row = idx / kWmmaN;
    int col = idx % kWmmaN;
    int n = n_base + warp_m * kWmmaM + row;
    int d = d_base + warp_n * kWmmaN + col;
    if (n < N && d < outNum) {
      half a0v = sAcc[warp_id][0][row][col];
      half a1v = sAcc[warp_id][1][row][col];
      half a2v = sAcc[warp_id][2][row][col];
      half a3v = sAcc[warp_id][3][row][col];
      half t0 = __hmul(hquarter(), __hadd(a0v, a2v));
      half t1 = __hmul(hquarter(), __hsub(a0v, a2v));
      half t2 = __hmul(hhalf(), a1v);
      half t3 = __hmul(hhalf(), a3v);
      half* y_ptr = Y + ((n * outNum + d) * kTranNum);
      y_ptr[0] = __hadd(t0, t2);
      y_ptr[1] = __hsub(t1, t3);
      y_ptr[2] = __hsub(t0, t2);
      y_ptr[3] = __hadd(t1, t3);
    }
  }
}

__global__ void flash_eq_linear_backward_dx_direct_gemm_fp16_kernel(
    const half* __restrict__ dY,
    const half* __restrict__ W,
    half* __restrict__ dX,
    int N,
    int inNum,
    int outNum) {
  __shared__ half sA[4][kDxTileM][kWmmaK];
  __shared__ half sB[5][kDxTileN][kWmmaK];
  __shared__ half sAcc[kDxWarps][4][kWmmaM][kWmmaN];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / kDxWarpsN;
  int warp_n = warp_id % kDxWarpsN;

  int n_base = blockIdx.y * kDxTileM;
  int c_base = blockIdx.x * kDxTileN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc0;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc1;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc2;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc3;
  wmma::fill_fragment(acc0, hzero());
  wmma::fill_fragment(acc1, hzero());
  wmma::fill_fragment(acc2, hzero());
  wmma::fill_fragment(acc3, hzero());

  for (int d_start = 0; d_start < outNum; d_start += kWmmaK) {
    for (int idx = threadIdx.x; idx < kDxTileM * kWmmaK; idx += kDxThreads) {
      int row = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int n = n_base + row;
      int d = d_start + kk;
      half out0 = hzero();
      half out1 = hzero();
      half out2 = hzero();
      half out3 = hzero();
      if (n < N && d < outNum) {
        const half* gy_ptr = dY + ((n * outNum + d) * kTranNum);
        load_dy_freq(gy_ptr, out0, out1, out2, out3);
      }
      sA[0][row][kk] = out0;
      sA[1][row][kk] = out1;
      sA[2][row][kk] = out2;
      sA[3][row][kk] = out3;
    }

    for (int idx = threadIdx.x; idx < kDxTileN * kWmmaK; idx += kDxThreads) {
      int col = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int c = c_base + col;
      int d = d_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      if (c < inNum && d < outNum) {
        const half* w_ptr = W + ((d * inNum + c) * kTranNum);
        w0 = w_ptr[0];
        w1 = w_ptr[1];
        w2 = w_ptr[2];
        w3 = w_ptr[3];
      }
      sB[0][col][kk] = w0;
      sB[1][col][kk] = w1;
      sB[2][col][kk] = w2;
      sB[3][col][kk] = w3;
      sB[4][col][kk] = __hneg(w3);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a2;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a3;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b0;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b1;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b2;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b3;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b3neg;

    int warp_row = warp_m * kWmmaM;
    int warp_col = warp_n * kWmmaN;

    wmma::load_matrix_sync(a0, &sA[0][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a1, &sA[1][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a2, &sA[2][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a3, &sA[3][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(b0, &sB[0][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b1, &sB[1][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b2, &sB[2][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b3, &sB[3][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b3neg, &sB[4][warp_col][0], kWmmaK);

    wmma::mma_sync(acc0, a0, b0, acc0);
    wmma::mma_sync(acc1, a1, b1, acc1);
    wmma::mma_sync(acc1, a3, b3, acc1);
    wmma::mma_sync(acc2, a2, b2, acc2);
    wmma::mma_sync(acc3, a1, b3neg, acc3);
    wmma::mma_sync(acc3, a3, b1, acc3);
    __syncthreads();
  }

  wmma::store_matrix_sync(&sAcc[warp_id][0][0][0], acc0, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][1][0][0], acc1, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][2][0][0], acc2, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][3][0][0], acc3, kWmmaN, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane_id; idx < kWmmaM * kWmmaN; idx += 32) {
    int row = idx / kWmmaN;
    int col = idx % kWmmaN;
    int n = n_base + warp_m * kWmmaM + row;
    int c = c_base + warp_n * kWmmaN + col;
    if (n < N && c < inNum) {
      half a0v = sAcc[warp_id][0][row][col];
      half a1v = sAcc[warp_id][1][row][col];
      half a2v = sAcc[warp_id][2][row][col];
      half a3v = sAcc[warp_id][3][row][col];
      half t0 = __hadd(a0v, a2v);
      half t1 = __hsub(a0v, a2v);
      half* dx_ptr = dX + ((n * inNum + c) * kTranNum);
      dx_ptr[0] = __hadd(t0, a1v);
      dx_ptr[1] = __hsub(t1, a3v);
      dx_ptr[2] = __hsub(t0, a1v);
      dx_ptr[3] = __hadd(t1, a3v);
    }
  }
}

__global__ void flash_eq_linear_backward_dw_direct_gemm_fp16_kernel(
    const half* __restrict__ dY,
    const half* __restrict__ X,
    half* __restrict__ dW_accum,
    int N,
    int inNum,
    int outNum,
    int chunk_size) {
  __shared__ half sA[4][kDwTileM][kWmmaK];
  __shared__ half sB[5][kDwTileN][kWmmaK];
  __shared__ half sAcc[kDwWarps][4][kWmmaM][kWmmaN];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / kDwWarpsN;
  int warp_n = warp_id % kDwWarpsN;

  int d_base = blockIdx.y * kDwTileM;
  int c_base = blockIdx.x * kDwTileN;
  int n_start = blockIdx.z * chunk_size;
  int n_end = min(n_start + chunk_size, N);

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc0;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc1;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc2;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc3;
  wmma::fill_fragment(acc0, hzero());
  wmma::fill_fragment(acc1, hzero());
  wmma::fill_fragment(acc2, hzero());
  wmma::fill_fragment(acc3, hzero());

  for (int n_iter = n_start; n_iter < n_end; n_iter += kWmmaK) {
    for (int idx = threadIdx.x; idx < kDwTileM * kWmmaK; idx += kDwThreads) {
      int row = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int d = d_base + row;
      int n = n_iter + kk;
      half out0 = hzero();
      half out1 = hzero();
      half out2 = hzero();
      half out3 = hzero();
      if (d < outNum && n < n_end) {
        const half* gy_ptr = dY + ((n * outNum + d) * kTranNum);
        load_dy_freq(gy_ptr, out0, out1, out2, out3);
      }
      sA[0][row][kk] = out0;
      sA[1][row][kk] = out1;
      sA[2][row][kk] = out2;
      sA[3][row][kk] = out3;
    }

    for (int idx = threadIdx.x; idx < kDwTileN * kWmmaK; idx += kDwThreads) {
      int col = idx / kWmmaK;
      int kk = idx % kWmmaK;
      int c = c_base + col;
      int n = n_iter + kk;
      half x0 = hzero();
      half x1 = hzero();
      half x2 = hzero();
      half x3 = hzero();
      if (c < inNum && n < n_end) {
        const half* x_ptr = X + ((n * inNum + c) * kTranNum);
        load_x_freq(x_ptr, x0, x1, x2, x3);
      }
      sB[0][col][kk] = x0;
      sB[1][col][kk] = x1;
      sB[2][col][kk] = x2;
      sB[3][col][kk] = x3;
      sB[4][col][kk] = __hneg(x3);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a0;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a1;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a2;
    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a3;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b0;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b1;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b2;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b3;
    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b3neg;

    int warp_row = warp_m * kWmmaM;
    int warp_col = warp_n * kWmmaN;

    wmma::load_matrix_sync(a0, &sA[0][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a1, &sA[1][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a2, &sA[2][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(a3, &sA[3][warp_row][0], kWmmaK);
    wmma::load_matrix_sync(b0, &sB[0][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b1, &sB[1][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b2, &sB[2][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b3, &sB[3][warp_col][0], kWmmaK);
    wmma::load_matrix_sync(b3neg, &sB[4][warp_col][0], kWmmaK);

    wmma::mma_sync(acc0, a0, b0, acc0);
    wmma::mma_sync(acc1, a1, b1, acc1);
    wmma::mma_sync(acc1, a3, b3, acc1);
    wmma::mma_sync(acc2, a2, b2, acc2);
    wmma::mma_sync(acc3, a3, b1, acc3);
    wmma::mma_sync(acc3, a1, b3neg, acc3);
    __syncthreads();
  }

  wmma::store_matrix_sync(&sAcc[warp_id][0][0][0], acc0, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][1][0][0], acc1, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][2][0][0], acc2, kWmmaN, wmma::mem_row_major);
  wmma::store_matrix_sync(&sAcc[warp_id][3][0][0], acc3, kWmmaN, wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane_id; idx < kWmmaM * kWmmaN; idx += 32) {
    int row = idx / kWmmaN;
    int col = idx % kWmmaN;
    int d = d_base + warp_m * kWmmaM + row;
    int c = c_base + warp_n * kWmmaN + col;
    if (d < outNum && c < inNum) {
      half* dw_ptr = dW_accum + ((d * inNum + c) * kTranNum);
      atomicAdd(dw_ptr + 0, sAcc[warp_id][0][row][col]);
      atomicAdd(dw_ptr + 1, sAcc[warp_id][1][row][col]);
      atomicAdd(dw_ptr + 2, sAcc[warp_id][2][row][col]);
      atomicAdd(dw_ptr + 3, sAcc[warp_id][3][row][col]);
    }
  }
}

template <typename Fn>
float measure_cuda_ms(Fn&& fn, int repeats, int warmup = 20) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return elapsed_ms / static_cast<float>(repeats);
}

void configure_kernel_attributes() {
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdTileM,
            kFwdTileN,
            kFwdWarpsM,
            kFwdWarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDTileM,
            kFwdSmallDTileN,
            kFwdSmallDWarpsM,
            kFwdSmallDWarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDM32N64TileM,
            kFwdSmallDM32N64TileN,
            kFwdSmallDM32N64WarpsM,
            kFwdSmallDM32N64WarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDTileM,
            kFwdSmallDTileN,
            kFwdSmallDWarpsM,
            kFwdSmallDWarpsN,
            true>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDM32N64TileM,
            kFwdSmallDM32N64TileN,
            kFwdSmallDM32N64WarpsM,
            kFwdSmallDM32N64WarpsN,
            true>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDM16N64TileM,
            kFwdSmallDM16N64TileN,
            kFwdSmallDM16N64WarpsM,
            kFwdSmallDM16N64WarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDM16N32TileM,
            kFwdSmallDM16N32TileN,
            kFwdSmallDM16N32WarpsM,
            kFwdSmallDM16N32WarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_forward_direct_gemm_fp16_kernel<
            kFwdSmallDM16N16TileM,
            kFwdSmallDM16N16TileN,
            kFwdSmallDM16N16WarpsM,
            kFwdSmallDM16N16WarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_backward_dx_direct_gemm_fp16_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    cudaFuncSetAttribute(
        flash_eq_linear_backward_dw_direct_gemm_fp16_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    configured = true;
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
torch::Tensor launch_forward_direct_gemm_fp16(torch::Tensor X, torch::Tensor W) {
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));

  auto Y = torch::empty({X.size(0), X.size(1), outNum, kTranNum}, X.options());
  constexpr int Threads = WarpsM * WarpsN * 32;
  dim3 block(Threads);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  flash_eq_linear_forward_direct_gemm_fp16_kernel<TileM, TileN, WarpsM, WarpsN>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
torch::Tensor launch_forward_direct_gemm_fp16_gauss(torch::Tensor X, torch::Tensor W) {
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));

  auto Y = torch::empty({X.size(0), X.size(1), outNum, kTranNum}, X.options());
  constexpr int Threads = WarpsM * WarpsN * 32;
  dim3 block(Threads);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  flash_eq_linear_forward_direct_gemm_fp16_kernel<TileM, TileN, WarpsM, WarpsN, true>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  CUDA_CHECK(cudaGetLastError());
  return Y;
}

torch::Tensor flash_eq_linear_forward_direct_gemm_fp16_ada4090(torch::Tensor X, torch::Tensor W) {
  return launch_forward_direct_gemm_fp16<kFwdTileM, kFwdTileN, kFwdWarpsM, kFwdWarpsN>(X, W);
}

torch::Tensor flash_eq_linear_forward_direct_gemm_fp16_variant_ada4090(
    torch::Tensor X,
    torch::Tensor W,
    const std::string& variant) {
  if (variant == "pure_old_m32n32_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDTileM,
        kFwdSmallDTileN,
        kFwdSmallDWarpsM,
        kFwdSmallDWarpsN>(X, W);
  }
  if (variant == "pure_old_m32n32_gauss_smalld") {
    return launch_forward_direct_gemm_fp16_gauss<
        kFwdSmallDTileM,
        kFwdSmallDTileN,
        kFwdSmallDWarpsM,
        kFwdSmallDWarpsN>(X, W);
  }
  if (variant == "pure_old_m32n64_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM32N64TileM,
        kFwdSmallDM32N64TileN,
        kFwdSmallDM32N64WarpsM,
        kFwdSmallDM32N64WarpsN>(X, W);
  }
  if (variant == "pure_old_m16n64_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM16N64TileM,
        kFwdSmallDM16N64TileN,
        kFwdSmallDM16N64WarpsM,
        kFwdSmallDM16N64WarpsN>(X, W);
  }
  if (variant == "pure_old_m16n32_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM16N32TileM,
        kFwdSmallDM16N32TileN,
        kFwdSmallDM16N32WarpsM,
        kFwdSmallDM16N32WarpsN>(X, W);
  }
  if (variant == "pure_old_m16n16_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM16N16TileM,
        kFwdSmallDM16N16TileN,
        kFwdSmallDM16N16WarpsM,
        kFwdSmallDM16N16WarpsN>(X, W);
  }
  if (variant == "pure_old_m32n64_smalld_pad64") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM32N64TileM,
        kFwdSmallDM32N64TileN,
        kFwdSmallDM32N64WarpsM,
        kFwdSmallDM32N64WarpsN>(X, W);
  }
  if (variant == "pure_old_m32n64_gauss_smalld_pad64") {
    return launch_forward_direct_gemm_fp16_gauss<
        kFwdSmallDM32N64TileM,
        kFwdSmallDM32N64TileN,
        kFwdSmallDM32N64WarpsM,
        kFwdSmallDM32N64WarpsN>(X, W);
  }
  if (variant == "pure_old_m64n64_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM64N64TileM,
        kFwdSmallDM64N64TileN,
        kFwdSmallDM64N64WarpsM,
        kFwdSmallDM64N64WarpsN>(X, W);
  }
  if (variant == "pure_old_m64n16_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM64N16TileM,
        kFwdSmallDM64N16TileN,
        kFwdSmallDM64N16WarpsM,
        kFwdSmallDM64N16WarpsN>(X, W);
  }
  if (variant == "pure_old_m32n128_smalld") {
    return launch_forward_direct_gemm_fp16<
        kFwdSmallDM32N128TileM,
        kFwdSmallDM32N128TileN,
        kFwdSmallDM32N128WarpsM,
        kFwdSmallDM32N128WarpsN>(X, W);
  }
  if (variant == "pure_old_direct") {
    return flash_eq_linear_forward_direct_gemm_fp16_ada4090(X, W);
  }
  TORCH_CHECK(false, "unknown old direct fp16 variant: ", variant);
}

std::vector<torch::Tensor> flash_eq_linear_backward_direct_gemm_fp16_ada4090(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W) {
  check_tensor_4d_half_cuda(dY, "dY");
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");
  TORCH_CHECK(dY.size(2) == W.size(0), "output channel mismatch");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));

  auto dX = torch::empty_like(X);
  auto dW = torch::zeros_like(W);

  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  dim3 block_dx(kDxThreads);
  dim3 grid_dx((inNum + kDxTileN - 1) / kDxTileN, (N + kDxTileM - 1) / kDxTileM);
  flash_eq_linear_backward_dx_direct_gemm_fp16_kernel<<<grid_dx, block_dx, 0, stream>>>(
      reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dX.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  CUDA_CHECK(cudaGetLastError());

  int split_k = 64;
  if (N <= 1024) split_k = 1;
  else if (N <= 4096) split_k = 8;
  else if (N <= 16384) split_k = 32;
  int chunk_size = (N + split_k - 1) / split_k;

  dim3 block_dw(kDwThreads);
  dim3 grid_dw((inNum + kDwTileN - 1) / kDwTileN, (outNum + kDwTileM - 1) / kDwTileM, split_k);
  flash_eq_linear_backward_dw_direct_gemm_fp16_kernel<<<grid_dw, block_dw, 0, stream>>>(
      reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(dW.data_ptr<at::Half>()),
      N,
      inNum,
      outNum,
      chunk_size);
  CUDA_CHECK(cudaGetLastError());

  return {dX, dW};
}

py::dict flash_eq_linear_profile_direct_gemm_fp16_ada4090(
    torch::Tensor X,
    torch::Tensor dY,
    torch::Tensor W,
    int repeats,
    int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_tensor_4d_half_cuda(dY, "dY");
  check_weight_3d_half_cuda(W, "W");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));

  auto Y = torch::empty({X.size(0), X.size(1), outNum, kTranNum}, X.options());
  auto dX = torch::empty_like(X);
  auto dW = torch::zeros_like(W);

  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  dim3 block_fwd(kFwdThreads);
  dim3 grid_fwd((outNum + kFwdTileN - 1) / kFwdTileN, (N + kFwdTileM - 1) / kFwdTileM);
  dim3 block_dx(kDxThreads);
  dim3 grid_dx((inNum + kDxTileN - 1) / kDxTileN, (N + kDxTileM - 1) / kDxTileM);

  int split_k = 64;
  if (N <= 1024) split_k = 1;
  else if (N <= 4096) split_k = 8;
  else if (N <= 16384) split_k = 32;
  int chunk_size = (N + split_k - 1) / split_k;

  dim3 block_dw(kDwThreads);
  dim3 grid_dw((inNum + kDwTileN - 1) / kDwTileN, (outNum + kDwTileM - 1) / kDwTileM, split_k);

  auto run_forward = [&] {
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdTileM,
        kFwdTileN,
        kFwdWarpsM,
        kFwdWarpsN><<<grid_fwd, block_fwd, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };

  auto run_dx = [&] {
    flash_eq_linear_backward_dx_direct_gemm_fp16_kernel<<<grid_dx, block_dx, 0, stream>>>(
        reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dX.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };

  auto run_dw_stage1 = [&] {
    dW.zero_();
    flash_eq_linear_backward_dw_direct_gemm_fp16_kernel<<<grid_dw, block_dw, 0, stream>>>(
        reinterpret_cast<const half*>(dY.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<half*>(dW.data_ptr<at::Half>()),
        N,
        inNum,
        outNum,
        chunk_size);
  };

  auto run_backward = [&] {
    auto out = flash_eq_linear_backward_direct_gemm_fp16_ada4090(dY, X, W);
    (void)out;
  };

  float forward_kernel_ms = measure_cuda_ms(run_forward, repeats, warmup);
  float dx_kernel_ms = measure_cuda_ms(run_dx, repeats, warmup);
  float dw_stage1_ms = measure_cuda_ms(run_dw_stage1, repeats, warmup);
  float dw_stage2_ms = 0.0f;
  float backward_total_ms = measure_cuda_ms(run_backward, repeats, warmup);

  py::dict result;
  result["forward_kernel_ms"] = forward_kernel_ms;
  result["dx_kernel_ms"] = dx_kernel_ms;
  result["dw_stage1_ms"] = dw_stage1_ms;
  result["dw_stage2_ms"] = dw_stage2_ms;
  result["backward_total_ms"] = backward_total_ms;
  result["split_k"] = split_k;
  result["chunk_size"] = chunk_size;
  return result;
}

py::dict flash_eq_linear_profile_forward_direct_gemm_fp16_variant_ada4090(
    torch::Tensor X,
    torch::Tensor W,
    const std::string& variant,
    int repeats,
    int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");
  configure_kernel_attributes();

  c10::cuda::CUDAGuard guard(X.device());
  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));
  auto Y = torch::empty({X.size(0), X.size(1), outNum, kTranNum}, X.options());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  auto run_default = [&] {
    constexpr int Threads = kFwdWarpsM * kFwdWarpsN * 32;
    dim3 block(Threads);
    dim3 grid((outNum + kFwdTileN - 1) / kFwdTileN, (N + kFwdTileM - 1) / kFwdTileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdTileM,
        kFwdTileN,
        kFwdWarpsM,
        kFwdWarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };

  auto run_smalld = [&] {
    constexpr int Threads = kFwdSmallDWarpsM * kFwdSmallDWarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDTileN - 1) / kFwdSmallDTileN,
        (N + kFwdSmallDTileM - 1) / kFwdSmallDTileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDTileM,
        kFwdSmallDTileN,
        kFwdSmallDWarpsM,
        kFwdSmallDWarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_gauss_smalld = [&] {
    constexpr int Threads = kFwdSmallDWarpsM * kFwdSmallDWarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDTileN - 1) / kFwdSmallDTileN,
        (N + kFwdSmallDTileM - 1) / kFwdSmallDTileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDTileM,
        kFwdSmallDTileN,
        kFwdSmallDWarpsM,
        kFwdSmallDWarpsN,
        true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m32n64 = [&] {
    constexpr int Threads = kFwdSmallDM32N64WarpsM * kFwdSmallDM32N64WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM32N64TileN - 1) / kFwdSmallDM32N64TileN,
        (N + kFwdSmallDM32N64TileM - 1) / kFwdSmallDM32N64TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM32N64TileM,
        kFwdSmallDM32N64TileN,
        kFwdSmallDM32N64WarpsM,
        kFwdSmallDM32N64WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_gauss_smalld_m32n64 = [&] {
    constexpr int Threads = kFwdSmallDM32N64WarpsM * kFwdSmallDM32N64WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM32N64TileN - 1) / kFwdSmallDM32N64TileN,
        (N + kFwdSmallDM32N64TileM - 1) / kFwdSmallDM32N64TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM32N64TileM,
        kFwdSmallDM32N64TileN,
        kFwdSmallDM32N64WarpsM,
        kFwdSmallDM32N64WarpsN,
        true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m16n64 = [&] {
    constexpr int Threads = kFwdSmallDM16N64WarpsM * kFwdSmallDM16N64WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM16N64TileN - 1) / kFwdSmallDM16N64TileN,
        (N + kFwdSmallDM16N64TileM - 1) / kFwdSmallDM16N64TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM16N64TileM,
        kFwdSmallDM16N64TileN,
        kFwdSmallDM16N64WarpsM,
        kFwdSmallDM16N64WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m16n32 = [&] {
    constexpr int Threads = kFwdSmallDM16N32WarpsM * kFwdSmallDM16N32WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM16N32TileN - 1) / kFwdSmallDM16N32TileN,
        (N + kFwdSmallDM16N32TileM - 1) / kFwdSmallDM16N32TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM16N32TileM,
        kFwdSmallDM16N32TileN,
        kFwdSmallDM16N32WarpsM,
        kFwdSmallDM16N32WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m16n16 = [&] {
    constexpr int Threads = kFwdSmallDM16N16WarpsM * kFwdSmallDM16N16WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM16N16TileN - 1) / kFwdSmallDM16N16TileN,
        (N + kFwdSmallDM16N16TileM - 1) / kFwdSmallDM16N16TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM16N16TileM,
        kFwdSmallDM16N16TileN,
        kFwdSmallDM16N16WarpsM,
        kFwdSmallDM16N16WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m64n64 = [&] {
    constexpr int Threads = kFwdSmallDM64N64WarpsM * kFwdSmallDM64N64WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM64N64TileN - 1) / kFwdSmallDM64N64TileN,
        (N + kFwdSmallDM64N64TileM - 1) / kFwdSmallDM64N64TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM64N64TileM,
        kFwdSmallDM64N64TileN,
        kFwdSmallDM64N64WarpsM,
        kFwdSmallDM64N64WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m64n16 = [&] {
    constexpr int Threads = kFwdSmallDM64N16WarpsM * kFwdSmallDM64N16WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM64N16TileN - 1) / kFwdSmallDM64N16TileN,
        (N + kFwdSmallDM64N16TileM - 1) / kFwdSmallDM64N16TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM64N16TileM,
        kFwdSmallDM64N16TileN,
        kFwdSmallDM64N16WarpsM,
        kFwdSmallDM64N16WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };
  auto run_smalld_m32n128 = [&] {
    constexpr int Threads = kFwdSmallDM32N128WarpsM * kFwdSmallDM32N128WarpsN * 32;
    dim3 block(Threads);
    dim3 grid(
        (outNum + kFwdSmallDM32N128TileN - 1) / kFwdSmallDM32N128TileN,
        (N + kFwdSmallDM32N128TileM - 1) / kFwdSmallDM32N128TileM);
    flash_eq_linear_forward_direct_gemm_fp16_kernel<
        kFwdSmallDM32N128TileM,
        kFwdSmallDM32N128TileN,
        kFwdSmallDM32N128WarpsM,
        kFwdSmallDM32N128WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  };

  float forward_kernel_ms = 0.0f;
  if (variant == "pure_old_m32n32_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld, repeats, warmup);
  } else if (variant == "pure_old_m32n32_gauss_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_gauss_smalld, repeats, warmup);
  } else if (variant == "pure_old_m32n64_smalld" || variant == "pure_old_m32n64_smalld_pad64") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m32n64, repeats, warmup);
  } else if (variant == "pure_old_m32n64_gauss_smalld_pad64") {
    forward_kernel_ms = measure_cuda_ms(run_gauss_smalld_m32n64, repeats, warmup);
  } else if (variant == "pure_old_m16n64_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m16n64, repeats, warmup);
  } else if (variant == "pure_old_m16n32_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m16n32, repeats, warmup);
  } else if (variant == "pure_old_m16n16_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m16n16, repeats, warmup);
  } else if (variant == "pure_old_m64n64_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m64n64, repeats, warmup);
  } else if (variant == "pure_old_m64n16_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m64n16, repeats, warmup);
  } else if (variant == "pure_old_m32n128_smalld") {
    forward_kernel_ms = measure_cuda_ms(run_smalld_m32n128, repeats, warmup);
  } else if (variant == "pure_old_direct") {
    forward_kernel_ms = measure_cuda_ms(run_default, repeats, warmup);
  } else {
    TORCH_CHECK(false, "unknown old direct fp16 profile variant: ", variant);
  }
  CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = forward_kernel_ms;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "flash_eq_linear_forward_direct_gemm_fp16_ada4090",
      &flash_eq_linear_forward_direct_gemm_fp16_ada4090,
      "Direct structured GEMM fp16 Tensor Core forward");
  m.def(
      "flash_eq_linear_forward_direct_gemm_fp16_variant_ada4090",
      &flash_eq_linear_forward_direct_gemm_fp16_variant_ada4090,
      py::arg("X"),
      py::arg("W"),
      py::arg("variant"),
      "Direct structured GEMM fp16 Tensor Core forward variants");
  m.def(
      "flash_eq_linear_backward_direct_gemm_fp16_ada4090",
      &flash_eq_linear_backward_direct_gemm_fp16_ada4090,
      "Direct structured GEMM fp16 Tensor Core backward");
  m.def(
      "flash_eq_linear_profile_direct_gemm_fp16_ada4090",
      &flash_eq_linear_profile_direct_gemm_fp16_ada4090,
      py::arg("X"),
      py::arg("dY"),
      py::arg("W"),
      py::arg("repeats") = 80,
      py::arg("warmup") = 20,
      "Direct structured GEMM fp16 Tensor Core profile");
  m.def(
      "flash_eq_linear_profile_forward_direct_gemm_fp16_variant_ada4090",
      &flash_eq_linear_profile_forward_direct_gemm_fp16_variant_ada4090,
      py::arg("X"),
      py::arg("W"),
      py::arg("variant"),
      py::arg("repeats") = 80,
      py::arg("warmup") = 20,
      "Forward-only direct structured GEMM fp16 Tensor Core variant profile");
}
