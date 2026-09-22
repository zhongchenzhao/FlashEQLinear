#pragma once

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <type_traits>

namespace flash_eq_fp16_pipeline {

namespace py = pybind11;
using namespace nvcuda;

constexpr int kTranNum = 4;
constexpr int kPackedTranNum = 5;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kMmaAtomM = 16;
constexpr int kMmaAtomN = 8;
constexpr int kMmaAtomK = 16;

#define FLASH_EQ_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define FLASH_EQ_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define FLASH_EQ_CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be float16")

#define FLASH_EQ_CUDA_CHECK(call) \
  do { \
    cudaError_t err__ = (call); \
    TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__)); \
  } while (0)

inline void check_tensor_4d_half_cuda(const torch::Tensor& t, const char* name) {
  FLASH_EQ_CHECK_CUDA(t);
  FLASH_EQ_CHECK_CONTIGUOUS(t);
  FLASH_EQ_CHECK_HALF(t);
  TORCH_CHECK(t.dim() == 4, name, " must be 4D");
  TORCH_CHECK(t.size(-1) == kTranNum, name, " last dim must be 4");
}

inline void check_weight_3d_half_cuda(const torch::Tensor& t, const char* name) {
  FLASH_EQ_CHECK_CUDA(t);
  FLASH_EQ_CHECK_CONTIGUOUS(t);
  FLASH_EQ_CHECK_HALF(t);
  TORCH_CHECK(t.dim() == 3, name, " must be 3D");
  TORCH_CHECK(t.size(-1) == kTranNum, name, " last dim must be 4");
}

inline void check_packed_weight_3d_half_cuda(const torch::Tensor& t, const char* name) {
  FLASH_EQ_CHECK_CUDA(t);
  FLASH_EQ_CHECK_CONTIGUOUS(t);
  FLASH_EQ_CHECK_HALF(t);
  TORCH_CHECK(t.dim() == 3, name, " must be 3D");
  TORCH_CHECK(t.size(0) == kPackedTranNum, name, " first dim must be 5");
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

template <typename T>
__device__ __forceinline__ T zero_value();

template <>
__device__ __forceinline__ half zero_value<half>() {
  return hzero();
}

template <>
__device__ __forceinline__ float zero_value<float>() {
  return 0.0f;
}

__device__ __forceinline__ uint32_t pack_half2_u32(half lo, half hi);

__device__ __forceinline__ void load_x_freq_half(
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

__device__ __forceinline__ void load_x_freq_half_vec4(
    const half* __restrict__ x_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3) {
  union Pack {
    uint64_t u;
    half h[4];
  };
  Pack p;
  p.u = *reinterpret_cast<const uint64_t*>(x_ptr);
  half x0 = p.h[0];
  half x1 = p.h[1];
  half x2 = p.h[2];
  half x3 = p.h[3];
  f0 = __hadd(__hadd(x0, x1), __hadd(x2, x3));
  f1 = __hsub(x0, x2);
  f2 = __hsub(__hadd(x0, x2), __hadd(x1, x3));
  f3 = __hsub(x3, x1);
}

__device__ __forceinline__ void load_x_freq_float(
    const half* __restrict__ x_ptr,
    float& f0,
    float& f1,
    float& f2,
    float& f3) {
  float x0 = __half2float(x_ptr[0]);
  float x1 = __half2float(x_ptr[1]);
  float x2 = __half2float(x_ptr[2]);
  float x3 = __half2float(x_ptr[3]);
  f0 = x0 + x1 + x2 + x3;
  f1 = x0 - x2;
  f2 = (x0 + x2) - (x1 + x3);
  f3 = x3 - x1;
}

template <typename AccT>
__device__ __forceinline__ void load_x_freq_to_half(
    const half* __restrict__ x_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3);

template <>
__device__ __forceinline__ void load_x_freq_to_half<half>(
    const half* __restrict__ x_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3) {
  load_x_freq_half(x_ptr, f0, f1, f2, f3);
}

template <>
__device__ __forceinline__ void load_x_freq_to_half<float>(
    const half* __restrict__ x_ptr,
    half& f0,
    half& f1,
    half& f2,
    half& f3) {
  float t0, t1, t2, t3;
  load_x_freq_float(x_ptr, t0, t1, t2, t3);
  f0 = __float2half_rn(t0);
  f1 = __float2half_rn(t1);
  f2 = __float2half_rn(t2);
  f3 = __float2half_rn(t3);
}

template <typename AccT>
__device__ __forceinline__ void write_spatial_output(
    const AccT& a0v,
    const AccT& a1v,
    const AccT& a2v,
    const AccT& a3v,
    half* __restrict__ y_ptr);

template <>
__device__ __forceinline__ void write_spatial_output<half>(
    const half& a0v,
    const half& a1v,
    const half& a2v,
    const half& a3v,
    half* __restrict__ y_ptr) {
  half t0 = __hmul(hquarter(), __hadd(a0v, a2v));
  half t1 = __hmul(hquarter(), __hsub(a0v, a2v));
  half t2 = __hmul(hhalf(), a1v);
  half t3 = __hmul(hhalf(), a3v);
  y_ptr[0] = __hadd(t0, t2);
  y_ptr[1] = __hsub(t1, t3);
  y_ptr[2] = __hsub(t0, t2);
  y_ptr[3] = __hadd(t1, t3);
}

__device__ __forceinline__ void write_spatial_output_half4(
    const half& a0v,
    const half& a1v,
    const half& a2v,
    const half& a3v,
    half* __restrict__ y_ptr) {
  half t0 = __hmul(hquarter(), __hadd(a0v, a2v));
  half t1 = __hmul(hquarter(), __hsub(a0v, a2v));
  half t2 = __hmul(hhalf(), a1v);
  half t3 = __hmul(hhalf(), a3v);
  uint32_t lo = pack_half2_u32(__hadd(t0, t2), __hsub(t1, t3));
  uint32_t hi = pack_half2_u32(__hsub(t0, t2), __hadd(t1, t3));
  uint64_t packed = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
  *reinterpret_cast<uint64_t*>(y_ptr) = packed;
}

template <>
__device__ __forceinline__ void write_spatial_output<float>(
    const float& a0v,
    const float& a1v,
    const float& a2v,
    const float& a3v,
    half* __restrict__ y_ptr) {
  float t0 = 0.25f * (a0v + a2v);
  float t1 = 0.25f * (a0v - a2v);
  float t2 = 0.5f * a1v;
  float t3 = 0.5f * a3v;
  y_ptr[0] = __float2half_rn(t0 + t2);
  y_ptr[1] = __float2half_rn(t1 - t3);
  y_ptr[2] = __float2half_rn(t0 - t2);
  y_ptr[3] = __float2half_rn(t1 + t3);
}

__global__ void pack_weight_fp16_pipeline_kernel(
    const half* __restrict__ W,
    half* __restrict__ Wp,
    int inNum,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outNum * inNum;
  if (idx >= total) {
    return;
  }
  int d = idx / inNum;
  int c = idx - d * inNum;
  const half* w = W + ((d * inNum + c) * kTranNum);
  int base = d * inNum + c;
  Wp[0 * outNum * inNum + base] = w[0];
  Wp[1 * outNum * inNum + base] = w[1];
  Wp[2 * outNum * inNum + base] = w[2];
  Wp[3 * outNum * inNum + base] = w[3];
  Wp[4 * outNum * inNum + base] = __hneg(w[3]);
}

__device__ __forceinline__ uint32_t pack_half2_u32(half lo, half hi) {
  union Pack {
    half h[2];
    uint32_t u;
  };
  Pack p;
  p.h[0] = lo;
  p.h[1] = hi;
  return p.u;
}

__device__ __forceinline__ half unpack_half_u32(uint32_t x, int idx) {
  union Pack {
    uint32_t u;
    half h[2];
  };
  Pack p;
  p.u = x;
  return p.h[idx];
}

__device__ __forceinline__ uint32_t shared_ptr_to_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src) {
  uint32_t smem = shared_ptr_to_u32(smem_dst);
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(smem), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::);
}

template <int BarrierId, int ThreadCount>
__device__ __forceinline__ void named_barrier_sync() {
  asm volatile("bar.sync %0, %1;\n" : : "n"(BarrierId), "n"(ThreadCount));
}

__device__ __forceinline__ int ldmatrix_swizzled_offset(int row8, int col8) {
  return row8 * 8 + col8;
}

__device__ __forceinline__ void ldmatrix_x4(const half* ptr, int lane, uint32_t (&dst)[4]) {
  int mat = lane >> 3;
  int row = lane & 7;
  uint32_t smem_ptr = shared_ptr_to_u32(ptr + (mat * 64 + row * 8));
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
      : "r"(smem_ptr));
}

__device__ __forceinline__ void ldmatrix_x2(const half* ptr, int lane, uint32_t (&dst)[2]) {
  int mat = (lane >> 3) & 1;
  int row = lane & 7;
  uint32_t smem_ptr = shared_ptr_to_u32(ptr + (mat * 64 + row * 8));
  asm volatile(
      "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(dst[0]), "=r"(dst[1])
      : "r"(smem_ptr));
}

__global__ void debug_ldmatrix_x4_kernel(float* __restrict__ out) {
  __shared__ half tile[4][8][8];
  int lane = threadIdx.x & 31;
  for (int idx = threadIdx.x; idx < 4 * 8 * 8; idx += blockDim.x) {
    int mat = idx / 64;
    int rem = idx - mat * 64;
    int row = rem >> 3;
    int col = rem & 7;
    tile[mat][row][col] = __float2half(static_cast<float>(mat * 100 + row * 10 + col));
  }
  __syncthreads();

  uint32_t regs[4];
  ldmatrix_x4(&tile[0][0][0], lane, regs);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    out[(lane * 4 + r) * 2 + 0] = __half2float(unpack_half_u32(regs[r], 0));
    out[(lane * 4 + r) * 2 + 1] = __half2float(unpack_half_u32(regs[r], 1));
  }
}

inline torch::Tensor debug_ldmatrix_x4_mapping() {
  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  auto out = torch::empty({32, 4, 2}, options);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  debug_ldmatrix_x4_kernel<<<1, 32, 0, stream>>>(out.data_ptr<float>());
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return out;
}

__device__ __forceinline__ int mma_a_layout_offset(int lane, int v) {
  int t0 = lane & 3;
  int t1 = lane >> 2;
  int v0 = v & 1;
  int v1 = (v >> 1) & 1;
  int v2 = (v >> 2) & 1;
  return t0 * 32 + t1 + v0 * 16 + v1 * 8 + v2 * 128;
}

__device__ __forceinline__ int mma_b_layout_offset(int lane, int v) {
  int t0 = lane & 3;
  int t1 = lane >> 2;
  int v0 = v & 1;
  int v1 = (v >> 1) & 1;
  return t0 * 16 + t1 + v0 * 8 + v1 * 64;
}

__device__ __forceinline__ int mma_c_layout_offset(int lane, int v) {
  int t0 = lane & 3;
  int t1 = lane >> 2;
  int v0 = v & 1;
  int v1 = (v >> 1) & 1;
  return t0 * 32 + t1 + v0 * 16 + v1 * 8;
}

template <int TileM>
__device__ __forceinline__ void load_mma_a_regs(
    const half (&sA)[TileM][kMmaAtomK],
    int warp_row,
    int lane,
    uint32_t (&a)[4]) {
  half frag[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int offset = mma_a_layout_offset(lane, i);
    int row = offset & 15;
    int kk = offset >> 4;
    frag[i] = sA[warp_row + row][kk];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    a[i] = pack_half2_u32(frag[2 * i], frag[2 * i + 1]);
  }
}

template <int TileN>
__device__ __forceinline__ void load_mma_b_regs(
    const half (&sB)[TileN][kMmaAtomK],
    int warp_col,
    int lane,
    uint32_t (&b)[2]) {
  half frag[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int offset = mma_b_layout_offset(lane, i);
    int col = offset & 7;
    int kk = offset >> 3;
    frag[i] = sB[warp_col + col][kk];
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    b[i] = pack_half2_u32(frag[2 * i], frag[2 * i + 1]);
  }
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    uint32_t (&c)[2],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
  uint32_t d0;
  uint32_t d1;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0, %1}, "
      "{%2, %3, %4, %5}, "
      "{%6, %7}, "
      "{%8, %9};\n"
      : "=r"(d0), "=r"(d1)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "r"(c[0]), "r"(c[1]));
  c[0] = d0;
  c[1] = d1;
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    float (&c)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2]) {
  float d0;
  float d1;
  float d2;
  float d3;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
  c[0] = d0;
  c[1] = d1;
  c[2] = d2;
  c[3] = d3;
}

template <typename AccT>
__device__ __forceinline__ void accumulate_formula_scalar(
    AccT& y0,
    AccT& y1,
    AccT& y2,
    AccT& y3,
    half x0h,
    half x1h,
    half x2h,
    half x3h,
    half w0h,
    half w1h,
    half w2h,
    half w3h);

template <>
__device__ __forceinline__ void accumulate_formula_scalar<half>(
    half& y0,
    half& y1,
    half& y2,
    half& y3,
    half x0,
    half x1,
    half x2,
    half x3,
    half w0,
    half w1,
    half w2,
    half w3) {
  half q = hquarter();
  half h = hhalf();
  half c00 = __hadd(__hadd(__hmul(q, w0), __hmul(h, w1)), __hmul(q, w2));
  half c01 = __hadd(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c02 = __hadd(__hsub(__hmul(q, w0), __hmul(h, w1)), __hmul(q, w2));
  half c03 = __hsub(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));

  half c10 = __hsub(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c11 = __hadd(__hadd(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w1));
  half c12 = __hadd(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c13 = __hadd(__hadd(__hmul(q, w0), __hmul(q, w2)), __hneg(__hmul(h, w1)));

  half c20 = __hadd(__hsub(__hmul(q, w0), __hmul(h, w1)), __hmul(q, w2));
  half c21 = __hsub(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c22 = __hadd(__hadd(__hmul(q, w0), __hmul(h, w1)), __hmul(q, w2));
  half c23 = __hadd(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));

  half c30 = __hadd(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c31 = __hsub(__hadd(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w1));
  half c32 = __hsub(__hsub(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w3));
  half c33 = __hadd(__hadd(__hmul(q, w0), __hmul(q, w2)), __hmul(h, w1));

  y0 = __hfma(x0, c00, __hfma(x1, c01, __hfma(x2, c02, __hfma(x3, c03, y0))));
  y1 = __hfma(x0, c10, __hfma(x1, c11, __hfma(x2, c12, __hfma(x3, c13, y1))));
  y2 = __hfma(x0, c20, __hfma(x1, c21, __hfma(x2, c22, __hfma(x3, c23, y2))));
  y3 = __hfma(x0, c30, __hfma(x1, c31, __hfma(x2, c32, __hfma(x3, c33, y3))));
}

template <>
__device__ __forceinline__ void accumulate_formula_scalar<float>(
    float& y0,
    float& y1,
    float& y2,
    float& y3,
    half x0h,
    half x1h,
    half x2h,
    half x3h,
    half w0h,
    half w1h,
    half w2h,
    half w3h) {
  float x0 = __half2float(x0h);
  float x1 = __half2float(x1h);
  float x2 = __half2float(x2h);
  float x3 = __half2float(x3h);
  float w0 = __half2float(w0h);
  float w1 = __half2float(w1h);
  float w2 = __half2float(w2h);
  float w3 = __half2float(w3h);

  float c00 = 0.25f * w0 + 0.5f * w1 + 0.25f * w2;
  float c01 = 0.25f * w0 - 0.25f * w2 + 0.5f * w3;
  float c02 = 0.25f * w0 - 0.5f * w1 + 0.25f * w2;
  float c03 = 0.25f * w0 - 0.25f * w2 - 0.5f * w3;

  float c10 = 0.25f * w0 - 0.25f * w2 - 0.5f * w3;
  float c11 = 0.25f * w0 + 0.25f * w2 + 0.5f * w1;
  float c12 = 0.25f * w0 - 0.25f * w2 + 0.5f * w3;
  float c13 = 0.25f * w0 + 0.25f * w2 - 0.5f * w1;

  float c20 = 0.25f * w0 - 0.5f * w1 + 0.25f * w2;
  float c21 = 0.25f * w0 - 0.25f * w2 - 0.5f * w3;
  float c22 = 0.25f * w0 + 0.5f * w1 + 0.25f * w2;
  float c23 = 0.25f * w0 - 0.25f * w2 + 0.5f * w3;

  float c30 = 0.25f * w0 - 0.25f * w2 + 0.5f * w3;
  float c31 = 0.25f * w0 + 0.25f * w2 - 0.5f * w1;
  float c32 = 0.25f * w0 - 0.25f * w2 - 0.5f * w3;
  float c33 = 0.25f * w0 + 0.25f * w2 + 0.5f * w1;

  y0 = fmaf(x0, c00, fmaf(x1, c01, fmaf(x2, c02, fmaf(x3, c03, y0))));
  y1 = fmaf(x0, c10, fmaf(x1, c11, fmaf(x2, c12, fmaf(x3, c13, y1))));
  y2 = fmaf(x0, c20, fmaf(x1, c21, fmaf(x2, c22, fmaf(x3, c23, y2))));
  y3 = fmaf(x0, c30, fmaf(x1, c31, fmaf(x2, c32, fmaf(x3, c33, y3))));
}

template <typename AccT>
__global__ void forward_formula_scalar_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outNum;
  if (idx >= total) {
    return;
  }

  int d = idx % outNum;
  int n = idx / outNum;
  int plane = outNum * inNum;
  AccT y0 = zero_value<AccT>();
  AccT y1 = zero_value<AccT>();
  AccT y2 = zero_value<AccT>();
  AccT y3 = zero_value<AccT>();

  for (int c = 0; c < inNum; ++c) {
    const half* x = X + ((n * inNum + c) * kTranNum);
    int base = d * inNum + c;
    accumulate_formula_scalar<AccT>(
        y0,
        y1,
        y2,
        y3,
        x[0],
        x[1],
        x[2],
        x[3],
        Wp[0 * plane + base],
        Wp[1 * plane + base],
        Wp[2 * plane + base],
        Wp[3 * plane + base]);
  }

  half* y = Y + ((n * outNum + d) * kTranNum);
  if constexpr (std::is_same<AccT, half>::value) {
    y[0] = y0;
    y[1] = y1;
    y[2] = y2;
    y[3] = y3;
  } else {
    y[0] = __float2half_rn(y0);
    y[1] = __float2half_rn(y1);
    y[2] = __float2half_rn(y2);
    y[3] = __float2half_rn(y3);
  }
}

template <int TileN, int WarpsN, int NFrags, int Threads>
__device__ __forceinline__ void load_w_tile_cpasync(
    const half* __restrict__ Wp,
    half (&sB)[5][WarpsN][NFrags][2][8][8],
    int d_base,
    int c_start,
    int inNum,
    int outNum) {
  int plane_size = outNum * inNum;
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

    if (d < outNum && c + 7 < inNum) {
      const half* src = Wp + plane_id * plane_size + d * inNum + c;
      cp_async_cg_16(dst, src);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        half value = hzero();
        if (d < outNum && c + i < inNum) {
          value = Wp[plane_id * plane_size + d * inNum + c + i];
        }
        dst[i] = value;
      }
    }
  }
}

template <int TileN, int WarpsN, int NFrags, int Threads>
__device__ __forceinline__ void load_w_tile_cpasync_threaded(
    const half* __restrict__ Wp,
    half (&sB)[5][WarpsN][NFrags][2][8][8],
    int d_base,
    int c_start,
    int inNum,
    int outNum,
    int thread_idx) {
  int plane_size = outNum * inNum;
  constexpr int GroupsK = 2;
  constexpr int TotalGroups = kPackedTranNum * TileN * GroupsK;
  for (int idx = thread_idx; idx < TotalGroups; idx += Threads) {
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

    if (d < outNum && c + 7 < inNum) {
      const half* src = Wp + plane_id * plane_size + d * inNum + c;
      cp_async_cg_16(dst, src);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        half value = hzero();
        if (d < outNum && c + i < inNum) {
          value = Wp[plane_id * plane_size + d * inNum + c + i];
        }
        dst[i] = value;
      }
    }
  }
}

template <int TileM, int WarpsM, int Threads>
__device__ __forceinline__ void load_a_tile_half(
    const half* __restrict__ X,
    half (&sA)[4][WarpsM][4][8][8],
    int n_base,
    int c_start,
    int N,
    int inNum) {
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
}

template <int TileM, int WarpsM, int Threads>
__device__ __forceinline__ void load_a_tile_half_threaded(
    const half* __restrict__ X,
    half (&sA)[4][WarpsM][4][8][8],
    int n_base,
    int c_start,
    int N,
    int inNum,
    int thread_idx) {
  for (int idx = thread_idx; idx < TileM * kMmaAtomK; idx += Threads) {
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
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_mma_pipe_f16_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
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

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[4][NFrags][2];
#pragma unroll
  for (int f = 0; f < 4; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      acc[f][nf][0] = 0;
      acc[f][nf][1] = 0;
    }
  }

  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
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

    for (int idx = threadIdx.x; idx < TileN * kMmaAtomK; idx += Threads) {
      int col = idx / kMmaAtomK;
      int kk = idx - col * kMmaAtomK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      half w3neg = hzero();
      if (d < outNum && c < inNum) {
        int base = d * inNum + c;
        int plane = outNum * inNum;
        w0 = Wp[0 * plane + base];
        w1 = Wp[1 * plane + base];
        w2 = Wp[2 * plane + base];
        w3 = Wp[3 * plane + base];
        w3neg = Wp[4 * plane + base];
      }
      int wn = col / (NFrags * kMmaAtomN);
      int nfrag = (col / kMmaAtomN) - wn * NFrags;
      int mat = kk >> 3;
      int off = ldmatrix_swizzled_offset(col & 7, kk & 7);
      int sr = off >> 3;
      int sc = off & 7;
      sB[0][wn][nfrag][mat][sr][sc] = w0;
      sB[1][wn][nfrag][mat][sr][sc] = w1;
      sB[2][wn][nfrag][mat][sr][sc] = w2;
      sB[3][wn][nfrag][mat][sr][sc] = w3;
      sB[4][wn][nfrag][mat][sr][sc] = w3neg;
    }
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
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void phase_x_transform_shared_kernel(
    const half* __restrict__ X,
    half* __restrict__ sink,
    int N,
    int inNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  __shared__ half sA[4][WarpsM][4][8][8];

  int n_base = blockIdx.y * TileM;
  half carry = hzero();
  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
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
    __syncthreads();
    if (threadIdx.x == 0) {
      carry = __hadd(carry, sA[0][0][0][0][0]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sink[blockIdx.y * gridDim.x + blockIdx.x] = carry;
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int OutTiles>
__global__ void forward_mma_pipe_f16_multi_n_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");
  static_assert(OutTiles >= 1 && OutTiles <= 4, "OutTiles must be in [1, 4]");

  __shared__ half sA[4][WarpsM][4][8][8];
  __shared__ half sB[5][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_group_base = blockIdx.x * (TileN * OutTiles);
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[OutTiles][4][NFrags][2];
#pragma unroll
  for (int ot = 0; ot < OutTiles; ++ot) {
#pragma unroll
    for (int f = 0; f < 4; ++f) {
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        acc[ot][f][nf][0] = 0;
        acc[ot][f][nf][1] = 0;
      }
    }
  }

  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
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
    __syncthreads();

    uint32_t a0[4], a1[4], a2[4], a3[4];
    ldmatrix_x4(&sA[0][warp_m][0][0][0], lane_id, a0);
    ldmatrix_x4(&sA[1][warp_m][0][0][0], lane_id, a1);
    ldmatrix_x4(&sA[2][warp_m][0][0][0], lane_id, a2);
    ldmatrix_x4(&sA[3][warp_m][0][0][0], lane_id, a3);

#pragma unroll
    for (int ot = 0; ot < OutTiles; ++ot) {
      int d_base = d_group_base + ot * TileN;
      for (int idx = threadIdx.x; idx < TileN * kMmaAtomK; idx += Threads) {
        int col = idx / kMmaAtomK;
        int kk = idx - col * kMmaAtomK;
        int d = d_base + col;
        int c = c_start + kk;
        half w0 = hzero();
        half w1 = hzero();
        half w2 = hzero();
        half w3 = hzero();
        half w3neg = hzero();
        if (d < outNum && c < inNum) {
          int base = d * inNum + c;
          int plane = outNum * inNum;
          w0 = Wp[0 * plane + base];
          w1 = Wp[1 * plane + base];
          w2 = Wp[2 * plane + base];
          w3 = Wp[3 * plane + base];
          w3neg = Wp[4 * plane + base];
        }
        int wn = col / (NFrags * kMmaAtomN);
        int nfrag = (col / kMmaAtomN) - wn * NFrags;
        int mat = kk >> 3;
        int off = ldmatrix_swizzled_offset(col & 7, kk & 7);
        int sr = off >> 3;
        int sc = off & 7;
        sB[0][wn][nfrag][mat][sr][sc] = w0;
        sB[1][wn][nfrag][mat][sr][sc] = w1;
        sB[2][wn][nfrag][mat][sr][sc] = w2;
        sB[3][wn][nfrag][mat][sr][sc] = w3;
        sB[4][wn][nfrag][mat][sr][sc] = w3neg;
      }
      __syncthreads();

#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        uint32_t b0[2], b1[2], b2[2], b3[2], b3neg[2];
        ldmatrix_x2(&sB[0][warp_n][nf][0][0][0], lane_id, b0);
        ldmatrix_x2(&sB[1][warp_n][nf][0][0][0], lane_id, b1);
        ldmatrix_x2(&sB[2][warp_n][nf][0][0][0], lane_id, b2);
        ldmatrix_x2(&sB[3][warp_n][nf][0][0][0], lane_id, b3);
        ldmatrix_x2(&sB[4][warp_n][nf][0][0][0], lane_id, b3neg);
        mma_m16n8k16_f16(acc[ot][0][nf], a0, b0);
        mma_m16n8k16_f16(acc[ot][1][nf], a1, b1);
        mma_m16n8k16_f16(acc[ot][1][nf], a3, b3neg);
        mma_m16n8k16_f16(acc[ot][2][nf], a2, b2);
        mma_m16n8k16_f16(acc[ot][3][nf], a1, b3);
        mma_m16n8k16_f16(acc[ot][3][nf], a3, b1);
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int ot = 0; ot < OutTiles; ++ot) {
    int d_base = d_group_base + ot * TileN;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      int offset = mma_c_layout_offset(lane_id, v);
      int row = offset & 15;
      int col = offset >> 4;
      int n = n_base + warp_row + row;
#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        int d = d_base + warp_col + nf * kMmaAtomN + col;
        half a0v = unpack_half_u32(acc[ot][0][nf][v >> 1], v & 1);
        half a1v = unpack_half_u32(acc[ot][1][nf][v >> 1], v & 1);
        half a2v = unpack_half_u32(acc[ot][2][nf][v >> 1], v & 1);
        half a3v = unpack_half_u32(acc[ot][3][nf][v >> 1], v & 1);
        if (n < N && d < outNum) {
          write_spatial_output<half>(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
        }
      }
    }
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void phase_w_load_shared_kernel(
    const half* __restrict__ Wp,
    half* __restrict__ sink,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  __shared__ half sB[5][WarpsN][NFrags][2][8][8];

  int d_base = blockIdx.x * TileN;
  half carry = hzero();
  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
    for (int idx = threadIdx.x; idx < TileN * kMmaAtomK; idx += Threads) {
      int col = idx / kMmaAtomK;
      int kk = idx - col * kMmaAtomK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      half w3neg = hzero();
      if (d < outNum && c < inNum) {
        int base = d * inNum + c;
        int plane = outNum * inNum;
        w0 = Wp[0 * plane + base];
        w1 = Wp[1 * plane + base];
        w2 = Wp[2 * plane + base];
        w3 = Wp[3 * plane + base];
        w3neg = Wp[4 * plane + base];
      }
      int wn = col / (NFrags * kMmaAtomN);
      int nfrag = (col / kMmaAtomN) - wn * NFrags;
      int mat = kk >> 3;
      int off = ldmatrix_swizzled_offset(col & 7, kk & 7);
      int sr = off >> 3;
      int sc = off & 7;
      sB[0][wn][nfrag][mat][sr][sc] = w0;
      sB[1][wn][nfrag][mat][sr][sc] = w1;
      sB[2][wn][nfrag][mat][sr][sc] = w2;
      sB[3][wn][nfrag][mat][sr][sc] = w3;
      sB[4][wn][nfrag][mat][sr][sc] = w3neg;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry = __hadd(carry, sB[0][0][0][0][0][0]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sink[blockIdx.y * gridDim.x + blockIdx.x] = carry;
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void phase_empty_sync_kernel(half* __restrict__ sink, int inNum) {
  half carry = hzero();
  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
    __syncthreads();
    if (threadIdx.x == 0) {
      carry = __hadd(carry, __float2half(static_cast<float>(c_start & 1)));
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sink[blockIdx.y * gridDim.x + blockIdx.x] = carry;
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void phase_epilogue_store_kernel(
    half* __restrict__ Y,
    int N,
    int outNum) {
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;
  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

#pragma unroll
  for (int v = 0; v < 4; ++v) {
    int offset = mma_c_layout_offset(lane_id, v);
    int row = offset & 15;
    int col = offset >> 4;
    int n = n_base + warp_row + row;
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      int d = d_base + warp_col + nf * kMmaAtomN + col;
      if (n < N && d < outNum) {
        write_spatial_output<half>(hzero(), hzero(), hzero(), hzero(), Y + ((n * outNum + d) * kTranNum));
      }
    }
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_mma_pipe_f16_gauss_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");

  __shared__ half sA[5][WarpsM][4][8][8];
  __shared__ half sB[5][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[5][NFrags][2];
#pragma unroll
  for (int f = 0; f < 5; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      acc[f][nf][0] = 0;
      acc[f][nf][1] = 0;
    }
  }

  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
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
      sA[4][wm][mat][sr][sc] = __hadd(x1, x3);
    }

    for (int idx = threadIdx.x; idx < TileN * kMmaAtomK; idx += Threads) {
      int col = idx / kMmaAtomK;
      int kk = idx - col * kMmaAtomK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      if (d < outNum && c < inNum) {
        int base = d * inNum + c;
        int plane = outNum * inNum;
        w0 = Wp[0 * plane + base];
        w1 = Wp[1 * plane + base];
        w2 = Wp[2 * plane + base];
        w3 = Wp[3 * plane + base];
      }
      int wn = col / (NFrags * kMmaAtomN);
      int nfrag = (col / kMmaAtomN) - wn * NFrags;
      int mat = kk >> 3;
      int off = ldmatrix_swizzled_offset(col & 7, kk & 7);
      int sr = off >> 3;
      int sc = off & 7;
      sB[0][wn][nfrag][mat][sr][sc] = w0;
      sB[1][wn][nfrag][mat][sr][sc] = w1;
      sB[2][wn][nfrag][mat][sr][sc] = w2;
      sB[3][wn][nfrag][mat][sr][sc] = w3;
      sB[4][wn][nfrag][mat][sr][sc] = __hadd(w1, w3);
    }
    __syncthreads();

    uint32_t a0[4], a1[4], a2[4], a3[4], a13[4];
    ldmatrix_x4(&sA[0][warp_m][0][0][0], lane_id, a0);
    ldmatrix_x4(&sA[1][warp_m][0][0][0], lane_id, a1);
    ldmatrix_x4(&sA[2][warp_m][0][0][0], lane_id, a2);
    ldmatrix_x4(&sA[3][warp_m][0][0][0], lane_id, a3);
    ldmatrix_x4(&sA[4][warp_m][0][0][0], lane_id, a13);
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      uint32_t b0[2], b1[2], b2[2], b3[2], b13[2];
      ldmatrix_x2(&sB[0][warp_n][nf][0][0][0], lane_id, b0);
      ldmatrix_x2(&sB[1][warp_n][nf][0][0][0], lane_id, b1);
      ldmatrix_x2(&sB[2][warp_n][nf][0][0][0], lane_id, b2);
      ldmatrix_x2(&sB[3][warp_n][nf][0][0][0], lane_id, b3);
      ldmatrix_x2(&sB[4][warp_n][nf][0][0][0], lane_id, b13);
      mma_m16n8k16_f16(acc[0][nf], a0, b0);
      mma_m16n8k16_f16(acc[1][nf], a1, b1);
      mma_m16n8k16_f16(acc[2][nf], a2, b2);
      mma_m16n8k16_f16(acc[3][nf], a3, b3);
      mma_m16n8k16_f16(acc[4][nf], a13, b13);
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
      half p = unpack_half_u32(acc[1][nf][v >> 1], v & 1);
      half a2v = unpack_half_u32(acc[2][nf][v >> 1], v & 1);
      half q = unpack_half_u32(acc[3][nf][v >> 1], v & 1);
      half t = unpack_half_u32(acc[4][nf][v >> 1], v & 1);
      half a1v = __hsub(p, q);
      half a3v = __hsub(__hsub(t, p), q);
      if (n < N && d < outNum) {
        write_spatial_output<half>(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
      }
    }
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_mma_pipe_f16_cpasync_w_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
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

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[4][NFrags][2];
#pragma unroll
  for (int f = 0; f < 4; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      acc[f][nf][0] = 0;
      acc[f][nf][1] = 0;
    }
  }

  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
    load_w_tile_cpasync<TileN, WarpsN, NFrags, Threads>(Wp, sB, d_base, c_start, inNum, outNum);
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
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_mma_pipe_f16_double_buffer_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");

  __shared__ half sA[2][4][WarpsM][4][8][8];
  __shared__ half sB[2][5][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[4][NFrags][2];
#pragma unroll
  for (int f = 0; f < 4; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      acc[f][nf][0] = 0;
      acc[f][nf][1] = 0;
    }
  }

  load_w_tile_cpasync<TileN, WarpsN, NFrags, Threads>(Wp, sB[0], d_base, 0, inNum, outNum);
  cp_async_commit_group();
  load_a_tile_half<TileM, WarpsM, Threads>(X, sA[0], n_base, 0, N, inNum);
  cp_async_wait_all();
  __syncthreads();

  for (int c_start = 0, tile = 0; c_start < inNum; c_start += kMmaAtomK, ++tile) {
    int buf = tile & 1;
    int next_buf = buf ^ 1;
    int next_c = c_start + kMmaAtomK;
    bool has_next = next_c < inNum;

    if (has_next) {
      load_w_tile_cpasync<TileN, WarpsN, NFrags, Threads>(Wp, sB[next_buf], d_base, next_c, inNum, outNum);
      cp_async_commit_group();
    }

    uint32_t a0[4], a1[4], a2[4], a3[4];
    ldmatrix_x4(&sA[buf][0][warp_m][0][0][0], lane_id, a0);
    ldmatrix_x4(&sA[buf][1][warp_m][0][0][0], lane_id, a1);
    ldmatrix_x4(&sA[buf][2][warp_m][0][0][0], lane_id, a2);
    ldmatrix_x4(&sA[buf][3][warp_m][0][0][0], lane_id, a3);

    if (has_next) {
      load_a_tile_half<TileM, WarpsM, Threads>(X, sA[next_buf], n_base, next_c, N, inNum);
    }

#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      uint32_t b0[2], b1[2], b2[2], b3[2], b3neg[2];
      ldmatrix_x2(&sB[buf][0][warp_n][nf][0][0][0], lane_id, b0);
      ldmatrix_x2(&sB[buf][1][warp_n][nf][0][0][0], lane_id, b1);
      ldmatrix_x2(&sB[buf][2][warp_n][nf][0][0][0], lane_id, b2);
      ldmatrix_x2(&sB[buf][3][warp_n][nf][0][0][0], lane_id, b3);
      ldmatrix_x2(&sB[buf][4][warp_n][nf][0][0][0], lane_id, b3neg);
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
      if (n < N && d < outNum) {
        write_spatial_output<half>(a0v, a1v, a2v, a3v, Y + ((n * outNum + d) * kTranNum));
      }
    }
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int ProducerWarps>
__global__ void forward_mma_pipe_f16_warpspec_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int ConsumerWarps = WarpsM * WarpsN;
  constexpr int ConsumerThreads = ConsumerWarps * 32;
  constexpr int ProducerThreads = ProducerWarps * 32;
  constexpr int Threads = ConsumerThreads + ProducerThreads;
  constexpr int NFrags = TileN / (WarpsN * kMmaAtomN);
  static_assert(TileN == WarpsN * NFrags * kMmaAtomN, "TileN must divide warp N fragments");
  static_assert(ProducerWarps > 0, "Need at least one producer warp");

  __shared__ half sA[2][4][WarpsM][4][8][8];
  __shared__ half sB[2][5][WarpsN][NFrags][2][8][8];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  bool is_consumer = warp_id < ConsumerWarps;
  bool is_producer = !is_consumer;
  int producer_thread = threadIdx.x - ConsumerThreads;

  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  uint32_t acc[4][NFrags][2];
#pragma unroll
  for (int f = 0; f < 4; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
      acc[f][nf][0] = 0;
      acc[f][nf][1] = 0;
    }
  }

  if (is_producer) {
    load_w_tile_cpasync_threaded<TileN, WarpsN, NFrags, ProducerThreads>(
        Wp, sB[0], d_base, 0, inNum, outNum, producer_thread);
    cp_async_commit_group();
    load_a_tile_half_threaded<TileM, WarpsM, ProducerThreads>(
        X, sA[0], n_base, 0, N, inNum, producer_thread);
    cp_async_wait_all();
  }
  named_barrier_sync<1, Threads>();

  for (int c_start = 0, tile = 0; c_start < inNum; c_start += kMmaAtomK, ++tile) {
    int buf = tile & 1;
    int next_buf = buf ^ 1;
    int next_c = c_start + kMmaAtomK;
    bool has_next = next_c < inNum;

    if (is_producer && has_next) {
      load_w_tile_cpasync_threaded<TileN, WarpsN, NFrags, ProducerThreads>(
          Wp, sB[next_buf], d_base, next_c, inNum, outNum, producer_thread);
      cp_async_commit_group();
      load_a_tile_half_threaded<TileM, WarpsM, ProducerThreads>(
          X, sA[next_buf], n_base, next_c, N, inNum, producer_thread);
      cp_async_wait_all();
    }

    if (is_consumer) {
      uint32_t a0[4], a1[4], a2[4], a3[4];
      ldmatrix_x4(&sA[buf][0][warp_m][0][0][0], lane_id, a0);
      ldmatrix_x4(&sA[buf][1][warp_m][0][0][0], lane_id, a1);
      ldmatrix_x4(&sA[buf][2][warp_m][0][0][0], lane_id, a2);
      ldmatrix_x4(&sA[buf][3][warp_m][0][0][0], lane_id, a3);

#pragma unroll
      for (int nf = 0; nf < NFrags; ++nf) {
        uint32_t b0[2], b1[2], b2[2], b3[2], b3neg[2];
        ldmatrix_x2(&sB[buf][0][warp_n][nf][0][0][0], lane_id, b0);
        ldmatrix_x2(&sB[buf][1][warp_n][nf][0][0][0], lane_id, b1);
        ldmatrix_x2(&sB[buf][2][warp_n][nf][0][0][0], lane_id, b2);
        ldmatrix_x2(&sB[buf][3][warp_n][nf][0][0][0], lane_id, b3);
        ldmatrix_x2(&sB[buf][4][warp_n][nf][0][0][0], lane_id, b3neg);
        mma_m16n8k16_f16(acc[0][nf], a0, b0);
        mma_m16n8k16_f16(acc[1][nf], a1, b1);
        mma_m16n8k16_f16(acc[1][nf], a3, b3neg);
        mma_m16n8k16_f16(acc[2][nf], a2, b2);
        mma_m16n8k16_f16(acc[3][nf], a1, b3);
        mma_m16n8k16_f16(acc[3][nf], a3, b1);
      }
    }

    if (has_next) {
      named_barrier_sync<1, Threads>();
    }
  }

  if (!is_consumer) {
    return;
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
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_mma_pipe_f32_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
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

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;
  int warp_row = warp_m * kMmaAtomM;
  int warp_col = warp_n * (NFrags * kMmaAtomN);

  float acc[4][NFrags][4];
#pragma unroll
  for (int f = 0; f < 4; ++f) {
#pragma unroll
    for (int nf = 0; nf < NFrags; ++nf) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        acc[f][nf][v] = 0.0f;
      }
    }
  }

  for (int c_start = 0; c_start < inNum; c_start += kMmaAtomK) {
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
        load_x_freq_to_half<float>(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
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

    for (int idx = threadIdx.x; idx < TileN * kMmaAtomK; idx += Threads) {
      int col = idx / kMmaAtomK;
      int kk = idx - col * kMmaAtomK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      half w3neg = hzero();
      if (d < outNum && c < inNum) {
        int base = d * inNum + c;
        int plane = outNum * inNum;
        w0 = Wp[0 * plane + base];
        w1 = Wp[1 * plane + base];
        w2 = Wp[2 * plane + base];
        w3 = Wp[3 * plane + base];
        w3neg = Wp[4 * plane + base];
      }
      int wn = col / (NFrags * kMmaAtomN);
      int nfrag = (col / kMmaAtomN) - wn * NFrags;
      int mat = kk >> 3;
      int off = ldmatrix_swizzled_offset(col & 7, kk & 7);
      int sr = off >> 3;
      int sc = off & 7;
      sB[0][wn][nfrag][mat][sr][sc] = w0;
      sB[1][wn][nfrag][mat][sr][sc] = w1;
      sB[2][wn][nfrag][mat][sr][sc] = w2;
      sB[3][wn][nfrag][mat][sr][sc] = w3;
      sB[4][wn][nfrag][mat][sr][sc] = w3neg;
    }
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
      mma_m16n8k16_f32(acc[0][nf], a0, b0);
      mma_m16n8k16_f32(acc[1][nf], a1, b1);
      mma_m16n8k16_f32(acc[1][nf], a3, b3neg);
      mma_m16n8k16_f32(acc[2][nf], a2, b2);
      mma_m16n8k16_f32(acc[3][nf], a1, b3);
      mma_m16n8k16_f32(acc[3][nf], a3, b1);
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
      if (n < N && d < outNum) {
        write_spatial_output<float>(
            acc[0][nf][v], acc[1][nf][v], acc[2][nf][v], acc[3][nf][v],
            Y + ((n * outNum + d) * kTranNum));
      }
    }
  }
}

template <typename AccT, int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void forward_fused_wmma_pipeline_kernel(
    const half* __restrict__ X,
    const half* __restrict__ Wp,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;

  __shared__ half sA[4][TileM][kWmmaK];
  __shared__ half sB[5][TileN][kWmmaK];
  __shared__ AccT sAcc[Warps][4][kWmmaM][kWmmaN];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id - warp_m * WarpsN;

  int n_base = blockIdx.y * TileM;
  int d_base = blockIdx.x * TileN;

  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, AccT> acc0;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, AccT> acc1;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, AccT> acc2;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, AccT> acc3;
  wmma::fill_fragment(acc0, zero_value<AccT>());
  wmma::fill_fragment(acc1, zero_value<AccT>());
  wmma::fill_fragment(acc2, zero_value<AccT>());
  wmma::fill_fragment(acc3, zero_value<AccT>());

  for (int c_start = 0; c_start < inNum; c_start += kWmmaK) {
    for (int idx = threadIdx.x; idx < TileM * kWmmaK; idx += Threads) {
      int row = idx / kWmmaK;
      int kk = idx - row * kWmmaK;
      int n = n_base + row;
      int c = c_start + kk;
      half x0 = hzero();
      half x1 = hzero();
      half x2 = hzero();
      half x3 = hzero();
      if (n < N && c < inNum) {
        load_x_freq_to_half<AccT>(X + ((n * inNum + c) * kTranNum), x0, x1, x2, x3);
      }
      sA[0][row][kk] = x0;
      sA[1][row][kk] = x1;
      sA[2][row][kk] = x2;
      sA[3][row][kk] = x3;
    }

    for (int idx = threadIdx.x; idx < TileN * kWmmaK; idx += Threads) {
      int col = idx / kWmmaK;
      int kk = idx - col * kWmmaK;
      int d = d_base + col;
      int c = c_start + kk;
      half w0 = hzero();
      half w1 = hzero();
      half w2 = hzero();
      half w3 = hzero();
      half w3neg = hzero();
      if (d < outNum && c < inNum) {
        int base = d * inNum + c;
        int plane = outNum * inNum;
        w0 = Wp[0 * plane + base];
        w1 = Wp[1 * plane + base];
        w2 = Wp[2 * plane + base];
        w3 = Wp[3 * plane + base];
        w3neg = Wp[4 * plane + base];
      }
      sB[0][col][kk] = w0;
      sB[1][col][kk] = w1;
      sB[2][col][kk] = w2;
      sB[3][col][kk] = w3;
      sB[4][col][kk] = w3neg;
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
    wmma::mma_sync(acc1, a3, b3neg, acc1);
    wmma::mma_sync(acc2, a2, b2, acc2);
    wmma::mma_sync(acc3, a1, b3, acc3);
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
    int col = idx - row * kWmmaN;
    int n = n_base + warp_m * kWmmaM + row;
    int d = d_base + warp_n * kWmmaN + col;
    if (n < N && d < outNum) {
      write_spatial_output<AccT>(
          sAcc[warp_id][0][row][col],
          sAcc[warp_id][1][row][col],
          sAcc[warp_id][2][row][col],
          sAcc[warp_id][3][row][col],
          Y + ((n * outNum + d) * kTranNum));
    }
  }
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline void configure_pipeline_kernels() {
  cudaFuncSetAttribute(
      forward_fused_wmma_pipeline_kernel<half, TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  cudaFuncSetAttribute(
      forward_fused_wmma_pipeline_kernel<float, TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline void configure_mma_ptx_kernels() {
  cudaFuncSetAttribute(
      forward_mma_pipe_f16_kernel<TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  cudaFuncSetAttribute(
      forward_mma_pipe_f16_cpasync_w_kernel<TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  cudaFuncSetAttribute(
      forward_mma_pipe_f16_double_buffer_kernel<TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  cudaFuncSetAttribute(
      forward_mma_pipe_f32_kernel<TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline void configure_mma_ptx_gauss_kernels() {
  cudaFuncSetAttribute(
      forward_mma_pipe_f16_gauss_kernel<TileM, TileN, WarpsM, WarpsN>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int ProducerWarps>
inline void configure_mma_warpspec_kernels() {
  cudaFuncSetAttribute(
      forward_mma_pipe_f16_warpspec_kernel<TileM, TileN, WarpsM, WarpsN, ProducerWarps>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
}

inline torch::Tensor pack_weight_fp16_pipeline(torch::Tensor W) {
  check_weight_3d_half_cuda(W, "W");
  c10::cuda::CUDAGuard device_guard(W.device());
  int outNum = static_cast<int>(W.size(0));
  int inNum = static_cast<int>(W.size(1));
  auto Wp = torch::empty({kPackedTranNum, outNum, inNum}, W.options());
  int elements = outNum * inNum;
  int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  pack_weight_fp16_pipeline_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Wp.data_ptr<at::Half>()),
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Wp;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline torch::Tensor forward_pipeline_impl(torch::Tensor X, torch::Tensor Wp, bool mixed) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (mixed) {
    forward_fused_wmma_pipeline_kernel<float, TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  } else {
    forward_fused_wmma_pipeline_kernel<half, TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  }
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline torch::Tensor forward_mma_ptx_impl(torch::Tensor X, torch::Tensor Wp, bool mixed) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (mixed) {
    forward_mma_pipe_f32_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  } else {
    forward_mma_pipe_f16_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  }
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int OutTiles>
inline torch::Tensor forward_mma_ptx_multi_n_impl(torch::Tensor X, torch::Tensor Wp) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN * OutTiles - 1) / (TileN * OutTiles), (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_mma_pipe_f16_multi_n_kernel<TileM, TileN, WarpsM, WarpsN, OutTiles><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_mma_ptx_phase_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  auto sink = torch::empty({grid.x * grid.y}, X.options());
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  auto measure = [&](auto launch) {
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
    FLASH_EQ_CUDA_CHECK(cudaGetLastError());
    return elapsed_ms / static_cast<float>(repeats);
  };

  float x_ms = measure([&]() {
    phase_x_transform_shared_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sink.data_ptr<at::Half>()),
        N,
        inNum);
  });
  float w_ms = measure([&]() {
    phase_w_load_shared_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(sink.data_ptr<at::Half>()),
        inNum,
        outNum);
  });
  float sync_ms = measure([&]() {
    phase_empty_sync_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(sink.data_ptr<at::Half>()),
        inNum);
  });
  float epilogue_ms = measure([&]() {
    phase_epilogue_store_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        outNum);
  });

  py::dict result;
  result["x_transform_shared_sync_ms"] = x_ms;
  result["w_load_shared_sync_ms"] = w_ms;
  result["empty_sync_ms"] = sync_ms;
  result["epilogue_store_ms"] = epilogue_ms;
  result["grid_x"] = static_cast<int>(grid.x);
  result["grid_y"] = static_cast<int>(grid.y);
  result["threads"] = WarpsM * WarpsN * 32;
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline torch::Tensor forward_mma_ptx_cpasync_w_impl(torch::Tensor X, torch::Tensor Wp) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_mma_pipe_f16_cpasync_w_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline torch::Tensor forward_mma_ptx_gauss_impl(torch::Tensor X, torch::Tensor Wp) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_mma_pipe_f16_gauss_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline torch::Tensor forward_mma_ptx_double_buffer_impl(torch::Tensor X, torch::Tensor Wp) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_mma_pipe_f16_double_buffer_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int ProducerWarps>
inline torch::Tensor forward_mma_ptx_warpspec_impl(torch::Tensor X, torch::Tensor Wp) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  constexpr int ConsumerWarps = WarpsM * WarpsN;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  dim3 block((ConsumerWarps + ProducerWarps) * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  forward_mma_pipe_f16_warpspec_kernel<TileM, TileN, WarpsM, WarpsN, ProducerWarps><<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

inline torch::Tensor forward_formula_scalar_impl(torch::Tensor X, torch::Tensor Wp, bool mixed) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  int N = batch * seq;
  int total = N * outNum;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  if (mixed) {
    forward_formula_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  } else {
    forward_formula_scalar_kernel<half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
  }
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_pipeline_impl(torch::Tensor X, torch::Tensor Wp, bool mixed, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    if (mixed) {
      forward_fused_wmma_pipeline_kernel<float, TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
    } else {
      forward_fused_wmma_pipeline_kernel<half, TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_mma_ptx_impl(torch::Tensor X, torch::Tensor Wp, bool mixed, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    if (mixed) {
      forward_mma_pipe_f32_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
    } else {
      forward_mma_pipe_f16_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int OutTiles>
inline py::dict profile_mma_ptx_multi_n_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN * OutTiles - 1) / (TileN * OutTiles), (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    forward_mma_pipe_f16_multi_n_kernel<TileM, TileN, WarpsM, WarpsN, OutTiles><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["grid_x"] = static_cast<int>(grid.x);
  result["grid_y"] = static_cast<int>(grid.y);
  result["out_tiles_per_cta"] = OutTiles;
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_mma_ptx_cpasync_w_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    forward_mma_pipe_f16_cpasync_w_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_mma_ptx_gauss_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    forward_mma_pipe_f16_gauss_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
inline py::dict profile_mma_ptx_double_buffer_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  dim3 block(WarpsM * WarpsN * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    forward_mma_pipe_f16_double_buffer_kernel<TileM, TileN, WarpsM, WarpsN><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <int TileM, int TileN, int WarpsM, int WarpsN, int ProducerWarps>
inline py::dict profile_mma_ptx_warpspec_impl(torch::Tensor X, torch::Tensor Wp, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());

  constexpr int ConsumerWarps = WarpsM * WarpsN;
  dim3 block((ConsumerWarps + ProducerWarps) * 32);
  dim3 grid((outNum + TileN - 1) / TileN, (N + TileM - 1) / TileM);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    forward_mma_pipe_f16_warpspec_kernel<TileM, TileN, WarpsM, WarpsN, ProducerWarps><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        N,
        inNum,
        outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

inline py::dict profile_formula_scalar_impl(torch::Tensor X, torch::Tensor Wp, bool mixed, int repeats, int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  int N = batch * seq;
  int total = N * outNum;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  auto Y = torch::empty({batch, seq, outNum, kTranNum}, X.options());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto launch = [&]() {
    if (mixed) {
      forward_formula_scalar_kernel<float><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
    } else {
      forward_formula_scalar_kernel<half><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(Wp.data_ptr<at::Half>()),
          reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
          N,
          inNum,
          outNum);
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
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

}  // namespace flash_eq_fp16_pipeline
