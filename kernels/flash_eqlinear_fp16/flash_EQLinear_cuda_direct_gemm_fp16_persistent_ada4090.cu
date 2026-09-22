#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <string>

namespace py = pybind11;
using namespace nvcuda;

namespace {

constexpr int kTranNum = 4;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

constexpr int kPersistentM32N64TileM = 32;
constexpr int kPersistentM32N64TileN = 64;
constexpr int kPersistentM32N64WarpsM = 2;
constexpr int kPersistentM32N64WarpsN = 4;
constexpr int kPersistentCtasAda4090 = 512;
constexpr int kPersistentCtasAda4090X2 = 288;
constexpr int kPersistentCtasAda4090X4 = 576;
constexpr int kPersistentCtasAda4090X8 = 1024;
constexpr int kPersistentCtasAda4090X16 = 2048;
constexpr int kPersistentCtasAda4090X32 = 4096;

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

template <int TileM, int TileN, int WarpsM, int WarpsN>
__global__ void flash_eq_linear_forward_persistent_direct_gemm_fp16_kernel(
    const half* __restrict__ X,
    const half* __restrict__ W,
    half* __restrict__ Y,
    int N,
    int inNum,
    int outNum,
    int total_tiles,
    int d_tiles) {
  constexpr int Warps = WarpsM * WarpsN;
  constexpr int Threads = Warps * 32;
  __shared__ half sA[4][TileM][kWmmaK];
  __shared__ half sB[5][TileN][kWmmaK];
  __shared__ half sAcc[Warps][4][kWmmaM][kWmmaN];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id / WarpsN;
  int warp_n = warp_id % WarpsN;
  int warp_row = warp_m * kWmmaM;
  int warp_col = warp_n * kWmmaN;

  for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
    int d_tile = tile_id % d_tiles;
    int n_tile = tile_id / d_tiles;
    int n_base = n_tile * TileM;
    int d_base = d_tile * TileN;

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc0;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc1;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc2;
    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, half> acc3;
    wmma::fill_fragment(acc0, hzero());
    wmma::fill_fragment(acc1, hzero());
    wmma::fill_fragment(acc2, hzero());
    wmma::fill_fragment(acc3, hzero());

    for (int c_start = 0; c_start < inNum; c_start += kWmmaK) {
      for (int idx = threadIdx.x; idx < TileM * kWmmaK; idx += Threads) {
        int row = idx / kWmmaK;
        int kk = idx % kWmmaK;
        int n = n_base + row;
        int c = c_start + kk;
        half f0 = hzero();
        half f1 = hzero();
        half f2 = hzero();
        half f3 = hzero();
        if (n < N && c < inNum) {
          const half* x_ptr = X + ((n * inNum + c) * kTranNum);
          load_x_freq(x_ptr, f0, f1, f2, f3);
        }
        sA[0][row][kk] = f0;
        sA[1][row][kk] = f1;
        sA[2][row][kk] = f2;
        sA[3][row][kk] = f3;
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
}

template <typename Fn>
float measure_cuda_ms(Fn&& fn, int repeats, int warmup) {
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
        flash_eq_linear_forward_persistent_direct_gemm_fp16_kernel<
            kPersistentM32N64TileM,
            kPersistentM32N64TileN,
            kPersistentM32N64WarpsM,
            kPersistentM32N64WarpsN>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100);
    configured = true;
  }
}

inline int ceil_div_int(int x, int y) {
  return (x + y - 1) / y;
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
void launch_persistent_kernel(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor Y,
    int persistent_ctas) {
  int N = static_cast<int>(X.size(0) * X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(W.size(0));
  int d_tiles = ceil_div_int(outNum, TileN);
  int n_tiles = ceil_div_int(N, TileM);
  int total_tiles = d_tiles * n_tiles;
  int ctas = std::max(1, persistent_ctas);

  constexpr int Threads = WarpsM * WarpsN * 32;
  dim3 block(Threads);
  dim3 grid(ctas);
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  flash_eq_linear_forward_persistent_direct_gemm_fp16_kernel<TileM, TileN, WarpsM, WarpsN>
      <<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      N,
      inNum,
      outNum,
      total_tiles,
      d_tiles);
}

template <int TileM, int TileN, int WarpsM, int WarpsN>
torch::Tensor forward_persistent_impl(torch::Tensor X, torch::Tensor W) {
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");
  TORCH_CHECK(X.size(0) * X.size(1) <= INT_MAX, "flattened sequence is too large");
  TORCH_CHECK(X.size(2) <= INT_MAX && W.size(0) <= INT_MAX, "channel count is too large");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  auto Y = torch::empty({X.size(0), X.size(1), W.size(0), kTranNum}, X.options());
  launch_persistent_kernel<TileM, TileN, WarpsM, WarpsN>(X, W, Y, kPersistentCtasAda4090);
  CUDA_CHECK(cudaGetLastError());
  return Y;
}

torch::Tensor forward(torch::Tensor X, torch::Tensor W, const std::string& variant) {
  if (variant == "pure_persistent_fused_m32n64" ||
      variant == "pure_persistent_fused_m32n64_cta288" ||
      variant == "pure_persistent_fused_m32n64_cta576" ||
      variant == "pure_persistent_fused_m32n64_cta1024" ||
      variant == "pure_persistent_fused_m32n64_cta2048" ||
      variant == "pure_persistent_fused_m32n64_cta4096") {
    int ctas = kPersistentCtasAda4090;
    if (variant == "pure_persistent_fused_m32n64_cta288") {
      ctas = kPersistentCtasAda4090X2;
    } else if (variant == "pure_persistent_fused_m32n64_cta576") {
      ctas = kPersistentCtasAda4090X4;
    } else if (variant == "pure_persistent_fused_m32n64_cta1024") {
      ctas = kPersistentCtasAda4090X8;
    } else if (variant == "pure_persistent_fused_m32n64_cta2048") {
      ctas = kPersistentCtasAda4090X16;
    } else if (variant == "pure_persistent_fused_m32n64_cta4096") {
      ctas = kPersistentCtasAda4090X32;
    }
    check_tensor_4d_half_cuda(X, "X");
    check_weight_3d_half_cuda(W, "W");
    TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");
    TORCH_CHECK(X.size(0) * X.size(1) <= INT_MAX, "flattened sequence is too large");
    TORCH_CHECK(X.size(2) <= INT_MAX && W.size(0) <= INT_MAX, "channel count is too large");

    configure_kernel_attributes();
    c10::cuda::CUDAGuard device_guard(X.device());

    auto Y = torch::empty({X.size(0), X.size(1), W.size(0), kTranNum}, X.options());
    launch_persistent_kernel<
        kPersistentM32N64TileM,
        kPersistentM32N64TileN,
        kPersistentM32N64WarpsM,
        kPersistentM32N64WarpsN>(X, W, Y, ctas);
    CUDA_CHECK(cudaGetLastError());
    return Y;
  }
  TORCH_CHECK(false, "unknown persistent direct fp16 variant: ", variant);
}

py::dict profile_forward(
    torch::Tensor X,
    torch::Tensor W,
    const std::string& variant,
    int repeats,
    int warmup) {
  check_tensor_4d_half_cuda(X, "X");
  check_weight_3d_half_cuda(W, "W");
  TORCH_CHECK(X.size(2) == W.size(1), "input channel mismatch");
  TORCH_CHECK(repeats > 0, "repeats must be positive");
  TORCH_CHECK(warmup >= 0, "warmup must be non-negative");
  TORCH_CHECK(X.size(0) * X.size(1) <= INT_MAX, "flattened sequence is too large");
  TORCH_CHECK(X.size(2) <= INT_MAX && W.size(0) <= INT_MAX, "channel count is too large");

  configure_kernel_attributes();
  c10::cuda::CUDAGuard device_guard(X.device());

  int N = static_cast<int>(X.size(0) * X.size(1));
  int outNum = static_cast<int>(W.size(0));
  auto Y = torch::empty({X.size(0), X.size(1), W.size(0), kTranNum}, X.options());

  float kernel_ms = 0.0f;
  int tile_m = 0;
  int tile_n = 0;
  int d_tiles = 0;
  int n_tiles = 0;
  int ctas = kPersistentCtasAda4090;

  if (variant == "pure_persistent_fused_m32n64" ||
      variant == "pure_persistent_fused_m32n64_cta288" ||
      variant == "pure_persistent_fused_m32n64_cta576" ||
      variant == "pure_persistent_fused_m32n64_cta1024" ||
      variant == "pure_persistent_fused_m32n64_cta2048" ||
      variant == "pure_persistent_fused_m32n64_cta4096") {
    tile_m = kPersistentM32N64TileM;
    tile_n = kPersistentM32N64TileN;
    d_tiles = ceil_div_int(outNum, tile_n);
    n_tiles = ceil_div_int(N, tile_m);
    if (variant == "pure_persistent_fused_m32n64_cta288") {
      ctas = kPersistentCtasAda4090X2;
    } else if (variant == "pure_persistent_fused_m32n64_cta576") {
      ctas = kPersistentCtasAda4090X4;
    } else if (variant == "pure_persistent_fused_m32n64_cta1024") {
      ctas = kPersistentCtasAda4090X8;
    } else if (variant == "pure_persistent_fused_m32n64_cta2048") {
      ctas = kPersistentCtasAda4090X16;
    } else if (variant == "pure_persistent_fused_m32n64_cta4096") {
      ctas = kPersistentCtasAda4090X32;
    }
    auto run = [&] {
      launch_persistent_kernel<
          kPersistentM32N64TileM,
          kPersistentM32N64TileN,
          kPersistentM32N64WarpsM,
          kPersistentM32N64WarpsN>(X, W, Y, ctas);
    };
    kernel_ms = measure_cuda_ms(run, repeats, warmup);
  } else {
    TORCH_CHECK(false, "unknown persistent direct fp16 profile variant: ", variant);
  }
  CUDA_CHECK(cudaGetLastError());

  py::dict result;
  result["forward_kernel_ms"] = kernel_ms;
  result["variant"] = variant;
  result["tile_m"] = tile_m;
  result["tile_n"] = tile_n;
  result["persistent_ctas"] = ctas;
  result["total_tiles"] = d_tiles * n_tiles;
  result["d_tiles"] = d_tiles;
  result["n_tiles"] = n_tiles;
  result["single_kernel"] = true;
  result["materializes_x_freq"] = false;
  result["materializes_y_freq"] = false;
  return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &forward,
      py::arg("X"),
      py::arg("W"),
      py::arg("variant"),
      "Single-launch persistent/grouped fused direct GEMM fp16 forward");
  m.def(
      "profile_forward",
      &profile_forward,
      py::arg("X"),
      py::arg("W"),
      py::arg("variant"),
      py::arg("repeats") = 80,
      py::arg("warmup") = 20,
      "Profile persistent/grouped fused direct GEMM fp16 forward");
}
