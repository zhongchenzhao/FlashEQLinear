#include <cute/tensor.hpp>

#include "flash_EQLinear_cuda_direct_gemm_fp16_pipeline_common_ada4090.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <examples/35_gemm_softmax/gemm_with_epilogue_visitor.h>

#include <string>

namespace {

constexpr int kCuteFusedTileM = 64;
constexpr int kCuteFusedTileN = 64;
constexpr int kCuteFusedWarpsM = 4;
constexpr int kCuteFusedWarpsN = 4;

using ElementOperand = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;

using CutlassEpilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute>;

// CUTLASS 2.x DefaultGemm accepts RowMajor or AffineRankN<2> for the SM80
// TensorOp epilogue. AffineRankN<2> lets one GEMM write a single spatial
// component directly into the final [M,D,4] tensor with row stride D*4 and
// column stride 4. ElementsPerAccess must be 1 because adjacent GEMM columns
// are not contiguous in memory once the spatial component stride is 4.
using CutlassDirectSpatialLayout = cutlass::layout::AffineRankN<2>;
using CutlassDirectSpatialEpilogue = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    1,
    ElementAccumulator,
    ElementCompute>;

using CutlassSpatialFanoutThreadblockSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

template <
    typename ThreadblockShape_,
    int ThreadCount,
    typename OutputTileIterator_>
class SpatialFanoutVisitor {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = cutlass::half_t;
  using ElementNorm = int;
  using ElementSum = int;
  using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

  struct Arguments {
    int out_num;
    int zero_existing;
    float coeff0;
    float coeff1;
    float coeff2;
    float coeff3;

    Arguments()
        : out_num(0), zero_existing(0), coeff0(0.0f), coeff1(0.0f), coeff2(0.0f), coeff3(0.0f) {}

    Arguments(
        int out_num_,
        bool zero_existing_,
        float coeff0_,
        float coeff1_,
        float coeff2_,
        float coeff3_)
        : out_num(out_num_),
          zero_existing(zero_existing_ ? 1 : 0),
          coeff0(coeff0_),
          coeff1(coeff1_),
          coeff2(coeff2_),
          coeff3(coeff3_) {}
  };

  struct Params {
    int out_num;
    int zero_existing;
    float coeff0;
    float coeff1;
    float coeff2;
    float coeff3;

    CUTLASS_HOST_DEVICE
    Params()
        : out_num(0), zero_existing(0), coeff0(0.0f), coeff1(0.0f), coeff2(0.0f), coeff3(0.0f) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : out_num(args.out_num),
          zero_existing(args.zero_existing),
          coeff0(args.coeff0),
          coeff1(args.coeff1),
          coeff2(args.coeff2),
          coeff3(args.coeff3) {}
  };

  struct SharedStorage {};

 private:
  Params params_;
  cutlass::MatrixCoord extent_;
  OutputTileIterator iterator_D_;
  ElementOutput* y_base_;

 public:
  CUTLASS_DEVICE
  SpatialFanoutVisitor(
      Params const& params,
      SharedStorage&,
      cutlass::MatrixCoord const& problem_size,
      int thread_idx,
      int,
      int,
      typename OutputTileIterator::Params,
      typename OutputTileIterator::Params params_D,
      ElementOutput*,
      ElementOutput* ptr_D,
      ElementNorm* = nullptr,
      ElementSum* = nullptr,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0),
      int = 0)
      : params_(params),
        extent_(problem_size),
        iterator_D_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        y_base_(ptr_D) {}

  CUTLASS_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    y_base_ += batch_idx * extent_.row() * params_.out_num * flash_eq_fp16_pipeline::kTranNum;
  }

  CUTLASS_DEVICE
  void begin_epilogue() {}

  CUTLASS_DEVICE
  void begin_step(int) {}

  CUTLASS_DEVICE
  void begin_row(int) {}

  CUTLASS_DEVICE
  void visit(
      int,
      int,
      int,
      int frag_idx,
      AccumulatorFragment const& accum) {
    cutlass::MatrixCoord thread_offset =
        iterator_D_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kElementsPerAccess; ++e) {
      int row = thread_offset.row();
      int col = thread_offset.column() + e;
      if (row < extent_.row() && col < extent_.column()) {
        float a = static_cast<float>(accum[e]);
        ElementOutput* y = y_base_ + (row * params_.out_num + col) * flash_eq_fp16_pipeline::kTranNum;
        store_component(y + 0, a, params_.coeff0);
        store_component(y + 1, a, params_.coeff1);
        store_component(y + 2, a, params_.coeff2);
        store_component(y + 3, a, params_.coeff3);
      }
    }
  }

  CUTLASS_DEVICE
  void end_row(int) {}

  CUTLASS_DEVICE
  void end_step(int) {
    ++iterator_D_;
  }

  CUTLASS_DEVICE
  void end_epilogue() {}

 private:
  CUTLASS_DEVICE
  void store_component(ElementOutput* dst, float accum, float coeff) const {
    if (coeff == 0.0f) {
      return;
    }
    float value = accum * coeff;
    if (!params_.zero_existing) {
      value += static_cast<float>(*dst);
    }
    *dst = ElementOutput(value);
  }
};

using CutlassStagedGemm = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    8,
    8,
    false>;

using CutlassDirectSpatialGemm = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    CutlassDirectSpatialLayout,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassDirectSpatialEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    8,
    8,
    false>;

using CutlassSpatialFanoutDefaultKernel = typename cutlass::gemm::kernel::DefaultGemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    8,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    8,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    CutlassSpatialFanoutThreadblockSwizzle,
    3,
    false,
    cutlass::arch::OpMultiplyAdd,
    cutlass::gemm::SharedMemoryClearOption::kNone>::GemmKernel;

using CutlassSpatialFanoutVisitor = SpatialFanoutVisitor<
    typename CutlassSpatialFanoutDefaultKernel::Mma::Shape,
    CutlassSpatialFanoutDefaultKernel::kThreadCount,
    typename CutlassSpatialFanoutDefaultKernel::Epilogue::OutputTileIterator>;

using CutlassSpatialFanoutEpilogue =
    typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        CutlassSpatialFanoutVisitor,
        typename CutlassSpatialFanoutDefaultKernel::Epilogue>::Epilogue;

using CutlassSpatialFanoutGemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    typename CutlassSpatialFanoutDefaultKernel::Mma,
    CutlassSpatialFanoutEpilogue,
    CutlassSpatialFanoutThreadblockSwizzle>;

using CutlassStagedGemmN64K64 = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    8,
    8,
    false>;

using CutlassStagedGemmK64 = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    3,
    8,
    8,
    false>;

using CutlassStagedGemmS4 = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    4,
    8,
    8,
    false>;

template <
    int ThreadblockM,
    int ThreadblockN,
    int ThreadblockK,
    int WarpM,
    int WarpN,
    int WarpK,
    int Stages>
using CutlassStagedGemmShape = cutlass::gemm::device::Gemm<
    ElementOperand,
    cutlass::layout::RowMajor,
    ElementOperand,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    CutlassEpilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8,
    8,
    false>;

using CutlassStagedGemmS2 = CutlassStagedGemmShape<128, 128, 32, 64, 64, 32, 2>;
using CutlassStagedGemmK64S2 = CutlassStagedGemmShape<128, 128, 64, 64, 64, 64, 2>;
using CutlassStagedGemmK64S4 = CutlassStagedGemmShape<128, 128, 64, 64, 64, 64, 4>;
using CutlassStagedGemmN64K64S2 = CutlassStagedGemmShape<128, 64, 64, 64, 32, 64, 2>;
using CutlassStagedGemmN64K64S4 = CutlassStagedGemmShape<128, 64, 64, 64, 32, 64, 4>;
using CutlassStagedGemmM64N128K64S2 = CutlassStagedGemmShape<64, 128, 64, 32, 64, 64, 2>;
using CutlassStagedGemmM64N128K64 = CutlassStagedGemmShape<64, 128, 64, 32, 64, 64, 3>;
using CutlassStagedGemmM64N128K64S4 = CutlassStagedGemmShape<64, 128, 64, 32, 64, 64, 4>;

using CutlassNoYfreqGemm = CutlassStagedGemm;


bool is_mixed_variant(const std::string& variant) {
  if (variant == "pure_cute_fused") {
    return false;
  }
  if (variant == "mixed_cute_fused") {
    return true;
  }
  TORCH_CHECK(false, "unknown CuTe fused variant: ", variant);
}

__global__ void materialize_x_freq_kernel(
    const half* __restrict__ X,
    half* __restrict__ Xf,
    int N,
    int inNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * inNum;
  if (idx >= total) {
    return;
  }
  half f0, f1, f2, f3;
  flash_eq_fp16_pipeline::load_x_freq_half(X + idx * flash_eq_fp16_pipeline::kTranNum, f0, f1, f2, f3);
  Xf[0 * total + idx] = f0;
  Xf[1 * total + idx] = f1;
  Xf[2 * total + idx] = f2;
  Xf[3 * total + idx] = f3;
}

__global__ void materialize_x_freq_gauss_kernel(
    const half* __restrict__ X,
    half* __restrict__ Xg,
    int N,
    int inNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * inNum;
  if (idx >= total) {
    return;
  }
  half f0, f1, f2, f3;
  flash_eq_fp16_pipeline::load_x_freq_half(X + idx * flash_eq_fp16_pipeline::kTranNum, f0, f1, f2, f3);
  Xg[0 * total + idx] = f0;
  Xg[1 * total + idx] = f2;
  Xg[2 * total + idx] = __hadd(f1, f3);
  Xg[3 * total + idx] = f1;
  Xg[4 * total + idx] = f3;
}

__global__ void pack_weight_fp16_gauss_kernel(
    const half* __restrict__ W,
    half* __restrict__ Wg,
    int inNum,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outNum * inNum;
  if (idx >= total) {
    return;
  }
  int d = idx / inNum;
  int c = idx - d * inNum;
  const half* w = W + ((d * inNum + c) * flash_eq_fp16_pipeline::kTranNum);
  int base = d * inNum + c;
  half w1 = w[1];
  half w3 = w[3];
  Wg[0 * total + base] = w[0];
  Wg[1 * total + base] = w[2];
  Wg[2 * total + base] = w1;
  Wg[3 * total + base] = __hsub(w3, w1);
  Wg[4 * total + base] = __hadd(w1, w3);
}

__global__ void spatial_epilogue_from_freq_kernel(
    const half* __restrict__ Yf,
    half* __restrict__ Y,
    int N,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outNum;
  if (idx >= total) {
    return;
  }
  half a0 = Yf[0 * total + idx];
  half a1 = Yf[1 * total + idx];
  half a2 = Yf[2 * total + idx];
  half a3 = Yf[3 * total + idx];
  flash_eq_fp16_pipeline::write_spatial_output<half>(
      a0,
      a1,
      a2,
      a3,
      Y + idx * flash_eq_fp16_pipeline::kTranNum);
}

__global__ void spatial_epilogue_from_gauss_kernel(
    const half* __restrict__ Yg,
    half* __restrict__ Y,
    int N,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outNum;
  if (idx >= total) {
    return;
  }
  half a0 = Yg[0 * total + idx];
  half a2 = Yg[1 * total + idx];
  half p = Yg[2 * total + idx];
  half q = Yg[3 * total + idx];
  half r = Yg[4 * total + idx];
  half a1 = __hsub(p, r);
  half a3 = __hadd(p, q);
  flash_eq_fp16_pipeline::write_spatial_output<half>(
      a0,
      a1,
      a2,
      a3,
      Y + idx * flash_eq_fp16_pipeline::kTranNum);
}

__global__ void copy_spatial_planar_to_interleaved_kernel(
    const half* __restrict__ Yp,
    half* __restrict__ Y,
    int N,
    int outNum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * outNum;
  if (idx >= total) {
    return;
  }
  half* y = Y + idx * flash_eq_fp16_pipeline::kTranNum;
  y[0] = Yp[0 * total + idx];
  y[1] = Yp[1 * total + idx];
  y[2] = Yp[2 * total + idx];
  y[3] = Yp[3 * total + idx];
}

template <typename GemmOp>
void run_cutlass_gemm_typed(
    const half* A,
    const half* B_col_major,
    half* C,
    int M,
    int N,
    int K,
    half alpha,
    half beta,
    cudaStream_t stream) {
  GemmOp gemm;
  ElementCompute alpha_c = ElementCompute(__half2float(alpha));
  ElementCompute beta_c = ElementCompute(__half2float(beta));
  typename GemmOp::Arguments args(
      {M, N, K},
      {reinterpret_cast<ElementOperand const*>(A), K},
      {reinterpret_cast<ElementOperand const*>(B_col_major), K},
      {reinterpret_cast<ElementOutput const*>(C), N},
      {reinterpret_cast<ElementOutput*>(C), N},
      {alpha_c, beta_c});
  cutlass::Status status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS staged GEMM launch failed");
}

template <typename GemmOp>
void run_cutlass_direct_spatial_gemm_typed(
    const half* A,
    const half* B_col_major,
    half* Y_component_base,
    int M,
    int N,
    int K,
    half alpha,
    half beta,
    cudaStream_t stream) {
  GemmOp gemm;
  ElementCompute alpha_c = ElementCompute(__half2float(alpha));
  ElementCompute beta_c = ElementCompute(__half2float(beta));
  CutlassDirectSpatialLayout spatial_layout(
      static_cast<CutlassDirectSpatialLayout::LongIndex>(N) * flash_eq_fp16_pipeline::kTranNum,
      static_cast<CutlassDirectSpatialLayout::LongIndex>(flash_eq_fp16_pipeline::kTranNum));
  typename GemmOp::Arguments args(
      {M, N, K},
      {reinterpret_cast<ElementOperand const*>(A), K},
      {reinterpret_cast<ElementOperand const*>(B_col_major), K},
      {reinterpret_cast<ElementOutput const*>(Y_component_base), spatial_layout},
      {reinterpret_cast<ElementOutput*>(Y_component_base), spatial_layout},
      {alpha_c, beta_c});
  cutlass::Status status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS direct spatial GEMM launch failed");
}

template <typename GemmKernel>
void run_cutlass_spatial_fanout_gemm_typed(
    const half* A,
    const half* B_col_major,
    half* Y,
    int M,
    int N,
    int K,
    bool zero_existing,
    float coeff0,
    float coeff1,
    float coeff2,
    float coeff3,
    cudaStream_t stream) {
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename GemmKernel::LayoutC;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  typename GemmKernel::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      cutlass::gemm::GemmCoord(M, N, K),
      1,
      {reinterpret_cast<ElementA*>(const_cast<half*>(A)), K},
      {reinterpret_cast<ElementB*>(const_cast<half*>(B_col_major)), K},
      TensorRefC(reinterpret_cast<ElementC*>(Y), LayoutC(N)),
      TensorRefC(reinterpret_cast<ElementC*>(Y), LayoutC(N)),
      nullptr,
      nullptr,
      0,
      0,
      typename GemmKernel::EpilogueVisitor::Arguments(
          N, zero_existing, coeff0, coeff1, coeff2, coeff3));

  typename GemmKernel::Params params(args);
  typename GemmKernel::ThreadblockSwizzle threadblock_swizzle;
  dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
  dim3 block(GemmKernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr_status = cudaFuncSetAttribute(
        cutlass::Kernel<GemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    TORCH_CHECK(attr_status == cudaSuccess, "CUTLASS fanout epilogue shared-memory attribute failed");
  }
  cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
}

template <typename GemmOp>
torch::Tensor forward_cutlass_staged_impl(torch::Tensor X, torch::Tensor Wp) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  int M = batch * seq;
  int K = inNum;
  int D = outNum;
  auto Xf = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, K}, X.options());
  auto Yf = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, D}, X.options());
  auto Y = torch::empty({batch, seq, D, flash_eq_fp16_pipeline::kTranNum}, X.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int x_total = M * K;
  int threads = 256;
  materialize_x_freq_kernel<<<(x_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Xf.data_ptr<at::Half>()),
      M,
      K);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  const half* xfp = reinterpret_cast<const half*>(Xf.data_ptr<at::Half>());
  const half* wp = reinterpret_cast<const half*>(Wp.data_ptr<at::Half>());
  half* yfp = reinterpret_cast<half*>(Yf.data_ptr<at::Half>());
  int x_plane = M * K;
  int w_plane = D * K;
  int y_plane = M * D;
  half one = __float2half(1.0f);
  half zero = __float2half(0.0f);

  run_cutlass_gemm_typed<GemmOp>(xfp + 0 * x_plane, wp + 0 * w_plane, yfp + 0 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xfp + 1 * x_plane, wp + 1 * w_plane, yfp + 1 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xfp + 3 * x_plane, wp + 4 * w_plane, yfp + 1 * y_plane, M, D, K, one, one, stream);
  run_cutlass_gemm_typed<GemmOp>(xfp + 2 * x_plane, wp + 2 * w_plane, yfp + 2 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xfp + 1 * x_plane, wp + 3 * w_plane, yfp + 3 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xfp + 3 * x_plane, wp + 1 * w_plane, yfp + 3 * y_plane, M, D, K, one, one, stream);

  int y_total = M * D;
  spatial_epilogue_from_freq_kernel<<<(y_total + threads - 1) / threads, threads, 0, stream>>>(
      yfp,
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      M,
      D);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <typename GemmOp>
torch::Tensor forward_cutlass_staged_gauss_impl(torch::Tensor X, torch::Tensor Wg) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wg, "Wg");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wg.device(), "X and Wg must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wg.size(1));
  TORCH_CHECK(Wg.size(2) == inNum, "Wg in dimension must match X inNum");

  int M = batch * seq;
  int K = inNum;
  int D = outNum;
  auto Xg = torch::empty({flash_eq_fp16_pipeline::kPackedTranNum, M, K}, X.options());
  auto Yg = torch::empty({flash_eq_fp16_pipeline::kPackedTranNum, M, D}, X.options());
  auto Y = torch::empty({batch, seq, D, flash_eq_fp16_pipeline::kTranNum}, X.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int x_total = M * K;
  int threads = 256;
  materialize_x_freq_gauss_kernel<<<(x_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Xg.data_ptr<at::Half>()),
      M,
      K);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  const half* xgp = reinterpret_cast<const half*>(Xg.data_ptr<at::Half>());
  const half* wgp = reinterpret_cast<const half*>(Wg.data_ptr<at::Half>());
  half* ygp = reinterpret_cast<half*>(Yg.data_ptr<at::Half>());
  int x_plane = M * K;
  int w_plane = D * K;
  int y_plane = M * D;
  half one = __float2half(1.0f);
  half zero = __float2half(0.0f);

  run_cutlass_gemm_typed<GemmOp>(xgp + 0 * x_plane, wgp + 0 * w_plane, ygp + 0 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xgp + 1 * x_plane, wgp + 1 * w_plane, ygp + 1 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xgp + 2 * x_plane, wgp + 2 * w_plane, ygp + 2 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xgp + 3 * x_plane, wgp + 3 * w_plane, ygp + 3 * y_plane, M, D, K, one, zero, stream);
  run_cutlass_gemm_typed<GemmOp>(xgp + 4 * x_plane, wgp + 4 * w_plane, ygp + 4 * y_plane, M, D, K, one, zero, stream);

  int y_total = M * D;
  spatial_epilogue_from_gauss_kernel<<<(y_total + threads - 1) / threads, threads, 0, stream>>>(
      ygp,
      reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
      M,
      D);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <typename GemmOp>
torch::Tensor forward_cutlass_no_yfreq_materialize_impl(torch::Tensor X, torch::Tensor Wp) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  int M = batch * seq;
  int K = inNum;
  int D = outNum;
  auto Xf = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, K}, X.options());
  auto Yp = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, D}, X.options());
  auto Y = torch::empty({batch, seq, D, flash_eq_fp16_pipeline::kTranNum}, X.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int x_total = M * K;
  int threads = 256;
  materialize_x_freq_kernel<<<(x_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Xf.data_ptr<at::Half>()),
      M,
      K);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  const half* xfp = reinterpret_cast<const half*>(Xf.data_ptr<at::Half>());
  const half* wp = reinterpret_cast<const half*>(Wp.data_ptr<at::Half>());
  half* ypp = reinterpret_cast<half*>(Yp.data_ptr<at::Half>());
  half* yp = reinterpret_cast<half*>(Y.data_ptr<at::Half>());
  int x_plane = M * K;
  int w_plane = D * K;
  int y_plane = M * D;
  half one = __float2half(1.0f);
  half zero = __float2half(0.0f);
  half quarter = __float2half(0.25f);
  half neg_quarter = __float2half(-0.25f);
  half halfv = __float2half(0.5f);
  half neg_half = __float2half(-0.5f);

  auto run_component = [&](int component, int x_idx, int w_idx, half alpha, half beta) {
    run_cutlass_gemm_typed<GemmOp>(
        xfp + x_idx * x_plane,
        wp + w_idx * w_plane,
        ypp + component * y_plane,
        M,
        D,
        K,
        alpha,
        beta,
        stream);
  };

  // This prototype avoids materializing Y_freq by accumulating directly into
  // interleaved spatial output components. It preserves the algebraic formula
  // but not the staged route's exact FP16 add ordering.
  run_component(0, 0, 0, quarter, zero);
  run_component(0, 1, 1, halfv, one);
  run_component(0, 3, 4, halfv, one);
  run_component(0, 2, 2, quarter, one);

  run_component(1, 0, 0, quarter, zero);
  run_component(1, 2, 2, neg_quarter, one);
  run_component(1, 1, 3, neg_half, one);
  run_component(1, 3, 1, neg_half, one);

  run_component(2, 0, 0, quarter, zero);
  run_component(2, 1, 1, neg_half, one);
  run_component(2, 3, 4, neg_half, one);
  run_component(2, 2, 2, quarter, one);

  run_component(3, 0, 0, quarter, zero);
  run_component(3, 2, 2, neg_quarter, one);
  run_component(3, 1, 3, halfv, one);
  run_component(3, 3, 1, halfv, one);

  int y_total = M * D;
  copy_spatial_planar_to_interleaved_kernel<<<(y_total + threads - 1) / threads, threads, 0, stream>>>(
      ypp,
      yp,
      M,
      D);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Y;
}

template <typename GemmOp>
torch::Tensor forward_cutlass_direct_spatial_impl(torch::Tensor X, torch::Tensor Wp) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  int M = batch * seq;
  int K = inNum;
  int D = outNum;
  auto Xf = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, K}, X.options());
  auto Y = torch::empty({batch, seq, D, flash_eq_fp16_pipeline::kTranNum}, X.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int x_total = M * K;
  int threads = 256;
  materialize_x_freq_kernel<<<(x_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Xf.data_ptr<at::Half>()),
      M,
      K);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  const half* xfp = reinterpret_cast<const half*>(Xf.data_ptr<at::Half>());
  const half* wp = reinterpret_cast<const half*>(Wp.data_ptr<at::Half>());
  half* yp = reinterpret_cast<half*>(Y.data_ptr<at::Half>());
  int x_plane = M * K;
  int w_plane = D * K;
  half one = __float2half(1.0f);
  half zero = __float2half(0.0f);
  half quarter = __float2half(0.25f);
  half neg_quarter = __float2half(-0.25f);
  half halfv = __float2half(0.5f);
  half neg_half = __float2half(-0.5f);

  auto run_component = [&](int component, int x_idx, int w_idx, half alpha, half beta) {
    run_cutlass_direct_spatial_gemm_typed<GemmOp>(
        xfp + x_idx * x_plane,
        wp + w_idx * w_plane,
        yp + component,
        M,
        D,
        K,
        alpha,
        beta,
        stream);
  };

  // POC limitation: CUTLASS DefaultGemm exposes one output matrix per launch,
  // so this writes each spatial component directly but still uses one GEMM per
  // component contribution. It removes Y_freq and the planar-to-interleaved
  // copy without pretending this is a true multi-output epilogue.
  run_component(0, 0, 0, quarter, zero);
  run_component(0, 1, 1, halfv, one);
  run_component(0, 3, 4, halfv, one);
  run_component(0, 2, 2, quarter, one);

  run_component(1, 0, 0, quarter, zero);
  run_component(1, 2, 2, neg_quarter, one);
  run_component(1, 1, 3, neg_half, one);
  run_component(1, 3, 1, neg_half, one);

  run_component(2, 0, 0, quarter, zero);
  run_component(2, 1, 1, neg_half, one);
  run_component(2, 3, 4, neg_half, one);
  run_component(2, 2, 2, quarter, one);

  run_component(3, 0, 0, quarter, zero);
  run_component(3, 2, 2, neg_quarter, one);
  run_component(3, 1, 3, halfv, one);
  run_component(3, 3, 1, halfv, one);

  return Y;
}

torch::Tensor forward_cutlass_spatial_fanout_impl(torch::Tensor X, torch::Tensor Wp) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  TORCH_CHECK(X.device() == Wp.device(), "X and Wp must be on same CUDA device");
  int batch = static_cast<int>(X.size(0));
  int seq = static_cast<int>(X.size(1));
  int inNum = static_cast<int>(X.size(2));
  int outNum = static_cast<int>(Wp.size(1));
  TORCH_CHECK(Wp.size(2) == inNum, "Wp in dimension must match X inNum");

  int M = batch * seq;
  int K = inNum;
  int D = outNum;
  auto Xf = torch::empty({flash_eq_fp16_pipeline::kTranNum, M, K}, X.options());
  auto Y = torch::empty({batch, seq, D, flash_eq_fp16_pipeline::kTranNum}, X.options());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  int x_total = M * K;
  int threads = 256;
  materialize_x_freq_kernel<<<(x_total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Xf.data_ptr<at::Half>()),
      M,
      K);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());

  const half* xfp = reinterpret_cast<const half*>(Xf.data_ptr<at::Half>());
  const half* wp = reinterpret_cast<const half*>(Wp.data_ptr<at::Half>());
  half* yp = reinterpret_cast<half*>(Y.data_ptr<at::Half>());
  int x_plane = M * K;
  int w_plane = D * K;

  auto run_fanout = [&](int x_idx, int w_idx, bool zero, float c0, float c1, float c2, float c3) {
    run_cutlass_spatial_fanout_gemm_typed<CutlassSpatialFanoutGemmKernel>(
        xfp + x_idx * x_plane,
        wp + w_idx * w_plane,
        yp,
        M,
        D,
        K,
        zero,
        c0,
        c1,
        c2,
        c3,
        stream);
  };

  run_fanout(0, 0, true, 0.25f, 0.25f, 0.25f, 0.25f);
  run_fanout(1, 1, false, 0.5f, 0.0f, -0.5f, 0.0f);
  run_fanout(3, 4, false, 0.5f, 0.0f, -0.5f, 0.0f);
  run_fanout(2, 2, false, 0.25f, -0.25f, 0.25f, -0.25f);
  run_fanout(1, 3, false, 0.0f, -0.5f, 0.0f, 0.5f);
  run_fanout(3, 1, false, 0.0f, -0.5f, 0.0f, 0.5f);

  return Y;
}

template <typename GemmOp>
flash_eq_fp16_pipeline::py::dict profile_cutlass_staged_exact_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    int repeats,
    int warmup) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i) {
    forward_cutlass_staged_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    forward_cutlass_staged_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  flash_eq_fp16_pipeline::py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <typename GemmOp>
flash_eq_fp16_pipeline::py::dict profile_cutlass_staged_gauss_typed_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    int repeats,
    int warmup) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i) {
    forward_cutlass_staged_gauss_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    forward_cutlass_staged_gauss_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  flash_eq_fp16_pipeline::py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  return result;
}

template <typename GemmOp>
flash_eq_fp16_pipeline::py::dict profile_cutlass_no_yfreq_typed_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    int repeats,
    int warmup) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i) {
    forward_cutlass_no_yfreq_materialize_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    forward_cutlass_no_yfreq_materialize_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  flash_eq_fp16_pipeline::py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  result["materializes_y_freq"] = false;
  result["gemms"] = 16;
  return result;
}

template <typename GemmOp>
flash_eq_fp16_pipeline::py::dict profile_cutlass_direct_spatial_typed_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    int repeats,
    int warmup) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i) {
    forward_cutlass_direct_spatial_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    forward_cutlass_direct_spatial_impl<GemmOp>(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  flash_eq_fp16_pipeline::py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  result["materializes_y_freq"] = false;
  result["writes_spatial_layout_direct"] = true;
  result["gemms"] = 16;
  result["epilogue_elements_per_access"] = 1;
  return result;
}

flash_eq_fp16_pipeline::py::dict profile_cutlass_spatial_fanout_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    int repeats,
    int warmup) {
  flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
  flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
  c10::cuda::CUDAGuard device_guard(X.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  for (int i = 0; i < warmup; ++i) {
    forward_cutlass_spatial_fanout_impl(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
  FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < repeats; ++i) {
    forward_cutlass_spatial_fanout_impl(X, Wp);
  }
  FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
  FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
  FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
  flash_eq_fp16_pipeline::py::dict result;
  result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
  result["repeats"] = repeats;
  result["warmup"] = warmup;
  result["materializes_y_freq"] = false;
  result["writes_spatial_layout_direct"] = true;
  result["custom_threadblock_epilogue"] = true;
  result["gemms"] = 6;
  result["epilogue_elements_per_access"] =
      CutlassSpatialFanoutVisitor::kElementsPerAccess;
  return result;
}

torch::Tensor forward_cute_impl(torch::Tensor X, torch::Tensor Wp, const std::string& variant) {
  if (variant == "pure_cutlass_direct_spatial" || variant == "pure_cutlass_spatial_epilogue") {
    return forward_cutlass_direct_spatial_impl<CutlassDirectSpatialGemm>(X, Wp);
  }
  if (variant == "pure_cutlass_spatial_fanout_epilogue") {
    return forward_cutlass_spatial_fanout_impl(X, Wp);
  }
  if (variant == "pure_cutlass_no_yfreq_materialize") {
    return forward_cutlass_no_yfreq_materialize_impl<CutlassNoYfreqGemm>(X, Wp);
  }
  if (variant == "pure_cutlass_staged") {
    return forward_cutlass_staged_impl<CutlassStagedGemm>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemm>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmN64K64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_n64k64") {
    return forward_cutlass_staged_impl<CutlassStagedGemmN64K64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_k64") {
    return forward_cutlass_staged_impl<CutlassStagedGemmK64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_k64") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmK64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_s4") {
    return forward_cutlass_staged_impl<CutlassStagedGemmS4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_s4") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmS4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_s2") {
    return forward_cutlass_staged_impl<CutlassStagedGemmS2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_s2") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmS2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_k64_s2") {
    return forward_cutlass_staged_impl<CutlassStagedGemmK64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_k64_s2") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmK64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_k64_s4") {
    return forward_cutlass_staged_impl<CutlassStagedGemmK64S4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_k64_s4") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmK64S4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_n64k64_s2") {
    return forward_cutlass_staged_impl<CutlassStagedGemmN64K64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64_s2") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmN64K64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_n64k64_s4") {
    return forward_cutlass_staged_impl<CutlassStagedGemmN64K64S4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64_s4") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmN64K64S4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_m64n128k64_s2") {
    return forward_cutlass_staged_impl<CutlassStagedGemmM64N128K64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64_s2") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmM64N128K64S2>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_m64n128k64") {
    return forward_cutlass_staged_impl<CutlassStagedGemmM64N128K64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmM64N128K64>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_m64n128k64_s4") {
    return forward_cutlass_staged_impl<CutlassStagedGemmM64N128K64S4>(X, Wp);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64_s4") {
    return forward_cutlass_staged_gauss_impl<CutlassStagedGemmM64N128K64S4>(X, Wp);
  }
  return flash_eq_fp16_pipeline::forward_pipeline_impl<
      kCuteFusedTileM,
      kCuteFusedTileN,
      kCuteFusedWarpsM,
      kCuteFusedWarpsN>(X, Wp, is_mixed_variant(variant));
}

flash_eq_fp16_pipeline::py::dict profile_cute_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    const std::string& variant,
    int repeats,
    int warmup) {
  if (variant == "pure_cutlass_direct_spatial" || variant == "pure_cutlass_spatial_epilogue") {
    return profile_cutlass_direct_spatial_typed_impl<CutlassDirectSpatialGemm>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_spatial_fanout_epilogue") {
    return profile_cutlass_spatial_fanout_impl(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_no_yfreq_materialize") {
    return profile_cutlass_no_yfreq_typed_impl<CutlassNoYfreqGemm>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_s2") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmS2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_s2") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmS2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_k64_s2") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmK64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_k64_s2") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmK64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_k64_s4") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmK64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_k64_s4") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmK64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_n64k64_s2") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmN64K64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64_s2") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmN64K64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_n64k64_s4") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmN64K64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64_s4") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmN64K64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_m64n128k64_s2") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmM64N128K64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64_s2") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmM64N128K64S2>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_m64n128k64") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmM64N128K64>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmM64N128K64>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_m64n128k64_s4") {
    return profile_cutlass_staged_exact_impl<CutlassStagedGemmM64N128K64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged_gauss_m64n128k64_s4") {
    return profile_cutlass_staged_gauss_typed_impl<CutlassStagedGemmM64N128K64S4>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_cutlass_staged") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemm>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemm>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_n64k64") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmN64K64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmN64K64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_k64") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmK64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmK64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_s4") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmS4>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_impl<CutlassStagedGemmS4>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_gauss") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemm>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemm>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_gauss_n64k64") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmN64K64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmN64K64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_gauss_k64") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmK64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmK64>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  if (variant == "pure_cutlass_staged_gauss_s4") {
    flash_eq_fp16_pipeline::check_tensor_4d_half_cuda(X, "X");
    flash_eq_fp16_pipeline::check_packed_weight_3d_half_cuda(Wp, "Wp");
    c10::cuda::CUDAGuard device_guard(X.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    for (int i = 0; i < warmup; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmS4>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&start));
    FLASH_EQ_CUDA_CHECK(cudaEventCreate(&stop));
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; ++i) {
      forward_cutlass_staged_gauss_impl<CutlassStagedGemmS4>(X, Wp);
    }
    FLASH_EQ_CUDA_CHECK(cudaEventRecord(stop, stream));
    FLASH_EQ_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    FLASH_EQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(start));
    FLASH_EQ_CUDA_CHECK(cudaEventDestroy(stop));
    flash_eq_fp16_pipeline::py::dict result;
    result["forward_kernel_ms"] = elapsed_ms / static_cast<float>(repeats);
    result["repeats"] = repeats;
    result["warmup"] = warmup;
    return result;
  }
  return flash_eq_fp16_pipeline::profile_pipeline_impl<
      kCuteFusedTileM,
      kCuteFusedTileN,
      kCuteFusedWarpsM,
      kCuteFusedWarpsN>(X, Wp, is_mixed_variant(variant), repeats, warmup);
}

void configure_cute_pipeline() {
  static bool configured = false;
  if (!configured) {
    flash_eq_fp16_pipeline::configure_pipeline_kernels<
        kCuteFusedTileM,
        kCuteFusedTileN,
        kCuteFusedWarpsM,
        kCuteFusedWarpsN>();
    configured = true;
  }
}

torch::Tensor pack_weight(torch::Tensor W) {
  configure_cute_pipeline();
  return flash_eq_fp16_pipeline::pack_weight_fp16_pipeline(W);
}

torch::Tensor pack_weight_gauss(torch::Tensor W) {
  configure_cute_pipeline();
  flash_eq_fp16_pipeline::check_weight_3d_half_cuda(W, "W");
  c10::cuda::CUDAGuard device_guard(W.device());
  int outNum = static_cast<int>(W.size(0));
  int inNum = static_cast<int>(W.size(1));
  auto Wg = torch::empty({flash_eq_fp16_pipeline::kPackedTranNum, outNum, inNum}, W.options());
  int total = outNum * inNum;
  int threads = 256;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  pack_weight_fp16_gauss_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
      reinterpret_cast<half*>(Wg.data_ptr<at::Half>()),
      inNum,
      outNum);
  FLASH_EQ_CUDA_CHECK(cudaGetLastError());
  return Wg;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pack_weight_fp16_cute_ada4090",
      &pack_weight,
      "Pack [out,in,4] fp16 EQ weight into [5,out,in] CuTe fused layout");
  m.def(
      "pack_weight_fp16_cute_gauss_ada4090",
      &pack_weight_gauss,
      "Pack [out,in,4] fp16 EQ weight into [5,out,in] CUTLASS Gauss staged layout");
  m.def(
      "flash_eq_linear_forward_direct_gemm_fp16_cute_ada4090",
      &forward_cute_impl,
      flash_eq_fp16_pipeline::py::arg("X"),
      flash_eq_fp16_pipeline::py::arg("Wp"),
      flash_eq_fp16_pipeline::py::arg("variant"),
      "Forward-only direct structured GEMM fp16 CuTe fused variants");
  m.def(
      "flash_eq_linear_profile_direct_gemm_fp16_cute_ada4090",
      &profile_cute_impl,
      flash_eq_fp16_pipeline::py::arg("X"),
      flash_eq_fp16_pipeline::py::arg("Wp"),
      flash_eq_fp16_pipeline::py::arg("variant"),
      flash_eq_fp16_pipeline::py::arg("repeats") = 80,
      flash_eq_fp16_pipeline::py::arg("warmup") = 20,
      "Profile forward-only direct structured GEMM fp16 CuTe fused variants");
}
