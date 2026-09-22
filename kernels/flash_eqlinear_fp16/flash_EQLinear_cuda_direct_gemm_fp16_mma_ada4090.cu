#include "flash_EQLinear_cuda_direct_gemm_fp16_pipeline_common_ada4090.cuh"

#include <string>

namespace {

constexpr int kMmaPipeTileM = 64;
constexpr int kMmaPipeTileN = 128;
constexpr int kMmaPipeWarpsM = 4;
constexpr int kMmaPipeWarpsN = 4;

constexpr int kMmaPipeV2PureTileM = 64;
constexpr int kMmaPipeV2PureTileN = 128;
constexpr int kMmaPipeV2PureWarpsM = 4;
constexpr int kMmaPipeV2PureWarpsN = 4;

constexpr int kMmaPipeV2MixedTileM = 64;
constexpr int kMmaPipeV2MixedTileN = 64;
constexpr int kMmaPipeV2MixedWarpsM = 4;
constexpr int kMmaPipeV2MixedWarpsN = 2;

constexpr int kMmaPipeM128N128TileM = 128;
constexpr int kMmaPipeM128N128TileN = 128;
constexpr int kMmaPipeM128N128WarpsM = 8;
constexpr int kMmaPipeM128N128WarpsN = 4;

constexpr int kMmaPipeM128N64TileM = 128;
constexpr int kMmaPipeM128N64TileN = 64;
constexpr int kMmaPipeM128N64WarpsM = 8;
constexpr int kMmaPipeM128N64WarpsN = 2;

constexpr int kMmaPipeM128N64W4TileM = 128;
constexpr int kMmaPipeM128N64W4TileN = 64;
constexpr int kMmaPipeM128N64W4WarpsM = 8;
constexpr int kMmaPipeM128N64W4WarpsN = 4;

constexpr int kMmaPipeM128N128W2TileM = 128;
constexpr int kMmaPipeM128N128W2TileN = 128;
constexpr int kMmaPipeM128N128W2WarpsM = 8;
constexpr int kMmaPipeM128N128W2WarpsN = 2;

constexpr int kMmaPipeM128N32TileM = 128;
constexpr int kMmaPipeM128N32TileN = 32;
constexpr int kMmaPipeM128N32WarpsM = 8;
constexpr int kMmaPipeM128N32WarpsN = 2;

constexpr int kMmaPipeM128N32ModcTileM = 128;
constexpr int kMmaPipeM128N32ModcTileN = 32;
constexpr int kMmaPipeM128N32ModcWarpsM = 8;
constexpr int kMmaPipeM128N32ModcWarpsN = 4;

constexpr int kMmaPipeM64N32ModcTileM = 64;
constexpr int kMmaPipeM64N32ModcTileN = 32;
constexpr int kMmaPipeM64N32ModcWarpsM = 4;
constexpr int kMmaPipeM64N32ModcWarpsN = 4;

constexpr int kMmaPipeM64N64TileM = 64;
constexpr int kMmaPipeM64N64TileN = 64;
constexpr int kMmaPipeM64N64WarpsM = 4;
constexpr int kMmaPipeM64N64WarpsN = 2;

constexpr int kMmaPipeM64N64W4TileM = 64;
constexpr int kMmaPipeM64N64W4TileN = 64;
constexpr int kMmaPipeM64N64W4WarpsM = 4;
constexpr int kMmaPipeM64N64W4WarpsN = 4;

constexpr int kMmaPipeM64N256TileM = 64;
constexpr int kMmaPipeM64N256TileN = 256;
constexpr int kMmaPipeM64N256WarpsM = 4;
constexpr int kMmaPipeM64N256WarpsN = 8;

constexpr int kMmaPipeM128N64X2DOutTiles = 2;
constexpr int kMmaPipeM128N64X4DOutTiles = 4;
constexpr int kMmaPipeM64N64X2DOutTiles = 2;

constexpr int kMmaPipeWarpSpecTileM = 64;
constexpr int kMmaPipeWarpSpecTileN = 96;
constexpr int kMmaPipeWarpSpecWarpsM = 4;
constexpr int kMmaPipeWarpSpecWarpsN = 3;
constexpr int kMmaPipeWarpSpecProducerWarps = 4;

constexpr int kMmaPipeWarpSpec15TileM = 48;
constexpr int kMmaPipeWarpSpec15TileN = 120;
constexpr int kMmaPipeWarpSpec15WarpsM = 3;
constexpr int kMmaPipeWarpSpec15WarpsN = 5;
constexpr int kMmaPipeWarpSpec15ProducerWarps = 1;

torch::Tensor forward_mma_impl(torch::Tensor X, torch::Tensor Wp, const std::string& variant) {
  if (variant == "pure_formula_scalar") {
    return flash_eq_fp16_pipeline::forward_formula_scalar_impl(X, Wp, false);
  }
  if (variant == "mixed_formula_scalar") {
    return flash_eq_fp16_pipeline::forward_formula_scalar_impl(X, Wp, true);
  }
  if (variant == "pure_mma_pipe_cpasync_w") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_cpasync_w_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp);
  }
  if (variant == "pure_mma_pipe_db") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_double_buffer_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp);
  }
  if (variant == "pure_mma_pipe_warpspec") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_warpspec_impl<
        kMmaPipeWarpSpecTileM,
        kMmaPipeWarpSpecTileN,
        kMmaPipeWarpSpecWarpsM,
        kMmaPipeWarpSpecWarpsN,
        kMmaPipeWarpSpecProducerWarps>(X, Wp);
  }
  if (variant == "pure_mma_pipe_warpspec15") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_warpspec_impl<
        kMmaPipeWarpSpec15TileM,
        kMmaPipeWarpSpec15TileN,
        kMmaPipeWarpSpec15WarpsM,
        kMmaPipeWarpSpec15WarpsN,
        kMmaPipeWarpSpec15ProducerWarps>(X, Wp);
  }
  if (variant == "pure_mma_pipe" || variant == "pure_mma_pipe_v2") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp, false);
  }
  if (variant == "mixed_mma_pipe_v2") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeV2MixedTileM,
        kMmaPipeV2MixedTileN,
        kMmaPipeV2MixedWarpsM,
        kMmaPipeV2MixedWarpsN>(X, Wp, true);
  }
  if (variant == "pure_mma_pipe_m128n128") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N128TileM,
        kMmaPipeM128N128TileN,
        kMmaPipeM128N128WarpsM,
        kMmaPipeM128N128WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n64") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n64_db") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_double_buffer_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp);
  }
  if (variant == "pure_mma_pipe_m128n64_w4") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n128_w2") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N128W2TileM,
        kMmaPipeM128N128W2TileN,
        kMmaPipeM128N128W2WarpsM,
        kMmaPipeM128N128W2WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n64_gauss") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_gauss_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp);
  }
  if (variant == "pure_mma_pipe_m128n64_w4_gauss") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_gauss_impl<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>(X, Wp);
  }
  if (variant == "pure_mma_pipe_m128n32") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N32TileM,
        kMmaPipeM128N32TileN,
        kMmaPipeM128N32WarpsM,
        kMmaPipeM128N32WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n32_modc") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM128N32ModcTileM,
        kMmaPipeM128N32ModcTileN,
        kMmaPipeM128N32ModcWarpsM,
        kMmaPipeM128N32ModcWarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m64n32_modc") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM64N32ModcTileM,
        kMmaPipeM64N32ModcTileN,
        kMmaPipeM64N32ModcWarpsM,
        kMmaPipeM64N32ModcWarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m64n64") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM64N64TileM,
        kMmaPipeM64N64TileN,
        kMmaPipeM64N64WarpsM,
        kMmaPipeM64N64WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m64n64_w4") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM64N64W4TileM,
        kMmaPipeM64N64W4TileN,
        kMmaPipeM64N64W4WarpsM,
        kMmaPipeM64N64W4WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m64n256") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
        kMmaPipeM64N256TileM,
        kMmaPipeM64N256TileN,
        kMmaPipeM64N256WarpsM,
        kMmaPipeM64N256WarpsN>(X, Wp, false);
  }
  if (variant == "pure_mma_pipe_m128n64_x2d") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_multi_n_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN,
        kMmaPipeM128N64X2DOutTiles>(X, Wp);
  }
  if (variant == "pure_mma_pipe_m128n64_x4d") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_multi_n_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN,
        kMmaPipeM128N64X4DOutTiles>(X, Wp);
  }
  if (variant == "pure_mma_pipe_m64n64_x2d") {
    return flash_eq_fp16_pipeline::forward_mma_ptx_multi_n_impl<
        kMmaPipeM64N64TileM,
        kMmaPipeM64N64TileN,
        kMmaPipeM64N64WarpsM,
        kMmaPipeM64N64WarpsN,
        kMmaPipeM64N64X2DOutTiles>(X, Wp);
  }
  if (variant != "mixed_mma_pipe") {
    TORCH_CHECK(false, "unknown mma pipeline variant: ", variant);
  }
  return flash_eq_fp16_pipeline::forward_mma_ptx_impl<
      kMmaPipeTileM,
      kMmaPipeTileN,
      kMmaPipeWarpsM,
      kMmaPipeWarpsN>(X, Wp, true);
}

flash_eq_fp16_pipeline::py::dict profile_mma_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    const std::string& variant,
    int repeats,
    int warmup) {
  if (variant == "pure_formula_scalar") {
    return flash_eq_fp16_pipeline::profile_formula_scalar_impl(X, Wp, false, repeats, warmup);
  }
  if (variant == "mixed_formula_scalar") {
    return flash_eq_fp16_pipeline::profile_formula_scalar_impl(X, Wp, true, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_cpasync_w") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_cpasync_w_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_db") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_double_buffer_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_warpspec") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_warpspec_impl<
        kMmaPipeWarpSpecTileM,
        kMmaPipeWarpSpecTileN,
        kMmaPipeWarpSpecWarpsM,
        kMmaPipeWarpSpecWarpsN,
        kMmaPipeWarpSpecProducerWarps>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_warpspec15") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_warpspec_impl<
        kMmaPipeWarpSpec15TileM,
        kMmaPipeWarpSpec15TileN,
        kMmaPipeWarpSpec15WarpsM,
        kMmaPipeWarpSpec15WarpsN,
        kMmaPipeWarpSpec15ProducerWarps>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe" || variant == "pure_mma_pipe_v2") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "mixed_mma_pipe_v2") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeV2MixedTileM,
        kMmaPipeV2MixedTileN,
        kMmaPipeV2MixedWarpsM,
        kMmaPipeV2MixedWarpsN>(X, Wp, true, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n128") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N128TileM,
        kMmaPipeM128N128TileN,
        kMmaPipeM128N128WarpsM,
        kMmaPipeM128N128WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_db") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_double_buffer_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_w4") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n128_w2") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N128W2TileM,
        kMmaPipeM128N128W2TileN,
        kMmaPipeM128N128W2WarpsM,
        kMmaPipeM128N128W2WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_gauss") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_gauss_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_w4_gauss") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_gauss_impl<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n32") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N32TileM,
        kMmaPipeM128N32TileN,
        kMmaPipeM128N32WarpsM,
        kMmaPipeM128N32WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n32_modc") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM128N32ModcTileM,
        kMmaPipeM128N32ModcTileN,
        kMmaPipeM128N32ModcWarpsM,
        kMmaPipeM128N32ModcWarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n32_modc") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM64N32ModcTileM,
        kMmaPipeM64N32ModcTileN,
        kMmaPipeM64N32ModcWarpsM,
        kMmaPipeM64N32ModcWarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n64") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM64N64TileM,
        kMmaPipeM64N64TileN,
        kMmaPipeM64N64WarpsM,
        kMmaPipeM64N64WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n64_w4") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM64N64W4TileM,
        kMmaPipeM64N64W4TileN,
        kMmaPipeM64N64W4WarpsM,
        kMmaPipeM64N64W4WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n256") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
        kMmaPipeM64N256TileM,
        kMmaPipeM64N256TileN,
        kMmaPipeM64N256WarpsM,
        kMmaPipeM64N256WarpsN>(X, Wp, false, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_x2d") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_multi_n_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN,
        kMmaPipeM128N64X2DOutTiles>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64_x4d") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_multi_n_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN,
        kMmaPipeM128N64X4DOutTiles>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n64_x2d") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_multi_n_impl<
        kMmaPipeM64N64TileM,
        kMmaPipeM64N64TileN,
        kMmaPipeM64N64WarpsM,
        kMmaPipeM64N64WarpsN,
        kMmaPipeM64N64X2DOutTiles>(X, Wp, repeats, warmup);
  }
  if (variant != "mixed_mma_pipe") {
    TORCH_CHECK(false, "unknown mma pipeline variant: ", variant);
  }
  return flash_eq_fp16_pipeline::profile_mma_ptx_impl<
      kMmaPipeTileM,
      kMmaPipeTileN,
      kMmaPipeWarpsM,
      kMmaPipeWarpsN>(X, Wp, true, repeats, warmup);
}

flash_eq_fp16_pipeline::py::dict profile_mma_phases_impl(
    torch::Tensor X,
    torch::Tensor Wp,
    const std::string& variant,
    int repeats,
    int warmup) {
  if (variant == "pure_mma_pipe_v2") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_phase_impl<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n128") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_phase_impl<
        kMmaPipeM128N128TileM,
        kMmaPipeM128N128TileN,
        kMmaPipeM128N128WarpsM,
        kMmaPipeM128N128WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n64") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_phase_impl<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m128n128_w2") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_phase_impl<
        kMmaPipeM128N128W2TileM,
        kMmaPipeM128N128W2TileN,
        kMmaPipeM128N128W2WarpsM,
        kMmaPipeM128N128W2WarpsN>(X, Wp, repeats, warmup);
  }
  if (variant == "pure_mma_pipe_m64n256") {
    return flash_eq_fp16_pipeline::profile_mma_ptx_phase_impl<
        kMmaPipeM64N256TileM,
        kMmaPipeM64N256TileN,
        kMmaPipeM64N256WarpsM,
        kMmaPipeM64N256WarpsN>(X, Wp, repeats, warmup);
  }
  TORCH_CHECK(false, "phase profile unsupported for variant: ", variant);
}

void configure_mma_pipeline() {
  static bool configured = false;
  if (!configured) {
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeTileM,
        kMmaPipeTileN,
        kMmaPipeWarpsM,
        kMmaPipeWarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeV2PureTileM,
        kMmaPipeV2PureTileN,
        kMmaPipeV2PureWarpsM,
        kMmaPipeV2PureWarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeV2MixedTileM,
        kMmaPipeV2MixedTileN,
        kMmaPipeV2MixedWarpsM,
        kMmaPipeV2MixedWarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N128TileM,
        kMmaPipeM128N128TileN,
        kMmaPipeM128N128WarpsM,
        kMmaPipeM128N128WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N128W2TileM,
        kMmaPipeM128N128W2TileN,
        kMmaPipeM128N128W2WarpsM,
        kMmaPipeM128N128W2WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_gauss_kernels<
        kMmaPipeM128N64TileM,
        kMmaPipeM128N64TileN,
        kMmaPipeM128N64WarpsM,
        kMmaPipeM128N64WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_gauss_kernels<
        kMmaPipeM128N64W4TileM,
        kMmaPipeM128N64W4TileN,
        kMmaPipeM128N64W4WarpsM,
        kMmaPipeM128N64W4WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N32TileM,
        kMmaPipeM128N32TileN,
        kMmaPipeM128N32WarpsM,
        kMmaPipeM128N32WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM128N32ModcTileM,
        kMmaPipeM128N32ModcTileN,
        kMmaPipeM128N32ModcWarpsM,
        kMmaPipeM128N32ModcWarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM64N32ModcTileM,
        kMmaPipeM64N32ModcTileN,
        kMmaPipeM64N32ModcWarpsM,
        kMmaPipeM64N32ModcWarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM64N64TileM,
        kMmaPipeM64N64TileN,
        kMmaPipeM64N64WarpsM,
        kMmaPipeM64N64WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM64N64W4TileM,
        kMmaPipeM64N64W4TileN,
        kMmaPipeM64N64W4WarpsM,
        kMmaPipeM64N64W4WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_ptx_kernels<
        kMmaPipeM64N256TileM,
        kMmaPipeM64N256TileN,
        kMmaPipeM64N256WarpsM,
        kMmaPipeM64N256WarpsN>();
    flash_eq_fp16_pipeline::configure_mma_warpspec_kernels<
        kMmaPipeWarpSpecTileM,
        kMmaPipeWarpSpecTileN,
        kMmaPipeWarpSpecWarpsM,
        kMmaPipeWarpSpecWarpsN,
        kMmaPipeWarpSpecProducerWarps>();
    flash_eq_fp16_pipeline::configure_mma_warpspec_kernels<
        kMmaPipeWarpSpec15TileM,
        kMmaPipeWarpSpec15TileN,
        kMmaPipeWarpSpec15WarpsM,
        kMmaPipeWarpSpec15WarpsN,
        kMmaPipeWarpSpec15ProducerWarps>();
    configured = true;
  }
}

torch::Tensor pack_weight(torch::Tensor W) {
  configure_mma_pipeline();
  return flash_eq_fp16_pipeline::pack_weight_fp16_pipeline(W);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "pack_weight_fp16_mma_ada4090",
      &pack_weight,
      "Pack [out,in,4] fp16 EQ weight into [5,out,in] mma pipeline layout");
  m.def(
      "flash_eq_linear_forward_direct_gemm_fp16_mma_ada4090",
      &forward_mma_impl,
      flash_eq_fp16_pipeline::py::arg("X"),
      flash_eq_fp16_pipeline::py::arg("Wp"),
      flash_eq_fp16_pipeline::py::arg("variant"),
      "Forward-only direct structured GEMM fp16 MMA pipeline variants");
  m.def(
      "flash_eq_linear_profile_direct_gemm_fp16_mma_ada4090",
      &profile_mma_impl,
      flash_eq_fp16_pipeline::py::arg("X"),
      flash_eq_fp16_pipeline::py::arg("Wp"),
      flash_eq_fp16_pipeline::py::arg("variant"),
      flash_eq_fp16_pipeline::py::arg("repeats") = 80,
      flash_eq_fp16_pipeline::py::arg("warmup") = 20,
      "Profile forward-only direct structured GEMM fp16 MMA pipeline variants");
  m.def(
      "flash_eq_linear_profile_phases_direct_gemm_fp16_mma_ada4090",
      &profile_mma_phases_impl,
      flash_eq_fp16_pipeline::py::arg("X"),
      flash_eq_fp16_pipeline::py::arg("Wp"),
      flash_eq_fp16_pipeline::py::arg("variant"),
      flash_eq_fp16_pipeline::py::arg("repeats") = 80,
      flash_eq_fp16_pipeline::py::arg("warmup") = 20,
      "Profile diagnostic non-MMA phase kernels for selected MMA pipeline variants");
  m.def(
      "debug_ldmatrix_x4_mapping",
      &flash_eq_fp16_pipeline::debug_ldmatrix_x4_mapping,
      "Return lane/register mapping for ldmatrix.x4 debug");
}
