#pragma once

#include "cute/layout.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::fmha::collective {

template <class Element_, class StrideO_>
struct Sm100FmhaGenEpilogueWarpspecialized {
  using Pipeline = cutlass::PipelineAsync<2>;

  using SmemLayoutO = Layout<Shape<_1, _1, _1>>;
  using SmemLayoutO_ = SmemLayoutO;
  using Element = Element_;
  using StrideOOrig = StrideO_;
  using StrideO = decltype(replace<0>(StrideOOrig{}, 0));

  struct TensorStorage {
    using SmemLayoutO = SmemLayoutO_;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
  };

  struct Arguments {
    Element* ptr_o;
    StrideO dO;
  };

  using Params = Arguments;

  const Params& params;

  CUTLASS_DEVICE Sm100FmhaGenEpilogueWarpspecialized(const Params& params) : params(params) {}

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace = nullptr) {
    return args;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) { /* no-op */ }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE auto store(BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
                            Params const& params, ParamsProblemShape const& params_problem_shape,
                            TensorStorage& shared_storage, Pipeline& pipeline,
                            typename Pipeline::PipelineState& pipeline_consumer_state) {
    /* no-op */
  }
};

}  // namespace cutlass::fmha::collective
