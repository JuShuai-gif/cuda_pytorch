#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

struct HostPrecomputedTileScheduler {
  struct Arguments {
    int* work_indptr;
    int* qo_tile_indices;
    int* qo_head_indices;
    int* batch_indices;
  };

  struct Params {
    int* work_indptr;
    int* qo_tile_indices;
    int* qo_head_indices;
    int* batch_indices;
    int num_sm;
  };

  Params params;
  int work_ptr;
  int work_ptr_end;
  int qo_tile_idx;
  int batch_idx;
  int qo_head_idx;
  bool is_valid_;

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler(Params const& params) {
    this->params = params;
    work_ptr = params.work_indptr[blockIdx.x];
    work_ptr_end = params.work_indptr[blockIdx.x + 1];
    if (work_ptr < work_ptr_end) {
      qo_tile_idx = params.qo_tile_indices[work_ptr];
      batch_idx = params.batch_indices[work_ptr];
      qo_head_idx = params.qo_head_indices[work_ptr];
    } else {
      qo_tile_idx = 0;
      batch_idx = 0;
      qo_head_idx = 0;
    }
    is_valid_ = true;
  }

  static Params to_underlying_arguments(Arguments const& args, KernelHardwareInfo hw_info) {
    return {args.work_indptr, args.qo_tile_indices, args.qo_head_indices, args.batch_indices,
            hw_info.sm_count};
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(params.num_sm);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() const { return is_valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    return make_coord(qo_tile_idx, _0{}, make_coord(qo_head_idx, batch_idx));
  }

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler& operator++() {
    work_ptr++;
    is_valid_ = work_ptr < work_ptr_end;
    if (is_valid_) {
      qo_tile_idx = params.qo_tile_indices[work_ptr];
      batch_idx = params.batch_indices[work_ptr];
      qo_head_idx = params.qo_head_indices[work_ptr];
    }
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
