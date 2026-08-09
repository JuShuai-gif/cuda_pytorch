// A parallel processing pipeline (synthesized from Ch11 parallel patterns).
//
// Pipeline: source -> map (parallel) -> filter (parallel) -> reduce.
// Uses std::execution policies so each stage is one call with a policy;
// stage sizes are chosen so every stage keeps all cores busy.

#ifndef CHP_PARALLEL_PIPELINE_HPP
#define CHP_PARALLEL_PIPELINE_HPP

#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>
#include <utility>
#include <vector>

namespace chp {

// map then filter then reduce, all parallel. `map_fn` and `filter_fn` must be
// element-independent (no shared mutable state) and thread-safe.
template <typename In, typename Out, typename MapFn, typename FilterFn,
          typename ReduceFn>
Out parallel_pipeline(const std::vector<In>& src, MapFn map_fn,
                      FilterFn filter_fn, ReduceFn reduce_fn, Out init) {
    std::vector<Out> mapped(src.size());
    std::transform(std::execution::par, src.begin(), src.end(), mapped.begin(),
                   map_fn);

    // Compact the filtered elements (sparse -> dense).
    std::vector<Out> filtered(src.size());
    const auto end = std::copy_if(std::execution::par, mapped.begin(),
                                  mapped.end(), filtered.begin(), filter_fn);
    filtered.erase(end, filtered.end());

    return std::reduce(std::execution::par, filtered.begin(), filtered.end(),
                       std::move(init), reduce_fn);
}

}  // namespace chp

#endif  // CHP_PARALLEL_PIPELINE_HPP
