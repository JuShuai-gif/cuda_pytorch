#pragma once

// Hand-rolled parallel algorithms from PDF Chapter 11 (p.319-331).
//
// These are teaching reimplementations, not copies of the book's source.
// They exist to show the *cost* of writing parallelism by hand; prefer the
// C++17 execution policies (see execution_policies/) in production code.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <future>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

namespace chp11 {

// Naive: split into hardware_concurrency() equal chunks, one async each
// (PDF p.302). Fixed chunk count is fragile: a slow chunk stalls everyone.
// Note the book's version misses the tail when n is not a multiple of
// num_tasks; iterating while start < n covers the whole range.
template <typename SrcIt, typename DstIt, typename Func>
void par_transform_naive(SrcIt first, SrcIt last, DstIt dst, Func f) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n == 0) {
        return;
    }
    const auto num_tasks =
        static_cast<std::size_t>(std::max(std::thread::hardware_concurrency(), 1u));
    const auto chunk_sz = std::max(n / num_tasks, std::size_t{1});

    std::vector<std::future<void>> futures;
    futures.reserve(num_tasks + 1);
    for (std::size_t start = 0; start < n; start += chunk_sz) {
        const auto stop = std::min(start + chunk_sz, n);
        futures.push_back(std::async(std::launch::async, [=] {
            std::transform(first + start, first + stop, dst + start, f);
        }));
    }
    for (auto& fut : futures) {
        fut.wait();
    }
}

// Divide and conquer: recurse splitting in half until chunk_sz, branch one
// half to a task and process the other on the calling thread (PDF p.306).
// A small chunk size yields many small tasks that the scheduler balances.
template <typename SrcIt, typename DstIt, typename Func>
void par_transform(SrcIt first, SrcIt last, DstIt dst, Func f,
                   std::size_t chunk_sz) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= chunk_sz) {
        std::transform(first, last, dst, f);
        return;
    }
    const auto src_mid = std::next(first, n / 2);
    auto future = std::async(std::launch::async, [=] {
        par_transform(first, src_mid, dst, f, chunk_sz);
    });
    par_transform(src_mid, last, std::next(dst, n / 2), f, chunk_sz);
    future.wait();
}

// Divide-and-conquer count_if: add the two half counts (PDF p.309).
template <typename It, typename Pred>
std::size_t par_count_if(It first, It last, Pred pred, std::size_t chunk_sz) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= chunk_sz) {
        return static_cast<std::size_t>(std::count_if(first, last, pred));
    }
    const auto middle = std::next(first, n / 2);
    auto future = std::async(std::launch::async, [=] {
        return par_count_if(first, middle, pred, chunk_sz);
    });
    return par_count_if(middle, last, pred, chunk_sz) + future.get();
}

// --- Parallel copy_if, strategy 1: atomic write index (PDF p.311) ---
// Correct but suffers false sharing: threads write adjacent destinations,
// so with a cheap predicate it is usually slower than serial.
template <typename SrcIt, typename DstIt, typename Pred>
void inner_copy_if_sync(SrcIt first, SrcIt last, DstIt dst,
                        std::atomic<std::size_t>& dst_idx, Pred pred,
                        std::size_t chunk_sz) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= chunk_sz) {
        std::for_each(first, last, [&](const auto& v) {
            if (pred(v)) {
                const auto write_idx = dst_idx.fetch_add(1);
                *std::next(dst, static_cast<std::ptrdiff_t>(write_idx)) = v;
            }
        });
        return;
    }
    const auto middle = std::next(first, n / 2);
    auto future = std::async(std::launch::async, [=, &pred, &dst_idx] {
        inner_copy_if_sync(first, middle, dst, dst_idx, pred, chunk_sz);
    });
    inner_copy_if_sync(middle, last, dst, dst_idx, pred, chunk_sz);
    future.wait();
}

template <typename SrcIt, typename DstIt, typename Pred>
DstIt par_copy_if_sync(SrcIt first, SrcIt last, DstIt dst, Pred pred,
                       std::size_t chunk_sz) {
    std::atomic<std::size_t> dst_idx{0};
    inner_copy_if_sync(first, last, dst, dst_idx, pred, chunk_sz);
    return std::next(dst, static_cast<std::ptrdiff_t>(dst_idx.load()));
}

// --- Parallel copy_if, strategy 2: sparse copy + compaction (PDF p.313) ---
// No shared writes: each chunk copies to its own sparse range, then the
// ranges are moved sequentially into place.
template <typename SrcIt, typename DstIt, typename Pred>
DstIt par_copy_if_split(SrcIt first, SrcIt last, DstIt dst, Pred pred,
                        std::size_t chunk_sz) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    using CopiedRange = std::pair<DstIt, DstIt>;
    using FutureType = std::future<CopiedRange>;

    std::vector<FutureType> futures;
    futures.reserve(n / chunk_sz + 1);
    for (std::size_t start = 0; start < n; start += chunk_sz) {
        const auto stop = std::min(start + chunk_sz, n);
        futures.push_back(std::async(std::launch::async, [=, &pred] {
            const auto dst_first = std::next(dst, static_cast<std::ptrdiff_t>(start));
            const auto dst_last =
                std::copy_if(std::next(first, static_cast<std::ptrdiff_t>(start)),
                             std::next(first, static_cast<std::ptrdiff_t>(stop)),
                             dst_first, pred);
            return std::make_pair(dst_first, dst_last);
        }));
    }

    DstIt new_end = futures.front().get().second;
    for (auto it = std::next(futures.begin()); it != futures.end(); ++it) {
        const auto chunk_rng = it->get();
        new_end = std::move(chunk_rng.first, chunk_rng.second, new_end);
    }
    return new_end;
}

}  // namespace chp11
