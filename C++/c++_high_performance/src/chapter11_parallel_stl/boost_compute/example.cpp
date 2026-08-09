// Boost.Compute: run STL-style algorithms on the GPU (PDF p.343-352).
//
// 1. transform-reduce: sum of circle areas, CPU vs GPU.
// 2. sort by radius with a GPU predicate, verified on the CPU.
// 3. Custom OpenCL kernel: a box filter on a 2D image, compared to CPU.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include <boost/compute.hpp>

namespace bc = boost::compute;

struct Circle {
    float x, y, r;
};

BOOST_COMPUTE_ADAPT_STRUCT(Circle, Circle, (x, y, r));

namespace {

std::vector<Circle> make_circles(std::size_t n) {
    std::mt19937 gen(2024u);
    std::uniform_real_distribution<float> xy(-100.0F, 100.0F);
    std::uniform_real_distribution<float> r(1.0F, 20.0F);
    std::vector<Circle> cs(n);
    for (auto& c : cs) {
        c.x = xy(gen);
        c.y = xy(gen);
        c.r = r(gen);
    }
    return cs;
}

float circle_area_cpu(const Circle& c) {
    constexpr float kPi = 3.14159F;
    return c.r * c.r * kPi;
}

BOOST_COMPUTE_FUNCTION(float, circle_area_gpu, (Circle c), {
    float pi = 3.14159f;
    return c.r * c.r * pi;
});

BOOST_COMPUTE_FUNCTION(bool, less_r_gpu, (Circle a, Circle b), {
    return a.r < b.r;
});

bool less_r_cpu(const Circle& a, const Circle& b) { return a.r < b.r; }

// CPU box filter reference (PDF p.333).
float box_filter_cpu(const std::vector<float>& src, int x, int y, int w, int r) {
    float sum = 0.0F;
    for (int yp = y - r; yp <= y + r; ++yp) {
        for (int xp = x - r; xp <= x + r; ++xp) {
            sum += src[static_cast<std::size_t>(yp * w + xp)];
        }
    }
    const auto n = static_cast<float>((r * 2 + 1) * (r * 2 + 1));
    return sum / n;
}

std::vector<float> box_filter_test_cpu(int w, int h, int r) {
    std::vector<float> src(static_cast<std::size_t>(w * h));
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i);
    }
    std::vector<float> dst(src.size(), 0.0F);
    for (int y = r; y < h - r; ++y) {
        for (int x = r; x < w - r; ++x) {
            dst[static_cast<std::size_t>(y * w + x)] =
                box_filter_cpu(src, x, y, w, r);
        }
    }
    return dst;
}

const char* kBoxFilterSource = R"(
kernel void box_filter(global const float* src, global float* dst,
                       int w, int r) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float sum = 0.0f;
    for (int yp = y - r; yp <= y + r; ++yp) {
        for (int xp = x - r; xp <= x + r; ++xp) {
            sum += src[yp * w + xp];
        }
    }
    float n = (float)((r * 2 + 1) * (r * 2 + 1));
    dst[y * w + x] = sum / n;
}
)";

}  // namespace

int main() {
    std::printf("== boost_compute ==\n");

    const auto device = bc::system::default_device();
    std::printf("GPU device: %s\n", device.name().c_str());
    auto context = bc::context(device);
    auto queue = bc::command_queue(context, device);

    // --- 1. transform-reduce: sum of circle areas ---
    {
        constexpr std::size_t n = 1'000'000;
        auto circles = make_circles(n);

        // CPU.
        double cpu_sum = 0.0;
        for (const auto& c : circles) {
            cpu_sum += circle_area_cpu(c);
        }

        // GPU: copy in, transform, reduce, copy out.
        auto gpu_circles = bc::vector<Circle>(n, context);
        bc::copy(circles.begin(), circles.end(), gpu_circles.begin(), queue);

        auto gpu_areas = bc::vector<float>(n, context);
        bc::transform(gpu_circles.begin(), gpu_circles.end(), gpu_areas.begin(),
                      circle_area_gpu, queue);

        auto gpu_sum = bc::vector<float>(1, context);
        bc::reduce(gpu_areas.begin(), gpu_areas.end(), gpu_sum.begin(), queue);

        float gpu_result = 0.0F;
        bc::copy(gpu_sum.begin(), gpu_sum.end(), &gpu_result, queue);

        std::printf("circle area sum: cpu=%g gpu=%g diff=%.4f%%\n", cpu_sum,
                    gpu_result, 100.0 * std::abs(cpu_sum - gpu_result) / cpu_sum);
    }

    // --- 2. sort by radius with a GPU predicate ---
    {
        constexpr std::size_t n = 100'000;
        auto circles = make_circles(n);
        auto gpu_circles = bc::vector<Circle>(n, context);
        bc::copy(circles.begin(), circles.end(), gpu_circles.begin(), queue);
        bc::sort(gpu_circles.begin(), gpu_circles.end(), less_r_gpu, queue);
        std::vector<Circle> back(n);
        bc::copy(gpu_circles.begin(), gpu_circles.end(), back.begin(), queue);
        const bool sorted = std::is_sorted(back.begin(), back.end(), less_r_cpu);
        std::printf("gpu sort by radius verified on cpu: %d\n", sorted);
    }

    // --- 3. custom kernel: 2D box filter ---
    {
        constexpr int kWidth = 200;
        constexpr int kHeight = 100;
        constexpr int kRadius = 2;

        // CPU reference.
        const auto cpu_dst = box_filter_test_cpu(kWidth, kHeight, kRadius);

        // GPU kernel.
        bc::program program = bc::program::create_with_source(kBoxFilterSource,
                                                              context);
        program.build();
        bc::kernel kernel(program, "box_filter");

        auto gpu_src = bc::vector<float>(static_cast<std::size_t>(kWidth * kHeight),
                                         context);
        bc::iota(gpu_src.begin(), gpu_src.end(), 0.0F, queue);
        auto gpu_dst = bc::vector<float>(static_cast<std::size_t>(kWidth * kHeight),
                                         context);
        bc::fill(gpu_dst.begin(), gpu_dst.end(), 0.0F, queue);

        const std::array<std::size_t, 2> offset{
            static_cast<std::size_t>(kRadius), static_cast<std::size_t>(kRadius)};
        const std::array<std::size_t, 2> elems{
            static_cast<std::size_t>(kWidth - 2 * kRadius),
            static_cast<std::size_t>(kHeight - 2 * kRadius)};

        kernel.set_arg(0, gpu_src);
        kernel.set_arg(1, gpu_dst);
        kernel.set_arg(2, kWidth);
        kernel.set_arg(3, kRadius);
        queue.enqueue_nd_range_kernel(kernel, 2, offset.data(), elems.data(),
                                      nullptr);

        std::vector<float> gpu_back(static_cast<std::size_t>(kWidth * kHeight));
        bc::copy(gpu_dst.begin(), gpu_dst.end(), gpu_back.begin(), queue);

        // Verify with epsilon tolerance (floating point, PDF p.335).
        std::size_t mismatches = 0;
        for (std::size_t i = 0; i < cpu_dst.size(); ++i) {
            if (std::abs(cpu_dst[i] - gpu_back[i]) > 1e-3F) {
                ++mismatches;
            }
        }
        std::printf("box filter mismatches (eps 1e-3): %zu / %zu\n",
                    mismatches, cpu_dst.size());
    }

    return 0;
}
