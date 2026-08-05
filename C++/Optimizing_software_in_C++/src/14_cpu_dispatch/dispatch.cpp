// 14_cpu_dispatch: CPUID detection + dispatch-on-first-call.
//
// PDF 13.1/13.5 (p135-141, Example 13.1): a function pointer initially
// points to a dispatcher; after the first call it points to the best
// implementation for this CPU. A --force flag lets you run any branch for
// correctness testing (PDF 13.4 p139).
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/benchmark.h"
#include "common/cpu_info.h"

void vadd_scalar(float* c, const float* a, const float* b, int n);
void vadd_sse2(float* c, const float* a, const float* b, int n);
void vadd_avx2(float* c, const float* a, const float* b, int n);

typedef void (*vadd_fn)(float*, const float*, const float*, int);

static vadd_fn selected = nullptr;

// --- implementations --------------------------------------------------------
void vadd_scalar(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

void vadd_sse2(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];  // SSE2 shown separately
}

// --- dispatcher (first call only) ------------------------------------------
// Chooses the best implementation once and caches it in `selected`.
static vadd_fn dispatch_impl() {
    int level = cpu_instruction_set_level();
    if (level >= 8) selected = vadd_avx2;
    else if (level >= 2) selected = vadd_sse2;
    else selected = vadd_scalar;
    return selected;
}

// Entry point: through a function pointer that the dispatcher rewrites.
static void vadd_dispatch(float* c, const float* a, const float* b, int n) {
    vadd_fn fn = selected ? selected : dispatch_impl();
    fn(c, a, b, n);
}

int main(int argc, char** argv) {
    const int n = 16'000'000;
    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f), ref(n, 0.0f);

    cpu_print_info();

    // optional --force <scalar|sse2|avx2> to test a specific branch
    if (argc == 3 && std::strcmp(argv[1], "--force") == 0) {
        if (std::strcmp(argv[2], "scalar") == 0) selected = vadd_scalar;
        else if (std::strcmp(argv[2], "sse2") == 0) selected = vadd_sse2;
        else if (std::strcmp(argv[2], "avx2") == 0) selected = vadd_avx2;
        else { std::printf("unknown branch: %s\n", argv[2]); return 2; }
    }

    vadd_dispatch(c.data(), a.data(), b.data(), n);
    vadd_scalar(ref.data(), a.data(), b.data(), n);

    bool ok = true;
    for (int i = 0; i < n; ++i) if (c[i] != ref[i]) { ok = false; break; }
    std::printf("checksum match with scalar reference: %s\n", ok ? "yes" : "no");

    // time the selected implementation
    bench("dispatched vadd", [&] {
        vadd_dispatch(c.data(), a.data(), b.data(), n);
        return c[n - 1];
    });
    return ok ? 0 : 1;
}
