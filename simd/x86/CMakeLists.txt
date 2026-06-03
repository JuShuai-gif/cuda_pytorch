cmake_minimum_required(VERSION 3.18)

# x86 SIMD targets (AVX2 / AVX-512)
# Requires BUILD_X86=ON (auto-detected on x86_64/amd64 hosts)

# Enable ASM language for assembly source files (.s)
enable_language(ASM)

# Sanity check: warn if building x86 targets on a non-x86 machine
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i686")
    message(WARNING "Building x86 SIMD targets on non-x86 host (${CMAKE_SYSTEM_PROCESSOR}). Cross-compilation assumed.")
endif()

# Export compile_commands.json for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Common flags shared by all x86 targets
# ---------------------------------------------------------------------------
set(X86_COMMON_OPTIONS
    -O2 -g
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion
    -fno-math-errno
)

# Path to common SIMD utility headers (relative to this directory)
set(COMMON_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")

# ---------------------------------------------------------------------------
# AVX2 targets
# ---------------------------------------------------------------------------
set(AVX2_SOURCES
    src/avx2_vector_add.cpp
    src/avx2_relu_clamp.cpp
    src/avx2_dot_product.cpp
    src/avx2_reduce_sum.cpp
    src/avx2_memcpy_like.cpp
    src/avx2_aos_to_soa.cpp
    src/avx2_rgb_to_gray.cpp
    src/avx2_conv1d.cpp
    src/avx2_int8_dot.cpp
    src/avx2_layernorm.cpp
    src/avx2_softmax_partial.cpp
    src/avx2_autovec_vs_intrinsics.cpp
    src/dispatch_demo.cpp
    src/edge_cases_demo.cpp
    src/avx2_gemm_micro.cpp
    src/complete_softmax_avx2.cpp
    src/avx2_string_search.cpp
    src/avx2_image_conv2d.cpp
    src/cpuid_full_demo.cpp
    src/avx2_gemm_production.cpp
    src/avx2_scaled_attention.cpp
    src/avx2_batchnorm.cpp
    src/avx2_winograd_f23.cpp
    src/avx2_int8_gemm_requantize.cpp
    src/avx2_math_functions.cpp
)

foreach(src ${AVX2_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_compile_options(${name} PRIVATE
        ${X86_COMMON_OPTIONS}
        -mavx2 -mfma
    )
    target_link_options(${name} PRIVATE -mavx2 -mfma)
    target_include_directories(${name} PRIVATE ${COMMON_HEADERS_DIR})

    # Emit vectorization remarks for the autovec-vs-intrinsics comparison
    if(name STREQUAL "avx2_autovec_vs_intrinsics")
        target_compile_options(${name} PRIVATE -fopt-info-vec -fopt-info-vec-missed)
    endif()

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86"
    )
endforeach()

# ---- portable_simd_demo: compile same source for multiple ISAs ----

# SSE target (128-bit) - demonstrate SSE basics
add_executable(sse_basics_demo src/sse_basics_demo.cpp)
target_compile_options(sse_basics_demo PRIVATE ${X86_COMMON_OPTIONS} -msse4.1)
target_link_options(sse_basics_demo PRIVATE -msse4.1)
target_include_directories(sse_basics_demo PRIVATE ${COMMON_HEADERS_DIR})
set_target_properties(sse_basics_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")

# CRC32C demo (SSE4.2)
add_executable(crc32c_demo src/crc32c_demo.cpp)
target_compile_options(crc32c_demo PRIVATE ${X86_COMMON_OPTIONS} -msse4.2)
target_link_options(crc32c_demo PRIVATE -msse4.2)
target_include_directories(crc32c_demo PRIVATE ${COMMON_HEADERS_DIR})
set_target_properties(crc32c_demo PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")

# ASM/Intrinsics bridge demo (C++ + assembly)
add_executable(asm_intrinsics_bridge
    src/asm_intrinsics_bridge.cpp
    src/x86_64_calling_convention.s
)
target_compile_options(asm_intrinsics_bridge PRIVATE ${X86_COMMON_OPTIONS} -mavx2 -mfma)
target_link_options(asm_intrinsics_bridge PRIVATE -mavx2 -mfma)
target_include_directories(asm_intrinsics_bridge PRIVATE ${COMMON_HEADERS_DIR})
set_target_properties(asm_intrinsics_bridge PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")

# ---- portable_simd_demo: compile same source for multiple ISAs ----

# AVX-512 arch flags (define early, used by portable_avx512 below)
set(AVX512_ARCH_FLAGS
    -mavx512f
    -mavx512bw
    -mavx512vl
    -mavx512dq
    -mavx512cd
    -mfma
)

# SSE variant (128-bit)
add_executable(portable_sse src/portable_simd_demo.cpp)
target_compile_options(portable_sse PRIVATE ${X86_COMMON_OPTIONS} -msse4.2)
target_link_options(portable_sse PRIVATE -msse4.2)
target_include_directories(portable_sse PRIVATE ${COMMON_HEADERS_DIR})
set_target_properties(portable_sse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")

# AVX2 variant (256-bit)
add_executable(portable_avx2 src/portable_simd_demo.cpp)
target_compile_options(portable_avx2 PRIVATE ${X86_COMMON_OPTIONS} -mavx2 -mfma)
target_link_options(portable_avx2 PRIVATE -mavx2 -mfma)
target_include_directories(portable_avx2 PRIVATE ${COMMON_HEADERS_DIR})
set_target_properties(portable_avx2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")

# AVX-512 variant (512-bit, build only if AVX-512 enabled)
if(BUILD_X86_AVX512)
    add_executable(portable_avx512 src/portable_simd_demo.cpp)
    target_compile_options(portable_avx512 PRIVATE ${X86_COMMON_OPTIONS} ${AVX512_ARCH_FLAGS})
    target_link_options(portable_avx512 PRIVATE ${AVX512_ARCH_FLAGS})
    target_include_directories(portable_avx512 PRIVATE ${COMMON_HEADERS_DIR})
    set_target_properties(portable_avx512 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86")
endif()

# ---------------------------------------------------------------------------
# AVX-512 targets (optional -- BUILD_X86_AVX512=ON)
# ---------------------------------------------------------------------------
if(BUILD_X86_AVX512)
    set(AVX512_SOURCES
        src/avx512_vector_add.cpp
        src/avx512_reduce_sum.cpp
        src/avx512_dot_product.cpp
        src/avx512_masked_tail.cpp
        src/avx512_gather_scatter.cpp
        src/avx512_byte_scan.cpp
        src/avx512_conflict_demo.cpp
        src/avx512_compress_expand_demo.cpp
        src/avx512_rounding_demo.cpp
    )

    foreach(src ${AVX512_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_compile_options(${name} PRIVATE
            ${X86_COMMON_OPTIONS}
            ${AVX512_ARCH_FLAGS}
        )
        target_link_options(${name} PRIVATE ${AVX512_ARCH_FLAGS})
        target_include_directories(${name} PRIVATE ${COMMON_HEADERS_DIR})

        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/x86"
        )
    endforeach()
endif()
