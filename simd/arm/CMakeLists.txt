# ARM SIMD targets (NEON / SVE)
# Requires BUILD_ARM=ON (auto-detected on aarch64/arm64 hosts)

# Sanity check: warn if building ARM targets on a non-ARM machine
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv7")
    message(WARNING "Building ARM SIMD targets on non-ARM host (${CMAKE_SYSTEM_PROCESSOR}). Cross-compilation assumed.")
endif()

# Export compile_commands.json for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Common flags shared by all ARM targets
# ---------------------------------------------------------------------------
set(ARM_COMMON_OPTIONS
    -O2 -g
    -Wall -Wextra -Wpedantic
    -Wshadow -Wconversion
    -fno-math-errno
)

# Path to common SIMD utility headers (relative to this directory)
set(COMMON_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../common")

# ---------------------------------------------------------------------------
# NEON targets (ARMv8.0-A + SIMD)
# ---------------------------------------------------------------------------
set(NEON_SOURCES
    src/neon_vector_add.cpp
    src/neon_relu_clamp.cpp
    src/neon_dot_product.cpp
    src/neon_reduce_sum.cpp
    src/neon_memcpy_like.cpp
    src/neon_aos_to_soa.cpp
    src/neon_rgb_to_gray.cpp
    src/neon_conv1d.cpp
    src/neon_int8_dot.cpp
    src/neon_layernorm.cpp
    src/neon_softmax_partial.cpp
    src/neon_autovec_vs_intrinsics.cpp
    src/dispatch_demo.cpp
)

foreach(src ${NEON_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_compile_options(${name} PRIVATE
        ${ARM_COMMON_OPTIONS}
        -march=armv8-a+simd
    )
    target_link_options(${name} PRIVATE -march=armv8-a+simd)
    target_include_directories(${name} PRIVATE ${COMMON_HEADERS_DIR})

    # Emit vectorization remarks for the autovec-vs-intrinsics comparison
    if(name STREQUAL "neon_autovec_vs_intrinsics")
        target_compile_options(${name} PRIVATE -fopt-info-vec -fopt-info-vec-missed)
    endif()

    # neon_int8_dot requires ARMv8.2-A dot-product extension
    if(name STREQUAL "neon_int8_dot")
        target_compile_options(${name} PRIVATE
            ${ARM_COMMON_OPTIONS}
            -march=armv8.2-a+dotprod
        )
        target_link_options(${name} PRIVATE -march=armv8.2-a+dotprod)
    endif()

    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arm"
    )
endforeach()

# ---------------------------------------------------------------------------
# SVE targets (optional -- BUILD_ARM_SVE=ON)
# ---------------------------------------------------------------------------
if(BUILD_ARM_SVE)
    set(SVE_SOURCES
        src/sve_vector_add.cpp
        src/sve_reduce_sum.cpp
        src/sve_dot_product.cpp
        src/sve_predicated_tail.cpp
        src/dispatch_demo.cpp
    )

    foreach(src ${SVE_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_compile_options(${name} PRIVATE
            ${ARM_COMMON_OPTIONS}
            -march=armv8-a+sve
        )
        target_link_options(${name} PRIVATE -march=armv8-a+sve)
        target_include_directories(${name} PRIVATE ${COMMON_HEADERS_DIR})

        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arm"
        )
    endforeach()
endif()
