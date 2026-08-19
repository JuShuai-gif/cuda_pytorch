cmake_minimum_required(VERSION 3.20)
project(LLVMCodeGeneration VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLVM_CODEGEN_STRICT_VERSION
    "Reject LLVM versions outside the book's 20.1.x API baseline" ON)
option(LLVM_CODEGEN_WARNINGS_AS_ERRORS
    "Compile examples with warnings treated as errors" OFF)

# -----------------------------------------------------------------------------
# Find LLVM. The book's snippets use LLVM 20.1.1. LLVM's C++ API is not
# source-stable across releases, so fail during configuration instead of
# producing dozens of missing-header/link errors later.
#
# Set LLVM_DIR to the matching package directory, e.g.:
#   cmake -S . -B build \
#     -DLLVM_DIR=/opt/llvm-20.1.1/lib/cmake/llvm
# -----------------------------------------------------------------------------
find_package(LLVM CONFIG REQUIRED)

if(LLVM_CODEGEN_STRICT_VERSION AND
   NOT LLVM_PACKAGE_VERSION MATCHES "^20\\.1(\\.|$)")
    message(FATAL_ERROR
        "LLVM ${LLVM_PACKAGE_VERSION} is outside the supported 20.1.x "
        "baseline. Use a matching LLVM_DIR, or configure with "
        "-DLLVM_CODEGEN_STRICT_VERSION=OFF for an explicit porting experiment.")
endif()

message(STATUS "LLVM package version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "LLVM package directory: ${LLVM_DIR}")
message(STATUS "LLVM tools directory: ${LLVM_TOOLS_BINARY_DIR}")

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

llvm_map_components_to_libnames(LLVM_LIBS
    core support irreader asmparser bitreader bitwriter
    analysis transformutils scalaropts ipo vectorize
    target targetparser mc mcparser codegen
    passes instrumentation objcarcopts
    native nativecodegen nativeasmparser nativeasmprinter
)
message(STATUS "LLVM libraries: ${LLVM_LIBS}")

# -----------------------------------------------------------------------------
# Helper function: add per-chapter example executables
# Usage: add_chapter(ChapterName file1.cpp file2.cpp ...)
# -----------------------------------------------------------------------------
function(add_chapter CHAPTER_NAME)
    file(GLOB CH_SRC CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/src/${CHAPTER_NAME}/*.cpp")
    foreach(SRC_FILE ${CH_SRC})
        get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
        set(TARGET_NAME "${CHAPTER_NAME}_${EXE_NAME}")
        add_executable(${TARGET_NAME} ${SRC_FILE})
        target_include_directories(${TARGET_NAME} PRIVATE
            ${LLVM_INCLUDE_DIRS}
        )
        target_compile_options(${TARGET_NAME} PRIVATE
            ${LLVM_DEFINITIONS_LIST}
            -Wall
            -Wextra
            -Wpedantic
        )
        if(LLVM_CODEGEN_WARNINGS_AS_ERRORS)
            target_compile_options(${TARGET_NAME} PRIVATE -Werror)
        endif()
        target_link_libraries(${TARGET_NAME} PRIVATE ${LLVM_LIBS})
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/bin/${CHAPTER_NAME}"
        )
    endforeach()
endfunction()

# -----------------------------------------------------------------------------
# Chapter 0: Prerequisite Knowledge
# -----------------------------------------------------------------------------
add_chapter(Chapter0_Prerequisite_Knowledge)

# -----------------------------------------------------------------------------
# Part 1: Getting Started with LLVM (Chapters 1-6)
# -----------------------------------------------------------------------------
add_chapter(Chapter1_Building_LLVM_and_Understanding_the_Directory_Structure)
add_chapter(Chapter2_Contributing_to_LLVM)
add_chapter(Chapter3_Compiler_Basics_and_How_They_Map_to_LLVM_APIs)
add_chapter(Chapter4_Writing_Your_First_Optimization)
add_chapter(Chapter5_Dealing_with_Pass_Managers)
add_chapter(Chapter6_TableGen_LLVM_Swiss_Army_Knife_for_Modeling)

# -----------------------------------------------------------------------------
# Part 2: Middle-End - LLVM IR to LLVM IR (Chapters 7-10)
# -----------------------------------------------------------------------------
add_chapter(Chapter7_Understanding_LLVM_IR)
add_chapter(Chapter8_Survey_of_the_Existing_Passes)
add_chapter(Chapter9_Introducing_Target_Specific_Constructs)
add_chapter(Chapter10_Hands_On_Debugging_LLVM_IR_Passes)

# -----------------------------------------------------------------------------
# Part 3: Introduction to the Backend (Chapters 11-13)
# -----------------------------------------------------------------------------
add_chapter(Chapter11_Getting_Started_with_the_Backend)
add_chapter(Chapter12_Getting_Started_with_the_Machine_Code_Layer)
add_chapter(Chapter13_The_Machine_Pass_Pipeline)

# -----------------------------------------------------------------------------
# Part 4: LLVM IR to Machine IR (Chapters 14-17)
# -----------------------------------------------------------------------------
add_chapter(Chapter14_Getting_Started_with_Instruction_Selection)
add_chapter(Chapter15_Instruction_Selection_The_IR_Building_Phase)
add_chapter(Chapter16_Instruction_Selection_The_Legalization_Phase)
add_chapter(Chapter17_Instruction_Selection_The_Selection_Phase_and_Beyond)

# -----------------------------------------------------------------------------
# Part 5: Final Lowering and Optimizations (Chapters 18-21)
# -----------------------------------------------------------------------------
add_chapter(Chapter18_Instruction_Scheduling)
add_chapter(Chapter19_Register_Allocation)
add_chapter(Chapter20_Lowering_of_the_Stack_Layout)
add_chapter(Chapter21_Getting_Started_with_the_Assembler)

message(STATUS "LLVM Code Generation project configured successfully.")
message(STATUS "Build examples with: cmake --build ${CMAKE_BINARY_DIR}")
