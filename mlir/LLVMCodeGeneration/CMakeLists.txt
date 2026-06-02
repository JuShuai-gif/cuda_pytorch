cmake_minimum_required(VERSION 3.14)
project(LLVMCodeGeneration VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# Find LLVM
# Set LLVM_DIR to your LLVM build or install prefix, e.g.:
#   cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm .
# -----------------------------------------------------------------------------
find_package(LLVM CONFIG QUIET)
if(NOT LLVM_FOUND)
    message(STATUS "LLVM not found via find_package. Trying to locate manually...")
    # Try common locations
    set(LLVM_HINTS
        /usr/lib/llvm-19
        /usr/lib/llvm-18
        /usr/lib/llvm-17
        /usr/lib/llvm-16
        /usr/lib/llvm
        /usr/local/llvm
    )
    find_path(LLVM_INCLUDE_DIRS llvm/IR/LLVMContext.h
        HINTS ${LLVM_HINTS}
        PATH_SUFFIXES include
    )
    if(LLVM_INCLUDE_DIRS)
        message(STATUS "  Found LLVM include: ${LLVM_INCLUDE_DIRS}")
    else()
        message(WARNING "LLVM headers not found. Set LLVM_DIR or LLVM_INCLUDE_DIRS manually.")
    endif()
else()
    message(STATUS "LLVM found: ${LLVM_PACKAGE_VERSION}")
    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})
endif()

# If LLVM was found via find_package, map library names
if(LLVM_FOUND)
    llvm_map_components_to_libnames(LLVM_LIBS
        core support irreader asmparser bitreader bitwriter
        analysis transformsutils scalaropts ipo vectorize
        target targetparser mc mcparser codegen
        passes instrumentation objcarcopts
    )
    message(STATUS "LLVM libraries: ${LLVM_LIBS}")
endif()

# -----------------------------------------------------------------------------
# Helper function: add per-chapter example executables
# Usage: add_chapter(ChapterName file1.cpp file2.cpp ...)
# -----------------------------------------------------------------------------
function(add_chapter CHAPTER_NAME)
    file(GLOB CH_SRC "${CMAKE_SOURCE_DIR}/src/${CHAPTER_NAME}/*.cpp")
    foreach(SRC_FILE ${CH_SRC})
        get_filename_component(EXE_NAME ${SRC_FILE} NAME_WE)
        set(TARGET_NAME "${CHAPTER_NAME}_${EXE_NAME}")
        add_executable(${TARGET_NAME} ${SRC_FILE})
        target_include_directories(${TARGET_NAME} PRIVATE
            ${LLVM_INCLUDE_DIRS}
        )
        if(LLVM_FOUND)
            target_link_libraries(${TARGET_NAME} ${LLVM_LIBS})
        endif()
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
message(STATUS "Build with: cmake --build .")
