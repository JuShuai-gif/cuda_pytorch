cmake_minimum_required(VERSION 3.22)
project(Chapter07_RNN_LSTM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

# --- Eigen 3.4+ ---
find_package(Eigen3 QUIET)

# ================================================================
# RNN: RNNCell with Eigen
# ================================================================

if(Eigen3_FOUND)
    add_executable(rnn_eigen 00_rnn_eigen.cpp)
    target_include_directories(rnn_eigen PRIVATE ${EIGEN3_INCLUDE_DIR})

    add_executable(bptt_demo 01_bptt_demo.cpp)
    target_include_directories(bptt_demo PRIVATE ${EIGEN3_INCLUDE_DIR})

    add_executable(lstm_eigen 02_lstm_eigen.cpp)
    target_include_directories(lstm_eigen PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(WARNING "Eigen3 not found -- skipping rnn_eigen, bptt_demo, lstm_eigen targets")
endif()

# ================================================================
# LSTM: Production LSTM with LibTorch
# ================================================================

if(Torch_FOUND)
    add_executable(lstm_libtorch 03_lstm_libtorch.cpp)
    target_link_libraries(lstm_libtorch ${TORCH_LIBRARIES})
    target_include_directories(lstm_libtorch PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(word2vec 05_word2vec_libtorch.cpp)
    target_link_libraries(word2vec ${TORCH_LIBRARIES})
    target_include_directories(word2vec PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping lstm_libtorch, word2vec targets")
endif()

# ================================================================
# Text Processing: pure STL, always builds
# ================================================================

add_executable(text_processing 04_text_processing.cpp)
