cmake_minimum_required(VERSION 3.22)
project(Chapter13_Explainability)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================================================================
# Chapter 13: Explainability and Transparency
# ================================================================

# --- Explainability Concepts (pure STL) ---
add_executable(explainability_concepts 01_explainability_concepts.cpp)

# --- LIME Tabular (needs Eigen) ---
find_package(Eigen3 REQUIRED)
add_executable(lime_tabular 02_lime_tabular.cpp)
target_include_directories(lime_tabular PRIVATE ${EIGEN3_INCLUDE_DIR})

# --- KernelSHAP (needs Eigen) ---
add_executable(shap_kernelshap 03_shap_kernelshap.cpp)
target_include_directories(shap_kernelshap PRIVATE ${EIGEN3_INCLUDE_DIR})

# --- Grad-CAM (needs LibTorch + OpenCV) ---
find_package(OpenCV QUIET)
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

if(OpenCV_FOUND AND Torch_FOUND)
    add_executable(gradcam_cnn 04_gradcam_cnn.cpp)
    target_link_libraries(gradcam_cnn ${TORCH_LIBRARIES} ${OpenCV_LIBS})
    target_include_directories(gradcam_cnn PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING "OpenCV or LibTorch not found -- skipping gradcam_cnn target")
endif()

# --- Uncertainty & Abstain (pure STL) ---
add_executable(uncertainty_and_abstain 05_uncertainty_and_abstain.cpp)

# --- Model Card & Ethics (pure STL) ---
add_executable(model_card_and_ethics 06_model_card_and_ethics.cpp)
