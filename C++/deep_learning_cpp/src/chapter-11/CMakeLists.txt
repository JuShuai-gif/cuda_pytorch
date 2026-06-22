cmake_minimum_required(VERSION 3.22)
project(Chapter11_Debugging_Retraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================================================================
# Chapter 11: Debugging and Retraining Deployed Models
# All examples are pure STL (C++17), no external library dependencies
# ================================================================

# Drift Detection: Histograms, PSI, KS, chi-squared, KL/JS, Welford
add_executable(ch11_drift_detection 01_drift_detection.cpp)

# Observability: Structured Logging
add_executable(ch11_structured_logging 02_structured_logging.cpp)

# Observability: In-Process Metrics
add_executable(ch11_inprocess_metrics 03_inprocess_metrics.cpp)

# Observability: Distributed Tracing
add_executable(ch11_distributed_tracing 04_distributed_tracing.cpp)

# Debugging: Reproducible Failures
add_executable(ch11_reproducible_debugging 05_reproducible_debugging.cpp)

# Debugging: Numerical Stability
add_executable(ch11_numerical_stability 06_numerical_stability.cpp)

# Serving: Micro-Batcher (async batching)
add_executable(ch11_micro_batcher 07_micro_batcher.cpp)

# Retraining: Decision Flow, Acceptance Gates, Rollout Ladder
add_executable(ch11_retraining_workflow 08_retraining_workflow.cpp)
