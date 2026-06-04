cmake_minimum_required(VERSION 3.22)
project(Chapter12_Monitoring)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================================================================
# Chapter 12: Monitoring Deployed Models
# All examples are pure STL (C++17), no external library dependencies
# (NVML is optional for GPU metrics, defaults to mock)
# ================================================================

# Basic monitoring: Percentile calculation
add_executable(percentile_calculation 01_percentile_calculation.cpp)

# Basic monitoring: Latency histogram
add_executable(latency_histogram 02_latency_histogram.cpp)

# Online quality: ECE and Brier
add_executable(online_calibration_ece 03_online_calibration_ece.cpp)

# Resource utilization: GPU, CPU, RSS, H2D/D2H
add_executable(gpu_utilization 04_gpu_utilization.cpp)

# C++ instrumentation: Prometheus-style metrics
add_executable(prometheus_metrics 05_prometheus_metrics.cpp)

# C++ instrumentation: Structured JSON logs
add_executable(structured_logs 06_structured_logs.cpp)

# C++ instrumentation: Traces and spans
add_executable(traces_spans 07_traces_spans.cpp)

# Online quality: Delayed label joiner
add_executable(delayed_labels 08_delayed_labels.cpp)

# Online quality: Leading indicators (entropy, margin, disagreement)
add_executable(leading_indicators 09_leading_indicators.cpp)

# Alerts and action: Alert policies, drift budgets, action routing
add_executable(alerts_and_action 10_alerts_and_action.cpp)
