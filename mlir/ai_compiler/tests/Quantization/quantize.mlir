// RUN: edge-quantize %s --edge-out=/tmp | FileCheck %s
// PTQ 量化: 三种校准对比 (full vs body SQNR) + 对常量权重的量化 + 混合精度.

// CHECK: Quantization Report
// CHECK: full SQNR(dB)
// CHECK: body SQNR(dB)
// CHECK-DAG: MinMax
// CHECK-DAG: Percentile
// CHECK-DAG: KL-divergence
// CHECK: Per-constant weight quantization
// CHECK: Latency Report
func.func @w() -> tensor<2x2xf32> {
  %0 = edge.constant dense<[[1.0, -2.0], [3.0, -4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
