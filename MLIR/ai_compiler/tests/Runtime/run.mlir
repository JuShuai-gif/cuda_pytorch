// RUN: edge-run %s --edge-fill=1.0 | FileCheck %s
// 运行时执行 + Profiling: matmul(全1)x(全1) 每元素=2, relu 不变, 4 元素校验和=8.

// CHECK: Edge Runtime Profiling Report
// CHECK-DAG: edge.matmul
// CHECK-DAG: edge.relu
// CHECK: Outputs
// CHECK: checksum=8.0000
func.func @g(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = edge.matmul %a, %b : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %1 = edge.relu %0 : tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}
