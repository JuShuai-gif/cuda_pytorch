// RUN: edge-opt %s --edge-lower-to-linalg | FileCheck %s
// Edge -> Linalg 降级 (dialect conversion, 目的地传递风格 DPS).

// CHECK-LABEL: func.func @mm
func.func @mm(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK-DAG: %[[E0:.*]] = tensor.empty() : tensor<4x16xf32>
  // CHECK: linalg.fill
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
  %0 = edge.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: linalg.yield
  %1 = edge.relu %0 : tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @cst
func.func @cst() -> tensor<2xf32> {
  // CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  // CHECK-NOT: edge.constant
  %0 = edge.constant dense<[1.0, 2.0]> : tensor<2xf32>
  return %0 : tensor<2xf32>
}
