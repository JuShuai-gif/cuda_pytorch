// RUN: edge-opt %s --edge-shape-inference --allow-unregistered-dialect | FileCheck %s
// 形状推断: 把动态维 ? 细化为静态维, 并沿 use-def 链传播.

// CHECK-LABEL: func.func @chain
func.func @chain(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<16x32xf32>) {
  // CHECK: edge.matmul {{.*}} -> tensor<4x16xf32>
  %0 = edge.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<?x?xf32>
  // CHECK: edge.matmul {{.*}} -> tensor<4x32xf32>
  %1 = edge.matmul %0, %c : (tensor<?x?xf32>, tensor<16x32xf32>) -> tensor<?x?xf32>
  "test.sink"(%1) : (tensor<?x?xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @conv
func.func @conv(%in: tensor<1x3x224x224xf32>, %w: tensor<64x3x7x7xf32>) {
  // (224 + 3 + 3 - 1*(7-1) - 1)/2 + 1 = 112
  // CHECK: edge.conv2d {{.*}} -> tensor<1x64x112x112xf32>
  %0 = edge.conv2d %in, %w {strides = array<i64: 2, 2>, pads = array<i64: 3, 3, 3, 3>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<?x?x?x?xf32>
  "test.sink"(%0) : (tensor<?x?x?x?xf32>) -> ()
  return
}
