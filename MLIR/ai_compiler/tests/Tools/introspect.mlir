// RUN: edge-introspect %s | FileCheck %s
// edge-introspect: 遍历并打印 IR 结构树与算子统计.

// CHECK: IR Structure Tree
// CHECK: builtin.module
// CHECK: func.func
// CHECK: edge.conv2d
// CHECK: edge.relu
// CHECK: Op Statistics
// CHECK-DAG: edge.conv2d : 1
// CHECK-DAG: edge.relu : 1
func.func @m(%x: tensor<1x3x4x4xf32>, %w: tensor<2x3x1x1xf32>) -> tensor<1x2x4x4xf32> {
  %c = edge.conv2d %x, %w {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x4x4xf32>, tensor<2x3x1x1xf32>) -> tensor<1x2x4x4xf32>
  %r = edge.relu %c : tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}
