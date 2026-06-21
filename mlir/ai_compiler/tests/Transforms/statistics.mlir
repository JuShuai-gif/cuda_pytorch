// RUN: edge-opt %s --edge-statistics --allow-unregistered-dialect | FileCheck %s
// 统计 pass: 打印 Markdown 报告 (算子计数 + MAC 估算).

// CHECK: Edge Graph Statistics Report
// CHECK: Total operations:
// CHECK: Estimated MACs:
// CHECK: | edge.matmul | 2 |
// CHECK: | edge.conv2d | 1 |
func.func @model(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<16x32xf32>,
                 %in: tensor<1x3x224x224xf32>, %w: tensor<64x3x7x7xf32>) {
  %0 = edge.matmul %a, %b : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = edge.matmul %0, %c : (tensor<4x16xf32>, tensor<16x32xf32>) -> tensor<4x32xf32>
  %2 = edge.conv2d %in, %w {strides = array<i64: 2, 2>, pads = array<i64: 3, 3, 3, 3>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
  "test.sink"(%1) : (tensor<4x32xf32>) -> ()
  "test.sink"(%2) : (tensor<1x64x112x112xf32>) -> ()
  return
}
