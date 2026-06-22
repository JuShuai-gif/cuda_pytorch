// RUN: edge-opt %s --allow-unregistered-dialect | edge-opt --allow-unregistered-dialect | FileCheck %s
// Edge 方言算子与自定义类型/属性的 roundtrip 测试.

// CHECK-LABEL: func.func @conv_bn_relu_chain
func.func @conv_bn_relu_chain(%input: tensor<1x3x224x224xf32>,
                              %weight: tensor<64x3x7x7xf32>,
                              %bn_scale: tensor<64xf32>,
                              %bn_bias: tensor<64xf32>,
                              %bn_mean: tensor<64xf32>,
                              %bn_var: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK: edge.conv2d
  %0 = edge.conv2d %input, %weight {strides = array<i64: 2, 2>, pads = array<i64: 3, 3, 3, 3>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
  // CHECK: edge.batch_norm
  %1 = edge.batch_norm %0, %bn_scale, %bn_bias, %bn_mean, %bn_var {epsilon = 1.000000e-05 : f64}
       : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
  // CHECK: edge.relu
  %2 = edge.relu %1 : tensor<1x64x112x112xf32>
  return %2 : tensor<1x64x112x112xf32>
}

// CHECK-LABEL: func.func @matmul_attention
func.func @matmul_attention(%q: tensor<1x8x128x64xf32>,
                            %k: tensor<1x8x128x64xf32>,
                            %v: tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32> {
  // CHECK: edge.attention
  %0 = edge.attention %q, %k, %v {scale = 0.125 : f64}
       : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32>
  return %0 : tensor<1x8x128x64xf32>
}

// CHECK-LABEL: func.func @fused
func.func @fused(%input: tensor<1x3x224x224xf32>,
                 %weight: tensor<64x3x7x7xf32>,
                 %bias: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK: edge.conv_bn_relu
  %0 = edge.conv_bn_relu %input, %weight, %bias {strides = array<i64: 2, 2>, pads = array<i64: 3, 3, 3, 3>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
  return %0 : tensor<1x64x112x112xf32>
}

// 自定义类型 (!edge.qtensor) 与自定义属性 (#edge.quant_params) 的 roundtrip.
// CHECK-LABEL: func.func @quant_type
func.func @quant_type(%arg0: !edge.qtensor<tensor<1x3x4x4xi8>, 7.812500e-03 : f64, 0>)
    -> !edge.qtensor<tensor<1x3x4x4xi8>, 7.812500e-03 : f64, 0> {
  // CHECK: edge.quant_params
  "test.use_attr"() {qp = #edge.quant_params<2.500000e-02 : f64, 128>} : () -> ()
  return %arg0 : !edge.qtensor<tensor<1x3x4x4xi8>, 7.812500e-03 : f64, 0>
}
