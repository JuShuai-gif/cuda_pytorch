// RUN: edge-opt %s --edge-fuse-conv-bn-relu | FileCheck %s
// Conv+BN+ReLU 融合: 当 weight 与 BN 参数为常量时, 把 BN 折叠进卷积权重/偏置.
//
// 输入: weight=1.0, scale=[2,3], bn_bias=[0.5,1.0], mean=[0,0], var=[0,0], eps=1.0
//   factor[c]   = scale[c]/sqrt(var[c]+eps) = [2, 3]
//   new_w[c]    = 1.0 * factor[c]          => ch0=2.0, ch1=3.0
//   new_bias[c] = (0-0)*factor + bn_bias    = [0.5, 1.0]

// CHECK-LABEL: func.func @fuse
// CHECK-NOT: edge.conv2d
// CHECK-NOT: edge.batch_norm
// CHECK-NOT: edge.relu
// CHECK-DAG: edge.constant dense<{{.*}}2.000000e+00{{.*}}3.000000e+00{{.*}}> : tensor<2x3x1x1xf32>
// CHECK-DAG: edge.constant dense<[5.000000e-01, 1.000000e+00]> : tensor<2xf32>
// CHECK: edge.conv_bn_relu
func.func @fuse(%x: tensor<1x3x4x4xf32>) -> tensor<1x2x4x4xf32> {
  %w = edge.constant dense<1.0> : tensor<2x3x1x1xf32>
  %scale = edge.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %bnbias = edge.constant dense<[0.5, 1.0]> : tensor<2xf32>
  %mean = edge.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %var = edge.constant dense<[0.0, 0.0]> : tensor<2xf32>
  %c = edge.conv2d %x, %w {strides = array<i64: 1, 1>, pads = array<i64: 0, 0, 0, 0>, dilations = array<i64: 1, 1>}
       : (tensor<1x3x4x4xf32>, tensor<2x3x1x1xf32>) -> tensor<1x2x4x4xf32>
  %b = edge.batch_norm %c, %scale, %bnbias, %mean, %var {epsilon = 1.000000e+00 : f64}
       : (tensor<1x2x4x4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x4x4xf32>
  %r = edge.relu %b : tensor<1x2x4x4xf32>
  return %r : tensor<1x2x4x4xf32>
}
