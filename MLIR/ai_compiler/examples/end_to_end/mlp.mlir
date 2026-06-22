// 端到端示例: 一个两层 MLP (matmul + relu), 可全程跑通
//   edge-shape-inference -> (fusion) -> edge-memplan -> edge-lower-to-llvm -> edge-run
func.func @mlp(%x: tensor<8x16xf32>, %w1: tensor<16x32xf32>, %w2: tensor<32x8xf32>) -> tensor<8x8xf32> {
  %0 = edge.matmul %x, %w1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %1 = edge.relu %0 : tensor<8x32xf32>
  %2 = edge.matmul %1, %w2 : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  %3 = edge.relu %2 : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}
