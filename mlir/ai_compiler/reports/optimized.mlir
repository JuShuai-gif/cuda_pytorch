module {
  func.func @mlp(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>, %arg2: tensor<32x8xf32>) -> tensor<8x8xf32> {
    %0 = edge.matmul %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
    %1 = edge.relu %0 : tensor<8x32xf32>
    %2 = edge.matmul %1, %arg2 : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
    %3 = edge.relu %2 : tensor<8x8xf32>
    return %3 : tensor<8x8xf32>
  }
}

