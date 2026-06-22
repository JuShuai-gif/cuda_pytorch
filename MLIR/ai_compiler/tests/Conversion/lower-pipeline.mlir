// RUN: edge-opt %s --edge-lower-to-loops | FileCheck %s --check-prefix=LOOPS
// RUN: edge-opt %s --edge-lower-to-llvm | FileCheck %s --check-prefix=LLVM
//
// 端到端 lowering 流水线: Edge -> Linalg -> bufferize -> 循环/LLVM.

// LOOPS-LABEL: func.func @relu
// LOOPS: memref
// LOOPS: scf.for

// LLVM: llvm.func
func.func @relu(%a: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %1 = edge.relu %a : tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}
