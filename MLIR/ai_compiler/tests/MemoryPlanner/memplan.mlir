// RUN: edge-memplan %s --edge-align=1 | FileCheck %s
// 内存规划: 生命周期不重叠的张量共享地址槽, 降低峰值内存.
//
// 4 个 100x100xf32 张量 (各 40000 B); 链式 relu, 同一时刻最多 2 个活跃,
// 故规划峰值 = 2 槽 = 80000 B, 相对朴素 160000 B 节省 50%.

// CHECK: Edge Memory Planning Report
// CHECK: Naive peak (no reuse): 160000 bytes
// CHECK: Planned peak (reuse) : 80000 bytes
// CHECK: Saving: 50%
func.func @m(%a: tensor<100x100xf32>) -> tensor<100x100xf32> {
  %0 = edge.relu %a : tensor<100x100xf32>
  %1 = edge.relu %0 : tensor<100x100xf32>
  %2 = edge.relu %1 : tensor<100x100xf32>
  return %2 : tensor<100x100xf32>
}
