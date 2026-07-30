// 问题 2.8：观察执行顺序。
// 连续运行两三次，对比 block 出现的先后顺序，回答 handout 里的问题。
//
// ===================== block 执行顺序的观察 =====================
//
// 现象：连续运行多次，block 报到顺序每次都不一样（非确定性）。
//
// 原因：GPU 的 block 由硬件 warp scheduler 动态分配到各 SM。
//   - 不保证 block 0 先跑、block 1 再跑，谁先抢到 SM 谁先执行。
//   - 不同 SM 推进速度可能不同（取决于 SM 上的负载、warp 切换节奏等）。
//   - printf 输出的顺序取决于哪个 block 的 printf buffer 先被 flush 回 host，
//     而不是 block 的 index 大小。
//
// 结论：CUDA 编程模型中，block 之间没有固定的执行顺序。
//       需要 block 间同步时，唯一安全的方式是让新的 kernel launch 提供栅栏。
// ================================================================
#include "common.h"

__global__ void whoami() {
    // 让每个 block 的 0 号线程报到。
    if (threadIdx.x == 0) {
        printf("block %d 报到\n", blockIdx.x);
    }
}

int main() {
    whoami<<<16, 32>>>();
    CUDA_CHECK_KERNEL();
    return 0;
}
