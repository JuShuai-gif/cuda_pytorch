#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

// 工作线程参数结构体
// 用于向线程函数传递输入参数并接收计算结果
typedef struct {
  // 控制此线程负责的簇范围 [start, end)
  int start, end;

  // 所有函数共享的数据
  double *data;               // M×N 数据矩阵
  double *clusterCentroids;   // K×N 聚类中心矩阵
  int *clusterAssignments;    // 长度为 M 的聚类分配数组
  double *currCost;           // 每个簇的当前代价
  int M, N, K;                // 数据点数量、维度、簇数量
} WorkerArgs;


/**
 * 检查算法是否已经收敛。
 * 收敛条件：所有簇的代价变化量都小于 epsilon 阈值。
 *
 * @param prevCost 指向上一次迭代各簇代价的数组（长度 K）
 * @param currCost 指向当前迭代各簇代价的数组（长度 K）
 * @param epsilon 预设的收敛阈值，用于判断算法何时收敛
 * @param K 聚类簇的数量
 *
 * 注意：不要修改此函数！！！
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * 计算两个 N 维数据点之间的欧几里得距离（L2 距离）。
 * 公式：sqrt(sum((x[i] - y[i])^2))
 *
 * @param x 指向第一个数据点数组起始位置的指针
 * @param y 指向第二个数据点数组起始位置的指针
 * @param nDim 每个数据点的维度（x 和 y 必须维度相同）
 * @return 两点之间的欧几里得距离
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * 将每个数据点分配到距离"最近"的聚类中心。
 * 遍历[start, end)范围内的聚类中心，对每个中心计算到所有数据点的距离，
 * 将每个数据点分配到距离最小的簇。
 */
void computeAssignments(WorkerArgs *const args) {
  // minDist[m] 记录数据点 m 当前已知的最小距离
  double *minDist = new double[args->M];
  
  // 初始化：最小距离设为极大值，分配设为 -1（未分配）
  for (int m =0; m < args->M; m++) {
    minDist[m] = 1e30;
    args->clusterAssignments[m] = -1;
  }

  // 对[start, end)范围内的每个聚类中心，计算到所有数据点的距离
  for (int k = args->start; k < args->end; k++) {
    for (int m = 0; m < args->M; m++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist[m]) {
        minDist[m] = d;
        args->clusterAssignments[m] = k;
      }
    }
  }

  delete[] minDist;
}

/**
 * 根据当前的聚类分配，重新计算每个簇的新中心位置。
 * 新中心 = 该簇所有数据点的平均值。
 */
void computeCentroids(WorkerArgs *const args) {
  // counts[k] 记录分配给簇 k 的数据点数量
  int *counts = new int[args->K];

  // 清零：计数器和各维度的累加和
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }

  // 累加每个数据点到其所属簇的各维度分量
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // 计算平均值（除以每个簇的数据点数量）
  // 防止出现空簇导致除零错误
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // 防止除以 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }

  delete[] counts;
}

/**
 * 计算每个簇的代价（该簇所有数据点到中心的距离之和）。
 * 用于判断算法是否收敛。
 */
void computeCost(WorkerArgs *const args) {
  // accum[k] 临时累加簇 k 的代价
  double *accum = new double[args->K];

  // 清零累加器
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // 对所有数据点，累加到其所属簇的代价中
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // 将[start, end)范围内簇的代价写入结果数组
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }

  delete[] accum;
}

/**
 * 使用 std::thread 并行执行 K-Means 聚类算法。
 *
 * 算法流程：
 * 1. 将每个数据点分配到距离最近的聚类中心
 * 2. 根据分配结果重新计算每个簇的中心（取平均值）
 * 3. 计算每个簇的代价（数据点到中心的距离之和）
 * 4. 检查是否收敛：若所有簇的代价变化量都小于 epsilon，则停止
 *
 * @param data 指向 M×N 数组的指针，表示 M 个 N 维数据点。
 *             数据按"数据点优先"排列：data[i*N] 是第 i 个数据点的起始位置。
 *             第 i 个数据点的 N 个值位于 data[i*N] ~ data[(i+1)*N] 范围内。
 * @param clusterCentroids 指向 K×N 数组的指针，表示 K 个 N 维聚类中心。
 *             数据排列方式与 data 相同。
 * @param clusterAssignments 指向长度为 M 的数组的指针。
 *             clusterAssignments[i] = j 表示数据点 i 被分配到聚类中心 j。
 * @param M 待聚类的数据点数量
 * @param N 数据点的维度
 * @param K 聚类中心的数量
 * @param epsilon 收敛判定阈值：当 |currCost[i] - prevCost[i]| < epsilon
 *             对所有 i=0,1,...,K-1 成立时，算法收敛
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // 用于跟踪收敛状态的代价数组
  double *prevCost = new double[K];  // 上一次迭代的各簇代价
  double *currCost = new double[K];  // 当前迭代的各簇代价

  // WorkerArgs 结构体用于向各函数传递输入参数并接收输出
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;


  CycleTimer timer;
  // 初始化代价数组：prevCost 设为极大值确保第一次迭代必然执行
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* K-Means 算法主循环 */
  int iter = 0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // 保存当前代价到 prevCost，用于下一轮收敛判断
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // 设置 args 结构体：当前由单线程处理所有 K 个簇
    args.start = 0;
    args.end = K;

    // 三步迭代：
    // 1. 分配：将每个数据点分配给最近的聚类中心
    // 2. 更新：根据新分配重新计算聚类中心
    // 3. 评估：计算每个簇的代价（用于收敛判断）

    computeAssignments(&args);
    
    computeCentroids(&args);
    
    computeCost(&args);

    iter++;
  }

  delete[] currCost;
  delete[] prevCost;
}
