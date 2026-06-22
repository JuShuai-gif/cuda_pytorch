#include <algorithm>
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "CycleTimer.h"

// 随机数种子
#define SEED 7
// 日志采样率：只有 1% 的数据点会被写入日志文件，用于可视化
#define SAMPLE_RATE 1e-2

using namespace std;

// 核心计算函数
// kMeansThread: 执行 K-Means 聚类的主函数（多线程版本）
// data: M×N 的数据矩阵，每行是一个 N 维数据点
// clusterCentroids: K×N 的聚类中心矩阵，每行是一个 N 维中心点
// clusterAssignments: 长度为 M 的数组，记录每个数据点被分配到哪个簇
// M: 数据点数量，N: 数据维度，K: 簇的数量
// epsilon: 收敛阈值
extern void kMeansThread(double *data, double *clusterCentroids,
                      int *clusterAssignments, int M, int N, int K,
                      double epsilon);
// dist: 计算两个 N 维点之间的欧几里得距离
extern double dist(double *x, double *y, int nDim);

// 工具函数
// logToFile: 将当前算法状态（采样后的数据点、聚类分配、聚类中心）写入日志文件
extern void logToFile(string filename, double sampleRate, double *data,
                      int *clusterAssignments, double *clusterCentroids, int M,
                      int N, int K);
// writeData: 将数据以二进制格式写入文件（用于持久化生成的测试数据）
extern void writeData(string filename, double *data, double *clusterCentroids,
                      int *clusterAssignments, int *M_p, int *N_p, int *K_p,
                      double *epsilon_p);
// readData: 从二进制文件中读取数据
extern void readData(string filename, double **data, double **clusterCentroids,
                     int **clusterAssignments, int *M_p, int *N_p, int *K_p,
                     double *epsilon_p);

// 生成 [0, 1) 范围内的随机浮点数
double randDouble() {
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

// 生成聚类数据：围绕 K 个随机中心，用高斯噪声生成 M 个数据点
void initData(double *data, int M, int N) {
  int K = 10;
  double *centers = new double[K * N];

  // 高斯噪声参数
  double mean = 0.0;
  double stddev = 0.5;
  std::default_random_engine generator;
  std::normal_distribution<double> normal_dist(mean, stddev);

  // 随机生成 K 个聚类中心
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      centers[k * N + n] = randDouble();
    }
  }

  // 围绕聚类中心生成数据点，添加高斯噪声
  for (int m = 0; m < M; m++) {
    int startingPoint = rand() % K; // 随机选择一个中心
    for (int n = 0; n < N; n++) {
      double noise = normal_dist(generator);
      data[m * N + n] = centers[startingPoint * N + n] + noise;
    }
  }

  delete[] centers;
}

// 初始化聚类中心：所有中心互相靠近（使迭代过程更有趣）
void initCentroids(double *clusterCentroids, int K, int N) {
  // 第一个中心为随机值
  for (int n = 0; n < N; n++) {
    clusterCentroids[n] = randDouble();
  }
  // 其余中心在第一个中心附近随机偏移
  for (int k = 1; k < K; k++) {
    for (int n = 0; n < N; n++) {
      clusterCentroids[k * N + n] =
          clusterCentroids[n] + (randDouble() - 0.5) * 0.1;
    }
  }
}

int main() {
  srand(SEED);

  int M, N, K;
  double epsilon;

  double *data;
  double *clusterCentroids;
  int *clusterAssignments;

  // 注意：评分时将使用 data.dat 中的数据，
  // 通过 readData 函数读取
  readData("./data.dat", &data, &clusterCentroids, &clusterAssignments, &M, &N,
           &K, &epsilon);

  // 如果希望自己生成数据进行测试，取消下面代码的注释
  
  // M = 1e6;
  // N = 100;
  // K = 3;
  // epsilon = 0.1;

  // data = new double[M * N];
  // clusterCentroids = new double[K * N];
  // clusterAssignments = new int[M];

  // // 初始化数据和聚类中心
  // initData(data, M, N);
  // initCentroids(clusterCentroids, K, N);

  // // 初始化聚类分配：将每个数据点分配到距离最近的初始聚类中心
  // for (int m = 0; m < M; m++) {
  //   double minDist = 1e30;
  //   int bestAssignment = -1;
  //   for (int k = 0; k < K; k++) {
  //     double d = dist(&data[m * N], &clusterCentroids[k * N], N);
  //     if (d < minDist) {
  //       minDist = d;
  //       bestAssignment = k;
  //     }
  //   }
  //   clusterAssignments[m] = bestAssignment;
  // }

  // // 取消注释以生成数据文件
  // writeData("./data.dat", data, clusterCentroids, clusterAssignments, &M, &N,
  //           &K, &epsilon);
  

  printf("Running K-means with: M=%d, N=%d, K=%d, epsilon=%f\n", M, N,
         K, epsilon);

  // 记录算法开始前的状态（用于可视化对比）
  logToFile("./start.log", SAMPLE_RATE, data, clusterAssignments,
            clusterCentroids, M, N, K);

  // 计时并运行 K-Means 算法
  double startTime = CycleTimer::currentSeconds();
  kMeansThread(data, clusterCentroids, clusterAssignments, M, N, K, epsilon);
  double endTime = CycleTimer::currentSeconds();
  printf("[Total Time]: %.3f ms\n", (endTime - startTime) * 1000);

  // 记录算法结束后的状态（用于可视化对比）
  logToFile("./end.log", SAMPLE_RATE, data, clusterAssignments,
            clusterCentroids, M, N, K);

  delete[] data;
  delete[] clusterCentroids;
  delete[] clusterAssignments;
  return 0;
}
