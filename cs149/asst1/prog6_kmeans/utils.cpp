#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

/**
 * 将当前 K-Means 算法的状态写入日志文件，用于后续可视化。
 * 按 sampleRate 的概率采样数据点写入，避免文件过大。
 *
 * @param filename 输出日志文件路径（例如 "./start.log"）
 * @param sampleRate 采样率，取值范围 [0, 1]，决定写入数据点的比例
 * @param data M×N 数据矩阵
 * @param clusterAssignments 长度为 M 的聚类分配数组
 * @param clusterCentroids K×N 聚类中心矩阵
 * @param M 数据点数量
 * @param N 数据维度
 * @param K 簇数量
 */
void logToFile(string filename, double sampleRate, double *data,
               int *clusterAssignments, double *clusterCentroids, int M, int N,
               int K) {
  ofstream logFile;
  logFile.open(filename);

  // 写入文件头：M, N, K（逗号分隔）
  logFile << M << "," << N << "," << K << endl;

  // 按采样率随机写入部分数据点的信息
  for (int m = 0; m < M; m++) {
    if (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) <
        sampleRate) {
      logFile << "Example " << m << ", cluster " << clusterAssignments[m]
              << ": ";
      for (int n = 0; n < N; n++) {
        logFile << data[m * N + n] << " ";
      }
      logFile << "\n";
    }
  }

  // 写入所有聚类中心的信息
  for (int k = 0; k < K; k++) {
    logFile << "Centroid " << k << ": ";
    for (int n = 0; n < N; n++) {
      logFile << clusterCentroids[k * N + n] << " ";
    }
    logFile << "\n";
  }

  logFile.close();
}

/**
 * 将数据以二进制格式写入文件（高效存储）。
 * 写入顺序：M, N, K, epsilon, data 数组, clusterCentroids 数组, clusterAssignments 数组。
 *
 * @param filename 输出文件路径
 * @param data 数据矩阵
 * @param clusterCentroids 聚类中心矩阵
 * @param clusterAssignments 聚类分配数组
 * @param M_p 指向 M（数据点数量）的指针
 * @param N_p 指向 N（维度）的指针
 * @param K_p 指向 K（簇数量）的指针
 * @param epsilon_p 指向 epsilon（收敛阈值）的指针
 */
void writeData(string filename, double *data, double *clusterCentroids,
               int *clusterAssignments, int *M_p, int *N_p, int *K_p,
               double *epsilon_p) {
  int M = *M_p;
  int N = *N_p;
  int K = *K_p;

  // 以二进制模式打开文件
  ofstream dataFile(filename, ios::out | ios::binary);
  // 写入元数据：M, N, K, epsilon
  dataFile.write((char *)M_p, sizeof(int));
  dataFile.write((char *)N_p, sizeof(int));
  dataFile.write((char *)K_p, sizeof(int));
  dataFile.write((char *)epsilon_p, sizeof(double));
  // 写入数据数组（整个内存块直接写入）
  dataFile.write((char *)data, sizeof(double) * M * N);
  dataFile.write((char *)clusterCentroids, sizeof(double) * K * N);
  dataFile.write((char *)clusterAssignments, sizeof(int) * M);
  dataFile.close();
}

/**
 * 从二进制文件中读取训练数据。
 * 读取顺序必须与 writeData 一致：M, N, K, epsilon, data, clusterCentroids, clusterAssignments。
 *
 * @param filename 输入文件路径
 * @param data 输出参数，指向数据矩阵的指针（函数内分配内存）
 * @param clusterCentroids 输出参数，指向聚类中心矩阵的指针（函数内分配内存）
 * @param clusterAssignments 输出参数，指向聚类分配数组的指针（函数内分配内存）
 * @param M_p 输出参数，接收 M 的值
 * @param N_p 输出参数，接收 N 的值
 * @param K_p 输出参数，接收 K 的值
 * @param epsilon_p 输出参数，接收 epsilon 的值
 */
void readData(string filename, double **data, double **clusterCentroids,
              int **clusterAssignments, int *M_p, int *N_p, int *K_p,
              double *epsilon_p) {
  cout << "Reading data.dat..." << endl;

  // 以二进制模式打开文件，若失败则退出
  ifstream dataFile(filename, ios::in | ios::binary);
  if (dataFile.fail()) {
      cout << "Couldn't open the file! Please make sure data.dat exists... Exiting." << endl;
      exit(EXIT_FAILURE);
  }

  // 读取元数据：M, N, K, epsilon
  dataFile.read((char *)M_p, sizeof(int));
  dataFile.read((char *)N_p, sizeof(int));
  dataFile.read((char *)K_p, sizeof(int));
  dataFile.read((char *)epsilon_p, sizeof(double));

  int M = *M_p;
  int N = *N_p;
  int K = *K_p;

  // 根据读取的维度分配内存
  *data = new double[M * N];
  *clusterCentroids = new double[K * N];
  *clusterAssignments = new int[M];

  // 读取数据数组（直接读入已分配的内存块）
  dataFile.read((char *)*data, sizeof(double) * M * N);
  dataFile.read((char *)*clusterCentroids, sizeof(double) * K * N);
  dataFile.read((char *)*clusterAssignments, sizeof(int) * M);
  dataFile.close();
}
