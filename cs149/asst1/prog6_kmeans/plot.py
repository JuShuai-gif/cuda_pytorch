import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 日志文件和输出图片的路径配置
START_LOGFILE = "./start.log"  # 算法开始前的日志文件
END_LOGFILE = "./end.log"  # 算法结束后的日志文件
START_PLOTFILE = "./start.png"  # 算法开始前的可视化图片
END_PLOTFILE = "./end.png"  # 算法结束后的可视化图片

# 分别对算法开始前和结束后的状态生成可视化图片
for title, logfile, plotfile in [
    ("Start", START_LOGFILE, START_PLOTFILE),
    ("End", END_LOGFILE, END_PLOTFILE),
]:
    assert os.path.exists(logfile), "Log file doesn't exist, try running the program."

    with open(logfile) as f:
        # 读取文件头：M（数据点数量）, N（维度）, K（簇数量）
        M, N, K = f.readline().split(",")
        M, N, K = int(M), int(N), int(K)

        # 读取日志中的数据
        data = []
        cluster_assignments = []
        cluster_centroids = []

        for line in f.readlines():
            prefix = line.split(":")[0].split(" ")[0]
            if prefix == "Example":
                # 解析数据点：提取聚类分配信息和坐标值
                cluster_assignments.append(line.split(":")[0].split(" ")[-1])
                datapoint = line.split(":")[1].strip().split(" ")
                data.append(np.asarray(datapoint, dtype=float))
                line = f.readline()
            elif prefix == "Centroid":
                # 解析聚类中心坐标
                centroid = line.split(":")[1].strip().split(" ")
                cluster_centroids.append(np.asarray(centroid, dtype=float))

    # 转换为 numpy 数组
    data = np.stack(data)
    cluster_assignments = np.asarray(cluster_assignments, dtype=int)
    cluster_centroids = np.stack(cluster_centroids)

    # 使用 PCA 降维到 2D，便于在平面图上可视化
    pca = PCA(n_components=2)
    pca.fit(data)

    data_2d = pca.transform(data)
    cluster_centroids_2d = pca.transform(cluster_centroids)

    # 绘制散点图：不同颜色代表不同簇
    plt.subplots(figsize=(10, 8))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_assignments)
    # 聚类中心用红色五角星标记，黑色边框突出显示
    plt.scatter(
        cluster_centroids_2d[:, 0],
        cluster_centroids_2d[:, 1],
        c="r",
        marker="*",
        edgecolors="black",
        s=1000,
    )
    plt.title(f"K-Means: {title}")

    plt.savefig(plotfile)
