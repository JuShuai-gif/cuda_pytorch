# Dijkstra Shortest Path - Dijkstra 最短路径

## 算法原理

Dijkstra 算法求解带权有向图中单源最短路径问题。核心思想是贪心策略：每次从未访问节点中选择距离源点最近的节点，松弛其邻接边。

**标准 Dijkstra (CPU 串行)**: 时间复杂度 O(V^2)，使用邻接矩阵。每轮必须找到全局最小距离节点，该步骤天然串行。

**并行 Bellman-Ford 松弛 (GPU)**: 每轮并行松弛所有节点的邻接边，通过 `atomicMin` 更新距离值。时间复杂度 O(V * E) 但每轮可并行。

## CUDA 并行化策略

- **并行边松弛**: 每个线程负责一个节点的所有出边松弛
- **原子更新**: 使用 `atomicMin` 在线程间安全更新距离值
- **收敛检测**: 当不再有距离更新时提前终止循环
- **邻接矩阵存储**: 适合稠密图，利用 GPU 内存带宽

## 适用场景

| 适用 | 不适用 |
|------|--------|
| 稀疏大图 (V 大, E 小) | 稠密小图 (GPU 并行化反而更慢) |
| 所有边权非负 | 负权边 (需用 Bellman-Ford) |
| 批量多源最短路径 | 单次单源 + 小图 (< 500 节点) |
| 栅格地图路径规划 | 动态图 (边权频繁变化) |
| 与物理/游戏引擎结合 | 需要实时交互 (< 1ms) |

## 性能说明

GPU 版本是 *并行 Bellman-Ford 松弛* 而非标准 Dijkstra (标准 Dijkstra 的贪心步骤无法并行化)。因此：

- **稠密图 (density > 50%)**: CPU Dijkstra 更快 (O(V^2) vs GPU 的 O(VE) 迭代开销)
- **稀疏大图 (density < 20%, V > 1000)**: GPU 版本显著更快 (可达 100x+)

## 实际落地案例

| 场景 | 具体案例 |
|------|----------|
| 高精地图路网规划 | 百度/高德地图引擎底层就是 Dijkstra + A* 变体 |
| OSPF 路由协议 | 你电脑到服务器的每一跳路径由 Dijkstra 算出 |
| 芯片 EDA 布线 | Cadence/Synopsys 自动布线工具的全局布线引擎 |
| 机器人全覆盖清扫 | 扫地机器人区域间转移路径 = Dijkstra |
| 网络流量工程 | MPLS-TE 的 CSPF (Constrained SPF) 基于 Dijkstra 扩展 |

## 改进方向

| 方向 | 来源 | 说明 |
|------|------|------|
| Delta-Stepping | 学术: Meyer & Sanders (2003) | GPU 原生并行 SSSP，比 Bellman-Ford 松弛更高效 |
| A* 启发式 | 工业: 地图引擎标准 | 加入目标方向启发函数，大幅裁剪搜索空间 |
| Contraction Hierarchies | 工业: OSRM/GraphHopper | 预计算节点层级，查询时 O(log V)，地图导航引擎标配 |
| 双向搜索 | 学术+工业 | 同时从起点和终点出发，相交时停止，空间减半 |
| Frontier BFS | 学术: GPU BFS 优化 | 只对活跃 frontier 节点做松弛，避免全图 sweep |
| Landmark A* (ALT) | 工业: 微软 Bing Maps | 预选地标 + 三角不等式下界，A* 启发式更紧 |
| 可定制收缩层级 (CCH) | 工业: routingkit | 支持实时路况权重更新，无需完全重建层级 |

## 运行方式

```bash
python robot/dijkstra/dijkstra.py
```
