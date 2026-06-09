#!/usr/bin/env python3
"""
用于性能分析的真实机器人感知流水线。

阶段：
  1. image_capture   - 生成合成 1920x1080x3 相机 + 360 度 LiDAR（10 万点）
  2. image_preprocess - RGB->YUV，缩放至 640x480，float32 归一化 [0,1]
  3. lidar_preprocess - 距离滤波，地面去除 (RANSAC)，体素降采样
  4. detection        - CNN 滑动窗口 + 欧几里得 LiDAR 聚类
  5. postprocess      - NMS，3D 包围盒，图像平面投影

所有阶段均执行实际的 numpy 运算。不使用 time.sleep() 或空循环。
"""

import numpy as np
from typing import Tuple, List

from timer import TimerContext
from tracker import LatencyTracker

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CAM_W, CAM_H = 1920, 1080
PRE_W, PRE_H = 640, 480
NUM_LIDAR_PTS = 100_000
LIDAR_RANGE_MIN = 0.5
LIDAR_RANGE_MAX = 120.0
VOXEL_SIZE = 0.1
CLUSTER_DIST_THRESH = 1.5
CLUSTER_MIN_PTS = 10
NMS_IOU_THRESH = 0.45
RANSAC_ITERS = 50
RANSAC_DIST_THRESH = 0.2
RANSAC_INLIER_RATIO = 0.6

# YUV 转换矩阵 (BT.601)
_RGB2YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# 阶段 1：传感器数据生成
# ---------------------------------------------------------------------------
def capture_image(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成传感器数据：
      - 相机: 1920x1080x3 uint8 随机图像
      - LiDAR: 10 万点，包含 x,y,z,intensity（形状 [N,4]）
    返回 (image, lidar_points)。
    """
    # 相机: 模拟 Bayer/ISP 输出的随机 uint8 图像
    image = rng.integers(0, 256, size=(CAM_H, CAM_W, 3), dtype=np.uint8)

    # 360 度 LiDAR: 均匀方位角 [0, 2*pi]，仰角以 0 为中心正态分布
    azimuth = rng.uniform(0, 2 * np.pi, size=NUM_LIDAR_PTS).astype(np.float32)
    elevation = rng.normal(0, 0.05, size=NUM_LIDAR_PTS).astype(np.float32)
    # 距离: 混合均匀近场和指数分布远场
    distance = rng.exponential(30.0, size=NUM_LIDAR_PTS).astype(np.float32)
    distance = np.clip(distance, 0.1, LIDAR_RANGE_MAX + 20)

    x = distance * np.cos(elevation) * np.cos(azimuth)
    y = distance * np.cos(elevation) * np.sin(azimuth)
    z = distance * np.sin(elevation)
    intensity = rng.uniform(0.0, 1.0, size=NUM_LIDAR_PTS).astype(np.float32)

    lidar = np.column_stack([x, y, z, intensity])
    return image, lidar


# ---------------------------------------------------------------------------
# 阶段 2a：图像预处理
# ---------------------------------------------------------------------------
def preprocess_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    将 RGB->YUV，通过类似双线性的子采样缩放至 640x480，
    float32 归一化到 [0,1]。

    返回形状为 (PRE_H, PRE_W, 3) 的 float32 张量。
    """
    # RGB -> float32 [0,1] -> YUV
    img_f32 = image.astype(np.float32) / 255.0
    # 应用 YUV 转换: (H,W,3) @ (3,3)^T = (H,W,3)
    yuv = img_f32 @ _RGB2YUV.T
    yuv[:, :, 1:] += 0.5  # U,V 偏移至 [0,1]

    # 缩放: 使用简单的块平均进行降采样
    # 计算缩放因子
    sy = CAM_H // PRE_H  # 2
    sx = CAM_W // PRE_W  # 3
    # 裁剪到缩放因子的倍数
    crop_h = sy * PRE_H
    crop_w = sx * PRE_W
    yuv_cropped = yuv[:crop_h, :crop_w, :]
    # 重塑并均值池化
    reshaped = yuv_cropped.reshape(PRE_H, sy, PRE_W, sx, 3)
    resized = reshaped.mean(axis=(1, 3))  # (PRE_H, PRE_W, 3)

    return resized.astype(np.float32)


# ---------------------------------------------------------------------------
# 阶段 2b：LiDAR 预处理
# ---------------------------------------------------------------------------
def preprocess_lidar(
    lidar: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. 按距离 [0.5m, 120m] 过滤点
    2. 通过简化 RANSAC 平面拟合进行地面去除
    3. 使用 0.1m 网格进行体素降采样

    返回:
      - voxel_centroids: (M, 3) float32 xyz 质心
      - voxel_intensities: (M,) float32 每体素平均强度
      - ground_plane: (4,) float32 [a,b,c,d]，其中 ax+by+cz+d=0
    """
    xyz = lidar[:, :3]
    intensity = lidar[:, 3]
    dist = np.linalg.norm(xyz, axis=1)

    # 距离滤波
    mask = (dist >= LIDAR_RANGE_MIN) & (dist <= LIDAR_RANGE_MAX)
    xyz = xyz[mask]
    intensity = intensity[mask]
    dist = dist[mask]

    if xyz.shape[0] < 100:
        return xyz, intensity, np.array([0, 0, 1, 0], dtype=np.float32)

    # 简化 RANSAC 地平面: 找到内点数最多的平面
    # 模型: ax + by + cz + d = 0，其中 c ~ 1（主要水平）
    best_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    best_inlier_count = 0
    n_pts = xyz.shape[0]

    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = xyz[idx[0]], xyz[idx[1]], xyz[idx[2]]
        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal /= norm_len
        # 确保法向量朝上
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, p1)
        plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

        # 统计内点数
        distances = np.abs(np.dot(xyz, plane[:3]) + plane[3])
        inliers = distances < RANSAC_DIST_THRESH
        n_inliers = np.sum(inliers)

        if n_inliers > best_inlier_count:
            best_inlier_count = n_inliers
            best_plane = plane

    # 去除地面点
    ground_dists = np.abs(np.dot(xyz, best_plane[:3]) + best_plane[3])
    non_ground_mask = ground_dists >= RANSAC_DIST_THRESH
    xyz_ng = xyz[non_ground_mask]
    intensity_ng = intensity[non_ground_mask]

    if xyz_ng.shape[0] < 10:
        return xyz_ng, intensity_ng, best_plane

    # 体素降采样: 量化为 0.1m 网格
    voxel_indices = np.floor(xyz_ng / VOXEL_SIZE).astype(np.int32)
    # 通过字典获取唯一体素键
    voxel_map = {}
    for i in range(xyz_ng.shape[0]):
        key = (voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2])
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(i)

    # 计算每个体素的质心
    centroids = []
    mean_intensities = []
    for key, indices in voxel_map.items():
        pts = xyz_ng[indices]
        centroids.append(pts.mean(axis=0))
        mean_intensities.append(intensity_ng[indices].mean())

    if not centroids:
        return xyz_ng, intensity_ng, best_plane

    voxel_centroids = np.array(centroids, dtype=np.float32)
    voxel_intensities_arr = np.array(mean_intensities, dtype=np.float32)

    return voxel_centroids, voxel_intensities_arr, best_plane


# ---------------------------------------------------------------------------
# 阶段 3：目标检测
# ---------------------------------------------------------------------------
def _conv2d(patch: np.ndarray, kernel: np.ndarray) -> float:
    """将 3x3 卷积核应用于一个图像块。"""
    return float(np.sum(patch * kernel))


def detect_objects(
    image_tensor: np.ndarray,
    lidar_centroids: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict]]:
    """
    在图像上模拟 CNN 滑动窗口 + LiDAR 欧几里得聚类。

    图像 CNN: 在 640x480 图像上以 stride=112 滑动 224x224 窗口，
              应用 3x3 随机卷积核。对响应图进行阈值处理
              以获取检测候选。

    LiDAR: 基于欧几里得距离的聚类。

    返回:
      - img_detections: 列表 {x, y, w, h, confidence, class_id}
      - lidar_detections: 列表 {x, y, z, w, h, d, confidence, class_id}
    """
    h, w, c = image_tensor.shape
    kernel = rng.normal(0, 0.5, size=(3, 3, c)).astype(np.float32)
    window_size = 224
    stride = 112

    # 在 Y 通道上进行 CNN 滑动窗口（通道 0）
    y_channel = image_tensor[:, :, 0]
    img_detections = []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = image_tensor[y : y + window_size, x : x + window_size, :]
            # 模拟卷积: 将卷积核应用于随机的 3x3 子块
            cy = rng.integers(0, window_size - 3)
            cx = rng.integers(0, window_size - 3)
            sub = patch[cy : cy + 3, cx : cx + 3, :]
            response = np.abs(_conv2d(sub, kernel))
            if response > 1.0:
                confidence = min(float(response) / 5.0, 0.99)
                img_detections.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "w": float(window_size),
                        "h": float(window_size),
                        "confidence": confidence,
                        "class_id": int(rng.integers(0, 4)),
                    }
                )

    # LiDAR 欧几里得聚类（网格加速）
    lidar_detections = []
    if lidar_centroids.shape[0] < CLUSTER_MIN_PTS:
        return img_detections, lidar_detections

    pts = lidar_centroids[:, :3].copy()
    n_pts = pts.shape[0]

    # 构建空间网格以实现 O(n) 邻居查询
    grid_cell_size = CLUSTER_DIST_THRESH
    grid_min = pts.min(axis=0) - grid_cell_size
    grid_indices = np.floor((pts - grid_min) / grid_cell_size).astype(np.int32)
    grid_map = {}
    for i in range(n_pts):
        key = (grid_indices[i, 0], grid_indices[i, 1], grid_indices[i, 2])
        grid_map.setdefault(key, []).append(i)

    visited = np.zeros(n_pts, dtype=bool)
    clusters = []

    for i in range(n_pts):
        if visited[i]:
            continue
        visited[i] = True
        cluster = [i]
        queue = [i]

        while queue:
            cur = queue.pop()
            gx, gy, gz = (
                grid_indices[cur, 0],
                grid_indices[cur, 1],
                grid_indices[cur, 2],
            )
            # 检查 27 个相邻单元格
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        key = (gx + dx, gy + dy, gz + dz)
                        for neighbor in grid_map.get(key, []):
                            if visited[neighbor]:
                                continue
                            dist_sq = np.sum((pts[cur] - pts[neighbor]) ** 2)
                            if dist_sq < CLUSTER_DIST_THRESH**2:
                                visited[neighbor] = True
                                cluster.append(neighbor)
                                queue.append(neighbor)

        if len(cluster) >= CLUSTER_MIN_PTS:
            clusters.append(cluster)

    for cluster in clusters:
        cluster_pts = pts[cluster]
        min_xyz = cluster_pts.min(axis=0)
        max_xyz = cluster_pts.max(axis=0)
        center = cluster_pts.mean(axis=0)
        dims = max_xyz - min_xyz
        lidar_detections.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "w": float(dims[0]),
                "h": float(dims[1]),
                "d": float(dims[2]),
                "confidence": min(0.95, len(cluster) / 200.0),
                "class_id": int(rng.integers(0, 4)),
            }
        )

    return img_detections, lidar_detections


# ---------------------------------------------------------------------------
# 阶段 4：后处理
# ---------------------------------------------------------------------------
def compute_iou(box_a: dict, box_b: dict) -> float:
    """计算两个 2D 框之间的 IoU。"""
    xa = max(box_a["x"], box_b["x"])
    ya = max(box_a["y"], box_b["y"])
    xb = min(box_a["x"] + box_a["w"], box_b["x"] + box_b["w"])
    yb = min(box_a["y"] + box_a["h"], box_b["y"] + box_b["h"])
    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter = inter_w * inter_h
    area_a = box_a["w"] * box_a["h"]
    area_b = box_b["w"] * box_b["h"]
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def postprocess(
    img_detections: List[dict],
    lidar_detections: List[dict],
    ground_plane: np.ndarray,
) -> Tuple[List[dict], List[dict]]:
    """
    1. 对图像检测进行 NMS
    2. 为 LiDAR 聚类计算 3D 包围盒
    3. 将 LiDAR 检测投影到图像平面（针孔模型）

    返回 (final_img_dets, final_lidar_dets)。
    """
    # 对图像检测进行 NMS，按置信度排序
    img_sorted = sorted(img_detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    suppressed = np.zeros(len(img_sorted), dtype=bool)
    for i in range(len(img_sorted)):
        if suppressed[i]:
            continue
        kept.append(img_sorted[i])
        for j in range(i + 1, len(img_sorted)):
            if suppressed[j]:
                continue
            iou = compute_iou(img_sorted[i], img_sorted[j])
            if iou > NMS_IOU_THRESH:
                suppressed[j] = True

    # 将 LiDAR 检测投影到图像平面
    # 简易针孔模型: fx=fy=800, cx=PRE_W/2, cy=PRE_H/2
    fx, fy = 800.0, 800.0
    cx, cy = PRE_W / 2.0, PRE_H / 2.0
    projected = []
    for det in lidar_detections:
        X, Y, Z = det["x"], det["y"], det["z"]
        if Z <= 0.1:
            continue
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        det_copy = dict(det)
        det_copy["image_u"] = float(u)
        det_copy["image_v"] = float(v)
        det_copy["is_in_frame"] = 0 <= u < PRE_W and 0 <= v < PRE_H
        projected.append(det_copy)

    return kept, projected


# ---------------------------------------------------------------------------
# 端到端流水线运行器
# ---------------------------------------------------------------------------
def run_pipeline_frame(
    rng: np.random.Generator,
    tracker: LatencyTracker,
) -> None:
    """执行一个完整的感知流水线帧。"""

    # 阶段 1：传感器采集
    with TimerContext("image_capture") as t:
        image, lidar = capture_image(rng)
    tracker.record("image_capture", t.elapsed_us)

    # 阶段 2a：图像预处理
    with TimerContext("image_preprocess") as t:
        image_tensor = preprocess_image(image, rng)
    tracker.record("image_preprocess", t.elapsed_us)

    # 阶段 2b：LiDAR 预处理
    with TimerContext("lidar_preprocess") as t:
        lidar_centroids, _, ground_plane = preprocess_lidar(lidar, rng)
    tracker.record("lidar_preprocess", t.elapsed_us)

    # 阶段 3：检测
    with TimerContext("detection") as t:
        img_dets, lidar_dets = detect_objects(image_tensor, lidar_centroids, rng)
    tracker.record("detection", t.elapsed_us)

    # 阶段 4：后处理
    with TimerContext("postprocess") as t:
        final_img, final_lidar = postprocess(img_dets, lidar_dets, ground_plane)
    tracker.record("postprocess", t.elapsed_us)
