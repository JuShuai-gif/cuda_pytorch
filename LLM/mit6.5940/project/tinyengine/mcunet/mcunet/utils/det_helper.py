# ============================================================================
# det_helper.py —— 目标检测辅助工具
#
# 本文件提供目标检测相关的后处理工具，主要包括：
#   1. bbox_iou        —— 计算两个边界框（bounding box）的交并比（IoU）
#   2. standard_nms    —— 标准非极大值抑制（NMS）算法实现
#   3. StandardNMS     —— 标准 NMS 的面向对象封装（支持配置化构建）
#   4. MergeNMS        —— 合并 NMS（对重叠框做加权平均融合）
#   5. Yolo3Output     —— YOLOv3 检测头的输出解码层
#
# 部分代码改编自 gluoncv: https://cv.gluon.ai/model_zoo/detection.html
#
# NMS 是目标检测中最重要的后处理步骤之一。模型通常会输出大量重叠的候选框，
# NMS 的作用是保留每个目标最合适的那个框，去除冗余的重叠检测。
# ============================================================================

import torch
import torch.nn as nn
import numpy as np

# __all__ 控制 from det_helper import * 时暴露的公共接口
__all__ = ["standard_nms", "StandardNMS", "MergeNMS", "Yolo3Output"]


# ============================================================================
# bbox_iou
# ============================================================================
# 功能：计算两组边界框之间的 IoU（Intersection over Union，交并比）。
#
# IoU 公式：
#   IoU = (box1 ∩ box2) / (box1 ∪ box2)
#        = 交集面积 / (box1面积 + box2面积 - 交集面积)
#
# 边界框表示格式：
#   (x1, y1, x2, y2) —— 左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)
#
# 参数：
#   box1  —— 第一组框，形状任意，但最后 4 维必须是 (x1, y1, x2, y2)
#   box2  —— 第二组框，形状与 box1 兼容（支持广播）
#   offset—— 一个小偏移量（通常为 0 或 1），用于调整面积计算
#             当 offset=1 时相当于计算像素点个数（边界框坐标是像素索引时使用）
#
# 返回值：
#   IoU 值张量，形状与输入兼容
# ============================================================================
def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, offset=0) -> torch.Tensor:
    # ---------------------------------------------------------------
    # 第一步：解析两个框的角点坐标
    # ---------------------------------------------------------------
    # box[..., 0] 取每个框的 x1，box[..., 1] 取 y1，依此类推
    # ... 表示前面的所有维度（支持 batch 处理）
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # ---------------------------------------------------------------
    # 第二步：计算交集区域的坐标
    # ---------------------------------------------------------------
    # 交集左上角 = max(两个框的左上角坐标)
    # 交集右下角 = min(两个框的右下角坐标)
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # ---------------------------------------------------------------
    # 第三步：计算交集面积和并集面积
    # ---------------------------------------------------------------
    # torch.clamp(..., min=0) 保证当两个框不重叠（交集为负）时面积为 0
    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1 + offset, min=0
    ) * torch.clamp(inter_rect_y2 - inter_rect_y1 + offset, min=0)

    # 每个框的面积 = 宽 * 高
    b1_area = (b1_x2 - b1_x1 + offset) * (b1_y2 - b1_y1 + offset)
    b2_area = (b2_x2 - b2_x1 + offset) * (b2_y2 - b2_y1 + offset)

    # ---------------------------------------------------------------
    # 第四步：计算 IoU = 交集 / 并集
    # ---------------------------------------------------------------
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


# ============================================================================
# standard_nms
# ============================================================================
# 功能：标准非极大值抑制算法。
#
# 算法流程（每张图片独立处理）：
#   1. 过滤掉置信度低于 valid_thres 的框
#   2. 按置信度降序排列
#   3. 选择当前置信度最高的框，计算它与其他所有框的 IoU
#   4. 移除与当前框 IoU 超过 nms_thres 且类别相同的框
#   5. 重复步骤 3-4，直到所有框都处理完
#   6. 剩余的框就是最终检测结果
#
# 参数：
#   prediction  —— 模型输出的原始检测结果，
#                  形状 (batch, num_boxes, 6)，每行格式：
#                  (class_id, confidence, x1, y1, x2, y2)
#   valid_thres —— 置信度阈值，低于此值的框被直接丢弃（默认 0.01）
#   nms_thres   —— NMS 的 IoU 阈值，超过此值的重叠框被抑制（默认 0.45）
#   topk        —— NMS 前保留的最高分框数量，-1 表示不限制（默认 -1）
#   merge       —— 是否合并重叠框（加权平均坐标），用于 MergeNMS（默认 False）
#   pad_val     —— 当没有有效检测时使用的填充值（默认 -1）
#
# 返回值：
#   列表，每个元素是形状 (num_kept_boxes, 6) 的张量，表示该图片的最终检测结果。
#   如果某张图没有检测到任何框，则返回全 pad_val 的形状 (num_boxes, 6) 张量。
# ============================================================================
def standard_nms(
    prediction: torch.Tensor,
    valid_thres=0.01,
    nms_thres=0.45,
    topk=-1,
    merge=False,
    pad_val=-1,
) -> list:
    # ---------------------------------------------------------------
    # 第一步：初始化每张图的输出（空张量）
    # ---------------------------------------------------------------
    # output 是一个列表，每张图片对应一个元素
    output = [torch.empty(0) for _ in range(len(prediction))]
    num_boxes = prediction.shape[1]  # 每张图的候选框总数

    # ---------------------------------------------------------------
    # 第二步：逐张图片执行 NMS
    # ---------------------------------------------------------------
    for image_i, image_pred in enumerate(prediction):
        # 过滤：只保留置信度 >= valid_thres 的框
        # image_pred[:, 1] 是每个框的置信度
        image_pred = image_pred[image_pred[:, 1] >= valid_thres]

        # 如果过滤后没有有效框，用 pad_val 填充并跳过
        if not image_pred.size(0):
            # 创建一个形状 (num_boxes, 6) 的张量，所有值设为 pad_val
            output[image_i] = torch.ones(num_boxes, 6, device=prediction.device).fill_(
                pad_val
            )
            continue

        # ---------------------------------------------------------------
        # 第三步：按置信度降序排列
        # ---------------------------------------------------------------
        # argsort 返回排序索引，取负号实现降序
        image_pred = image_pred[(-image_pred[:, 1]).argsort()]
        if topk > 0:
            # 如果 topk 限制，只保留前 topk 个最高分框
            image_pred = image_pred[:topk]

        detections = image_pred
        keep_boxes = []  # 存放 NMS 后保留的框

        # ---------------------------------------------------------------
        # 第四步：循环 NMS 核心逻辑
        # ---------------------------------------------------------------
        # 每次迭代处理当前最高分的框，然后移除与它重叠过大的框
        n_remaining = detections.size(0)
        for i in range(n_remaining):
            # 计算当前最高分框（detections[0]）与所有框的 IoU
            # detections[0, 2:6] 是当前最高分框的 (x1,y1,x2,y2)
            # .unsqueeze(0) 增加 batch 维度以匹配 detections[:, 2:6] 的形状
            # torch.gt: greater than，判断 IoU 是否超过阈值
            large_overlap = torch.gt(
                bbox_iou(detections[0, 2:6].unsqueeze(0), detections[:, 2:6]), nms_thres
            )

            # 判断类别是否相同（类别相同的框才会互相抑制）
            # detections[0, 0] 是最高分框的类别 ID
            label_match = detections[0, 0] == detections[:, 0]

            # 标记需要抑制的框：与当前框类别相同且 IoU 过大
            invalid = large_overlap & label_match

            if merge:
                # 合并模式（MergeNMS）：用加权平均融合所有重叠框的坐标
                # 权重为每个框的置信度（detections[invalid, 4:5]）
                # 注意这里用的 invalid 包含了当前框本身
                weights = detections[invalid, 4:5]
                # 加权平均坐标：(w1 * box1 + w2 * box2 + ...) / (w1 + w2 + ...)
                detections[0, 2:6] = (weights * detections[invalid, 2:6]).sum(
                    0
                ) / weights.sum()

            # 保留当前最高分框（合并后的或原始的）
            keep_boxes += [detections[0]]
            # 移除被抑制的框（~invalid 表示未被抑制的框）
            detections = detections[~invalid]
            if detections.size(0) == 0:
                break

        # ---------------------------------------------------------------
        # 第五步：将保留的框堆叠为张量
        # ---------------------------------------------------------------
        if keep_boxes:
            # torch.stack 将列表中的张量沿新维度堆叠
            output[image_i] = torch.stack(keep_boxes)

    return output


# ============================================================================
# StandardNMS
# ============================================================================
# 功能：标准 NMS 的面向对象封装类。
#
# 设计意图：
#   将 NMS 的参数（valid_thres, nms_thres, topk 等）封装为对象属性，
#   便于在配置系统中使用。支持通过 build_from_config 从字典配置创建实例。
#
#   在 MCUNet 的检测模型部署流水线中，检测头输出原始结果后需要经过 NMS
#   后处理，这个类提供了可配置的 NMS 模块。
# ============================================================================
class StandardNMS(object):
    def __init__(
        self,
        nms_valid_thres=0.01,  # 置信度过滤阈值
        nms_thres=0.45,  # NMS 的 IoU 阈值
        nms_topk=-1,  # NMS 前保留的最高分框数
        post_nms=100,  # NMS 后保留的最大框数
        pad_val=-1,  # 填充值
    ):
        self.nms_valid_thres = nms_valid_thres
        self.nms_thres = nms_thres
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.pad_val = pad_val

    @property
    def merge(self):
        """是否启用合并模式（StandardNMS 默认为 False，MergeNMS 会重写此属性）"""
        return False

    def set_nms(
        self, nms_valid_thres=None, nms_thres=None, nms_topk=None, post_nms=None
    ):
        """动态修改 NMS 参数，支持仅更新部分参数"""
        if nms_valid_thres is not None:
            self.nms_valid_thres = nms_valid_thres
        if nms_thres is not None:
            self.nms_thres = nms_thres
        if nms_topk is not None:
            self.nms_topk = nms_topk
        if post_nms is not None:
            self.post_nms = post_nms

    def __call__(self, detections: torch.Tensor):
        """执行 NMS 后处理
        参数：
          detections —— 原始检测结果，形状 (batch, num_boxes, 6)
                        每行: (class_id, confidence, x1, y1, x2, y2)
        返回值：
          (ids, scores, bboxes) 元组，分别代表类别 ID、置信度和边界框坐标
        """
        if 0 < self.nms_thres < 1:
            box_num = detections.size(1)

            # 调用标准 NMS 函数
            detections = standard_nms(
                prediction=detections,
                valid_thres=self.nms_valid_thres,
                nms_thres=self.nms_thres,
                topk=self.nms_topk,
                merge=self.merge,
                pad_val=self.pad_val,
            )

            # 将每张图的检测结果补齐到统一数量 box_num
            # 因为 NMS 后每张图保留的框数可能不同，但后续处理需要统一形状
            for idx, det in enumerate(detections):
                if det.size(0) < box_num:
                    # 用 pad_val 填充到 box_num 个框
                    detections[idx] = torch.cat(
                        [
                            det,
                            torch.ones(
                                box_num - det.size(0), det.size(1), device=det.device
                            )
                            * self.pad_val,
                        ],
                        dim=0,
                    )
            detections = torch.stack(detections)

            # 如果设置了 post_nms，截取前 post_nms 个框
            if self.post_nms > 0:
                detections = detections[:, 0 : self.post_nms, :]

        # 拆分为类别 ID、置信度和边界框坐标三个部分返回
        ids = detections[..., 0:1]  # (batch, num_boxes, 1)
        scores = detections[..., 1:2]  # (batch, num_boxes, 1)
        bboxes = detections[..., 2:6]  # (batch, num_boxes, 4)
        return ids, scores, bboxes

    @staticmethod
    def build_from_config(config) -> "StandardNMS":
        """从配置字典创建 StandardNMS 实例"""
        return StandardNMS(
            nms_valid_thres=config.get("nms_valid_thres", 0.01),
            nms_thres=config.get("nms_thres", 0.45),
            nms_topk=config.get("nms_topk", 400),
            post_nms=config.get("post_nms", 100),
            pad_val=config.get("pad_val", -1),
        )


# ============================================================================
# MergeNMS
# ============================================================================
# 功能：合并 NMS，继承自 StandardNMS。
#
# 与标准 NMS 的区别：
#   标准 NMS 在抑制重叠框时直接丢弃，而 MergeNMS 会将所有重叠框的坐标
#   按置信度加权平均融合为一个框。这种方法通常能提高定位精度，因为融合
#   后的框综合了多个高响应框的信息。
#
#   具体实现差异体现在 merge 属性返回 True，standard_nms 函数在 merge=True
#   时会执行加权平均的逻辑。
# ============================================================================
class MergeNMS(StandardNMS):
    @property
    def merge(self):
        """覆盖父类属性，启用合并模式"""
        return True

    @staticmethod
    def build_from_config(config) -> "MergeNMS":
        """从配置字典创建 MergeNMS 实例"""
        return MergeNMS(
            nms_valid_thres=config.get("nms_valid_thres", 0.01),
            nms_thres=config.get("nms_thres", 0.45),
            nms_topk=config.get("nms_topk", 400),
            post_nms=config.get("post_nms", 100),
            pad_val=config.get("pad_val", -1),
        )


# ============================================================================
# Yolo3Output
# ============================================================================
# 功能：YOLOv3 检测头的输出解码层，将特征图映射为具体的检测结果。
#
# 设计背景：
#   YOLOv3 的检测头在每个特征图位置预测 B 个 anchor 框，每个 anchor
#   预测 (4 + 1 + C) 个值：
#     - 4: 边界框坐标（中心 tx, ty 和尺度 tw, th）
#     - 1: 物体性（objectness，是否有目标）
#     - C: C 个类别的分数（C = num_class）
#
#   网络的原始输出是编码后的（raw）值，需要通过这个模块解码为真实的
#   边界框坐标。解码公式如下：
#     box_center.x = (sigmoid(tx) + grid_x) * stride
#     box_center.y = (sigmoid(ty) + grid_y) * stride
#     box_width    = exp(tw) * anchor_w
#     box_height   = exp(th) * anchor_h
#     最终框 = [center - wh/2, center + wh/2]（即 (x1, y1, x2, y2) 格式）
#
# 参数：
#   num_class —— 检测类别数
#   anchors   —— anchor 框的尺寸列表，如 [[w1, h1], [w2, h2], [w3, h3]]
#   stride    —— 特征图相对于输入图像的步长
#   alloc_size—— 预分配的网格大小（用于缓存网格偏移量），默认 [128, 128]
# ============================================================================
class Yolo3Output(nn.Module):
    def __init__(self, num_class: int, anchors: list, stride: int, alloc_size=None):
        super(Yolo3Output, self).__init__()

        self.num_class = num_class
        self.anchors = anchors
        self.stride = stride
        self.alloc_size = [128, 128] if alloc_size is None else alloc_size

        # 将 anchor 列表转为 numpy 数组
        np_anchors = np.array(anchors).astype("float32")
        # 每个 anchor 的预测值数量：1 (obj) + 4 (box) + C (class)
        self._num_pred = 1 + 4 + num_class
        # anchor 的数量（列表长度 / 2）
        self._num_anchors = np_anchors.size // 2

        # ---------------------------------------------------------------
        # 注册 anchor buffer
        # ---------------------------------------------------------------
        # reshape 为 (1, 1, num_anchors, 2) 方便广播
        # 维度含义：(batch, 空间位置, anchor 索引, (w, h))
        np_anchors = np_anchors.reshape((1, 1, -1, 2))  # (1, 1, 3, 2)
        # register_buffer 将张量注册为模块的持久化缓冲区（会随模型保存/加载）
        self.register_buffer(
            "anchors_buffer", torch.from_numpy(np_anchors)
        )  # (1, 1, 3, 2)

        # ---------------------------------------------------------------
        # 预计算网格偏移量并注册为 buffer
        # ---------------------------------------------------------------
        # 对于特征图的每个位置 (gx, gy)，其锚框的中心偏移基准就是 (gx, gy)
        # 预先生成网格坐标，避免每次 forward 都重新计算
        grid_x = np.arange(self.alloc_size[1])
        grid_y = np.arange(self.alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # 堆叠为 (H, W, 2) 的偏移量
        offsets = np.concatenate(
            (grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1
        )
        # 扩展维度到 (1, 1, H, W, 2)
        offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
        # 注册为 buffer，自动管理 device 移动
        self.register_buffer(
            "offsets_buffer", torch.from_numpy(offsets)
        )  # (1, 1, 128, 128, 2)

    @property
    def total_pred_num(self):
        """每个空间位置的总预测值数 = num_anchors * (1 + 4 + num_class)"""
        return self._num_anchors * self._num_pred

    def forward(self, x):
        """前向传播：解码特征图到检测框

        参数：
          x —— 输入特征图，形状 (batch, C, H, W)
               其中 C = num_anchors * (5 + num_class)

        训练模式返回：
          (解码框, 原始中心, 原始尺度, 物体性, 类别分数, anchor, 偏移量)
          包含训练损失计算所需的全部中间变量

        推理模式返回：
          形状 (batch, H*W*num_anchors, 6) 的张量，每行 (class_id, conf, x1, y1, x2, y2)
        """
        # ---------------------------------------------------------------
        # 第一步：重塑特征图为可解析格式
        # ---------------------------------------------------------------
        # 原始 x.shape = (batch, num_anchors * (5+num_class), H, W)
        # 先 reshape 为 (batch, total_pred_num, H*W)
        pred = x.reshape((x.size(0), self.total_pred_num, -1))
        # 转置为 (batch, H*W, total_pred_num)
        pred = pred.permute(0, 2, 1)
        # 最终形状: (batch, H*W, num_anchors, 5+num_class)
        pred = pred.view(pred.size(0), -1, self._num_anchors, self._num_pred)

        # ---------------------------------------------------------------
        # 第二步：解析各分量
        # ---------------------------------------------------------------
        raw_box_centers = pred[..., 0:2]  # (tx, ty)，中心坐标的偏移
        raw_box_scales = pred[..., 2:4]  # (tw, th)，宽高的尺度
        objness = pred[..., 4:5]  # 物体性（有无目标）
        class_pred = pred[..., 5:]  # 各类别分数

        # ---------------------------------------------------------------
        # 第三步：解码边界框坐标
        # ---------------------------------------------------------------
        # 根据实际特征图尺寸裁剪偏移量（因为输入特征图可能小于 alloc_size）
        offsets = self.offsets_buffer[:, :, 0 : x.size(2), 0 : x.size(3), :]
        offsets = offsets.reshape((1, -1, 1, 2))

        # 解码中心坐标：sigmoid(raw) + grid_offset，再乘以 stride 映射回原图
        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self.stride
        # 解码宽高：exp(raw) * anchor_size
        box_scales = torch.exp(raw_box_scales) * self.anchors_buffer
        # 转换为中心+宽高表示 → 左上角+右下角表示
        wh = box_scales / 2.0
        bbox = torch.cat([box_centers - wh, box_centers + wh], dim=-1)

        # ---------------------------------------------------------------
        # 第四步：根据训练/推理模式返回不同格式
        # ---------------------------------------------------------------
        if self.training:
            # 训练模式：返回解码所需的全部中间变量，供损失函数使用
            return (
                bbox.reshape((bbox.size(0), -1, 4)),  # 解码后的边界框
                raw_box_centers,  # 原始中心偏移（用于损失计算）
                raw_box_scales,  # 原始尺度（用于损失计算）
                objness,  # 原始物体性分数
                class_pred,  # 原始类别分数
                self.anchors_buffer,  # anchor 尺寸
                offsets,  # 网格偏移
            )

        # 推理模式：组合为统一格式的输出
        # 先计算置信度 = sigmoid(objectness)
        confidence = torch.sigmoid(objness)
        # 类别分数 = sigmoid(class) * confidence（乘上物体性，消除没有目标的误检）
        class_score = torch.sigmoid(class_pred) * confidence

        # 将边界框复制 num_class 份，每份对应一个类别
        # bbox 形状 (batch, H*W, num_anchors, 4) → unsqueeze(0) → (1, ...)
        # repeat_interleave 沿第 0 维复制 num_class 次
        bboxes = torch.repeat_interleave(
            bbox.unsqueeze(0), repeats=self.num_class, dim=0
        )
        # scores 形状: (num_class, batch, H*W, num_anchors, 1)
        scores = class_score.permute(3, 0, 1, 2).unsqueeze(axis=-1)
        # ids: 为每个类别生成对应的索引
        # (num_class, 1, 1, 1, 1) → 广播到与 scores 相同形状
        ids = scores * 0 + torch.arange(0, self.num_class, device=x.device).reshape(
            (self.num_class, 1, 1, 1, 1)
        )
        # 拼接为 (num_class, batch, H*W, num_anchors, 6) 格式
        detections = torch.cat([ids, scores, bboxes], dim=-1)
        # 调整维度顺序: (batch, num_class, H*W, num_anchors, 6)
        detections = detections.permute(1, 0, 2, 3, 4)
        # 合并 num_class、H*W、num_anchors 三个维度
        # 最终形状: (batch, num_class * H * W * num_anchors, 6)
        detections = detections.reshape(detections.size(0), -1, 6)
        return detections

    @staticmethod
    def build_from_config(config) -> "Yolo3Output":
        """从配置字典创建 Yolo3Output 实例"""
        return Yolo3Output(
            num_class=config["output"]["num_class"],
            anchors=config["output"]["anchors"],
            stride=config["output"]["stride"],
            alloc_size=config["output"]["alloc_size"],
        )
