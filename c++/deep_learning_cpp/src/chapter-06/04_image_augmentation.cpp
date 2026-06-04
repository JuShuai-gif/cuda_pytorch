/*
 * 04_image_augmentation.cpp - 第 6 章：卷积神经网络
 * 基于 LibTorch 实现 8 种图像数据增强技术（对应原书第 191-202 页）
 *
 * 演示内容：
 *   1. 旋转 (rotate)   - 使用 affine_grid + grid_sample 手动实现
 *   2. 平移 (translate) - 填充 + 切片方式
 *   3. 裁剪 (crop)      - 使用 tensor.index({Slice...})
 *   4. 缩放 (scale)     - 使用 interpolate + bilinear
 *   5. 缩放裁剪 (zoom)  - 中心裁剪后 resize 回原尺寸
 *   6. 翻转 (flip)      - 使用 tensor.flip(dim)
 *   7. 填充 (pad)       - 使用 functional::pad（constant/replicate/reflect）
 *   8. 重采样 (resample)- 使用 interpolate（nearest/bilinear/bicubic）
 *   9. 组合变换 (combined) - 平移→旋转→翻转→重采样流水线
 *
 * 注意：不依赖 torchvision，所有变换均用 LibTorch 核心 API 实现
 */

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>

/* =========================== 1. 旋转变换 =========================== */

/* 使用 affine_grid + grid_sample 实现图像旋转（绕中心点）
 * @param image: 输入图像张量 [C, H, W]
 * @param angle: 旋转角度（度数）
 * @return 旋转后的图像 [C, H, W]
 */
torch::Tensor rotate_image(const torch::Tensor &image, float angle = 45.0f) {
    // 将角度转换为弧度
    float rad = angle * M_PI / 180.0f;
    float cos_a = std::cos(rad);
    float sin_a = std::sin(rad);

    // 构造 2×3 仿射矩阵：先平移中心到原点→旋转→平移回中心
    auto theta = torch::zeros({1, 2, 3}, image.options());
    theta[0][0][0] = cos_a;
    theta[0][0][1] = -sin_a;
    theta[0][0][2] = (1 - cos_a) / 2 + sin_a / 2;
    theta[0][1][0] = sin_a;
    theta[0][1][1] = cos_a;
    theta[0][1][2] = (1 - cos_a) / 2 - sin_a / 2;

    int H = image.size(1);
    int W = image.size(2);

    // 生成采样网格
    auto grid = torch::affine_grid_generator(theta, {1, image.size(0), H, W}, false);
    // 添加 batch 维度 → [1, C, H, W]
    auto img_batch = image.unsqueeze(0);
    // 使用双线性插值采样（0 填充边界外区域）
    auto rotated = torch::grid_sampler(img_batch, grid, 0, 0, false);
    return rotated.squeeze(0);
}

/* =========================== 2. 平移变换 =========================== */

/* 使用填充 + 切片方式实现图像平移
 * @param image: 输入图像张量 [C, H, W]
 * @param dx: 水平位移（正值向右）
 * @param dy: 垂直位移（正值向下）
 * @return 平移后的图像 [C, H, W]
 */
torch::Tensor translate_image(const torch::Tensor &image, int dx, int dy) {
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);

    // 先在周围填充足够的 0，再裁剪出平移后的区域
    int pad_left = std::max(0, -dx);
    int pad_right = std::max(0, dx);
    int pad_top = std::max(0, -dy);
    int pad_bottom = std::max(0, dy);

    auto padded = torch::nn::functional::pad(
        image.unsqueeze(0),
        torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
            .mode(torch::kConstant)
            .value(0.0f));

    int start_x = pad_left + dx; // 裁剪起始列
    int start_y = pad_top + dy;  // 裁剪起始行

    return padded.index({torch::indexing::Slice(),
                         torch::indexing::Slice(),
                         torch::indexing::Slice(start_y, start_y + H),
                         torch::indexing::Slice(start_x, start_x + W)})
        .squeeze(0);
}

/* =========================== 3. 裁剪变换 =========================== */

/* 使用 tensor.index 实现图像裁剪
 * @param image: 输入图像张量 [C, H, W]
 * @param x: 裁剪起始行坐标
 * @param y: 裁剪起始列坐标
 * @param h: 裁剪高度
 * @param w: 裁剪宽度
 * @return 裁剪后的图像 [C, h, w]
 */
torch::Tensor crop_image(const torch::Tensor &image, int x, int y, int h, int w) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(x, x + h),
                        torch::indexing::Slice(y, y + w)});
}

/* =========================== 4. 缩放变换 =========================== */

/* 使用 interpolate(bilinear) 实现图像缩放
 * @param image: 输入图像张量 [C, H, W]
 * @param scale_factor: 缩放因子 (>1 放大, <1 缩小)
 * @return 缩放后的图像 [C, H*scale, W*scale]
 */
torch::Tensor scale_image(const torch::Tensor &image, float scale_factor) {
    int H = image.size(1);
    int W = image.size(2);
    int new_H = static_cast<int>(H * scale_factor);
    int new_W = static_cast<int>(W * scale_factor);

    auto img_batch = image.unsqueeze(0); // [1, C, H, W]
    auto scaled = torch::nn::functional::interpolate(
        img_batch,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_H, new_W})
            .mode(torch::kBilinear)
            .align_corners(false));
    return scaled.squeeze(0);
}

/* =========================== 5. 缩放裁剪变换 =========================== */

/* 中心区域裁剪后 resize 回原尺寸（模拟相机变焦效果）
 * @param image: 输入图像张量 [C, H, W]
 * @param zoom_factor: 变焦因子 (>1 放大)
 * @return zoom 后的图像 [C, H, W]
 */
torch::Tensor zoom_image(const torch::Tensor &image, float zoom_factor) {
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);

    int crop_H = static_cast<int>(H / zoom_factor);
    int crop_W = static_cast<int>(W / zoom_factor);
    int start_x = (H - crop_H) / 2;
    int start_y = (W - crop_W) / 2;

    // 中心裁剪
    auto cropped = image.index({torch::indexing::Slice(),
                                torch::indexing::Slice(start_x, start_x + crop_H),
                                torch::indexing::Slice(start_y, start_y + crop_W)})
                       .unsqueeze(0); // [1, C, crop_H, crop_W]
    // 双线性插值 resize 回原尺寸
    auto zoomed = torch::nn::functional::interpolate(
        cropped,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H, W})
            .mode(torch::kBilinear)
            .align_corners(false));
    return zoomed.squeeze(0);
}

/* =========================== 6. 翻转变换 =========================== */

/* 使用 tensor.flip 实现水平/垂直翻转
 * @param image: 输入图像张量 [C, H, W]
 * @param horizontal: 是否水平翻转（左右镜像）
 * @param vertical: 是否垂直翻转（上下颠倒）
 * @return 翻转后的图像 [C, H, W]
 */
torch::Tensor flip_image(const torch::Tensor &image,
                         bool horizontal = true, bool vertical = false) {
    auto result = image;
    if (horizontal) {
        result = result.flip(-1); // 沿宽度维度翻转（左右）
    }
    if (vertical) {
        result = result.flip(-2); // 沿高度维度翻转（上下）
    }
    return result;
}

/* =========================== 7. 填充变换 =========================== */

/* 使用 functional::pad 实现边界填充（支持多种模式）
 * @param image: 输入图像张量 [C, H, W]
 * @param pad_h: 垂直方向填充量
 * @param pad_w: 水平方向填充量
 * @return 填充后的图像 [C, H+2*pad_h, W+2*pad_w]
 */
torch::Tensor pad_image(const torch::Tensor &image, int pad_h, int pad_w) {
    auto img_batch = image.unsqueeze(0); // [1, C, H, W]

    // constant 模式：用常数值填充
    auto pad_constant = torch::nn::functional::pad(
        img_batch,
        torch::nn::functional::PadFuncOptions({pad_w, pad_w, pad_h, pad_h})
            .mode(torch::kConstant)
            .value(0.0f));
    std::cout << "  常量填充 (constant)    形状: "
              << pad_constant.sizes() << std::endl;

    // replicate 模式：复制边界像素
    auto pad_replicate = torch::nn::functional::pad(
        img_batch,
        torch::nn::functional::PadFuncOptions({pad_w, pad_w, pad_h, pad_h})
            .mode(torch::kReplicate));
    std::cout << "  边界复制 (replicate)   形状: "
              << pad_replicate.sizes() << std::endl;

    // reflect 模式：镜像反射边界像素
    auto pad_reflect = torch::nn::functional::pad(
        img_batch,
        torch::nn::functional::PadFuncOptions({pad_w, pad_w, pad_h, pad_h})
            .mode(torch::kReflect));
    std::cout << "  镜像反射 (reflect)     形状: "
              << pad_reflect.sizes() << std::endl;

    return pad_constant.squeeze(0);
}

/* =========================== 8. 重采样变换 =========================== */

/* 使用 interpolate 实现多种插值模式的重采样
 * @param image: 输入图像张量 [C, H, W]
 * @param new_size: 目标尺寸 {new_H, new_W}
 * @param mode: 插值模式 ("nearest" / "bilinear" / "bicubic")
 * @return 重采样后的图像 [C, new_H, new_W]
 */
torch::Tensor resample_image(const torch::Tensor &image,
                             const std::vector<int64_t> &new_size,
                             const std::string &mode = "bilinear") {
    auto img_batch = image.unsqueeze(0);
    torch::Tensor result;

    if (mode == "nearest") {
        result = torch::nn::functional::interpolate(
            img_batch,
            torch::nn::functional::InterpolateFuncOptions()
                .size(new_size)
                .mode(torch::kNearest));
    } else if (mode == "bicubic") {
        result = torch::nn::functional::interpolate(
            img_batch,
            torch::nn::functional::InterpolateFuncOptions()
                .size(new_size)
                .mode(torch::kBicubic)
                .align_corners(false));
    } else {
        result = torch::nn::functional::interpolate(
            img_batch,
            torch::nn::functional::InterpolateFuncOptions()
                .size(new_size)
                .mode(torch::kBilinear)
                .align_corners(false));
    }
    return result.squeeze(0);
}

/* =========================== 9. 组合变换流水线 =========================== */

/* 将多种增强组合为流水线：平移 → 旋转 → 翻转 → 重采样
 * @param image: 输入图像张量 [C, H, W]
 * @return 组合增强后的图像 [C, H, W]
 */
torch::Tensor apply_combined_transforms(const torch::Tensor &image) {
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);

    // 1. 平移 (dx=10, dy=5)
    auto t = translate_image(image, 10, 5);
    // 2. 旋转 15 度
    t = rotate_image(t, 15.0f);
    // 3. 水平翻转
    t = flip_image(t, true, false);
    // 4. 双线性重采样回原尺寸
    t = resample_image(t, {H, W}, "bilinear");

    return t;
}

/* ================================ main =================================== */

int main() {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════╗\n"
              << "║     第 6 章：卷积神经网络 — 图像数据增强技术演示          ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    /* ------- 创建示例输入图像 ------- */
    // 3 通道 × 256 × 256 的随机 RGB 图像（模拟真实图片张量）
    auto demo_image = torch::rand({3, 256, 256});
    std::cout << "【示例图像】形状: " << demo_image.sizes()
              << "  (C×H×W, 3 通道 RGB, 256×256 像素)\n"
              << std::endl;

    /* ------- 1. 旋转变换 ------- */
    std::cout << "【1. 旋转 (Rotate)】—— 绕图像中心旋转 45 度" << std::endl;
    auto rotated = rotate_image(demo_image, 45.0f);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << rotated.sizes() << "\n"
              << std::endl;

    /* ------- 2. 平移变换 ------- */
    std::cout << "【2. 平移 (Translate)】—— 向右 30px, 向下 20px" << std::endl;
    auto translated = translate_image(demo_image, 30, 20);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << translated.sizes() << "\n"
              << std::endl;

    /* ------- 3. 裁剪变换 ------- */
    std::cout << "【3. 裁剪 (Crop)】—— 从 (50,50) 开始裁剪 128×128 区域" << std::endl;
    auto cropped = crop_image(demo_image, 50, 50, 128, 128);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << cropped.sizes() << "\n"
              << std::endl;

    /* ------- 4. 缩放变换 ------- */
    std::cout << "【4. 缩放 (Scale)】—— 缩放因子 0.5 (缩小一半)" << std::endl;
    auto scaled = scale_image(demo_image, 0.5f);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << scaled.sizes() << "\n"
              << std::endl;

    /* ------- 5. 缩放裁剪变换 ------- */
    std::cout << "【5. 缩放裁剪 (Zoom)】—— 变焦因子 1.5 (中心裁剪后放大)" << std::endl;
    auto zoomed = zoom_image(demo_image, 1.5f);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << zoomed.sizes()
              << "\n  效果: 中心区域被裁剪后插值回原尺寸，模拟镜头拉近\n"
              << std::endl;

    /* ------- 6. 翻转变换 ------- */
    std::cout << "【6. 翻转 (Flip)】—— 水平翻转（左右镜像）" << std::endl;
    auto flipped = flip_image(demo_image, true, false);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << flipped.sizes()
              << "\n  水平翻转: flip(dim=-1) 沿宽度轴反转像素顺序\n"
              << std::endl;

    /* ------- 7. 填充变换 ------- */
    std::cout << "【7. 填充 (Pad)】—— 各方向填充 10 像素 (演示 3 种模式)" << std::endl;
    auto padded = pad_image(demo_image, 10, 10);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "\n  constant: 用 0 填充新区域"
              << "\n  replicate: 用边缘像素值复制填充"
              << "\n  reflect: 用边缘镜像反射填充\n"
              << std::endl;

    /* ------- 8. 重采样变换 ------- */
    std::cout << "【8. 重采样 (Resample)】—— 尺寸变为 64×64 (演示 3 种插值)" << std::endl;
    auto resampled_near = resample_image(demo_image, {64, 64}, "nearest");
    auto resampled_bili = resample_image(demo_image, {64, 64}, "bilinear");
    auto resampled_bicu = resample_image(demo_image, {64, 64}, "bicubic");
    std::cout << "  输入形状: " << demo_image.sizes() << std::endl;
    std::cout << "  nearest  (最近邻) : " << resampled_near.sizes() << std::endl;
    std::cout << "  bilinear (双线性) : " << resampled_bili.sizes() << std::endl;
    std::cout << "  bicubic  (双三次) : " << resampled_bicu.sizes() << std::endl;
    std::cout << "  说明: 最近邻最快但锯齿明显，双线性平衡速度与质量，双三次最平滑但计算量大\n"
              << std::endl;

    /* ------- 9. 组合变换流水线 ------- */
    std::cout << "【9. 组合变换 (Combined)】—— 平移(10,5) → 旋转15° → 水平翻转 → 双线性重采样" << std::endl;
    auto combined = apply_combined_transforms(demo_image);
    std::cout << "  输入形状: " << demo_image.sizes()
              << "  →  输出形状: " << combined.sizes() << std::endl;
    std::cout << "  流水线: 数据增强实践中常将多种变换串联使用，提升模型泛化能力\n"
              << std::endl;

    /* ------- 总结 ------- */
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "  数据增强总结:\n";
    std::cout << "  · 几何变换（旋转/平移/缩放/翻转）不改像素语义但增加位置多样性\n";
    std::cout << "  · 裁剪与 Zoom 模拟不同视野和构图\n";
    std::cout << "  · 填充模式影响边界区域特征提取\n";
    std::cout << "  · 插值方法影响缩放后图像质量与训练速度\n";
    std::cout << "  · 所有变换均支持 GPU 加速（.to(device) 后直接使用）\n";
    std::cout << "════════════════════════════════════════════════════════════\n"
              << std::endl;

    return 0;
}
