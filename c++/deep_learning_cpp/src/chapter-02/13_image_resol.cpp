/*
 * image_resol.cpp
 * 第2章: C++中的数据准备与预处理
 *
 * 图像预处理将原始像素数据转换为归一化的、尺寸一致的输入，
 * 适用于卷积神经网络(CNN)。预处理步骤对模型收敛和泛化至关重要，
 * 与训练时应用的变换保持一致。
 *
 * 技术: 图像预处理流水线 (缩放 -> 裁剪 -> 均衡化 -> 边缘检测)
 * 对于生产级流水线，还需考虑:
 *   - 归一化: 减去均值，除以标准差 (按通道)
 *   - 数据增强: 随机翻转、旋转、颜色抖动
 *   - 批量处理: 对大规模数据集使用DataLoader
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    std::string inputPath;
    if (argc >= 2) {
        inputPath = argv[1];
    } else {
        // 如果未提供输入，生成一张演示图像
        std::cout << "No input image provided. Generating a demo image.\n";
        std::cout << "Usage: " << argv[0] << " <input_image.jpg>\n\n";

        cv::Mat demo(256, 256, CV_8UC3);
        cv::randu(demo, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::imwrite("/tmp/demo_input.jpg", demo);
        inputPath = "/tmp/demo_input.jpg";
        std::cout << "Generated demo image: " << inputPath << "\n";
    }

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
        std::cerr << "ERROR: Failed to load image: " << inputPath << "\n";
        return 1;
    }
    std::cout << "Loaded: " << inputPath
              << " (" << img.cols << "x" << img.rows << ")\n";

    // --- 步骤1: 缩放至256x256 ---
    // 许多CNN架构(ResNet、VGG等)的标准预处理流程
    // 首先缩放到稍大的尺寸，然后中心裁剪到目标尺寸。
    cv::Mat resized;
    cv::resize(img, resized, {256, 256});
    cv::imwrite("/tmp/resized.jpg", resized);
    std::cout << "[1] Resized to 256x256 -> /tmp/resized.jpg\n";

    // --- 步骤2: 中心裁剪至224x224 ---
    // 224x224裁剪是ImageNet预训练模型的标准尺寸。
    // 移除边缘像素使模型聚焦于中心主体，并消除
    // 缩放操作产生的边缘伪影。
    int cropSize = 224;
    int x = (resized.cols - cropSize) / 2;
    int y = (resized.rows - cropSize) / 2;
    cv::Rect roi(x, y, cropSize, cropSize);
    cv::Mat crop = resized(roi);
    cv::imwrite("/tmp/crop.jpg", crop);
    std::cout << "[2] Center-cropped to 224x224 -> /tmp/crop.jpg\n";

    // --- 步骤3: 对亮度通道进行直方图均衡化 ---
    // 通过拉伸最频繁出现的强度值来改善对比度。
    // 仅应用于亮度通道(Y)以避免色彩失真。
    // 适用于光照不足或对比度低的图像。
    cv::Mat ycc, equalized;
    cv::cvtColor(crop, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> channels;
    cv::split(ycc, channels);
    cv::equalizeHist(channels[0], channels[0]); // 仅均衡化Y通道
    cv::merge(channels, ycc);
    cv::cvtColor(ycc, equalized, cv::COLOR_YCrCb2BGR);
    cv::imwrite("/tmp/equalized.jpg", equalized);
    std::cout << "[3] Histogram equalization -> /tmp/equalized.jpg\n";

    // --- 步骤4: Canny边缘检测 ---
    // 从图像中提取结构边缘。可用作额外的
    // 输入通道，或用于图像分割、边界检测等任务。
    // 阈值(100, 200)控制灵敏度。
    cv::Mat gray, edges;
    cv::cvtColor(equalized, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 100, 200);
    cv::imwrite("/tmp/edges.jpg", edges);
    std::cout << "[4] Canny edge detection -> /tmp/edges.jpg\n";

    std::cout << "\nAll outputs saved to /tmp/ (resized.jpg, crop.jpg, "
              << "equalized.jpg, edges.jpg)\n";

    return 0;
}
