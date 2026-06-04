/*
 * advanced_tech.cpp
 * 第2章: C++中的数据准备与预处理
 *
 * 面向生产级深度学习流水线的高级预处理技术。
 * 这些技术超越了基础的数据清洗和缩放，解决了
 * 大规模、分布式和实时系统中的挑战。
 *
 * 涵盖主题:
 *   - 数据增强: 生成合成训练样本。
 *     * 数值型: 高斯噪声、特征随机丢弃
 *     * 图像: 高斯噪声 (基于OpenCV, PDF第71页)
 *     * 文本: Token随机丢弃 (PDF第71页)
 *   - 分层采样: 在训练/验证/测试集划分中保持类别分布。
 *     对不平衡数据集至关重要，随机划分可能导致
 *     稀有类别在验证/测试集中没有代表样本。
 *   - 数据版本化与模式验证: 跟踪预处理流水线
 *     版本与模型版本以保障可复现性。在喂入模型前
 *     验证输入模式。
 *   - 流水线缓存: 缓存中间预处理结果以避免
 *     跨实验的重复计算。
 *
 * 何时应用:
 *   - 增强: 当训练数据有限时(图像<10k样本,
 *     表格数据<1k)。对计算机视觉至关重要，对文本/音频也很有用。
 *   - 分层划分: 对于类别不平衡的分类问题始终使用。
 *   - 版本化: 需要模型和预处理完全可复现的生产系统。
 */

#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

// ================================================================
// 1. 数据增强 (表格数据)
// ================================================================

// 向数值特征添加高斯噪声。
// 模拟测量误差，使模型对微小扰动具有鲁棒性。
// noise_std: 噪声相对于特征尺度的标准差。
// 归一化数据使用0.01-0.05; 原始数据使用更高值。
std::vector<std::vector<double>> addGaussianNoise(
    const std::vector<std::vector<double>> &data, double noiseStd) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, noiseStd);

    auto augmented = data;
    for (auto &row : augmented) {
        for (auto &val : row) {
            val += dist(gen);
        }
    }
    return augmented;
}

// 特征随机丢弃: 随机将某些特征置为0。
// 类似于神经网络中的dropout，但应用于输入层面。
// 强制模型不依赖任何单一特征。
std::vector<std::vector<double>> featureDropout(
    const std::vector<std::vector<double>> &data, double dropProb) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto augmented = data;
    for (auto &row : augmented) {
        for (auto &val : row) {
            if (dist(gen) < dropProb) val = 0.0;
        }
    }
    return augmented;
}

// ================================================================
// 1b. 文本增强: Token随机丢弃 (PDF第71页)
// ================================================================

// 随机从序列中丢弃token以提高对缺失/噪声词的鲁棒性。
// 在NLP增强流水线中很常见。
// dropProb: 丢弃每个token的概率 (通常0.1-0.3)。
std::vector<std::string> tokenDropout(
    const std::vector<std::string> &tokens, double dropProb) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution keep(1.0 - dropProb);

    std::vector<std::string> result;
    for (const auto &token : tokens) {
        if (keep(gen)) {
            result.push_back(token);
        }
    }
    return result;
}

// ================================================================
// 1c. 图像增强: 高斯噪声 (PDF第71页, 基于OpenCV)
// ================================================================

// 此函数需要OpenCV。使用#ifdef保护以避免
// 硬依赖。请安装OpenCV以启用。
// 向每个像素独立添加高斯噪声。
// 均值=0, 标准差=10-25 对于uint8图像 [0,255] 是典型值。

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>

cv::Mat addGaussianNoise(const cv::Mat &image, double mean, double stddev) {
    cv::Mat noise(image.size(), CV_32F);
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(
        static_cast<float>(mean),
        static_cast<float>(stddev));

    for (int i = 0; i < noise.rows; ++i) {
        for (int j = 0; j < noise.cols; ++j) {
            noise.at<float>(i, j) = distribution(generator);
        }
    }

    cv::Mat noisyImage;
    image.convertTo(noisyImage, CV_32F);
    noisyImage += noise;

    // 钳制到有效范围
    cv::Mat result;
    cv::threshold(noisyImage, result, 255.0, 255.0, cv::THRESH_TRUNC);
    cv::threshold(result, result, 0.0, 0.0, cv::THRESH_TOZERO);
    result.convertTo(result, image.type());

    return result;
}
#endif // HAS_OPENCV

// ================================================================
// 2. 分层训练/测试集划分
// ================================================================

// 在保持类别比例的同时将索引划分为训练/测试集。
// 确保每个类别以相同比例出现在两个划分中。
// 对于不平衡数据集至关重要(例如欺诈检测中
// 欺诈案例占数据<1%)。
std::pair<std::vector<size_t>, std::vector<size_t>>
stratifiedSplit(const std::vector<int> &labels, double testRatio) {
    // 按类别分组索引
    std::map<int, std::vector<size_t>> classIndices;
    for (size_t i = 0; i < labels.size(); ++i) {
        classIndices[labels[i]].push_back(i);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<size_t> trainIdx, testIdx;
    for (auto &[cls, indices] : classIndices) {
        std::shuffle(indices.begin(), indices.end(), gen);
        size_t testCount = (size_t)(indices.size() * testRatio);
        if (testCount < 1 && indices.size() > 1) testCount = 1;

        for (size_t i = 0; i < testCount; ++i)
            testIdx.push_back(indices[i]);
        for (size_t i = testCount; i < indices.size(); ++i)
            trainIdx.push_back(indices[i]);
    }

    std::shuffle(trainIdx.begin(), trainIdx.end(), gen);
    std::shuffle(testIdx.begin(), testIdx.end(), gen);

    return {trainIdx, testIdx};
}

// ================================================================
// 3. 简单数据验证
// ================================================================

// 在喂入模型前验证数据批次
struct DataValidator {
    size_t expectedFeatures;
    double minVal, maxVal;

    bool validate(const std::vector<std::vector<double>> &batch) {
        for (const auto &row : batch) {
            if (row.size() != expectedFeatures) {
                std::cerr << "ERROR: Expected " << expectedFeatures
                          << " features, got " << row.size() << "\n";
                return false;
            }
            for (const auto &val : row) {
                if (std::isnan(val) || std::isinf(val)) {
                    std::cerr << "ERROR: NaN/Inf detected\n";
                    return false;
                }
                if (val < minVal || val > maxVal) {
                    std::cerr << "WARNING: Value " << val
                              << " outside [" << minVal << ", "
                              << maxVal << "]\n";
                }
            }
        }
        return true;
    }
};

// 辅助函数: 打印二维数据
void print2D(const std::string &label,
             const std::vector<std::vector<double>> &mat,
             int maxRows = 4) {
    std::cout << label << ":\n";
    for (size_t i = 0; i < std::min(mat.size(), (size_t)maxRows); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < mat[i].size(); ++j) {
            std::cout << mat[i][j];
            if (j + 1 < mat[i].size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    if (mat.size() > (size_t)maxRows) std::cout << "  ...\n";
}

int main() {
    std::cout << "=== Advanced Preprocessing Techniques ===\n\n";

    // 样本数据: 10个样本 × 3个特征
    std::vector<std::vector<double>> X = {
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 4.0},
        {3.0, 4.0, 5.0},
        {4.0, 5.0, 6.0},
        {5.0, 6.0, 7.0},
        {6.0, 7.0, 8.0},
        {7.0, 8.0, 9.0},
        {8.0, 9.0, 10.0},
        {9.0, 10.0, 11.0},
        {10.0, 11.0, 12.0}};
    std::vector<int> labels = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2};

    // --- 1. 数据增强 ---
    std::cout << "[Data Augmentation] Generate synthetic samples.\n";
    print2D("Original data (first 3)", X, 3);

    auto noisy = addGaussianNoise(X, 0.05);
    std::cout << "\n   After Gaussian noise (std=0.05), first 3:\n";
    for (int i = 0; i < 3; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 3; ++j) {
            std::cout << noisy[i][j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n   When: small datasets, CV tasks (flip/rotate/crop images).\n";
    std::cout << "   Avoid: When transformations change the label (e.g., flipping\n";
    std::cout << "          '6' to '9' in digit recognition).\n\n";

    // --- 2. 分层划分 ---
    std::cout << "[Stratified Split] Maintain class proportions.\n";
    auto [train, test] = stratifiedSplit(labels, 0.3);

    auto countClass = [&](const std::vector<size_t> &indices) {
        std::map<int, int> counts;
        for (auto idx : indices) counts[labels[idx]]++;
        return counts;
    };

    std::cout << "  Total: " << labels.size() << " samples, "
              << "classes: {0:3, 1:3, 2:4}\n";
    std::cout << "  Train: " << train.size() << " samples, classes: ";
    auto trainCounts = countClass(train);
    for (auto [c, n] : trainCounts)
        std::cout << c << ":" << n << " ";
    std::cout << "\n  Test:  " << test.size() << " samples, classes: ";
    auto testCounts = countClass(test);
    for (auto [c, n] : testCounts)
        std::cout << c << ":" << n << " ";
    std::cout << "\n\n  Critical for imbalanced datasets! Without stratification,\n";
    std::cout << "  a random split could miss rare classes in test set.\n\n";

    // --- 3. 数据验证 ---
    std::cout << "[Data Validation] Check schema before model ingestion.\n";
    DataValidator validator{3, -100.0, 100.0};
    bool ok = validator.validate(X);
    std::cout << "  Validation result: " << (ok ? "PASS" : "FAIL") << "\n";
    std::cout << "  Checks: feature count, NaN/Inf, value range.\n";
    std::cout << "  Use for: catching data pipeline bugs before training.\n\n";

    // --- 4. 特征随机丢弃 ---
    std::cout << "[Feature Dropout] Randomly zero out features (dropProb=0.3).\n";
    auto dropped = featureDropout(X, 0.3);
    std::cout << "  First 4 rows after dropout:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "    [";
        for (int j = 0; j < 3; ++j) {
            std::cout << dropped[i][j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "  Forces model to not rely on any single feature.\n\n";

    // --- 5. Token随机丢弃 (NLP增强) ---
    std::cout << "[Token Dropout] Randomly drop tokens (dropProb=0.3).\n";
    std::vector<std::string> tokens = {
        "the", "deep", "learning", "model", "processes", "images", "efficiently"};
    auto droppedTokens = tokenDropout(tokens, 0.3);
    std::cout << "  Before: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n  After:  [";
    for (size_t i = 0; i < droppedTokens.size(); ++i) {
        std::cout << droppedTokens[i];
        if (i + 1 < droppedTokens.size()) std::cout << ", ";
    }
    std::cout << "]\n\n";

    // --- 6. 图像增强 (基于OpenCV) ---
#ifdef HAS_OPENCV
    std::cout << "[Image Gaussian Noise] Add noise to image pixels.\n";
    std::cout << "  See addGaussianNoise() above. Usage:\n";
    std::cout << "    cv::Mat noisy = addGaussianNoise(img, 0.0, 10.0);\n";
    std::cout << "  Denoising autoencoders use this for training.\n";
#else
    std::cout << "[Image Gaussian Noise] (needs OpenCV)\n";
    std::cout << "  PDF p71 shows how to add Gaussian noise to images\n";
    std::cout << "  for denoising autoencoders and robustness training.\n";
#endif

    return 0;
}
