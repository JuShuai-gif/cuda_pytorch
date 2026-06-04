/*
 * gradcam_cnn.cpp
 * Chapter 13: Explainability and Transparency
 *
 * Grad-CAM (Gradient-weighted Class Activation Mapping) localizes
 * spatial evidence for a class in a CNN. It answers: "Where did
 * the network look when making this prediction?"
 *
 * This file covers:
 *   - SmallCNN with exposed last conv activation (retain_grad)
 *   - gradcam_map: backward from class score, channel weights, ReLU sum
 *   - OpenCV helpers: image→tensor, heatmap overlay
 *   - End-to-end demo (load image, compute CAM, save overlay)
 *
 * PDF pages: 536-542 (book pp. 536-542)
 *
 * Needs: LibTorch + OpenCV 4.x
 */

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

// ================================================================
// 1. Small CNN with exposed last conv activation (PDF p. 537)
// ================================================================

struct SmallCNN : torch::nn::Module {
    SmallCNN(int num_classes = 5) : conv1(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
                                    conv2(torch::nn::Conv2dOptions(16, 32, 3).padding(1)),
                                    fc(32, num_classes) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc", fc);
    }

    torch::Tensor last_conv_activation; // [N, 32, H, W]

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        last_conv_activation = x;           // capture for Grad-CAM
        last_conv_activation.retain_grad(); // keep gradients
        x = torch::adaptive_avg_pool2d(x, {1, 1});
        x = x.view({x.size(0), -1});
        return fc->forward(x); // logits [N, num_classes]
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc{nullptr};
};

// ================================================================
// 2. Image → Tensor conversion (PDF p. 538)
// ================================================================

torch::Tensor load_image_tensor(const std::string &path, int H = 224, int W = 224) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path);
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(W, H));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    auto t = torch::from_blob(resized.data, {H, W, 3}, torch::kFloat32).clone();
    t = t.permute({2, 0, 1}).unsqueeze(0); // [1, 3, H, W]
    return t;
}

// ================================================================
// 3. Heatmap overlay (PDF p. 538)
// ================================================================

cv::Mat overlay_heatmap(const torch::Tensor &heat01,
                        const cv::Mat &base_rgb,
                        double alpha = 0.5) {
    torch::Tensor h = heat01.clamp(0, 1).mul(255).to(torch::kU8).contiguous();
    cv::Mat heat(static_cast<int>(h.size(0)), static_cast<int>(h.size(1)),
                 CV_8U, h.data_ptr<uint8_t>());
    cv::Mat heat_color;
    cv::applyColorMap(heat, heat_color, cv::COLORMAP_JET);
    cv::Mat base_u8;
    base_rgb.convertTo(base_u8, CV_8UC3, 255.0);
    cv::Mat overlay;
    cv::addWeighted(heat_color, alpha, base_u8, 1.0 - alpha, 0.0, overlay);
    return overlay; // BGR 8-bit
}

// ================================================================
// 4. Grad-CAM map computation (PDF pp. 539-540)
// ================================================================

torch::Tensor gradcam_map(SmallCNN &model,
                          torch::Tensor input,
                          int target_class,
                          torch::Device device) {
    model.to(device);
    input = input.to(device);

    model.eval();
    auto logits = model.forward(input);           // [1, C]
    auto score = logits.index({0, target_class}); // scalar

    // Backpropagate only this class score to get gradients
    score.backward();

    // Activations and gradients at last conv block
    auto A = model.last_conv_activation.detach();           // [1, K, Hc, Wc]
    auto dAdy = model.last_conv_activation.grad().detach(); // [1, K, Hc, Wc]

    // Channel weights: α_k = mean_{h,w} grad[k, h, w]
    auto weights = dAdy.mean({2, 3}); // [1, K]

    // Weighted sum over channels + ReLU
    auto cam = torch::relu(
        torch::sum(A * weights.unsqueeze(-1).unsqueeze(-1), 1)); // [1, Hc, Wc]
    cam = cam.squeeze(0);                                        // [Hc, Wc]

    // Normalize to [0, 1]
    auto cam_min = std::get<0>(cam.min(0));
    auto cam_max = std::get<0>(cam.max(0));
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8);

    // Upsample to input size
    auto H = input.size(2);
    auto W = input.size(3);
    cam = torch::nn::functional::interpolate(
              cam.unsqueeze(0).unsqueeze(0),
              torch::nn::functional::InterpolateFuncOptions()
                  .size(std::vector<int64_t>{H, W})
                  .mode(torch::kBilinear)
                  .align_corners(false))
              .squeeze(); // [H, W]

    return cam.to(torch::kCPU);
}

// ================================================================
// 5. Print usage info
// ================================================================

void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " <image_path> <target_class>\n";
    std::cout << "Example: " << prog << " cat.jpg 3\n";
    std::cout << "\nModel uses random weights for demo.\n";
    std::cout << "Replace SmallCNN with a trained TorchScript model for real use.\n";
}

// ================================================================
// Main (PDF pp. 540-541)
// ================================================================

int main(int argc, char **argv) {
    std::cout << "=== Chapter 13: Grad-CAM ===\n\n";

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::string img_path = argv[1];
        int target_class = std::stoi(argv[2]);

        torch::Device device(
            torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Device: "
                  << (torch::cuda::is_available() ? "CUDA" : "CPU") << "\n";

        // Load image
        std::cout << "Loading image: " << img_path << "\n";
        cv::Mat bgr = cv::imread(img_path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            throw std::runtime_error("Failed to read: " + img_path);
        }
        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::Mat rgb_resized;
        cv::resize(rgb, rgb_resized, cv::Size(224, 224));
        cv::Mat rgb_f;
        rgb_resized.convertTo(rgb_f, CV_32F, 1.0 / 255.0);

        auto x = load_image_tensor(img_path, 224, 224);

        // Create model (random weights for demo)
        SmallCNN net(5);
        net.to(device);

        // Produce Grad-CAM heatmap
        std::cout << "Computing Grad-CAM for class " << target_class << "...\n";
        auto heat = gradcam_map(net, x, target_class, device);

        // Render overlay and save
        cv::Mat overlay = overlay_heatmap(heat, rgb_f, 0.5);
        std::string out_path = "gradcam_overlay.png";
        cv::imwrite(out_path, overlay);
        std::cout << "Saved: " << out_path << "\n";

        // Grad-CAM implementation notes
        std::cout << "\nGrad-CAM implementation notes:\n";
        std::cout << "  - Target layer: last convolutional block (conv2).\n";
        std::cout << "  - For real backbones, expose that activation or register a hook.\n";
        std::cout << "  - Use model.eval() to freeze batch-norm and dropout.\n";
        std::cout << "  - One backward pass adds < 10ms on GPU for common backbones.\n";
        std::cout << "  - Normalize CAM to [0,1] per image before colormapping.\n";
        std::cout << "  - Limitations: coarse (feature-map resolution), class-discriminative\n";
        std::cout << "    but doesn't explain WHY in feature space (combine with SHAP).\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }

    std::cout << "\n=== Grad-CAM demo complete ===\n";
    return 0;
}
