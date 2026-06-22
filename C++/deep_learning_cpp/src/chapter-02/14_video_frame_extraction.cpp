/*
 * video_frame_extraction.cpp
 * 第2章: C++中的数据准备与预处理
 *
 * PDF参考: 第67页, "非结构化数据 — 图像、音频和视频"
 *
 * 视频预处理从视频文件中提取单独帧，用于动作识别、
 * 目标跟踪和视频分类等深度学习任务。关键技术:
 *   - 帧采样: 以固定间隔提取帧
 *   - 跳帧: 每N帧处理一帧以减少冗余
 *   - 时间裁剪: 提取特定时间段
 *
 * 依赖: OpenCV 4.x (sudo apt install libopencv-dev)
 *
 * 用法: ./video_frame_extraction [video_file]
 *   如果未提供文件，则生成合成测试视频。
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>

// ----------------------------------------------------------------
// 从视频中以固定间隔提取帧。
// step=1: 每帧都提取; step=30: 每秒提取一帧(假设30fps)。
// 返回提取的帧文件路径列表。
// ----------------------------------------------------------------
int extractFrames(const std::string &videoPath, int step,
                  const std::string &outDir,
                  int maxFrames = 100) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open video: " << videoPath << "\n";
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double duration = totalFrames / fps;

    std::cout << "Video: " << videoPath << "\n";
    std::cout << "  FPS: " << fps << ", Total frames: " << totalFrames
              << ", Duration: " << std::fixed << std::setprecision(1)
              << duration << "s\n";
    std::cout << "  Sampling every " << step << " frame(s)";
    if (fps > 0 && step > 0)
        std::cout << " (~every "
                  << std::setprecision(2) << step / fps << "s)";
    std::cout << "\n\n";

    cv::Mat frame;
    int frameIdx = 0;
    int saved = 0;

    while (cap.read(frame) && saved < maxFrames) {
        if (frameIdx % step == 0) {
            std::ostringstream oss;
            oss << outDir << "/frame_" << std::setw(4)
                << std::setfill('0') << saved << ".jpg";
            cv::imwrite(oss.str(), frame);
            saved++;
        }
        frameIdx++;
    }

    cap.release();
    std::cout << "  Saved " << saved << " frames to " << outDir << "/\n";
    return saved;
}

// ----------------------------------------------------------------
// 生成合成视频(彩色移动矩形)用于演示
// ----------------------------------------------------------------
void generateDemoVideo(const std::string &path, int numFrames,
                       int width, int height, double fps) {
    cv::VideoWriter writer(path,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "ERROR: Cannot create video file\n";
        return;
    }

    for (int i = 0; i < numFrames; ++i) {
        cv::Mat frame(height, width, CV_8UC3, cv::Scalar(30, 30, 30));

        // 移动的彩色矩形
        int x = (i * 5) % (width - 80);
        int y = (height - 60) / 2;
        cv::Scalar color(
            (i * 3) % 255,      // B
            (i * 5 + 85) % 255, // G
            (i * 7 + 170) % 255 // R
        );
        cv::rectangle(frame,
                      cv::Point(x, y),
                      cv::Point(x + 80, y + 60),
                      color, cv::FILLED);

        // 帧计数器文本
        cv::putText(frame,
                    "Frame " + std::to_string(i),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255, 255, 255), 2);

        writer.write(frame);
    }
    writer.release();
    std::cout << "Generated demo video: " << path
              << " (" << numFrames << " frames)\n";
}

int main(int argc, char **argv) {
    std::cout << "=== Video Frame Extraction Demo (OpenCV) ===\n\n";
    std::cout << "PDF page 67: Unstructured data preprocessing for video\n\n";

    std::string videoPath;
    bool isDemo = false;

    if (argc >= 2) {
        videoPath = argv[1];
    } else {
        // 生成演示视频
        isDemo = true;
        videoPath = "/tmp/demo_video.avi";
        generateDemoVideo(videoPath, 90, 320, 240, 30.0);
    }

    // 创建输出目录
    std::string outDir = "/tmp/video_frames";
    std::system(("mkdir -p " + outDir).c_str());

    // 每10帧提取一帧(对于30fps视频，约3帧/秒)
    int step = 10;
    int count = extractFrames(videoPath, step, outDir);

    if (count < 0) return 1;

    std::cout << "\n--- Frame Extraction Strategies ---\n";
    std::cout << "  Uniform sampling: every Nth frame (used above)\n";
    std::cout << "  Keyframe extraction: I-frames from compressed video\n";
    std::cout << "  Scene change detection: frames where content changes significantly\n";
    std::cout << "  Temporal cropping: extract frames from [start_time, end_time]\n";

    if (isDemo) {
        std::remove(videoPath.c_str());
    }

    return 0;
}
