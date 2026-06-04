/*
 * audio_video.cpp
 * 第2章: C++中的数据准备与预处理
 *
 * 音频预处理将原始波形转换为深度学习模型可以处理的
 * 频域表示(频谱图)。频谱图揭示了原始音频中不可见的时频模式。
 *
 * 技术: 短时傅里叶变换(STFT)频谱图
 *   1. 将信号分帧为重叠的窗口
 *   2. 应用窗函数(Hann窗)以减少频谱泄漏
 *   3. 对每一帧计算DFT
 *   4. 转换为对数幅度(dB刻度)以获得感知相关性
 *
 * 参数:
 *   N = DFT大小(512): 频率分辨率; 越大频率分箱越精细
 *   H = 跳跃大小(160): 帧之间的步长; 越小时间分辨率越精细
 *   B = N/2 + 1: 频率分箱数(实信号DFT对称)
 *
 * 注意: 本实现使用朴素的O(N^2) DFT以保证可移植性。
 * 生产环境中请安装FFTW3: sudo apt install libfftw3-dev
 * 并使用fftw_plan_dft_r2c_1d获得O(N log N)性能。
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

// ----------------------------------------------------------------
// 生成合成测试信号: 正弦波 + 噪声
// ----------------------------------------------------------------
std::vector<double> generateTestSignal(int numSamples, double sampleRate) {
    std::vector<double> signal(numSamples);
    for (int i = 0; i < numSamples; ++i) {
        double t = (double)i / sampleRate;
        // 440 Hz (A4音符)，振幅0.8
        double tone = 0.8 * std::sin(2.0 * M_PI * 440.0 * t);
        // 添加一些噪声
        double noise = 0.05 * ((double)rand() / RAND_MAX - 0.5) * 2.0;
        signal[i] = tone + noise;
    }
    return signal;
}

// ----------------------------------------------------------------
// 朴素DFT(用于可移植性; O(N^2)复杂度)
// 生产环境中，请使用FFTW: fftw_plan_dft_r2c_1d获得O(N log N)性能
// ----------------------------------------------------------------
std::vector<std::complex<double>> computeDFT(
    const std::vector<double> &x) {
    size_t N = x.size();
    std::vector<std::complex<double>> result(N);
    for (size_t k = 0; k < N; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            double angle = -2.0 * M_PI * k * n / N;
            sum += x[n] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        result[k] = sum;
    }
    return result;
}

// ----------------------------------------------------------------
// 使用STFT计算对数幅度频谱图
// 返回二维向量: [时间帧][频率分箱]
// ----------------------------------------------------------------
std::vector<std::vector<double>> computeSpectrogram(
    const std::vector<double> &x,
    int N, // DFT大小
    int H  // 跳跃大小(帧之间的步长)
) {
    const int B = N / 2 + 1; // 频率分箱数(实信号)

    // Hann窗: 平滑帧边缘以减少频谱泄漏
    // w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
    std::vector<double> win(N);
    for (int n = 0; n < N; ++n) {
        win[n] = 0.5 * (1.0 - std::cos(2.0 * M_PI * n / (N - 1)));
    }

    // 时间帧数
    int T = (int)((x.size() + H - 1) / H);

    // 频谱图矩阵: T个时间帧 × B个频率分箱
    std::vector<std::vector<double>> spectrogram(
        T, std::vector<double>(B, 0.0));

    std::vector<double> frame(N, 0.0);
    for (int t = 0; t < T; ++t) {
        size_t start = (size_t)t * H;

        // 应用窗函数并提取帧
        for (int n = 0; n < N; ++n) {
            double sample = (start + n < x.size()) ? x[start + n] : 0.0;
            frame[n] = sample * win[n];
        }

        // 计算此帧的DFT
        auto dftResult = computeDFT(frame);

        // 转换为dB刻度: 20 * log10(|X| + epsilon)
        // 只保留前B个分箱(实信号是对称的)
        for (int k = 0; k < B; ++k) {
            double magnitude = std::abs(dftResult[k]);
            // dB刻度; epsilon=1e-12避免log(0)
            spectrogram[t][k] = 20.0 * std::log10(magnitude + 1e-12);
        }
    }

    return spectrogram;
}

// 辅助函数: 计算时间平均频谱
std::vector<double> timeAveragedSpectrum(
    const std::vector<std::vector<double>> &spec) {
    if (spec.empty()) return {};
    size_t T = spec.size();
    size_t B = spec[0].size();
    std::vector<double> avg(B, 0.0);
    for (size_t t = 0; t < T; ++t)
        for (size_t b = 0; b < B; ++b)
            avg[b] += spec[t][b];
    for (size_t b = 0; b < B; ++b)
        avg[b] /= (double)T;
    return avg;
}

int main() {
    std::cout << "=== Audio Spectrogram Demo (Portable DFT) ===\n\n";

    // 参数
    const double sampleRate = 16000.0; // 16 kHz (典型语音采样率)
    const int durationSec = 1;         // 1秒音频
    const int numSamples = (int)(sampleRate * durationSec);
    const int N = 512;       // DFT大小
    const int H = 160;       // 跳跃大小 (16kHz下约10ms)
    const int B = N / 2 + 1; // 频率分箱数

    std::cout << "Parameters: Fs=" << sampleRate << " Hz, "
              << "N=" << N << " (DFT size), "
              << "H=" << H << " (hop size)\n";
    std::cout << "Frequency bins: " << B
              << " (0 to " << sampleRate / 2 << " Hz)\n";
    std::cout << "Note: Using O(N^2) DFT for portability. Install FFTW3\n"
              << "      for O(N log N) production performance.\n\n";

    // 生成测试信号
    std::cout << "Generating 1-second 440 Hz tone with noise...\n";
    srand(42);
    auto signal = generateTestSignal(numSamples, sampleRate);

    // 计算频谱图
    std::cout << "Computing STFT spectrogram...\n";
    auto spec = computeSpectrogram(signal, N, H);

    int T = (int)spec.size();
    std::cout << "Spectrogram shape: [" << T << " time frames x "
              << B << " freq bins]\n";

    // 打印前几个时间帧
    std::cout << "\nSpectrogram (first 5 frames, first 6 bins):\n";
    for (int t = 0; t < std::min(T, 5); ++t) {
        std::cout << "  t=" << t << ": [";
        for (int k = 0; k < std::min(B, 6); ++k) {
            if (spec[t][k] > -80.0)
                std::cout << (int)spec[t][k] << " dB";
            else
                std::cout << "-inf";
            if (k + 1 < std::min(B, 6)) std::cout << ", ";
        }
        std::cout << "...]\n";
    }

    // 时间平均频谱(应在440 Hz处显示峰值)
    std::cout << "\nTime-averaged spectrum (peak detection):\n";
    auto avgSpec = timeAveragedSpectrum(spec);
    double maxMag = -200.0;
    int maxBin = 0;
    for (int k = 1; k < B; ++k) { // 跳过直流分量 (k=0)
        if (avgSpec[k] > maxMag) {
            maxMag = avgSpec[k];
            maxBin = k;
        }
    }
    double freqResolution = sampleRate / N; // 每个分箱对应的Hz数
    double detectedFreq = maxBin * freqResolution;
    std::cout << "  Peak at bin " << maxBin
              << " -> ~" << detectedFreq << " Hz"
              << " (expected 440 Hz, resolution "
              << freqResolution << " Hz)\n";

    std::cout << "\nUse case: Spectrograms are input to audio models\n"
              << "  (speech recognition, music classification).\n";

    return 0;
}
