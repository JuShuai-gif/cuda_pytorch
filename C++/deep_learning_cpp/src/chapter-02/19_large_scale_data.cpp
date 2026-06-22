/*
 * large_scale_data.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * PDF 参考：第 73-74 页，"管理大规模数据集"
 *
 * 当数据集超出可用 RAM 时，必须重新设计预处理以支持核外处理。
 * PDF 中的技术：
 *   - 分片（Sharding）：将数据拆分为较小文件，以便并行 I/O
 *   - 缓存（Caching）：保存中间预处理结果以避免重复计算
 *   - 流式处理（Streaming）：按块处理数据以限制内存使用
 *   - mmap：零拷贝文件访问（参见 memory_mapping.cpp）
 *
 * 本文件使用合成数据演示分片（拆分 + 读取）和流式处理（按块处理）。
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <sys/stat.h>

// ----------------------------------------------------------------
// 分片：将大型数据集拆分为 N 个分片文件。
// 每个分片自包含，可独立加载。
// 优势：并行 I/O、分布式训练、容错能力。
//
// PDF 第 73 页："分片将数据集拆分为较小的块，
// 这些块可以独立或跨工作线程并行处理。"
// ----------------------------------------------------------------
void createShards(const std::string &basename, int numShards,
                  int samplesPerShard, int numFeatures) {
    struct stat st;
    if (stat("/tmp/shard_data", &st) != 0) {
        mkdir("/tmp/shard_data", 0755);
    }

    for (int s = 0; s < numShards; ++s) {
        std::ostringstream path;
        path << "/tmp/shard_data/" << basename << "_shard_"
             << std::setw(2) << std::setfill('0') << s << ".csv";

        std::ofstream out(path.str());
        for (int i = 0; i < samplesPerShard; ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                // 基于分片和位置生成确定性数据
                double val = (s * 1000.0 + i * 10.0 + j) / 100.0;
                out << val;
                if (j + 1 < numFeatures) out << ",";
            }
            out << "\n";
        }
    }
    std::cout << "  Created " << numShards << " shards, "
              << samplesPerShard << " samples each -> "
              << numShards * samplesPerShard << " total samples\n";
}

// ----------------------------------------------------------------
// 流式加载器：每次产生一个块，内存使用有界。
// 一次处理一部分行——从不加载整个数据集。
//
// PDF 第 73 页："流式处理以固定大小的块处理数据，
// 无论数据集总大小如何，保持内存使用恒定。"
// ----------------------------------------------------------------
class StreamLoader {
public:
    StreamLoader(const std::string &basename, int numShards,
                 int chunkSize) : basename_(basename), numShards_(numShards),
                                  chunkSize_(chunkSize), currentShard_(0) {
    }

    // 加载下一个块；完成后返回空 vector
    std::vector<std::vector<double>> nextChunk() {
        std::vector<std::vector<double>> chunk;
        chunk.reserve(chunkSize_);

        while (chunk.size() < (size_t)chunkSize_) {
            if (!currentStream_.is_open()) {
                if (currentShard_ >= numShards_) break;
                openNextShard();
            }

            std::string line;
            while (chunk.size() < (size_t)chunkSize_ && std::getline(currentStream_, line)) {
                std::vector<double> row;
                std::istringstream iss(line);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    row.push_back(std::stod(token));
                }
                chunk.push_back(row);
                totalLoaded_++;
            }

            if (currentStream_.eof()) {
                currentStream_.close();
            }
        }
        return chunk;
    }

    size_t totalLoaded() const {
        return totalLoaded_;
    }

private:
    void openNextShard() {
        std::ostringstream path;
        path << "/tmp/shard_data/" << basename_ << "_shard_"
             << std::setw(2) << std::setfill('0')
             << currentShard_++ << ".csv";
        currentStream_.open(path.str());
        if (!currentStream_.is_open()) {
            std::cerr << "WARN: Cannot open shard " << path.str() << "\n";
        }
    }

    std::string basename_;
    int numShards_;
    int chunkSize_;
    int currentShard_;
    std::ifstream currentStream_;
    size_t totalLoaded_ = 0;
};

// ----------------------------------------------------------------
// 简单缓存：保存/加载中间预处理结果。
// 避免在不同运行中重复计算昂贵的转换操作。
//
// PDF 第 74 页："缓存中间预处理结果以避免
// 在不同实验中重复计算。"
// ----------------------------------------------------------------
void demoCaching() {
    std::cout << "[Caching] Save intermediate results to avoid recomputation.\n";

    const std::string cachePath = "/tmp/preprocess_cache.bin";

    // 模拟昂贵的预处理操作并缓存结果
    {
        std::vector<double> raw = {1.0, 2.0, 3.0, 4.0, 5.0};
        // "昂贵"的操作：计算均值和标准差
        double mean = std::accumulate(raw.begin(), raw.end(), 0.0) / raw.size();
        double variance = 0.0;
        for (auto v : raw) variance += (v - mean) * (v - mean);
        variance /= raw.size();
        double stddev = std::sqrt(variance);

        std::ofstream cache(cachePath, std::ios::binary);
        cache.write(reinterpret_cast<const char *>(&mean), sizeof(mean));
        cache.write(reinterpret_cast<const char *>(&stddev), sizeof(stddev));
        cache.close();

        std::cout << "  Cached: mean=" << mean << ", stddev=" << stddev << "\n";
    }

    // 之后：从缓存加载，无需重新计算
    {
        double mean = 0, stddev = 0;
        std::ifstream cache(cachePath, std::ios::binary);
        cache.read(reinterpret_cast<char *>(&mean), sizeof(mean));
        cache.read(reinterpret_cast<char *>(&stddev), sizeof(stddev));

        std::cout << "  Loaded from cache: mean=" << mean
                  << ", stddev=" << stddev << "\n";
        std::cout << "  Pattern: compute once, reuse across experiments.\n";
    }

    std::remove(cachePath.c_str());
}

int main() {
    std::cout << "=== Large-Scale Data Management ===\n";
    std::cout << "PDF pages 73-74: Sharding, caching, streaming\n\n";

    const std::string basename = "data";
    const int numShards = 4;
    const int samplesPerShard = 500;
    const int numFeatures = 5;
    const int chunkSize = 200;

    // ===========================================
    // 1. 分片
    // ===========================================
    std::cout << "[Sharding] Split data into independent files.\n";
    createShards(basename, numShards, samplesPerShard, numFeatures);
    std::cout << "  Each shard can be loaded by a different worker.\n";
    std::cout << "  Parallel I/O for distributed training.\n\n";

    // ===========================================
    // 2. 流式处理
    // ===========================================
    std::cout << "[Streaming] Process in chunks, constant memory.\n";
    std::cout << "  Chunk size: " << chunkSize << " rows\n";

    StreamLoader loader(basename, numShards, chunkSize);

    int chunkNum = 0;
    long runningSum = 0;
    while (true) {
        auto chunk = loader.nextChunk();
        if (chunk.empty()) break;

        // 模拟处理：累加所有值的总和
        for (const auto &row : chunk) {
            for (double val : row) {
                runningSum += (long)val;
            }
        }
        chunkNum++;
    }

    std::cout << "  Processed " << chunkNum << " chunks, "
              << loader.totalLoaded() << " total rows\n";
    std::cout << "  Memory used: ~chunk_size rows (vs loading all "
              << numShards * samplesPerShard << " rows)\n\n";

    // ===========================================
    // 3. 缓存
    // ===========================================
    demoCaching();

    // ===========================================
    // 4. 总结
    // ===========================================
    std::cout << "\n--- Large-Scale Data Patterns (PDF p73-74) ---\n";
    std::cout << "  Sharding:  Split -> parallel I/O -> merge\n";
    std::cout << "  Streaming: Load chunk -> process -> discard -> next\n";
    std::cout << "  Caching:   Preprocess once -> cache -> reuse\n";
    std::cout << "  mmap:      Map file -> access on demand (see memory_mapping.cpp)\n";
    std::cout << "  Validation: Check schema before training (see advanced_tech.cpp)\n";

    // 清理
    for (int s = 0; s < numShards; ++s) {
        std::ostringstream path;
        path << "/tmp/shard_data/" << basename << "_shard_"
             << std::setw(2) << std::setfill('0') << s << ".csv";
        std::remove(path.str().c_str());
    }

    return 0;
}
