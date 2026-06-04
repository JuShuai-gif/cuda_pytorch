/*
 * memory_mapping.cpp
 * 第2章：C++ 中的数据准备与预处理
 *
 * 内存映射（mmap）将文件直接映射到进程的虚拟地址空间中。
 * 这使得零拷贝 I/O 成为可能，操作系统透明地管理数据页的换入/换出，
 * 绕过用户空间缓冲区。
 *
 * 对深度学习数据管道的优势：
 *   - 零拷贝：数据直接从磁盘缓存读取到模型，无需中间缓冲区分配。
 *   - 延迟加载：只有被访问的页面才会从磁盘加载（按需分页）。
 *     非常适合大于物理内存的数据集。
 *   - 共享内存：多个进程可以共享相同的物理页面，
 *     对并行数据加载（多 worker DataLoader）很有用。
 *   - 高吞吐量：避免了重复 read() 调用的系统调用开销。
 *
 * 使用场景：在数据集太大无法装入内存时进行训练（例如视频帧、
 * 大型文本语料库）。映射文件一次，按需访问随机偏移量，
 * 无需将所有内容加载到 RAM 中。
 *
 * 注意事项：
 *   - mmap 以页粒度操作（通常为 4KB）
 *   - 文件必须在受支持的文件系统上
 *   - 无内置加密/压缩功能（可使用文件系统级别的支持）
 */

#include <iostream>
#include <string>
#include <stdexcept>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct MappedFile {
    void *data = nullptr;
    size_t size = 0;
    int fd = -1;

    // 以只读方式打开并内存映射文件
    explicit MappedFile(const std::string &path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("open failed: " + path);

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed: " + path);
        }
        size = static_cast<size_t>(st.st_size);

        // MAP_PRIVATE：写时复制（不会修改原始文件）
        // MAP_SHARED 则允许其他进程看到我们的写入
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed: " + path);
        }

        // 向内核提示我们将顺序访问此区域
        ::madvise(data, size, MADV_SEQUENTIAL);
    }

    // 禁止拷贝
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // 移动构造函数
    MappedFile(MappedFile &&other) noexcept
        : data(other.data), size(other.size), fd(other.fd) {
        other.data = nullptr;
        other.size = 0;
        other.fd = -1;
    }

    // 析构时取消映射并关闭
    ~MappedFile() {
        if (data && data != MAP_FAILED) {
            ::munmap(data, size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    // 以 const char 指针形式访问原始数据
    const char *asChars() const {
        return static_cast<const char *>(data);
    }
};

int main() {
    std::cout << "=== Memory Mapping (mmap) Demo ===\n\n";

    // 创建一个临时测试文件
    const std::string testPath = "/tmp/mmap_test_data.bin";
    {
        std::cout << "Creating test file: " << testPath << "\n";
        FILE *f = std::fopen(testPath.c_str(), "wb");
        if (!f) {
            std::cerr << "Failed to create test file\n";
            return 1;
        }
        // 写入 4000 个 float32 值（16 KB）
        const int N = 4000;
        for (int i = 0; i < N; ++i) {
            float val = (float)i * 0.1f;
            std::fwrite(&val, sizeof(float), 1, f);
        }
        std::fclose(f);
    }

    // 内存映射该文件
    try {
        MappedFile mf(testPath);

        std::cout << "File size: " << mf.size << " bytes ("
                  << mf.size / sizeof(float) << " floats)\n";
        std::cout << "Memory-mapped at address: " << mf.data << "\n";

        // 以 float 数组形式访问（零拷贝）
        const float *floats = static_cast<const float *>(mf.data);
        size_t numFloats = mf.size / sizeof(float);

        std::cout << "\nSampling data via memory map (zero-copy access):\n";
        std::cout << "  First 5: [";
        for (size_t i = 0; i < std::min(numFloats, (size_t)5); ++i) {
            std::cout << floats[i];
            if (i + 1 < std::min(numFloats, (size_t)5)) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Last 5:  [";
        for (size_t i = numFloats - 5; i < numFloats; ++i) {
            std::cout << floats[i];
            if (i + 1 < numFloats) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "\nKey benefits for DL data pipelines:\n";
        std::cout << "  - Zero-copy: data flows OS page cache -> model\n";
        std::cout << "  - Lazy loading: only accessed pages fault into RAM\n";
        std::cout << "  - Shared memory: multiple workers share same pages\n";
        std::cout << "  - Works with datasets larger than physical RAM\n";

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // 清理
    std::remove(testPath.c_str());

    return 0;
}
