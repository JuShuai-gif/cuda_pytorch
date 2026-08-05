// Experiment 22: Memory mapping (read vs mmap).
//
// Creates a large temp file, then reads it sequentially and randomly using
// pread() and mmap(). Reports time for each. Does NOT touch the system page
// cache deliberately (no drop_caches); results reflect warm cache unless
// the file is large enough to exceed it.
//
// Reference: PDF 7.5, 4.x (memory mapping); note/29.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "benchmark.h"

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t FILE_SIZE = 256 * MB;

int main() {
    const char* path = "/tmp/cpu_memory_map_test.bin";
    std::printf("Experiment 22: memory mapping (file %zu MB)\n", FILE_SIZE / MB);

    // Create the file.
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }
    // Fill with a pattern via fallocate/truncate + write.
    if (ftruncate(fd, (off_t)FILE_SIZE) != 0) {
        std::perror("ftruncate");
        return 1;
    }
    // Write a small buffer repeatedly to make sure blocks are allocated.
    std::vector<char> chunk(1 << 20, 0x5a);
    for (size_t off = 0; off < FILE_SIZE; off += chunk.size())
        if (write(fd, chunk.data(), chunk.size()) < 0) {
            std::perror("write");
            return 1;
        }

    size_t npages = FILE_SIZE / (size_t)sysconf(_SC_PAGESIZE);

    // Random order of pages.
    std::vector<size_t> order(npages);
    for (size_t i = 0; i < npages; ++i) order[i] = i;
    std::mt19937 rng(42);
    std::shuffle(order.begin(), order.end(), rng);

    auto read_seq = [&] {
        off_t off = 0;
        std::vector<char> buf(1 << 20);
        while (off < (off_t)FILE_SIZE) {
            ssize_t n = pread(fd, buf.data(), buf.size(), off);
            if (n <= 0) break;
            off += n;
        }
        bm::compiler_barrier();
    };
    auto read_rand = [&] {
        char c;
        for (size_t i = 0; i < npages; ++i) {
            off_t off = (off_t)(order[i] * (size_t)sysconf(_SC_PAGESIZE));
            pread(fd, &c, 1, off);
        }
        bm::do_not_optimize(c);
    };

    void* mp = mmap(nullptr, FILE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
    auto mmap_seq = [&] {
        const char* p = (const char*)mp;
        volatile char sink = 0;
        for (size_t i = 0; i < FILE_SIZE; ++i) sink += p[i];
        bm::do_not_optimize(sink);
    };
    auto mmap_rand = [&] {
        const char* p = (const char*)mp;
        volatile char sink = 0;
        for (size_t i = 0; i < npages; ++i)
            sink += p[order[i] * (size_t)sysconf(_SC_PAGESIZE)];
        bm::do_not_optimize(sink);
    };

    struct Mode { const char* name; std::function<void()> fn; };
    Mode modes[] = {{"pread_seq", read_seq},
                    {"pread_rand", read_rand},
                    {"mmap_seq", mmap_seq},
                    {"mmap_rand", mmap_rand}};

    std::printf("%-14s %-12s\n", "mode", "time_ms");
    for (auto& m : modes) {
        m.fn();
        auto res = bm::time_rounds(3, m.fn);
        std::printf("%-14s %-12.3f\n", m.name, res.median_ms);
    }
    munmap(mp, FILE_SIZE);
    close(fd);
    unlink(path);
    return 0;
}
