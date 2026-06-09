# System Bottleneck Identification - Robot Workloads

Realistic system bottleneck benchmarks using robot perception and control workloads instead of synthetic counters.

## File Structure

```
03_system_bottleneck/
├── CMakeLists.txt
├── README.md
├── timer.h              # High-resolution Timer utility
├── cache_bench.h        # Kalman filter false sharing + image cache thrashing
├── cache_bench.cpp      # Implementations with real matrix math
├── lock_bench.h         # NMS lock contention demo
├── lock_bench.cpp       # Spinlock, mutex, lock-free partitioned NMS
├── memory_bench.h       # Camera frame memory copy overhead
├── memory_bench.cpp     # memcpy vs zero-copy vs ring buffer
└── main.cpp             # Entry point + JSON writer
```

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./bottleneck_demos
```

Outputs `bottleneck_metrics.json` with benchmark descriptions and results.

## Benchmarks

1. **False Sharing**: Two threads run Kalman filter predict on 8 separate object tracks. Unpadded layout causes cache line ping-pong; padded layout eliminates false sharing.

2. **Lock Contention**: NMS (Non-Maximum Suppression) on 800 detection boxes under spinlock, std::mutex, and lock-free partitioned approaches. Demonstrates scalability differences of synchronization primitives under real robot perception workloads.

3. **Cache Thrashing**: 3x3 box blur on a 640x480x3 image comparing row-major (cache-friendly) vs column-major (cache-hostile) traversal. Shows why memory access patterns matter for image processing.

4. **Memory Copy Overhead**: Copies of 1920x1080x3 camera frames (~6.2 MB) comparing std::memcpy, zero-copy pointer swap, and camera ring buffer with swap semantics. Demonstrates why zero-copy design is essential for multi-camera pipelines.
