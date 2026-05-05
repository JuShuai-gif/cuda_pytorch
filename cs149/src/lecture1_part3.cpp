// lecture1_part3.cpp - Cache Simulation: LRU, Temporal & Spatial Locality
// =============================================================================
// Key concepts from CS149 Lecture 1:
//  - Memory hierarchy: L1 → L2 → L3 → DRAM (latency increases, size increases)
//  - Cache is on-chip storage that maintains a copy of a subset of memory
//  - Caches operate at the granularity of "cache lines"
//  - LRU (Least Recently Used) replacement policy
//  - Temporal locality: repeated accesses to the same address → cache hits
//  - Spatial locality: loading a cache line preloads nearby addresses
//  - Cache misses: cold miss (first access), capacity miss (cache too small),
//    conflict miss (set-associativity issues)
//  - Data movement energy costs: integer op ~1pJ, FP op ~20pJ, DRAM read ~1200pJ
//
// Compile: g++ -std=c++17 -O2 lecture1_part3.cpp -o lecture1_part3
// =============================================================================

#include <iostream>
#include <vector>
#include <list>
#include <unordered_map>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <random>
#include <chrono>
#include <cmath>

// ---------------------------------------------------------------------------
// A configurable LRU cache simulator
// ---------------------------------------------------------------------------
class LRUCache {
public:
    struct Config {
        int cache_size;       // total cache capacity in bytes
        int line_size;        // bytes per cache line
        int word_size;        // bytes per word (typically 4 for int/float)
        int access_latency;   // cycles for a hit
        int miss_penalty;     // extra cycles for a miss
    };

    struct Stats {
        int hits = 0;
        int misses = 0;
        int cold_misses = 0;
        int capacity_misses = 0;
        int total_accesses = 0;
        long long total_latency = 0; // cumulative access cost in cycles

        double hit_rate() const {
            return total_accesses > 0 
                ? static_cast<double>(hits) / total_accesses * 100.0 : 0.0;
        }
        double avg_access_time() const {
            return total_accesses > 0 
                ? static_cast<double>(total_latency) / total_accesses : 0.0;
        }
    };

    LRUCache(const Config& cfg) : config_(cfg) {
        num_lines_ = cfg.cache_size / cfg.line_size;
        std::cout << "    Cache initialized: " << num_lines_ << " lines, "
                  << cfg.cache_size << " bytes total, "
                  << cfg.line_size << " bytes/line\n";
    }

    // Access a single byte at the given address
    void access(unsigned int address) {
        unsigned int line_addr = address / config_.line_size;
        int offset = address % config_.line_size;
        (void)offset; // word-level access within line is abstracted

        stats_.total_accesses++;

        auto it = line_map_.find(line_addr);
        if (it != line_map_.end()) {
            // Cache hit! Move this line to the front of LRU list
            stats_.hits++;
            lru_list_.erase(it->second);
            lru_list_.push_front(line_addr);
            line_map_[line_addr] = lru_list_.begin();
            stats_.total_latency += config_.access_latency;
        } else {
            // Cache miss
            stats_.misses++;
            stats_.total_latency += config_.access_latency + config_.miss_penalty;

            if (static_cast<int>(lru_list_.size()) < num_lines_) {
                // Cold miss: cache not yet full
                stats_.cold_misses++;
            } else {
                // Capacity miss: need to evict the LRU line
                stats_.capacity_misses++;
                unsigned int evicted = lru_list_.back();
                lru_list_.pop_back();
                line_map_.erase(evicted);
            }

            // Insert new line at front
            lru_list_.push_front(line_addr);
            line_map_[line_addr] = lru_list_.begin();
        }
    }

    // Access a range of addresses sequentially (demonstrates spatial locality)
    void access_range(unsigned int start, unsigned int count) {
        for (unsigned int i = 0; i < count; i++) {
            access(start + i);
        }
    }

    const Stats& stats() const { return stats_; }
    const Config& config() const { return config_; }

    void reset_stats() {
        stats_ = Stats();
        lru_list_.clear();
        line_map_.clear();
    }

private:
    Config config_;
    int num_lines_;
    Stats stats_;
    std::list<unsigned int> lru_list_;               // front = most recently used
    std::unordered_map<unsigned int, 
        std::list<unsigned int>::iterator> line_map_;
};

// ---------------------------------------------------------------------------
// Cache access latency reference (Kaby Lake CPU, cycles at 4 GHz):
// L1: ~4 cycles, L2: ~12 cycles, L3: ~38 cycles, DRAM: ~248 cycles
// ---------------------------------------------------------------------------
void print_latency_reference() {
    std::cout << "    Real-world latency reference (Kaby Lake @ 4 GHz):\n";
    std::cout << "    -----------------------------------------------\n";
    std::cout << "    L1 cache hit:   ~4 cycles\n";
    std::cout << "    L2 cache hit:   ~12 cycles\n";
    std::cout << "    L3 cache hit:   ~38 cycles\n";
    std::cout << "    DRAM access:    ~248 cycles (best case)\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Demonstrate temporal and spatial locality with a cache
// Replicates the lecture examples from slides
// ---------------------------------------------------------------------------
void demo_cache_example(LRUCache& cache, 
                         const std::vector<unsigned int>& access_pattern,
                         const std::string& description) 
{
    std::cout << "    Pattern: " << description << "\n";
    std::cout << "    Access sequence: ";
    for (size_t i = 0; i < access_pattern.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << "0x" << std::hex << access_pattern[i] << std::dec;
    }
    std::cout << "\n\n";

    // Print header
    std::cout << "    " << std::setw(6) << "Addr" 
              << std::setw(6) << "Line"
              << std::setw(12) << "Result" 
              << std::setw(12) << "Cache State" << std::endl;
    std::cout << "    " << std::string(45, '-') << std::endl;

    for (unsigned int addr : access_pattern) {
        int hits_before = cache.stats().hits;
        cache.access(addr);
        bool is_hit = (cache.stats().hits > hits_before);
        
        std::cout << "    " << std::setw(6) << "0x" << std::hex << addr << std::dec
                  << std::setw(6) << (addr / cache.config().line_size)
                  << std::setw(12) << (is_hit ? "HIT" : "MISS") 
                  << "     ..." << std::endl;
    }

    std::cout << "\n    Results: " 
              << cache.stats().hits << " hits, "
              << cache.stats().misses << " misses "
              << "(" << std::fixed << std::setprecision(1) << cache.stats().hit_rate() 
              << "% hit rate)\n" << std::endl;
}

// ---------------------------------------------------------------------------
// Demonstrate energy cost of data movement
// ---------------------------------------------------------------------------
void demo_energy_costs() {
    std::cout << "[3] Energy cost of data movement\n" << std::endl;

    std::cout << "    Ballpark energy costs:\n";
    std::cout << "    -------------------------------------------------\n";
    std::cout << "    Integer operation:            ~1 pJ\n";
    std::cout << "    Floating point operation:     ~20 pJ\n";
    std::cout << "    Read 64 bits from on-chip SRAM: ~26 pJ\n";
    std::cout << "    Read 64 bits from mobile DRAM:  ~1200 pJ\n\n";

    double int_op = 1.0;
    double fp_op = 20.0;
    double sram_read = 26.0;
    double dram_read = 1200.0;

    std::cout << "    Relative cost (compared to integer op):\n";
    std::cout << "    Integer op:      1x\n";
    std::cout << "    FP op:           " << fp_op / int_op << "x\n";
    std::cout << "    SRAM read 64b:   " << sram_read / int_op << "x\n";
    std::cout << "    DRAM read 64b:   " << dram_read / int_op << "x\n" << std::endl;

    // Calculate bandwidth energy cost
    std::cout << "    Reading 10 GB/sec from memory:\n";
    std::cout << "    " << 10.0e9 / 8.0 * dram_read * 1e-12 
              << " watts (for mobile LPDDR)\n" << std::endl;

    std::cout << "    iPhone battery capacity: ~7 watt-hours\n";
    std::cout << "    → At 10 GB/s memory bandwidth, battery lasts ~4 hours\n";
    std::cout << "    → Exploiting locality matters for power!\n";
}

// =============================================================================
int main() {
    std::cout << "=== CS149 Lecture 1: Cache Simulation & Memory Hierarchy ===\n" << std::endl;

    // ---- Part 0: Latency Reference ----
    print_latency_reference();

    // ---- Part 1: Temporal & Spatial Locality (Cache Example 1) ----
    std::cout << "[1] Temporal & Spatial Locality Demonstration\n" << std::endl;

    // Configure: 8-byte cache, 4-byte lines → 2 lines (matches lecture slide)
    LRUCache::Config cfg;
    cfg.cache_size = 8;
    cfg.line_size = 4;
    cfg.word_size = 1;
    cfg.access_latency = 4;
    cfg.miss_penalty = 50;

    {
        LRUCache cache(cfg);
        
        // Lecture example 1: good spatial + temporal locality
        // Addresses 0x0-0x3 (line 0), then 0x2, 0x1 (temporal locality in line 0)
        // then 0x4-0x7 (line 1, good spatial locality)
        // then 0x1 again (still in cache)
        std::vector<unsigned int> pattern1 = {
            0x0, 0x1, 0x2, 0x3,  // line 0: cold miss, then 3 hits (spatial)
            0x2, 0x1,            // temporal locality: still hits
            0x4,                 // line 1: cold miss (capacity still OK, line 0 stays)
            0x1                  // temporal locality: hit
        };
        
        demo_cache_example(cache, pattern1, 
            "Good locality: sequential reads + repeated access");
    }

    // ---- Part 2: Capacity Misses (Cache Example 2) ----
    std::cout << "[2] Capacity Misses: sequential scan of large array\n" << std::endl;

    {
        LRUCache cache(cfg);
        
        // Lecture example 2: scan entire 16-byte array with 8-byte cache
        // First scan puts data in cache, second half of array evicts first half
        // Last access to 0x0 is a capacity miss (was evicted by 0x8)
        std::vector<unsigned int> pattern2 = {
            0x0, 0x1, 0x2, 0x3,  // line 0 loaded (cold)
            0x4, 0x5, 0x6, 0x7,  // line 1 loaded (cold)
            0x8, 0x9, 0xA, 0xB,  // line 2 loaded, evicts line 0 (capacity)
            0xC, 0xD, 0xE, 0xF,  // line 3 loaded, evicts line 1 (capacity)
            0x0                   // line 0 reloaded, evicts line 2 (capacity miss!)
        };
        
        demo_cache_example(cache, pattern2, 
            "Sequential scan → capacity misses on re-access");
    }

    // ---- Part 2b: If cache had 4 lines instead of 2 ----
    std::cout << "    [Comparison] Same pattern with 4-line cache (16 bytes):\n" << std::endl;
    {
        LRUCache::Config cfg4 = cfg;
        cfg4.cache_size = 16; // 4 lines of 4 bytes
        LRUCache cache4(cfg4);
        
        std::vector<unsigned int> pattern2 = {
            0x0, 0x1, 0x2, 0x3,
            0x4, 0x5, 0x6, 0x7,
            0x8, 0x9, 0xA, 0xB,
            0xC, 0xD, 0xE, 0xF,
            0x0
        };
        
        demo_cache_example(cache4, pattern2, 
            "Now: no capacity misses (4 lines hold all data)");
        std::cout << "    → Larger cache eliminates capacity misses\n" << std::endl;
    }

    // ---- Part 3: Energy Costs ----
    demo_energy_costs();

    // ---- Part 4: Simulating larger workloads ----
    std::cout << "[4] Large workload simulation: summing an array\n" << std::endl;

    LRUCache::Config l1_cfg;
    l1_cfg.cache_size = 32 * 1024;       // 32 KB L1 cache
    l1_cfg.line_size = 64;               // 64-byte cache lines (typical L1)
    l1_cfg.word_size = 4;                // 4-byte ints
    l1_cfg.access_latency = 4;           // L1 hit: 4 cycles
    l1_cfg.miss_penalty = 12;            // L2 hit: extra 12 cycles

    const int ARRAY_SIZE = 1'000'000;    // 1M ints = 4 MB

    // Sequential access (excellent spatial locality)
    {
        LRUCache cache(l1_cfg);
        std::cout << "    Sequential array sum (4 MB, ints):\n";
        for (int i = 0; i < ARRAY_SIZE; i++) {
            cache.access(i * 4); // each int is 4 bytes
        }
        auto& s = cache.stats();
        std::cout << "    Accesses: " << s.total_accesses 
                  << " | Hits: " << s.hits << " (" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%) | Avg latency: " << std::setprecision(1) 
                  << s.avg_access_time() << " cycles\n";
        std::cout << "    Cache line size (64B) = 16 ints per line → high spatial locality\n\n";
    }

    // Strided access (poor spatial locality)
    {
        LRUCache cache(l1_cfg);
        const int STRIDE = 256; // jump 256 ints = 1024 bytes between accesses
        std::cout << "    Strided array access (stride=" << STRIDE << " ints):\n";
        for (int i = 0; i < ARRAY_SIZE / STRIDE; i++) {
            cache.access(i * STRIDE * 4);
        }
        auto& s = cache.stats();
        std::cout << "    Accesses: " << s.total_accesses 
                  << " | Hits: " << s.hits << " (" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%) | Avg latency: " << std::setprecision(1) 
                  << s.avg_access_time() << " cycles\n";
        std::cout << "    Large stride → each access likely misses → poor locality\n\n";
    }

    // Random access (worst-case locality)
    {
        LRUCache cache(l1_cfg);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, ARRAY_SIZE - 1);
        const int NUM_ACCESSES = 100'000;
        
        std::cout << "    Random array access (" << NUM_ACCESSES << " accesses):\n";
        for (int i = 0; i < NUM_ACCESSES; i++) {
            cache.access(dist(rng) * 4);
        }
        auto& s = cache.stats();
        std::cout << "    Accesses: " << s.total_accesses 
                  << " | Hits: " << s.hits << " (" << std::fixed << std::setprecision(1) 
                  << s.hit_rate() << "%) | Avg latency: " << std::setprecision(1) 
                  << s.avg_access_time() << " cycles\n";
        std::cout << "    Random access → mostly misses → worst performance\n\n";
    }

    // ---- Part 5: Key Takeaways ----
    std::cout << "[5] Key Takeaways\n" << std::endl;
    std::cout << "    - Cache hierarchy: L1 (fast, small) → L2 → L3 → DRAM (slow, large)\n";
    std::cout << "    - Temporal locality: re-use data recently accessed\n";
    std::cout << "    - Spatial locality: access contiguous data (cache lines preload neighbors)\n";
    std::cout << "    - Cold miss: first access to data\n";
    std::cout << "    - Capacity miss: working set > cache size\n";
    std::cout << "    - LRU: least recently used eviction (common hardware policy)\n";
    std::cout << "    - Data movement dominates energy cost (~1200x more than integer ops)\n";
    std::cout << "    - Efficient programs minimize data movement → exploit locality\n";

    return 0;
}
