// lecture12_part2.cpp
// DRAM Simulator: Banks, Row Buffer, Burst Mode, Memory Controller
// Models DRAM internals as described in Lecture 12
// Stanford CS149, Fall 2025 - Lecture 12: Mapping AI to the AI Datacenter

#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>

// Time in nanoseconds
using Time = double;

// DRAM timing parameters (DDR4-like)
struct DRAMTiming {
    double tRC_ns = 45.0;     // Row cycle time (PRE + RAS + CAS)
    double tRAS_ns = 32.0;    // Row access strobe
    double tRP_ns = 13.0;     // Row precharge time
    double tCAS_ns = 13.0;    // Column access strobe
    double tBURST_ns = 4.0;   // Burst transfer (8 beats)
    double tRRD_ns = 6.0;     // Row activate to row activate delay
    double tFAW_ns = 30.0;    // Four activate window
    double tCCD_ns = 5.0;     // Column to column delay
    int burstLength = 8;      // Number of data beats per column access
    int dataBusWidth = 8;     // Bits per DRAM chip
    int numBanks = 8;         // Banks per DRAM chip
    int rowsPerBank = 16384;  // Rows per bank
    int colsPerRow = 1024;    // Columns per row (each col = burstLength * dataBusWidth bits)
};

// A single DRAM bank
class DRAMBank {
public:
    DRAMBank(int bankId, const DRAMTiming& timing)
        : bankId_(bankId), timing_(timing),
          rowBuffer_(timing.colsPerRow * timing.burstLength * timing.dataBusWidth / 8, 0),
          openRow_(-1), rowBufferValid_(false),
          busyUntil_(0.0), stats_{0, 0} {}

    // Service a read request to this bank
    // Returns the latency in ns
    Time read(int row, int col, Time currentTime) {
        Time startTime = std::max(currentTime, busyUntil_);
        Time latency = 0.0;

        if (openRow_ == row && rowBufferValid_) {
            // Row buffer hit: only need CAS
            latency = timing_.tCAS_ns + timing_.tBURST_ns;
            stats_.rowHits++;
        } else {
            // Row buffer miss: need PRE (if row open) + RAS + CAS
            if (rowBufferValid_) {
                // Write back current row buffer
                latency += timing_.tRP_ns;  // Precharge
            }
            latency += timing_.tRAS_ns;      // Activate new row
            latency += timing_.tCAS_ns;      // Column access
            latency += timing_.tBURST_ns;    // Burst transfer

            openRow_ = row;
            rowBufferValid_ = true;
            stats_.rowMisses++;
        }

        busyUntil_ = startTime + latency;
        stats_.totalRequests++;
        return latency;
    }

    void precharge(Time currentTime) {
        if (rowBufferValid_) {
            busyUntil_ = std::max(currentTime, busyUntil_) + timing_.tRP_ns;
            rowBufferValid_ = false;
            openRow_ = -1;
        }
    }

    bool isReady(Time currentTime) const {
        return busyUntil_ <= currentTime;
    }

    int openRow() const { return openRow_; }
    bool rowBufferValid() const { return rowBufferValid_; }
    int bankId() const { return bankId_; }

    struct Stats {
        long long totalRequests = 0;
        long long rowHits = 0;
        long long rowMisses = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    int bankId_;
    DRAMTiming timing_;
    std::vector<uint8_t> rowBuffer_;
    int openRow_;
    bool rowBufferValid_;
    Time busyUntil_;
    Stats stats_;
};

// Memory request from LLC
struct MemRequest {
    int id;
    int bankId;
    int row;
    int col;
    Time arrivalTime;
    Time completionTime;
    bool isRead;
};

// Memory Controller with FR-FCFS scheduling
// FR-FCFS: First-Ready, First-Come-First-Serve
// 1. Service requests to currently OPEN ROW first (maximize row locality)
// 2. Service other requests in FIFO order
class MemoryController {
public:
    MemoryController(int numBanks, const DRAMTiming& timing)
        : timing_(timing), currentTime_(0.0) {
        for (int b = 0; b < numBanks; ++b) {
            banks_.emplace_back(b, timing);
        }
    }

    // Submit a request to the controller
    void submitRequest(int bankId, int row, int col, bool isRead = true) {
        requestQueue_.push_back({nextReqId_++, bankId, row, col, currentTime_, 0.0, isRead});
    }

    // Process all pending requests
    void processAll() {
        while (!requestQueue_.empty()) {
            // FR-FCFS: prioritize row-buffer hits
            auto hitIt = requestQueue_.end();
            auto firstIt = requestQueue_.begin();

            // Find first request that is a row-buffer hit
            for (auto it = requestQueue_.begin(); it != requestQueue_.end(); ++it) {
                int b = it->bankId;
                if (banks_[b].isReady(currentTime_) &&
                    banks_[b].rowBufferValid() &&
                    banks_[b].openRow() == it->row) {
                    hitIt = it;
                    break;
                }
            }

            // Find first request to any ready bank (FIFO order)
            auto readyIt = requestQueue_.end();
            if (hitIt == requestQueue_.end()) {
                for (auto it = requestQueue_.begin(); it != requestQueue_.end(); ++it) {
                    if (banks_[it->bankId].isReady(currentTime_)) {
                        readyIt = it;
                        break;
                    }
                }
            }

            auto chosenIt = (hitIt != requestQueue_.end()) ? hitIt : readyIt;

            if (chosenIt == requestQueue_.end()) {
                // No request can be serviced now; advance time to next bank ready
                Time nextReady = 1e9;
                for (auto& b : banks_) {
                    if (!b.isReady(currentTime_)) {
                        nextReady = std::min(nextReady, b.isReady(currentTime_) ?
                                             currentTime_ : currentTime_ + 1.0);
                    }
                }
                currentTime_ = std::max(currentTime_ + 1.0, nextReady);
                continue;
            }

            // Service the chosen request
            int b = chosenIt->bankId;
            Time latency = banks_[b].read(chosenIt->row, chosenIt->col, currentTime_);
            currentTime_ += latency;
            chosenIt->completionTime = currentTime_;
            completedRequests_.push_back(*chosenIt);
            requestQueue_.erase(chosenIt);
        }
        totalTime_ = currentTime_;
    }

    void printStats() const {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Memory Controller Stats (FR-FCFS Scheduling)\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "Total time: " << std::fixed << std::setprecision(1)
                  << totalTime_ << " ns\n";
        std::cout << "Requests completed: " << completedRequests_.size() << "\n\n";

        std::cout << std::left
                  << std::setw(10) << "Bank"
                  << std::setw(14) << "Requests"
                  << std::setw(10) << "Row Hits"
                  << std::setw(12) << "Row Misses"
                  << std::setw(10) << "Hit Rate\n";
        std::cout << std::string(56, '-') << "\n";

        long long totalReqs = 0, totalHits = 0, totalMisses = 0;
        for (const auto& bank : banks_) {
            auto& s = bank.stats();
            double hitRate = s.totalRequests > 0 ?
                100.0 * s.rowHits / s.totalRequests : 0.0;
            std::cout << std::left
                      << std::setw(10) << bank.bankId()
                      << std::setw(14) << s.totalRequests
                      << std::setw(10) << s.rowHits
                      << std::setw(12) << s.rowMisses
                      << std::fixed << std::setprecision(1) << hitRate << "%\n";
            totalReqs += s.totalRequests;
            totalHits += s.rowHits;
            totalMisses += s.rowMisses;
        }

        std::cout << std::string(56, '-') << "\n";
        double overallHitRate = totalReqs > 0 ? 100.0 * totalHits / totalReqs : 0.0;
        std::cout << std::left
                  << std::setw(10) << "TOTAL"
                  << std::setw(14) << totalReqs
                  << std::setw(10) << totalHits
                  << std::setw(12) << totalMisses
                  << std::fixed << std::setprecision(1) << overallHitRate << "%\n\n";
    }

    double effectiveBandwidth() const {
        if (totalTime_ == 0) return 0.0;
        double totalBytes = completedRequests_.size() *
                            timing_.burstLength * timing_.dataBusWidth / 8.0;
        return totalBytes / (totalTime_ * 1e-9);  // Bytes/sec
    }

    void printTimingBreakdown() const {
        std::cout << "Timing Parameters:\n";
        std::cout << "  Row cycle (tRC):    " << timing_.tRC_ns << " ns\n";
        std::cout << "  Row activate (tRAS): " << timing_.tRAS_ns << " ns\n";
        std::cout << "  Precharge (tRP):    " << timing_.tRP_ns << " ns\n";
        std::cout << "  Column access (tCAS): " << timing_.tCAS_ns << " ns\n";
        std::cout << "  Burst transfer:     " << timing_.tBURST_ns << " ns ("
                  << timing_.burstLength << " beats)\n\n";

        std::cout << "Best case latency (row hit):  CAS + Burst = "
                  << (timing_.tCAS_ns + timing_.tBURST_ns) << " ns\n";
        std::cout << "Worst case latency (row miss): PRE + RAS + CAS + Burst = "
                  << (timing_.tRP_ns + timing_.tRAS_ns + timing_.tCAS_ns + timing_.tBURST_ns)
                  << " ns\n\n";
    }

    double totalTime() const { return totalTime_; }

private:
    DRAMTiming timing_;
    std::vector<DRAMBank> banks_;
    std::vector<MemRequest> requestQueue_;
    std::vector<MemRequest> completedRequests_;
    Time currentTime_;
    Time totalTime_;
    int nextReqId_ = 0;
};

// Simulate different access patterns
void simulateSequentialAccess(MemoryController& mc, int numRequests) {
    // Sequential access: consecutive columns in same row → high hit rate
    for (int i = 0; i < numRequests; ++i) {
        int bank = i % 8;
        int row = i / 1024;
        int col = i % 1024;
        mc.submitRequest(bank, row, col);
    }
}

void simulateRandomAccess(MemoryController& mc, int numRequests,
                          std::mt19937& rng) {
    // Random access: random rows → low hit rate
    std::uniform_int_distribution<int> bankDist(0, 7);
    std::uniform_int_distribution<int> rowDist(0, 16383);
    std::uniform_int_distribution<int> colDist(0, 1023);

    for (int i = 0; i < numRequests; ++i) {
        mc.submitRequest(bankDist(rng), rowDist(rng), colDist(rng));
    }
}

void simulateStridedAccess(MemoryController& mc, int numRequests, int stride) {
    // Strided access: skip `stride` rows each time → mixed hit rate
    for (int i = 0; i < numRequests; ++i) {
        int bank = (i * stride) % 8;
        int row = (i * stride / 8) % 16384;
        int col = i % 1024;
        mc.submitRequest(bank, row, col);
    }
}

// Energy cost analysis (from lecture data)
void analyzeEnergyCost() {
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Data Movement Energy Cost Analysis\n";
    std::cout << "══════════════════════════════════════════════════════════════\n\n";

    struct EnergyEntry {
        std::string operation;
        double energy_pJ;
        std::string note;
    };

    std::vector<EnergyEntry> costs = {
        {"FP32 math op",             0.9,  "45nm CMOS (Han, ICLR 2016)"},
        {"Local SRAM access",        5.0,  "On-chip, ~1mm distance"},
        {"Load 32b from LPDDR",      640.0,"Off-chip DRAM access"},
        {"Read 64b from SRAM",       26.0, "On-chip, Bill Dally numbers"},
        {"Read 64b from LPDDR",      1200.0,"Off-chip, mobile DRAM"},
        {"Read 10 GB/s from DRAM",   1.6,"~1.6 watts total (per second)"},
    };

    std::cout << std::left
              << std::setw(30) << "Operation"
              << std::setw(15) << "Energy"
              << "Note\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& c : costs) {
        std::cout << std::left
                  << std::setw(30) << c.operation
                  << std::setw(15) << std::fixed << std::setprecision(1)
                  << c.energy_pJ << " pJ"
                  << c.note << "\n";
    }

    std::cout << "\nKey ratios:\n";
    std::cout << "  DRAM/SRAM energy ratio: " << 1200.0/26.0 << "x\n";
    std::cout << "  DRAM/FP32 compute ratio: " << 640.0/0.9 << "x\n";
    std::cout << "\nImplication: recomputing values is often cheaper than\n";
    std::cout << "storing and reloading them from DRAM!\n\n";
}

int main() {
    std::cout << "=== Lecture 12: DRAM Simulator ===\n";
    std::cout << "Stanford CS149 - Mapping AI to the AI Datacenter\n\n";

    DRAMTiming timing;
    std::random_device rd;
    std::mt19937 rng(rd());

    // Scenario 1: Sequential access (good locality)
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Scenario 1: Sequential Access (high locality)\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        MemoryController mc(timing.numBanks, timing);
        simulateSequentialAccess(mc, 64);
        mc.processAll();
        mc.printStats();
        std::cout << "Effective bandwidth: " << std::fixed << std::setprecision(1)
                  << mc.effectiveBandwidth() / 1e9 << " GB/s\n\n";
        mc.printTimingBreakdown();
    }

    // Scenario 2: Random access (poor locality)
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Scenario 2: Random Access (low locality)\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        MemoryController mc(timing.numBanks, timing);
        simulateRandomAccess(mc, 64, rng);
        mc.processAll();
        mc.printStats();
        std::cout << "Effective bandwidth: " << std::fixed << std::setprecision(1)
                  << mc.effectiveBandwidth() / 1e9 << " GB/s\n\n";

        std::cout << "Comparison: sequential vs random\n";
        std::cout << "  Sequential: high row-buffer hit rate → low latency\n";
        std::cout << "  Random: row-buffer thrashing → high latency, low BW\n";
        std::cout << "  FR-FCFS helps by prioritizing open-row requests\n\n";
    }

    // Scenario 3: Model real-world GPU memory access patterns
    {
        std::cout << "══════════════════════════════════════════════════════════════\n";
        std::cout << "Scenario 3: GPU-style Memory Access\n";
        std::cout << "══════════════════════════════════════════════════════════════\n\n";

        std::cout << "CPU Memory (DRAM):\n";
        std::cout << "  - 64-bit memory bus per channel\n";
        std::cout << "  - DDR4 2400: 19.2 GB/s per channel, 2 channels ≈ 38.4 GB/s\n";
        std::cout << "  - ~13 ns CAS latency\n\n";

        std::cout << "GPU Memory (HBM):\n";
        std::cout << "  - H100: 6 HBM3 stacks × 1024-bit = 6144-bit interface\n";
        std::cout << "  - Peak BW: 3.2 TB/s (83x wider than dual-channel DDR4!)\n";
        std::cout << "  - 3D stacked DRAM: TSV connections through chips\n";
        std::cout << "  - Silicon interposer for high-BW interconnect\n\n";

        std::cout << "DIMM Organization:\n";
        std::cout << "  - 8 DRAM chips → 64-bit bus (one rank)\n";
        std::cout << "  - Physical addresses interleaved across chips at byte granularity\n";
        std::cout << "  - 64-byte cache line: 8 bursts × 64 bits across all chips\n";
        std::cout << "  - Memory controller maps physical address → bank, row, column\n\n";

        std::cout << "HBM Advantages:\n";
        std::cout << "  - More bandwidth: 1024-bit per stack (vs 64-bit DDR4)\n";
        std::cout << "  - Higher power efficiency: shorter wires, less capacitance\n";
        std::cout << "  - Smaller form factor: 3D stacking reduces PCB area\n";
        std::cout << "  - 94% less energy per bit vs GDDR5 (AMD estimate)\n\n";
    }

    // Energy cost analysis
    analyzeEnergyCost();

    // Summary
    std::cout << "══════════════════════════════════════════════════════════════\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. DRAM latency varies with row buffer state (hit vs miss)\n";
    std::cout << "2. FR-FCFS scheduling prioritizes open-row requests for throughput\n";
    std::cout << "3. Multiple banks enable request pipelining → higher pin utilization\n";
    std::cout << "4. HBM: 3D stacking + wide interfaces → 3.2 TB/s on H100\n";
    std::cout << "5. Data movement dominates energy: DRAM access ~700x cost of FP32 op\n";
    std::cout << "6. Key principle: bring data closer to processor, move less data\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    return 0;
}
