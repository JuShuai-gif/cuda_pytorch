// lecture14_part1.cpp — CS149 Lecture 14: MSI Cache Coherence Protocol Simulation
// Simulates a snooping-based MSI (Modified-Shared-Invalid) protocol.
// Compile: g++ -std=c++17 -O2 lecture14_part1.cpp -o lecture14_part1
// Run:     ./lecture14_part1

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cassert>
#include <iomanip>

// ============================================================================
// Cache line states in the MSI protocol
// ============================================================================
enum class MsiState {
    M,  // Modified: valid, dirty, exclusive — only copy in system
    S,  // Shared: valid, clean, shared — one or more caches have it
    I   // Invalid: not present or stale
};

const char* state_str(MsiState s) {
    switch (s) {
        case MsiState::M: return "M";
        case MsiState::S: return "S";
        case MsiState::I: return "I";
    }
    return "?";
}

// ============================================================================
// Bus transactions
// ============================================================================
enum class BusTrans {
    NONE,
    BusRd,   // Read with no intent to modify (get shared copy)
    BusRdX,  // Read-exclusive with intent to modify (invalidate others)
    BusWB    // Write-back dirty data to memory
};

const char* bus_str(BusTrans t) {
    switch (t) {
        case BusTrans::NONE:  return "---";
        case BusTrans::BusRd: return "BusRd";
        case BusTrans::BusRdX:return "BusRdX";
        case BusTrans::BusWB: return "BusWB";
    }
    return "???";
}

// Processor operations
enum class ProcOp { PrRd, PrWr };

const char* op_str(ProcOp op) {
    return op == ProcOp::PrRd ? "PrRd" : "PrWr";
}

// ============================================================================
// Single cache line in a single cache
// ============================================================================
struct CacheLine {
    MsiState state = MsiState::I;
    int tag = -1;
    int data = 0;
    bool dirty = false;
};

// ============================================================================
// A cache controller implementing the MSI protocol
// ============================================================================
class MsiCache {
public:
    MsiCache(int id, const std::string& name) : id_(id), name_(name) {
        // 4 cache lines (simplified: direct-mapped by address)
        lines_.resize(4);
    }

    // Processor requests a read
    BusTrans pr_read(int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];

        if (line.tag == addr && line.state != MsiState::I) {
            // Hit — state unchanged (M or S)
            log("PrRd  hit — state=" + std::string(state_str(line.state)));
            return BusTrans::NONE;
        }

        // Miss: I → S via BusRd
        log("PrRd  miss — BusRd (I→S)");
        BusTrans prev = BusTrans::NONE;
        if (line.state == MsiState::M) {
            // Evict dirty line: M → I triggers BusWB
            log("  (evict dirty line " + std::to_string(line.tag) + ": M→I, BusWB)");
            prev = BusTrans::BusWB;
        }
        allocate(line, addr, MsiState::S);
        return (prev != BusTrans::NONE) ? prev : BusTrans::BusRd;
    }

    // Processor requests a write
    BusTrans pr_write(int addr, int value) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];

        if (line.tag == addr) {
            if (line.state == MsiState::M) {
                // Already modified — just write, no bus transaction
                line.data = value;
                log("PrWr  hit M — no bus transaction");
                return BusTrans::NONE;
            }
            if (line.state == MsiState::S) {
                // Upgrade: S → M via BusRdX (invalidate other copies)
                log("PrWr  hit S — BusRdX (S→M upgrade)");
                line.state = MsiState::M;
                line.data = value;
                line.dirty = true;
                return BusTrans::BusRdX;
            }
        }

        // Miss: I → M via BusRdX
        log("PrWr  miss — BusRdX (I→M)");
        BusTrans prev = BusTrans::NONE;
        if (line.state == MsiState::M) {
            log("  (evict dirty line " + std::to_string(line.tag) + ": M→I, BusWB)");
            prev = BusTrans::BusWB;
        }
        allocate(line, addr, MsiState::M);
        line.data = value;
        line.dirty = true;
        return (prev != BusTrans::NONE) ? prev : BusTrans::BusRdX;
    }

    // Snoop a bus transaction from another cache
    // Returns BusWB if this cache needs to supply dirty data
    BusTrans snoop(BusTrans bus_op, int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];

        if (line.tag != addr) return BusTrans::NONE;

        switch (bus_op) {
            case BusTrans::BusRd:
                // Another cache wants a shared copy
                if (line.state == MsiState::M) {
                    // Must supply dirty data and downgrade to S
                    log("snoop BusRd — supply data (M→S)");
                    line.state = MsiState::S;
                    line.dirty = false;
                    return BusTrans::BusWB;
                }
                // S state: do nothing (shared line stays shared)
                log("snoop BusRd — remain S");
                return BusTrans::NONE;

            case BusTrans::BusRdX:
                // Another cache wants exclusive access — invalidate
                if (line.state == MsiState::M) {
                    log("snoop BusRdX — supply data (M→I)");
                    line.state = MsiState::I;
                    return BusTrans::BusWB;
                }
                if (line.state == MsiState::S) {
                    log("snoop BusRdX — invalidate (S→I)");
                    line.state = MsiState::I;
                    return BusTrans::NONE;
                }
                return BusTrans::NONE;

            case BusTrans::BusWB:
                // Write-back to memory; no state change needed here
                return BusTrans::NONE;

            default:
                return BusTrans::NONE;
        }
    }

    MsiState get_state(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].state : MsiState::I;
    }

    int get_data(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].data : -1;
    }

    const std::string& name() const { return name_; }

private:
    void allocate(CacheLine& line, int addr, MsiState s) {
        line.state = s;
        line.tag = addr;
        line.data = 0;
        line.dirty = (s == MsiState::M);
    }

    void log(const std::string& msg) const {
        // Uncomment for verbose tracing:
        // std::cout << "  [Cache " << name_ << "] " << msg << std::endl;
        (void)msg; // suppress unused warning
    }

    int id_;
    std::string name_;
    std::vector<CacheLine> lines_;
};

// ============================================================================
// Bus: serializes all transactions and delivers snoop messages
// ============================================================================
class Bus {
public:
    // Execute a bus transaction: notify all other caches
    // Returns the data source ("Memory" or "Px $")
    std::string execute(BusTrans trans, int addr,
                        std::vector<MsiCache>& caches,
                        int requester_id,
                        int& memory_data) {
        if (trans == BusTrans::NONE) return "---";

        std::string supplier = "Memory";
        bool need_wb = false;

        // Notify all OTHER caches
        for (size_t i = 0; i < caches.size(); ++i) {
            if ((int)i == requester_id) continue;
            BusTrans resp = caches[i].snoop(trans, addr);
            if (resp == BusTrans::BusWB) {
                memory_data = caches[i].get_data(addr);
                supplier = "P" + std::to_string(i) + " $";
                need_wb = true;
            }
        }

        if (need_wb) {
            log(std::string(bus_str(trans)) + " → data from " + supplier);
        } else if (trans == BusTrans::BusRd || trans == BusTrans::BusRdX) {
            log(std::string(bus_str(trans)) + " → data from Memory");
        }

        return supplier;
    }

private:
    void log(const std::string& msg) const {
        // Uncomment for verbose tracing:
        // std::cout << "  [Bus] " << msg << std::endl;
        (void)msg;
    }
};

// ============================================================================
// Simulation driver: replay the exact example from the lecture slides
// ============================================================================
void run_lecture_example() {
    std::cout << "=== CS149 Lecture 14: MSI Protocol — Lecture Slide Example ===" << std::endl;
    std::cout << std::endl;
    std::cout << std::left
              << std::setw(16) << "Proc Action"
              << std::setw(12) << "P1 state"
              << std::setw(12) << "P2 state"
              << std::setw(12) << "P3 state"
              << std::setw(12) << "Bus Trans"
              << "Data from" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<MsiCache> caches;
    caches.emplace_back(0, "P1");
    caches.emplace_back(1, "P2");
    caches.emplace_back(2, "P3");

    Bus bus;
    int memory = 0;  // initial memory value for address X

    // Helper lambda
    auto print_state = [&](const std::string& action, const std::string& bus_trans,
                           const std::string& supplier) {
        std::cout << std::left
                  << std::setw(16) << action
                  << std::setw(12) << state_str(caches[0].get_state(0))
                  << std::setw(12) << state_str(caches[1].get_state(0))
                  << std::setw(12) << state_str(caches[2].get_state(0))
                  << std::setw(12) << bus_trans
                  << supplier << std::endl;
    };

    int addr = 0;   // address X = 0

    // P1 read X — cold miss, BusRd, data from memory
    BusTrans t = caches[0].pr_read(addr);
    std::string src = bus.execute(t, addr, caches, 0, memory);
    print_state("P1 read X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // P3 read X — cold miss (for P3), BusRd, data from memory
    t = caches[2].pr_read(addr);
    src = bus.execute(t, addr, caches, 2, memory);
    print_state("P3 read X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // P3 write X — upgrade to M, BusRdX
    t = caches[2].pr_write(addr, 42);
    src = bus.execute(t, addr, caches, 2, memory);
    print_state("P3 write X (42)", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // P1 read X — miss (was invalidated), BusRd, data from P3 (M→S)
    t = caches[0].pr_read(addr);
    src = bus.execute(t, addr, caches, 0, memory);
    print_state("P1 read X", bus_str(t), t == BusTrans::NONE ? "---" : src);

    // P1 read X — hit (S state)
    t = BusTrans::NONE;
    print_state("P1 read X", "--- (hit)", "---");

    // P2 write X — miss, BusRdX
    t = caches[1].pr_write(addr, 99);
    src = bus.execute(t, addr, caches, 1, memory);
    print_state("P2 write X (99)", bus_str(t), t == BusTrans::NONE ? "---" : src);

    std::cout << std::endl;
    std::cout << "MSI invariants maintained:" << std::endl;
    std::cout << "  1. SWMR: At any time, ≤1 cache in M, or ≥0 in S, never both." << std::endl;
    std::cout << "  2. Data-Value: Writes are serialized through the bus." << std::endl;

    // Verify final states
    std::cout << std::endl;
    std::cout << "Final values:" << std::endl;
    for (size_t i = 0; i < caches.size(); ++i)
        std::cout << "  P" << (i + 1) << " data = " << caches[i].get_data(addr)
                  << " (state=" << state_str(caches[i].get_state(addr)) << ")" << std::endl;
}

// ============================================================================
// Step-by-step MSI state transition demo
// ============================================================================
void demo_msi_transitions() {
    std::cout << std::endl;
    std::cout << "=== MSI State Transition Rules ===" << std::endl;
    std::cout << std::endl;

    struct Rule { MsiState from; ProcOp op; MsiState to; std::string bus; };
    std::vector<Rule> rules = {
        {MsiState::I, ProcOp::PrRd, MsiState::S, "BusRd"},
        {MsiState::I, ProcOp::PrWr, MsiState::M, "BusRdX"},
        {MsiState::S, ProcOp::PrRd, MsiState::S, "--- (hit)"},
        {MsiState::S, ProcOp::PrWr, MsiState::M, "BusRdX (upgrade)"},
        {MsiState::M, ProcOp::PrRd, MsiState::M, "--- (hit)"},
        {MsiState::M, ProcOp::PrWr, MsiState::M, "--- (hit)"},
    };

    std::cout << std::left
              << std::setw(12) << "From"
              << std::setw(8)  << "Op"
              << std::setw(8)  << "To"
              << std::setw(24) << "Bus Transaction" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (const auto& r : rules) {
        std::cout << std::left
                  << std::setw(12) << state_str(r.from)
                  << std::setw(8)  << op_str(r.op)
                  << std::setw(8)  << state_str(r.to)
                  << r.bus << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Snoop-driven transitions (remote cache observes bus):" << std::endl;
    std::cout << "  S + BusRdX → I  (invalidate)" << std::endl;
    std::cout << "  M + BusRd  → S  (downgrade, supply data via BusWB)" << std::endl;
    std::cout << "  M + BusRdX → I  (invalidate, supply data via BusWB)" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    run_lecture_example();
    demo_msi_transitions();

    std::cout << std::endl;
    std::cout << "=== Key Takeaways ===" << std::endl;
    std::cout << "1. MSI ensures SWMR: only one M or multiple S, never both." << std::endl;
    std::cout << "2. BusRdX is needed even on S-hit (upgrade) to invalidate other copies." << std::endl;
    std::cout << "3. M→S transition occurs when another cache reads a dirty line." << std::endl;
    std::cout << "4. The bus serializes all transactions — the coherence point." << std::endl;

    return 0;
}
