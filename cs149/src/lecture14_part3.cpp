// lecture14_part3.cpp — CS149 Lecture 14: Directory-Based Cache Coherence Simulation
// Simulates a simple directory-based coherence protocol (vs. snooping).
// Compile: g++ -std=c++17 -O2 lecture14_part3.cpp -o lecture14_part3
// Run:     ./lecture14_part3

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <iomanip>
#include <cassert>

// ============================================================================
// Snooping vs Directory: Conceptual Comparison
// ============================================================================

// In snooping-based protocols, ALL coherence messages are broadcast to ALL caches.
// Every cache controller "snoops" the bus and checks every transaction.
//
// In directory-based protocols, the directory (centralized or distributed) tracks
// which caches hold each cache line. Coherence messages are sent ONLY to caches
// that actually contain the line — point-to-point messages, not broadcasts.

// ============================================================================
// Directory entry states (simplified, similar to MSI on directory)
// ============================================================================
enum class DirState {
    U,   // Uncached: no processor has the line
    S,   // Shared: one or more processors have read-only copies
    M    // Modified: exactly one processor has a dirty copy
};

const char* dir_state_str(DirState s) {
    switch (s) {
        case DirState::U: return "U (Uncached)";
        case DirState::S: return "S (Shared)";
        case DirState::M: return "M (Modified)";
    }
    return "?";
}

// ============================================================================
// Directory entry for a single cache line
// ============================================================================
struct DirEntry {
    DirState state = DirState::U;
    int owner = -1;                   // which core holds the line in M (if any)
    std::set<int> sharers;            // set of cores holding S copies
};

// ============================================================================
// Per-core cache line (simplified)
// ============================================================================
struct CoreLine {
    bool valid = false;
    int tag = -1;
    DirState state = DirState::U;   // local state mirrors directory
    int data = 0;
    bool dirty = false;
};

// ============================================================================
// Core (processor + private cache)
// ============================================================================
class Core {
public:
    Core(int id) : id_(id) {
        lines_.resize(4);  // 4 direct-mapped cache lines
    }

    int id() const { return id_; }

    // Read request: returns true if needs directory lookup
    bool read(int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr) {
            log("read hit, state=" + std::string(dir_state_str(line.state)));
            return false;  // hit, no directory request needed
        }
        log("read miss → need directory lookup");
        return true;
    }

    // Write request
    bool write(int addr, int val) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr && line.state == DirState::M) {
            line.data = val;
            log("write hit (M), no directory request");
            return false;
        }
        log("write miss or upgrade → need directory lookup");
        return true;
    }

    // Directory tells core to load line in given state
    void load_line(int addr, DirState st, int data) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        line.valid = true;
        line.tag = addr;
        line.state = st;
        line.data = data;
        line.dirty = (st == DirState::M);
        log("loaded line, state=" + std::string(dir_state_str(st)));
    }

    // Invalidate this line (snoop-like, but received as point-to-point msg)
    void invalidate(int addr) {
        int idx = addr % lines_.size();
        auto& line = lines_[idx];
        if (line.valid && line.tag == addr) {
            log("invalidate request received → I");
            line.valid = false;
            if (line.dirty)
                log("  (dirty data returned to directory)");
        }
    }

    DirState line_state(int addr) const {
        int idx = addr % lines_.size();
        const auto& line = lines_[idx];
        return (line.valid && line.tag == addr) ? line.state : DirState::U;
    }

    int line_data(int addr) const {
        int idx = addr % lines_.size();
        return lines_[idx].tag == addr ? lines_[idx].data : -1;
    }

private:
    void log(const std::string& msg) const {
        // Uncomment for verbose tracing:
        // std::cout << "  [Core " << id_ << "] " << msg << std::endl;
        (void)msg;
    }
    int id_;
    std::vector<CoreLine> lines_;
};

// ============================================================================
// Directory: centralized coherence controller
// Tracks per-cache-line state: which cores have it, in what state.
// Sends point-to-point invalidation/forward messages (NO broadcasts).
// ============================================================================
class Directory {
public:
    Directory(int num_cores, int num_lines)
        : num_cores_(num_cores) {
        entries_.resize(num_lines);
    }

    // Handle a read request from a core
    // In a real system, this is a message exchange:
    //   core → directory (request)
    //   directory → memory/owner (forward)
    //   directory → core (response with data)
    int handle_read(int core_id, int addr, int memory_val,
                    std::vector<Core>& cores, bool verbose) {
        auto& entry = entries_[addr % entries_.size()];

        if (verbose) {
            std::cout << "  Dir[" << addr << "]: state=" << dir_state_str(entry.state);
            if (entry.owner >= 0)
                std::cout << ", owner=P" << entry.owner;
            if (!entry.sharers.empty()) {
                std::cout << ", sharers={";
                bool first = true;
                for (int s : entry.sharers) {
                    if (!first) std::cout << ",";
                    std::cout << "P" << s;
                    first = false;
                }
                std::cout << "}";
            }
            std::cout << std::endl;
        }

        switch (entry.state) {
            case DirState::U:
                // No one has it — load from memory, enter S
                entry.state = DirState::S;
                entry.sharers.insert(core_id);
                cores[core_id].load_line(addr, DirState::S, memory_val);
                if (verbose) std::cout << "    → U→S, data from Memory" << std::endl;
                return memory_val;

            case DirState::S:
                // One or more have it shared — add to sharers, get from memory
                entry.sharers.insert(core_id);
                cores[core_id].load_line(addr, DirState::S, memory_val);
                if (verbose) std::cout << "    → remain S, data from Memory, added P" << core_id << " as sharer" << std::endl;
                return memory_val;

            case DirState::M: {
                // Another core has it modified!
                // 1. Tell owner to supply data and downgrade to S
                int owner = entry.owner;
                int dirty_data = cores[owner].line_data(addr);
                cores[owner].load_line(addr, DirState::S, dirty_data);  // downgrade M→S
                // 2. Update directory: M→S, add both to sharers
                entry.state = DirState::S;
                entry.owner = -1;
                entry.sharers.insert(owner);
                entry.sharers.insert(core_id);
                // 3. Give data to requester
                cores[core_id].load_line(addr, DirState::S, dirty_data);
                if (verbose) std::cout << "    → M→S (downgrade P" << owner << "), data from P" << owner << " cache" << std::endl;
                return dirty_data;
            }
        }
        return memory_val;
    }

    // Handle a write request from a core
    int handle_write(int core_id, int addr, int val, int memory_val,
                     std::vector<Core>& cores, bool verbose) {
        auto& entry = entries_[addr % entries_.size()];

        if (verbose) {
            std::cout << "  Dir[" << addr << "]: state=" << dir_state_str(entry.state);
            if (entry.owner >= 0)
                std::cout << ", owner=P" << entry.owner;
            if (!entry.sharers.empty()) {
                std::cout << ", sharers={";
                bool first = true;
                for (int s : entry.sharers) {
                    if (!first) std::cout << ",";
                    std::cout << "P" << s;
                    first = false;
                }
                std::cout << "}";
            }
            std::cout << std::endl;
        }

        switch (entry.state) {
            case DirState::U:
                // No one has it — give exclusive access
                entry.state = DirState::M;
                entry.owner = core_id;
                cores[core_id].load_line(addr, DirState::M, val);
                if (verbose) std::cout << "    → U→M, P" << core_id << " is owner" << std::endl;
                return val;

            case DirState::S: {
                // Invalidate all sharers (point-to-point messages, not broadcast!)
                for (int s : entry.sharers) {
                    if (s != core_id) {
                        cores[s].invalidate(addr);
                        if (verbose) std::cout << "    → invalidate P" << s << " (point-to-point)" << std::endl;
                    }
                }
                entry.sharers.clear();
                entry.state = DirState::M;
                entry.owner = core_id;
                cores[core_id].load_line(addr, DirState::M, val);
                if (verbose) std::cout << "    → S→M, invalidated all sharers" << std::endl;
                return val;
            }

            case DirState::M: {
                // Another core owns it — invalidate owner, then give to requester
                int old_owner = entry.owner;
                if (old_owner != core_id) {
                    int dirty_data = cores[old_owner].line_data(addr);
                    cores[old_owner].invalidate(addr);
                    entry.owner = core_id;
                    cores[core_id].load_line(addr, DirState::M, val);
                    if (verbose) std::cout << "    → M→M (invalidate P" << old_owner << "), new owner P" << core_id << std::endl;
                    return val;
                }
                // Same core already owns it — just write
                cores[core_id].load_line(addr, DirState::M, val);
                return val;
            }
        }
        return val;
    }

    void print_state(int addr, std::vector<Core>& cores) const {
        const auto& entry = entries_[addr % entries_.size()];
        std::cout << "  Directory[" << addr << "]: " << dir_state_str(entry.state);
        if (entry.owner >= 0)
            std::cout << " owner=P" << entry.owner;
        if (!entry.sharers.empty()) {
            std::cout << " sharers={";
            for (auto it = entry.sharers.begin(); it != entry.sharers.end(); ++it) {
                if (it != entry.sharers.begin()) std::cout << ",";
                std::cout << "P" << *it;
            }
            std::cout << "}";
        }
        std::cout << std::endl;
        for (size_t i = 0; i < cores.size(); ++i) {
            auto st = cores[i].line_state(addr);
            std::cout << "  P" << i << " cache: " << dir_state_str(st);
            if (st != DirState::U)
                std::cout << " data=" << cores[i].line_data(addr);
            std::cout << std::endl;
        }
    }

private:
    int num_cores_;
    std::vector<DirEntry> entries_;
};

// ============================================================================
// Demo: Directory-based coherence with the same example as MSI lecture
// ============================================================================
void run_directory_example() {
    std::cout << "=== CS149 Lecture 14: Directory-Based Cache Coherence ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Unlike snooping, directory sends coherence messages ONLY to" << std::endl;
    std::cout << "caches that actually contain the line (point-to-point)." << std::endl;
    std::cout << "No broadcast — scales to many cores." << std::endl;
    std::cout << std::endl;

    const int NUM_CORES = 3;
    const int NUM_LINES = 4;

    std::vector<Core> cores;
    for (int i = 0; i < NUM_CORES; ++i)
        cores.emplace_back(i);

    Directory dir(NUM_CORES, NUM_LINES);
    int memory = 0;
    int addr = 0;

    std::cout << "Initial state:" << std::endl;
    dir.print_state(addr, cores);
    std::cout << std::endl;

    auto do_read = [&](int core_id, bool verbose) {
        std::cout << "P" << core_id << " read X" << std::endl;
        bool need_dir = cores[core_id].read(addr);
        if (need_dir) {
            dir.handle_read(core_id, addr, memory, cores, verbose);
        } else {
            std::cout << "  (cache hit)" << std::endl;
        }
        dir.print_state(addr, cores);
        std::cout << std::endl;
    };

    auto do_write = [&](int core_id, int val, bool verbose) {
        std::cout << "P" << core_id << " write X = " << val << std::endl;
        bool need_dir = cores[core_id].write(addr, val);
        if (need_dir) {
            dir.handle_write(core_id, addr, val, memory, cores, verbose);
        } else {
            std::cout << "  (cache hit, M state)" << std::endl;
        }
        dir.print_state(addr, cores);
        std::cout << std::endl;
    };

    // Same sequence as the MSI lecture example
    do_read(0, true);                    // P1 read X  → U→S, memory
    do_read(2, true);                    // P3 read X  → add to S
    do_write(2, 42, true);               // P3 write 42 → S→M, invalidate P1
    do_read(0, true);                    // P1 read X  → M→S, downgrade P3
    do_read(0, true);                    // P1 read X  → hit in S
    do_write(1, 99, true);               // P2 write 99 → invalidate P1 and P3, M

    std::cout << "================================================================" << std::endl;
    std::cout << "Comparison: Snooping vs Directory" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Property"
              << std::setw(30) << "Snooping"
              << "Directory" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::left
              << std::setw(20) << "Message delivery"
              << std::setw(30) << "Broadcast to all"
              << "Point-to-point" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Scalability"
              << std::setw(30) << "Limited (bus BW)"
              << "Scales well" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Bus requirement"
              << std::setw(30) << "Ordered broadcast bus"
              << "Any interconnect" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Storage overhead"
              << std::setw(30) << "None (stateless)"
              << "P-bit vector per line" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Serialization point"
              << std::setw(30) << "Bus"
              << "Directory" << std::endl;
    std::cout << std::left
              << std::setw(20) << "Example"
              << std::setw(30) << "Older SMP systems"
              << "Intel Core i7 (L3 dir)" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main() {
    run_directory_example();
    return 0;
}
