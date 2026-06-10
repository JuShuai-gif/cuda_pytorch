#include <algorithm>
#include <cassert>
#include <deque>
#include <iostream>
#include <string>
#include <vector>

// CPU Pipeline Simulator inspired by LLVM scheduling model concepts.
// Models: superscalar dispatch, reservation stations, functional units,
// in-order/out-of-order execution, latency vs resource release.

// ---- Enums ----

enum class UnitKind { ALU,
                      FPU,
                      LSU,
                      DIV };

enum class CoreType { InOrder,
                      OutOfOrder };

enum class OpCode { ADD,
                    MUL,
                    FMA,
                    DIV,
                    LOAD,
                    STORE };

static const char *opcode_name(OpCode op) {
    switch (op) {
    case OpCode::ADD: return "ADD ";
    case OpCode::MUL: return "MUL ";
    case OpCode::FMA: return "FMA ";
    case OpCode::DIV: return "DIV ";
    case OpCode::LOAD: return "LOAD";
    case OpCode::STORE: return "STOR";
    }
    return "UNKN";
}

// ---- Instruction ----

struct Instruction {
    int id;
    OpCode op;
    int latency;                                          // result available after this many cycles
    std::vector<std::pair<UnitKind, int>> resource_usage; // (unit, ReleaseAtCycles)

    int dispatched_at = -1;
    int issued_at = -1;
    int completed_at = -1;
    int writeback_at = -1;

    // Dependencies: instruction IDs that must complete before this executes
    std::vector<int> raw_deps;
};

// ---- Functional Unit ----

struct FunctionalUnit {
    UnitKind kind;
    int num_pipes;   // number of parallel pipelines (ProcResource<N>)
    int buffer_size; // reservation station size (-1 = unified, 0 = in-order, 1 = latency device, >1 = decoupled)
    std::string name;

    // State tracking
    int busy_pipes = 0;
    std::deque<int> ready_queue; // instruction IDs ready to issue
};

// ---- Pipeline Simulator ----

class PipelineSimulator {
public:
    CoreType core_type;
    int issue_width;          // max uops dispatched per cycle
    int micro_op_buffer_size; // unified buffer size (for unified RS)
    int rob_size;             // reorder buffer size

    std::vector<FunctionalUnit> units;
    std::vector<Instruction> instructions;

    // State
    std::deque<int> dispatch_queue;
    std::deque<int> rob; // instruction IDs in ROB ordered by dispatch

    PipelineSimulator(CoreType ct, int iw) : core_type(ct), issue_width(iw) {
    }

    void add_unit(UnitKind kind, int num_pipes, int buffer_size, const std::string &name) {
        units.push_back({kind, num_pipes, buffer_size, name});
    }

    void add_instruction(OpCode op, int latency,
                         const std::vector<std::pair<UnitKind, int>> &res,
                         const std::vector<int> &deps) {
        int id = static_cast<int>(instructions.size());
        Instruction inst;
        inst.id = id;
        inst.op = op;
        inst.latency = latency;
        inst.resource_usage = res;
        inst.dispatched_at = -1;
        inst.issued_at = -1;
        inst.completed_at = -1;
        inst.writeback_at = -1;
        inst.raw_deps = deps;
        instructions.push_back(inst);
    }

    FunctionalUnit *find_unit(UnitKind kind) {
        for (auto &u : units) {
            if (u.kind == kind)
                return &u;
        }
        return nullptr;
    }

    bool deps_satisfied(const Instruction &inst, int cycle) {
        for (int dep_id : inst.raw_deps) {
            // Dependency satisfied if producer's result is ready
            // Result ready at: producer.issued_at + producer.latency
            if (instructions[dep_id].issued_at < 0)
                return false;
            if (instructions[dep_id].issued_at + instructions[dep_id].latency > cycle)
                return false;
        }
        return true;
    }

    bool can_dispatch(const Instruction &inst) {
        for (auto &[kind, release] : inst.resource_usage) {
            auto *unit = find_unit(kind);
            if (!unit)
                return false;
            // Check if a pipe is available
            if (unit->busy_pipes >= unit->num_pipes)
                return false;
            // For in-order units (buffer_size == 0), also check ready_queue
            if (unit->buffer_size == 0 && !unit->ready_queue.empty())
                return false;
        }
        return true;
    }

    void dispatch(Instruction &inst, int cycle) {
        inst.dispatched_at = cycle;
        for (auto &[kind, release] : inst.resource_usage) {
            auto *unit = find_unit(kind);
            unit->busy_pipes++;
            unit->ready_queue.push_back(inst.id);
        }
        if (rob_size > 0) {
            rob.push_back(inst.id);
            if (static_cast<int>(rob.size()) > rob_size) {
                // ROB full: retire oldest
                int retire_id = rob.front();
                rob.pop_front();
                instructions[retire_id].writeback_at = cycle;
            }
        }
    }

    void issue(Instruction &inst, int cycle) {
        inst.issued_at = cycle;
    }

    void release(Instruction &inst, int cycle) {
        for (auto &[kind, release_cycles] : inst.resource_usage) {
            auto *unit = find_unit(kind);
            unit->busy_pipes--;
            // Remove from ready queue
            auto &q = unit->ready_queue;
            for (auto it = q.begin(); it != q.end(); ++it) {
                if (*it == inst.id) {
                    q.erase(it);
                    break;
                }
            }
        }
        inst.completed_at = cycle;
    }

    void run() {
        int cycle = 0;
        size_t completed_count = 0;
        const size_t total = instructions.size();

        for (size_t i = 0; i < total; i++) {
            dispatch_queue.push_back(static_cast<int>(i));
        }

        while (completed_count < total) {
            // --- Release completed instructions ---
            for (auto &inst : instructions) {
                if (inst.issued_at >= 0 && inst.completed_at < 0) {
                    bool all_released = true;
                    for (auto &[kind, r] : inst.resource_usage) {
                        if (cycle < inst.issued_at + r) {
                            all_released = false;
                            break;
                        }
                    }
                    if (all_released) {
                        release(inst, cycle);
                        completed_count++;
                    }
                }
            }

            // --- Issue ready instructions ---
            for (auto &unit : units) {
                auto &q = unit.ready_queue;
                for (auto it = q.begin(); it != q.end();) {
                    int inst_id = *it;
                    auto &inst = instructions[inst_id];
                    if (inst.issued_at >= 0) {
                        ++it;
                        continue;
                    }
                    // For OoO: check dependencies. For InOrder: dispatch==issue already.
                    if (core_type == CoreType::OutOfOrder && !deps_satisfied(inst, cycle)) {
                        ++it;
                        continue;
                    }
                    issue(inst, cycle);
                    ++it;
                }
            }

            // --- Dispatch new instructions ---
            int dispatched_this_cycle = 0;
            auto it = dispatch_queue.begin();
            while (it != dispatch_queue.end() && dispatched_this_cycle < issue_width) {
                int inst_id = *it;
                auto &inst = instructions[inst_id];

                if (!can_dispatch(inst)) {
                    ++it;
                    continue;
                }
                // In-order: stall dispatch if any unit dependency not satisfied
                if (core_type == CoreType::InOrder && !deps_satisfied(inst, cycle)) {
                    break; // stop dispatching this cycle (in-order stall)
                }

                dispatch(inst, cycle);
                dispatched_this_cycle++;
                if (core_type == CoreType::InOrder) {
                    // In-order: issue immediately
                    issue(inst, cycle);
                }
                it = dispatch_queue.erase(it);
            }

            cycle++;
            if (cycle > 200) {
                std::cerr << "Deadlock at cycle " << cycle << "\n";
                break;
            }
        }

        std::cout << "Simulation complete. Total cycles: " << cycle << "\n\n";
        print_schedule();
    }

    void print_schedule() {
        std::cout << "ID | OpCode | Latency | Dispatch | Issue | Complete | Writeback\n";
        std::cout << "--------------------------------------------------------------\n";
        for (const auto &inst : instructions) {
            std::cout << inst.id << "  | " << opcode_name(inst.op) << " | " << inst.latency
                      << "       | " << inst.dispatched_at << "        | " << inst.issued_at
                      << "     | " << inst.completed_at << "        | " << inst.writeback_at << "\n";
        }
    }
};

// ---- Test: Dependency chain vs parallel instructions ----

void test_inorder() {
    std::cout << "========== In-Order Core Simulation ==========\n\n";

    PipelineSimulator sim(CoreType::InOrder, 2); // issue_width=2
    sim.add_unit(UnitKind::ALU, 2, 0, "ALU");    // 2 ALU pipes, in-order
    sim.add_unit(UnitKind::FPU, 1, 0, "FPU");    // 1 FPU pipe, in-order
    sim.add_unit(UnitKind::DIV, 1, 1, "DIV");    // latency device (BufferSize=1)
    sim.micro_op_buffer_size = 0;

    // Dependencies: 1→2→3, 4 independent
    std::vector<int> no_deps;
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, no_deps);
    sim.add_instruction(OpCode::MUL, 3, {{UnitKind::ALU, 1}}, {0});
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, {1});
    sim.add_instruction(OpCode::FMA, 4, {{UnitKind::FPU, 4}}, no_deps);
    sim.add_instruction(OpCode::DIV, 10, {{UnitKind::DIV, 10}}, no_deps);

    sim.run();
}

void test_outoforder() {
    std::cout << "========== Out-of-Order Core Simulation ==========\n\n";

    PipelineSimulator sim(CoreType::OutOfOrder, 4); // issue_width=4
    sim.add_unit(UnitKind::ALU, 4, 16, "ALU");      // 4 ALU pipes, each with 16-entry RS
    sim.add_unit(UnitKind::FPU, 2, 10, "FPU");      // 2 FPU pipes, 10-entry RS
    sim.add_unit(UnitKind::DIV, 1, 1, "DIV");       // latency device
    sim.add_unit(UnitKind::LSU, 3, 28, "LSU");      // 3 LS pipes, 28-entry RS
    sim.rob_size = 160;

    // Chain 1: dependent ADD chain
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, {});
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, {0});
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, {1});

    // Chain 2: MUL chain
    sim.add_instruction(OpCode::MUL, 3, {{UnitKind::ALU, 1}}, {});
    sim.add_instruction(OpCode::MUL, 3, {{UnitKind::ALU, 1}}, {3});

    // Independent instructions: OoO can execute these while waiting on chains
    sim.add_instruction(OpCode::LOAD, 4, {{UnitKind::LSU, 1}}, {});
    sim.add_instruction(OpCode::STORE, 1, {{UnitKind::LSU, 1}}, {});
    sim.add_instruction(OpCode::ADD, 1, {{UnitKind::ALU, 1}}, {});

    // DIV instruction (long latency, non-pipelined)
    sim.add_instruction(OpCode::DIV, 66, {{UnitKind::DIV, 66}}, {});
    sim.add_instruction(OpCode::DIV, 66, {{UnitKind::DIV, 66}}, {});

    sim.run();
}

void test_resource_conflict() {
    std::cout << "========== Resource Conflict Simulation ==========\n\n";
    std::cout << "Model: 3 ALU pipes, where only 2 can do MUL, 1 can do DIV\n";
    std::cout << "This demonstrates the ProcResGroup concept from LLVM scheduling model.\n\n";

    PipelineSimulator sim(CoreType::OutOfOrder, 3);
    sim.add_unit(UnitKind::ALU, 3, 8, "ALU"); // IEX: ProcResource<3>
    sim.add_unit(UnitKind::DIV, 1, 1, "DIV"); // IEX1: ProcResource<1>
    sim.rob_size = 32;

    // Attempt to dispatch 2 MULs + 1 DIV in same cycle
    // MUL can use 2 of 3 ALU pipes, DIV uses the remaining specific pipe
    // If both MULs occupy the pipes that can also do DIV, DIV should fail
    sim.add_instruction(OpCode::MUL, 3, {{UnitKind::ALU, 3}}, {});
    sim.add_instruction(OpCode::MUL, 3, {{UnitKind::ALU, 3}}, {});
    sim.add_instruction(OpCode::DIV, 10, {{UnitKind::DIV, 10}}, {});

    sim.run();
}

int main() {
    test_inorder();
    std::cout << "\n\n";
    test_outoforder();
    std::cout << "\n\n";
    test_resource_conflict();

    return 0;
}
