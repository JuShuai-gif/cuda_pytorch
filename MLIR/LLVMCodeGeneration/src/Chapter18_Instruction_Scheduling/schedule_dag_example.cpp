//===----------------------------------------------------------------------===//
// Chapter 18 - Instruction Scheduling
// Example: ScheduleDAG Concepts - Building and Mutating a Dependency Graph
//===----------------------------------------------------------------------===//
//
// This example demonstrates the core concepts of LLVM's instruction scheduling:
// - Building a ScheduleDAG from a sequence of instructions
// - Understanding data dependencies and the ready queue
// - Applying mutations to modify the scheduling constraints
//
// NOTE: This is an educational example showing the conceptual structure.
// In a real LLVM backend, these concepts are implemented within the proper
// subclass framework. Compile with: clang++ -std=c++17 example.cpp
//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated LLVM classes for educational purposes
//------------------------------------------------------------------------------

// Represents a scheduling unit (an instruction in the DAG)
struct SUnit {
  unsigned NodeNum;               // Unique identifier
  std::string Name;               // Instruction mnemonic
  bool isScheduled;               // Whether this node has been scheduled
  bool isAvailable;               // Whether this node is in the ready queue
  unsigned Latency;               // Execution latency in cycles

  // Scheduling properties
  bool mayLoad;                   // Has memory load side-effect
  bool mayStore;                  // Has memory store side-effect

  SUnit(unsigned Num, const std::string &N)
    : NodeNum(Num), Name(N), isScheduled(false), isAvailable(false),
      Latency(1), mayLoad(false), mayStore(false) {}
};

// A directed edge between two SUnits
struct SDep {
  SUnit *From;                    // Predecessor
  SUnit *To;                      // Successor (depends on From)
  bool isDataDependency;          // True for data, false for ordering
  unsigned Latency;               // Latency along this edge

  SDep(SUnit *F, SUnit *T, bool IsData = true, unsigned L = 0)
    : From(F), To(T), isDataDependency(IsData), Latency(L) {}
};

// Simulated ScheduleDAGInstrs
class ScheduleDAG {
private:
  std::vector<SUnit> Nodes;
  std::vector<SDep> Edges;

public:
  // Build the DAG from a sequence of instructions
  void addNode(const std::string &Name, unsigned Latency = 1) {
    Nodes.emplace_back(Nodes.size(), Name);
    Nodes.back().Latency = Latency;
    std::cout << "  Added node " << Nodes.back().NodeNum
              << ": " << Name << " (latency=" << Latency << ")\n";
  }

  // Add a dependency edge: From must execute before To
  void addEdge(unsigned FromIdx, unsigned ToIdx, bool isData = true,
               unsigned EdgeLatency = 0) {
    if (FromIdx < Nodes.size() && ToIdx < Nodes.size()) {
      Edges.emplace_back(&Nodes[FromIdx], &Nodes[ToIdx], isData, EdgeLatency);
      std::cout << "  Added edge: " << Nodes[FromIdx].Name
                << " -> " << Nodes[ToIdx].Name
                << (isData ? " (data)" : " (order)")
                << " latency=" << EdgeLatency << "\n";
    }
  }

  // Find nodes that have no unscheduled successors (ready for top-down schedule)
  std::vector<SUnit*> getReadyQueue() {
    std::vector<SUnit*> Ready;
    for (auto &Node : Nodes) {
      if (Node.isScheduled) continue;
      bool allSuccScheduled = true;
      for (auto &Edge : Edges) {
        if (Edge.From == &Node && !Edge.To->isScheduled) {
          allSuccScheduled = false;
          break;
        }
      }
      if (allSuccScheduled) {
        Ready.push_back(&Node);
        Node.isAvailable = true;
      }
    }
    return Ready;
  }

  // Print the current DAG state
  void print() const {
    std::cout << "\n  ScheduleDAG State:\n";
    std::cout << "  Nodes:\n";
    for (auto &N : Nodes) {
      std::cout << "    [" << (N.isScheduled ? "X" : " ") << "] "
                << N.NodeNum << ": " << N.Name;
      if (N.mayLoad) std::cout << " [mayLoad]";
      if (N.mayStore) std::cout << " [mayStore]";
      std::cout << "\n";
    }
    std::cout << "  Edges:\n";
    for (auto &E : Edges) {
      std::cout << "    " << E.From->Name << " -> " << E.To->Name;
      if (!E.isDataDependency) std::cout << " [ordering]";
      std::cout << "\n";
    }
  }

  // Simulate scheduling one node
  void scheduleNode(SUnit *Node) {
    Node->isScheduled = true;
    Node->isAvailable = false;
    std::cout << "  Scheduling: " << Node->Name << "\n";
  }

  // Apply a mutation (simulated)
  void applyMutation(const std::string &name) {
    std::cout << "\n  Applying mutation: " << name << "\n";
  }
};

//------------------------------------------------------------------------------
// Mutation concept: Add ordering constraints between specific instructions
//------------------------------------------------------------------------------
class ScheduleDAGMutation {
public:
  virtual ~ScheduleDAGMutation() = default;
  virtual void apply(ScheduleDAG &DAG) = 0;
  virtual std::string getName() const = 0;
};

// Example mutation: Cluster all load instructions together
class LoadClusterMutation : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAG &DAG) override {
    std::cout << "  [LoadClusterMutation] Clustering load instructions\n";
    // In a real implementation, this would add ordering edges between
    // consecutive loads to keep them adjacent in the schedule
    DAG.applyMutation(getName());
  }
  std::string getName() const override { return "LoadClusterMutation"; }
};

// Example mutation: Priority constraint between specific opcodes
class PriorityConstraintMutation : public ScheduleDAGMutation {
  std::string HighPrioOpcode;
  std::string LowPrioOpcode;
public:
  PriorityConstraintMutation(const std::string &High, const std::string &Low)
    : HighPrioOpcode(High), LowPrioOpcode(Low) {}

  void apply(ScheduleDAG &DAG) override {
    std::cout << "  [PriorityConstraint] " << HighPrioOpcode
              << " must schedule before " << LowPrioOpcode << "\n";
    DAG.applyMutation(getName());
  }
  std::string getName() const override { return "PriorityConstraintMutation"; }
};

//------------------------------------------------------------------------------
// Scheduling directions
//------------------------------------------------------------------------------
enum class SchedulingDirection {
  TopDown,        // Schedule from region start to end
  BottomUp,       // Schedule from region end to start
  Bidirectional   // Pick from either end (LLVM default)
};

const char *directionToString(SchedulingDirection Dir) {
  switch (Dir) {
  case SchedulingDirection::TopDown:      return "Top-Down";
  case SchedulingDirection::BottomUp:     return "Bottom-Up";
  case SchedulingDirection::Bidirectional: return "Bidirectional";
  }
  return "Unknown";
}

//------------------------------------------------------------------------------
// MachineSchedPolicy simulated
//------------------------------------------------------------------------------
struct MachineSchedPolicy {
  SchedulingDirection Direction = SchedulingDirection::Bidirectional;
  bool OnlyTopDown = false;
  bool OnlyBottomUp = false;
  bool TrackLaneMasks = false;
  bool TrackLiveness = true;

  void configure(SchedulingDirection Dir) {
    Direction = Dir;
    OnlyTopDown = (Dir == SchedulingDirection::TopDown);
    OnlyBottomUp = (Dir == SchedulingDirection::BottomUp);
  }

  void print() const {
    std::cout << "  Policy: direction=" << directionToString(Direction)
              << ", trackLiveness=" << (TrackLiveness ? "yes" : "no") << "\n";
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 18: Instruction Scheduling Concepts ===\n\n";

  // Build a simple DDG simulating a basic block
  std::cout << "--- Building a ScheduleDAG ---\n";
  ScheduleDAG DAG;

  // Create instructions
  DAG.addNode("LOAD r1, [addr]", /*Latency=*/4);
  DAG.addNode("ADD r2, r1, #1",  /*Latency=*/1);
  DAG.addNode("MUL r3, r1, r2",  /*Latency=*/2);
  DAG.addNode("STORE [addr2], r3", /*Latency=*/1);

  // Add data dependencies
  // LOAD -> ADD (ADD reads r1 produced by LOAD)
  DAG.addEdge(0, 1, true, 0);
  // LOAD -> MUL (MUL reads r1 produced by LOAD)
  DAG.addEdge(0, 2, true, 0);
  // ADD -> MUL (MUL reads r2 produced by ADD)
  DAG.addEdge(1, 2, true, 0);
  // MUL -> STORE (STORE reads r3 produced by MUL)
  DAG.addEdge(2, 3, true, 0);

  DAG.print();

  // Demonstrate the ready queue concept
  std::cout << "\n--- Ready Queue Demonstration (Top-Down Scheduling) ---\n";
  std::cout << "Note: In top-down scheduling, the ready queue contains nodes\n"
            << "      whose successors have all been scheduled.\n\n";

  // Simulate a simple scheduling sequence
  auto ready = DAG.getReadyQueue();
  std::cout << "Initial ready queue (nodes with no successors):\n";
  for (auto *N : ready) {
    std::cout << "  - " << N->Name << "\n";
  }

  // Schedule STORE first (bottom of DAG in top-down scheduling)
  if (!ready.empty()) {
    DAG.scheduleNode(ready[0]);
  }

  ready = DAG.getReadyQueue();
  std::cout << "Ready queue after scheduling STORE:\n";
  for (auto *N : ready) {
    std::cout << "  - " << N->Name << "\n";
  }

  // Demonstrate mutations
  std::cout << "\n--- Mutation Demonstration ---\n";

  LoadClusterMutation loadCluster;
  loadCluster.apply(DAG);

  PriorityConstraintMutation prio("MUL", "ADD");
  prio.apply(DAG);

  std::cout << "\nNote: Mutations modify the DDG after construction\n"
            << "to add domain-specific constraints. Be careful not to\n"
            << "introduce cycles, which make scheduling impossible.\n";

  // Demonstrate scheduling policy
  std::cout << "\n--- Scheduling Policy ---\n";

  MachineSchedPolicy policy;
  policy.print();

  policy.configure(SchedulingDirection::TopDown);
  std::cout << "After configuring for top-down:\n";
  policy.print();

  policy.configure(SchedulingDirection::BottomUp);
  std::cout << "After configuring for bottom-up:\n";
  policy.print();

  std::cout << "\n--- Summary ---\n";
  std::cout << "Key concepts demonstrated:\n";
  std::cout << "  1. ScheduleDAG: Represents instruction dependencies\n";
  std::cout << "  2. Ready Queue: Nodes eligible for scheduling\n";
  std::cout << "  3. Mutations: Post-construction DDG modifications\n";
  std::cout << "  4. Scheduling Directions: Top-down, bottom-up, bidirectional\n";

  return 0;
}
