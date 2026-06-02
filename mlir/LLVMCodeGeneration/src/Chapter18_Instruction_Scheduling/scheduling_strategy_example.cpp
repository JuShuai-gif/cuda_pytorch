//===----------------------------------------------------------------------===//
// Chapter 18 - Instruction Scheduling
// Example: Custom Scheduling Strategy - MachineSchedStrategy Implementation
//===----------------------------------------------------------------------===//
//
// This example demonstrates how to implement a custom scheduling strategy:
// - Extending GenericScheduler for pre-RA scheduling
// - Overriding tryCandidate to prioritize specific instructions
// - Configuring scheduling policy (direction, liveness tracking)
// - Creating the scheduler through TargetPassConfig
//
// NOTE: In a real LLVM backend, this code lives in the target-specific
// scheduling strategy class. This is a simulation for educational purposes.
//

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated scheduling candidates and strategy infrastructure
//------------------------------------------------------------------------------

// Reason why a candidate was selected (matches MachineSchedStrategy::CandReason)
enum class CandReason {
  NoCand,        // No other candidates
  NodeOrder,     // Next node in original block order (tiebreaker)
  Stall,         // Chosen to avoid pipeline stall
  RegCritical,   // Chosen to reduce register pressure
  Clustering,    // Chosen to cluster similar operations
  Weak,          // Weak reason - default for custom heuristics
};

const char *reasonToString(CandReason R) {
  switch (R) {
  case CandReason::NoCand:      return "NoCand";
  case CandReason::NodeOrder:   return "NodeOrder";
  case CandReason::Stall:        return "Stall";
  case CandReason::RegCritical:  return "RegCritical";
  case CandReason::Clustering:   return "Clustering";
  case CandReason::Weak:         return "Weak";
  }
  return "?";
}

// Scheduling unit (instruction in the ready queue)
struct SchedCandidate {
  std::string Name;        // Instruction name
  CandReason Reason;       // Why it was selected
  int Priority;            // Custom priority value
  unsigned ResourceCost;   // Resource usage cost

  SchedCandidate(const std::string &N = "")
    : Name(N), Reason(CandReason::NoCand), Priority(0), ResourceCost(0) {}
};

// Scheduling boundary (top or bottom of region)
struct SchedBoundary {
  bool IsTop;
  std::vector<std::string> ScheduledOrder;

  SchedBoundary(bool Top) : IsTop(Top) {}
};

// Base scheduler
class GenericScheduler {
public:
  virtual ~GenericScheduler() = default;

  // Default heuristic: compare two candidates
  virtual bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                            SchedBoundary *Zone) const {
    // Default: prefer the candidate with higher priority
    if (TryCand.Priority > Cand.Priority) {
      TryCand.Reason = CandReason::NodeOrder;
      Cand = TryCand;
      return true;
    }
    return false;
  }

  virtual void pickNode(SchedBoundary *Zone) {
    std::cout << "  GenericScheduler::pickNode called\n";
  }

  virtual const char *getName() const { return "GenericScheduler"; }
};

//------------------------------------------------------------------------------
// Custom pre-register-allocation scheduling strategy
// Demonstrates prioritizing loads and specific opcodes
//------------------------------------------------------------------------------
class CustomPreRASchedStrategy : public GenericScheduler {
private:
  // Opcodes we want to prioritize
  std::vector<std::string> PriorityOpcodes;
  bool PreferLoads;        // Whether to prioritize load instructions

public:
  CustomPreRASchedStrategy(bool PrefLoads = true)
    : PreferLoads(PrefLoads) {}

  void addPriorityOpcode(const std::string &Opc) {
    PriorityOpcodes.push_back(Opc);
  }

  // Override the candidate comparison
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override {
    // Step 1: Let the base class do its default comparison
    bool BetterCand = GenericScheduler::tryCandidate(Cand, TryCand, Zone);

    // Step 2: If base class found TryCand better, check the reason
    if (BetterCand) {
      // If the reason is strong (not just NodeOrder or NoCand), keep it
      if (TryCand.Reason != CandReason::NodeOrder &&
          TryCand.Reason != CandReason::NoCand) {
        return true;
      }
    }

    // Step 3: Apply our custom heuristics
    if (Zone != nullptr) {
      // Heuristic 1: Prioritize load instructions (hide memory latency)
      if (PreferLoads && isLoad(TryCand.Name)) {
        std::cout << "    [Custom] Prioritizing load: " << TryCand.Name << "\n";
        TryCand.Reason = CandReason::Stall;
        Cand = TryCand;
        return true;
      }

      // Heuristic 2: Prioritize specific performance-critical opcodes
      if (isPriorityOpcode(TryCand.Name)) {
        std::cout << "    [Custom] Prioritizing critical opcode: "
                  << TryCand.Name << "\n";
        TryCand.Reason = CandReason::Weak;
        Cand = TryCand;
        return true;
      }

      // Heuristic 3: De-prioritize store instructions (schedule later)
      if (isStore(TryCand.Name) && isALU(Cand.Name)) {
        std::cout << "    [Custom] Preferring ALU over store: "
                  << Cand.Name << " > " << TryCand.Name << "\n";
        return false; // Keep current candidate
      }
    }

    // If no custom heuristic applied, keep the TryCand if it was better
    return TryCand.Reason != CandReason::NoCand;
  }

  const char *getName() const override { return "CustomPreRASchedStrategy"; }

private:
  bool isLoad(const std::string &Name) const {
    return Name.find("LOAD") != std::string::npos ||
           Name.find("LDR") != std::string::npos;
  }

  bool isStore(const std::string &Name) const {
    return Name.find("STORE") != std::string::npos ||
           Name.find("STR") != std::string::npos;
  }

  bool isALU(const std::string &Name) const {
    return Name.find("ADD") != std::string::npos ||
           Name.find("SUB") != std::string::npos ||
           Name.find("MUL") != std::string::npos;
  }

  bool isPriorityOpcode(const std::string &Name) const {
    for (auto &Opc : PriorityOpcodes) {
      if (Name.find(Opc) != std::string::npos) return true;
    }
    return false;
  }
};

//------------------------------------------------------------------------------
// Post-register-allocation scheduling strategy
//------------------------------------------------------------------------------
class CustomPostRASchedStrategy {
private:
  bool AvoidHazards;   // Whether to avoid pipeline hazards

public:
  CustomPostRASchedStrategy(bool Avoid = true) : AvoidHazards(Avoid) {}

  const char *getName() const { return "CustomPostRASchedStrategy"; }

  bool avoidHazards() const { return AvoidHazards; }
};

//------------------------------------------------------------------------------
// Simulated TargetPassConfig
//------------------------------------------------------------------------------
class TargetPassConfig {
private:
  CustomPreRASchedStrategy *PreRAStrategy;
  CustomPostRASchedStrategy *PostRAStrategy;

public:
  TargetPassConfig()
    : PreRAStrategy(nullptr), PostRAStrategy(nullptr) {}

  // Create the pre-RA scheduling strategy (called during pipeline setup)
  void createMachineScheduler() {
    // In a real backend, this creates a ScheduleDAGMILive with the strategy
    std::cout << "\n  [TargetPassConfig] Creating MachineScheduler with "
              << "CustomPreRASchedStrategy\n";

    PreRAStrategy = new CustomPreRASchedStrategy(true);
    PreRAStrategy->addPriorityOpcode("WIDENING_MUL");
    PreRAStrategy->addPriorityOpcode("DIV");
  }

  // Create the post-RA scheduling strategy
  void createPostMachineScheduler() {
    std::cout << "  [TargetPassConfig] Creating PostMachineScheduler with "
              << "CustomPostRASchedStrategy\n";
    PostRAStrategy = new CustomPostRASchedStrategy(true);
  }

  CustomPreRASchedStrategy *getPreRAStrategy() const { return PreRAStrategy; }
  CustomPostRASchedStrategy *getPostRAStrategy() const { return PostRAStrategy; }

  ~TargetPassConfig() {
    delete PreRAStrategy;
    delete PostRAStrategy;
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 18: Custom Scheduling Strategy ===\n";

  // Create the pass configuration with custom strategies
  TargetPassConfig PassConfig;
  PassConfig.createMachineScheduler();
  PassConfig.createPostMachineScheduler();

  auto *Strategy = PassConfig.getPreRAStrategy();
  std::cout << "\nPre-RA Strategy: " << Strategy->getName() << "\n";
  std::cout << "Post-RA Strategy: "
            << PassConfig.getPostRAStrategy()->getName() << "\n";

  // Simulate candidate selection
  std::cout << "\n--- Simulating tryCandidate ---\n";

  SchedCandidate Current("ADD r1, r2, r3");
  Current.Priority = 10;
  Current.Reason = CandReason::NodeOrder;

  SchedCandidate Alternative("LOAD r4, [sp+8]");
  Alternative.Priority = 10;
  Alternative.Reason = CandReason::NoCand;

  SchedBoundary TopZone(true);

  std::cout << "  Current candidate: " << Current.Name
            << " (priority=" << Current.Priority
            << ", reason=" << reasonToString(Current.Reason) << ")\n";
  std::cout << "  Alternative:       " << Alternative.Name
            << " (priority=" << Alternative.Priority
            << ", reason=" << reasonToString(Alternative.Reason) << ")\n";

  bool swapped = Strategy->tryCandidate(Current, Alternative, &TopZone);

  std::cout << "\n  Result: " << (swapped ? "Swapped to alternative" : "Kept current")
            << "\n  Winner: " << Current.Name
            << " (reason=" << reasonToString(Current.Reason) << ")\n";

  // Simulate another comparison with a critical opcode
  std::cout << "\n--- Comparing with Critical Opcode ---\n";

  SchedCandidate WidenMul("WIDENING_MUL r5, r6, r7");
  WidenMul.Priority = 10;
  WidenMul.Reason = CandReason::NoCand;

  SchedCandidate NormalAdd("ADD r8, r9, r10");
  NormalAdd.Priority = 10;
  NormalAdd.Reason = CandReason::NodeOrder;

  std::cout << "  Current:    " << NormalAdd.Name
            << " (reason=" << reasonToString(NormalAdd.Reason) << ")\n";
  std::cout << "  Alternative:" << WidenMul.Name
            << " (reason=" << reasonToString(WidenMul.Reason) << ")\n";

  swapped = Strategy->tryCandidate(NormalAdd, WidenMul, &TopZone);
  std::cout << "\n  Result: " << (swapped ? "Swapped to WIDENING_MUL" : "Kept ADD")
            << "\n  Winner: " << NormalAdd.Name
            << " (reason=" << reasonToString(NormalAdd.Reason) << ")\n";

  // Scheduling policy demonstration
  std::cout << "\n--- Scheduling Policy Configuration ---\n";
  std::cout << "  The scheduling direction is controlled via TargetSubtargetInfo:\n";
  std::cout << "    void overrideSchedPolicy(MachineSchedPolicy &Policy,\n";
  std::cout << "                             unsigned NumRegionInstrs) const {\n";
  std::cout << "      Policy.OnlyTopDown = true;\n";
  std::cout << "      Policy.OnlyBottomUp = false;\n";
  std::cout << "    }\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Inherit from GenericScheduler for pre-RA scheduling\n";
  std::cout << "  2. Override tryCandidate() for custom candidate comparisons\n";
  std::cout << "  3. Use CandReason to document why candidates are chosen\n";
  std::cout << "  4. Create scheduler in TargetPassConfig::createMachineScheduler\n";
  std::cout << "  5. Use overrideSchedPolicy for scheduling direction\n";

  return 0;
}
