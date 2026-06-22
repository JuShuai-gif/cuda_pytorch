//===----------------------------------------------------------------------===//
// Chapter 18 - Instruction Scheduling
// Example: Scheduling Model - Processing Units, Events, and Bindings
//===----------------------------------------------------------------------===//
//
// This example demonstrates how a scheduling model is structured:
// - Processing resources (ALU, Memory Unit, etc.)
// - Scheduling events (reads and writes)
// - Bindings between events and resources
// - How instructions consume resources during scheduling
//
// NOTE: In a real LLVM backend, this is defined in TableGen (.td files).
// This C++ example simulates the runtime behavior.
//

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Processing Units (ProcResource in TableGen)
//------------------------------------------------------------------------------
struct ProcessorResource {
  std::string Name;
  unsigned NumUnits;             // How many identical units exist
  unsigned BufferSize;           // 0 = in-order, -1 = unlimited out-of-order
  bool IsInOrder;

  ProcessorResource(const std::string &N, unsigned Num, unsigned Buf = 0)
    : Name(N), NumUnits(Num), BufferSize(Buf),
      IsInOrder(Buf == 0) {}

  void print() const {
    std::cout << "    Resource [" << Name << "]: " << NumUnits
              << " units, buffer=" << (BufferSize == 0 ? "in-order" :
                  (BufferSize == (unsigned)-1 ? "out-of-order" :
                   "buffered(" + std::to_string(BufferSize) + ")")) << "\n";
  }
};

//------------------------------------------------------------------------------
// Scheduling Events (SchedReadWrite in TableGen)
//------------------------------------------------------------------------------
enum class EventType { Read, Write };

struct SchedulingEvent {
  std::string Name;
  EventType Type;
  unsigned Latency;              // Cycles the event takes

  SchedulingEvent(const std::string &N, EventType T, unsigned L = 0)
    : Name(N), Type(T), Latency(L) {}

  void print() const {
    std::cout << "    Event [" << Name << "]: "
              << (Type == EventType::Read ? "Read" : "Write")
              << ", latency=" << Latency << "\n";
  }
};

//------------------------------------------------------------------------------
// Scheduling Bindings (WriteRes, ReadAdvance in TableGen)
//------------------------------------------------------------------------------
struct ResourceBinding {
  std::string EventName;          // Which event this binds
  std::vector<std::string> Resources; // Which resources are used
  unsigned Cost;                  // Resource cycles consumed
  int ReadAdvanceCycles;          // Forwarding: negative = forwarding path

  static ResourceBinding createWrite(const std::string &Event,
      std::vector<std::string> Res, unsigned Latency) {
    ResourceBinding RB;
    RB.EventName = Event;
    RB.Resources = std::move(Res);
    RB.Cost = Latency;
    RB.ReadAdvanceCycles = 0;
    return RB;
  }

  static ResourceBinding createReadAdvance(const std::string &Event,
      int ForwardCycles) {
    ResourceBinding RB;
    RB.EventName = Event;
    RB.Cost = 0;
    RB.ReadAdvanceCycles = ForwardCycles;
    // ReadAdvance with negative = forwarding (absorb cycles)
    // ReadAdvance with positive = additional penalty
    return RB;
  }

  void print() const {
    std::cout << "    Binding: event=" << EventName;
    if (!Resources.empty()) {
      std::cout << ", resources=[";
      for (size_t i = 0; i < Resources.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << Resources[i];
      }
      std::cout << "]";
    }
    std::cout << ", cost=" << Cost;
    if (ReadAdvanceCycles != 0) {
      std::cout << ", advance=" << ReadAdvanceCycles
                << (ReadAdvanceCycles < 0 ? " (forwarding)" : " (penalty)");
    }
    std::cout << "\n";
  }
};

//------------------------------------------------------------------------------
// Simulated Instruction (as would be defined in TableGen)
//------------------------------------------------------------------------------
struct SimulatedInstruction {
  std::string Opcode;
  std::vector<std::string> ReadEvents;   // SchedRead events for operands
  std::vector<std::string> WriteEvents;  // SchedWrite events for definitions
  bool mayLoad;
  bool mayStore;
  bool isReMaterializable;

  SimulatedInstruction(const std::string &Opc) : Opcode(Opc),
    mayLoad(false), mayStore(false), isReMaterializable(false) {}

  void addReadEvent(const std::string &Evt) { ReadEvents.push_back(Evt); }
  void addWriteEvent(const std::string &Evt) { WriteEvents.push_back(Evt); }

  void print() const {
    std::cout << "  Instruction [" << Opcode << "]:\n";
    std::cout << "    Reads: ";
    if (ReadEvents.empty()) std::cout << "(none)";
    for (auto &R : ReadEvents) std::cout << R << " ";
    std::cout << "\n    Writes: ";
    if (WriteEvents.empty()) std::cout << "(none)";
    for (auto &W : WriteEvents) std::cout << W << " ";
    if (mayLoad) std::cout << "\n    [mayLoad]";
    if (mayStore) std::cout << "\n    [mayStore]";
    if (isReMaterializable) std::cout << "\n    [isReMaterializable]";
    std::cout << "\n";
  }
};

//------------------------------------------------------------------------------
// Scheduling Model (SchedMachineModel in TableGen)
//------------------------------------------------------------------------------
class SchedulingModel {
private:
  std::string Name;
  unsigned IssueWidth;             // Max micro-ops per cycle
  unsigned LoadLatency;            // Default latency for loads
  unsigned HighLatency;            // Default for high-latency ops
  bool CompleteModel;
  bool IsOutOfOrder;

  std::vector<ProcessorResource> Resources;
  std::vector<SchedulingEvent> Events;
  std::vector<ResourceBinding> Bindings;
  std::vector<SimulatedInstruction> Instructions;

  // Track resource usage per cycle (for simulation)
  std::map<std::string, std::vector<unsigned>> ResourceUsage;

public:
  SchedulingModel(const std::string &N)
    : Name(N), IssueWidth(1), LoadLatency(4), HighLatency(10),
      CompleteModel(false), IsOutOfOrder(false) {}

  // Configuration
  void setIssueWidth(unsigned W) { IssueWidth = W; }
  void setLoadLatency(unsigned L) { LoadLatency = L; }
  void setOutOfOrder(bool OOO) { IsOutOfOrder = OOO; }
  void setComplete() { CompleteModel = true; }

  // Add resources
  void addResource(const std::string &Name, unsigned NumUnits,
                   unsigned BufferSize = 0) {
    Resources.emplace_back(Name, NumUnits, BufferSize);
  }

  // Add scheduling events
  void addReadEvent(const std::string &Name, unsigned Latency = 0) {
    Events.emplace_back(Name, EventType::Read, Latency);
  }
  void addWriteEvent(const std::string &Name, unsigned Latency = 0) {
    Events.emplace_back(Name, EventType::Write, Latency);
  }

  // Add bindings
  void addWriteBinding(const std::string &Event,
      std::vector<std::string> Resources, unsigned Latency) {
    Bindings.push_back(ResourceBinding::createWrite(Event,
        std::move(Resources), Latency));
  }
  void addReadAdvance(const std::string &Event, int Cycles) {
    Bindings.push_back(ResourceBinding::createReadAdvance(Event, Cycles));
  }

  // Register an instruction with its events
  void addInstruction(const SimulatedInstruction &Inst) {
    Instructions.push_back(Inst);
  }

  // Print the complete model
  void print() const {
    std::cout << "\n=== Scheduling Model: " << Name << " ===\n";
    std::cout << "  IssueWidth=" << IssueWidth
              << ", LoadLatency=" << LoadLatency
              << ", OutOfOrder=" << (IsOutOfOrder ? "yes" : "no")
              << ", Complete=" << (CompleteModel ? "yes" : "no") << "\n";

    std::cout << "\n  --- Processing Units (" << Resources.size() << ") ---\n";
    for (auto &R : Resources) R.print();

    std::cout << "\n  --- Scheduling Events (" << Events.size() << ") ---\n";
    for (auto &E : Events) E.print();

    std::cout << "\n  --- Bindings (" << Bindings.size() << ") ---\n";
    for (auto &B : Bindings) B.print();

    std::cout << "\n  --- Instructions (" << Instructions.size() << ") ---\n";
    for (auto &I : Instructions) I.print();
  }

  // Simulate scheduling decisions based on the model
  unsigned getInstructionLatency(const SimulatedInstruction &Inst) const {
    // Naively: sum the max latency of write events
    unsigned MaxLat = 0;
    for (auto &W : Inst.WriteEvents) {
      for (auto &B : Bindings) {
        if (B.EventName == W) {
          MaxLat = std::max(MaxLat, B.Cost);
        }
      }
    }
    // Default latency
    if (MaxLat == 0) {
      MaxLat = Inst.mayLoad ? LoadLatency : 1;
    }
    return MaxLat;
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 18: Scheduling Model Implementation ===\n";

  // Build a scheduling model for a simple in-order processor
  SchedulingModel model("SimpleInOrderModel");

  // Configure overall capabilities
  model.setIssueWidth(1);         // Single-issue
  model.setLoadLatency(4);        // Loads take 4 cycles
  model.setOutOfOrder(false);     // In-order processor

  // Define processing resources
  model.addResource("ALU", 1, 0);           // 1 ALU, in-order
  model.addResource("MulUnit", 1, 0);      // 1 Multiplier, in-order
  model.addResource("MemUnit", 1, 0);      // 1 Memory unit, in-order
  model.addResource("BranchUnit", 1, 0);   // 1 Branch unit

  // Define scheduling events for reads
  model.addReadEvent("ReadALUArg0");
  model.addReadEvent("ReadALUArg1");
  model.addReadEvent("ReadMulArg0");
  model.addReadEvent("ReadMulArg1");
  model.addReadEvent("ReadLoadAddr");
  model.addReadEvent("ReadStoreAddr");
  model.addReadEvent("ReadStoreVal");

  // Define scheduling events for writes
  model.addWriteEvent("WriteALU", 1);
  model.addWriteEvent("WriteMul", 2);       // Multiply has 2-cycle latency
  model.addWriteEvent("WriteLoad", 4);      // Load has 4-cycle latency
  model.addWriteEvent("WriteStore", 1);

  // Bind write events to resources
  model.addWriteBinding("WriteALU", {"ALU"}, 1);
  model.addWriteBinding("WriteMul", {"MulUnit"}, 2);
  model.addWriteBinding("WriteLoad", {"MemUnit"}, 4);
  model.addWriteBinding("WriteStore", {"MemUnit"}, 1);

  // Add read advance bindings (forwarding paths)
  model.addReadAdvance("ReadALUArg0", 0);   // No forwarding for ALU reads
  model.addReadAdvance("ReadALUArg1", 0);
  model.addReadAdvance("ReadMulArg0", -1);  // 1-cycle forwarding for mul
  model.addReadAdvance("ReadMulArg1", -1);
  model.addReadAdvance("ReadStoreVal", -1); // Store value forwarded

  // Define instructions with their scheduling events
  SimulatedInstruction ADD("ADD");
  ADD.addReadEvent("ReadALUArg0");
  ADD.addReadEvent("ReadALUArg1");
  ADD.addWriteEvent("WriteALU");

  SimulatedInstruction MUL("MUL");
  MUL.addReadEvent("ReadMulArg0");
  MUL.addReadEvent("ReadMulArg1");
  MUL.addWriteEvent("WriteMul");

  SimulatedInstruction LOAD("LOAD");
  LOAD.addReadEvent("ReadLoadAddr");
  LOAD.addWriteEvent("WriteLoad");
  LOAD.mayLoad = true;

  SimulatedInstruction STORE("STORE");
  STORE.addReadEvent("ReadStoreAddr");
  STORE.addReadEvent("ReadStoreVal");
  STORE.addWriteEvent("WriteStore");
  STORE.mayStore = true;

  model.addInstruction(ADD);
  model.addInstruction(MUL);
  model.addInstruction(LOAD);
  model.addInstruction(STORE);

  // Mark model as complete
  model.setComplete();

  // Print the model
  model.print();

  // Simulate querying latencies
  std::cout << "\n--- Instruction Latency Queries ---\n";
  std::cout << "  ADD latency:  " << model.getInstructionLatency(ADD)
            << " cycles\n";
  std::cout << "  MUL latency:  " << model.getInstructionLatency(MUL)
            << " cycles\n";
  std::cout << "  LOAD latency: " << model.getInstructionLatency(LOAD)
            << " cycles\n";
  std::cout << "  STORE latency:" << model.getInstructionLatency(STORE)
            << " cycles\n";

  std::cout << "\n--- Key Concepts ---\n";
  std::cout << "  1. Processor resources are the hardware units (ALUs, memory ports)\n";
  std::cout << "  2. Scheduling events describe reads/writes per instruction operand\n";
  std::cout << "  3. WriteRes binds write events to resources with latency\n";
  std::cout << "  4. ReadAdvance models forwarding paths (negative = forwarding)\n";
  std::cout << "  5. InstRW (or WriteRes/ReadAdvance patterns) decorate instructions\n";
  std::cout << "  6. SchedMachineModel glues everything for a processor model\n";

  return 0;
}
