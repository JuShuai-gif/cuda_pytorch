//===----------------------------------------------------------------------===//
// Chapter 20 - Lowering of the Stack Layout
// Example: Frame Index Expansion and Register Scavenging
//===----------------------------------------------------------------------===//
//
// This example demonstrates:
// - eliminateFrameIndex: Converting frame indices to register+offset addressing
// - Frame index replacement with SP/FP base register
// - Register scavenging for temporary registers when all are allocated
// - Emergency spill slots for guaranteed spill space
//
// NOTE: In real LLVM, eliminateFrameIndex is a virtual method on TargetFrameLowering.
// RegScavenger is a helper class used during prologue/epilogue insertion.
//

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated MachineOperand and MachineInstr
//------------------------------------------------------------------------------
enum class OperandType { Register, Immediate, FrameIndex };

struct MachineOperand {
  OperandType Type;
  std::string RegName;    // Valid if Type == Register
  int ImmVal;             // Valid if Type == Immediate
  int FrameIdxVal;        // Valid if Type == FrameIndex

  static MachineOperand createReg(const std::string &R) {
    MachineOperand MO;
    MO.Type = OperandType::Register;
    MO.RegName = R;
    return MO;
  }

  static MachineOperand createImm(int Val) {
    MachineOperand MO;
    MO.Type = OperandType::Immediate;
    MO.ImmVal = Val;
    return MO;
  }

  static MachineOperand createFrameIdx(int Idx) {
    MachineOperand MO;
    MO.Type = OperandType::FrameIndex;
    MO.FrameIdxVal = Idx;
    return MO;
  }

  // In real LLVM: ChangeToRegister / ChangeToImmediate
  void changeToRegister(const std::string &R) {
    Type = OperandType::Register;
    RegName = R;
  }

  void changeToImmediate(int Val) {
    Type = OperandType::Immediate;
    ImmVal = Val;
  }

  void print() const {
    switch (Type) {
    case OperandType::Register:
      std::cout << RegName; break;
    case OperandType::Immediate:
      std::cout << "#" << ImmVal; break;
    case OperandType::FrameIndex:
      std::cout << "%stack." << FrameIdxVal; break;
    }
  }
};

struct MachineInstr {
  std::string Opcode;
  std::vector<MachineOperand> Operands;

  MachineInstr(const std::string &Opc) : Opcode(Opc) {}

  void addOperand(const MachineOperand &MO) { Operands.push_back(MO); }

  // Get a mutable reference to an operand
  MachineOperand &getOperand(unsigned Idx) { return Operands[Idx]; }
  const MachineOperand &getOperand(unsigned Idx) const { return Operands[Idx]; }
  unsigned getNumOperands() const { return Operands.size(); }

  void print() const {
    std::cout << "  " << Opcode << " ";
    for (size_t i = 0; i < Operands.size(); ++i) {
      if (i > 0) std::cout << ", ";
      Operands[i].print();
    }
    std::cout << "\n";
  }
};

//------------------------------------------------------------------------------
// Simulated MachineFrameInfo
//------------------------------------------------------------------------------
class SimFrameInfo {
private:
  struct Obj {
    int Idx;
    int Offset; // Offset from SP
    unsigned Size;
  };
  std::vector<Obj> Objects;

public:
  int CreateStackObject(unsigned Size, unsigned Align) {
    int Idx = Objects.size();
    Obj O;
    O.Idx = Idx;
    O.Offset = 0; // Will be computed during layout
    O.Size = Size;
    Objects.push_back(O);
    return Idx;
  }

  void setObjectOffset(int Idx, int Offset) {
    Objects[Idx].Offset = Offset;
  }

  int getObjectOffset(int Idx) const {
    return Objects[Idx].Offset;
  }

  unsigned getObjectSize(int Idx) const {
    return Objects[Idx].Size;
  }

  unsigned getStackSize() const {
    unsigned Total = 0;
    for (auto &O : Objects) Total += O.Size;
    return Total;
  }
};

//------------------------------------------------------------------------------
// Frame Index Elimination
//------------------------------------------------------------------------------
class FrameIndexEliminator {
private:
  bool UsesFramePointer;
  int StackSize;

public:
  FrameIndexEliminator(bool useFP, int SS)
    : UsesFramePointer(useFP), StackSize(SS) {}

  // Eliminate a frame index reference in an instruction
  // Returns true if a temporary register was needed (for scavenging)
  bool eliminateFrameIndex(MachineInstr &MI, unsigned FIOperandNum,
                           const SimFrameInfo &MFI) {
    int FrameIndex = MI.getOperand(FIOperandNum).FrameIdxVal;
    int ObjectOffset = MFI.getObjectOffset(FrameIndex);

    // Determine base register and final offset
    std::string BaseReg;
    int FinalOffset;

    if (UsesFramePointer) {
      BaseReg = "FP";
      // FP-relative: objects are below FP, so offsets are negative
      // For positive offsets from SP, compute: FP_offset = SP_offset + StackSize
      // (This is target-specific - ARM vs x86 differ in convention)
      FinalOffset = ObjectOffset - StackSize;
    } else {
      BaseReg = "SP";
      FinalOffset = ObjectOffset;
    }

    std::cout << "  [eliminateFrameIndex] frameIdx=" << FrameIndex
              << " -> " << BaseReg << " + " << FinalOffset << "\n";

    // Check if the offset fits in the immediate field
    bool needsScavenging = (FinalOffset > 255 || FinalOffset < -255);

    // Replace the frame index operand with the base register
    MI.getOperand(FIOperandNum).changeToRegister(BaseReg);
    // Replace the offset operand
    if (FIOperandNum + 1 < MI.getNumOperands()) {
      if (!needsScavenging) {
        MI.getOperand(FIOperandNum + 1).changeToImmediate(FinalOffset);
      } else {
        // Offset too large: need a temporary register to compute address
        std::cout << "    Offset " << FinalOffset
                  << " too large for immediate field - need scavenging!\n";
        MI.getOperand(FIOperandNum + 1).changeToImmediate(0);
        // In real code: expand into ADDri + load/store with new register
        return true;
      }
    }

    return false;
  }
};

//------------------------------------------------------------------------------
// Register Scavenger
//------------------------------------------------------------------------------
class RegScavenger {
private:
  std::set<std::string> UsedRegs;     // Currently live registers
  std::set<std::string> AvailableRegs; // Registers that may be free
  int EmergencySpillSlot;              // Frame index of emergency spill slot
  bool HasEmergencySlot;

public:
  RegScavenger() : EmergencySpillSlot(-1), HasEmergencySlot(false) {
    // Initialize available register pool
    AvailableRegs = {"r0", "r1", "r2", "r3", "r12"};
  }

  // Set which registers are currently in use
  void setUsedRegisters(const std::set<std::string> &Used) {
    UsedRegs = Used;
  }

  // Add an emergency spill slot for worst-case scenarios
  void addScavengingFrameIndex(int FI) {
    EmergencySpillSlot = FI;
    HasEmergencySlot = true;
    std::cout << "  [Scavenger] Emergency spill slot: #" << FI << "\n";
  }

  // Find an unused register of the specified class (simulated)
  std::string findUnusedReg() const {
    for (auto &Reg : AvailableRegs) {
      if (UsedRegs.find(Reg) == UsedRegs.end()) {
        std::cout << "  [Scavenger] Found unused register: " << Reg << "\n";
        return Reg;
      }
    }
    return ""; // No free register
  }

  // Scavenge a register - may spill if necessary
  std::string scavengeRegister() {
    std::string FreeReg = findUnusedReg();

    if (!FreeReg.empty()) {
      return FreeReg;
    }

    // No free registers - need to spill one
    if (!HasEmergencySlot) {
      std::cerr << "  [Scavenger] ERROR: No free registers and no emergency slot!\n";
      return "";
    }

    std::cout << "  [Scavenger] No free registers - spilling to emergency slot\n";
    std::cout << "  [Scavenger] Spilling " << *AvailableRegs.begin()
              << " to frameIdx #" << EmergencySpillSlot << "\n";

    // In real code: emit spill instructions for the spilled register
    std::string ScavengedReg = *AvailableRegs.begin();
    AvailableRegs.erase(AvailableRegs.begin());

    return ScavengedReg;
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 20: Frame Index Expansion and Register Scavenging ===\n";

  // Set up a stack frame
  SimFrameInfo MFI;

  std::cout << "\n--- Creating Stack Objects ---\n";
  int LocalA = MFI.CreateStackObject(4, 4);
  MFI.setObjectOffset(LocalA, 0);

  int LocalB = MFI.CreateStackObject(8, 8);
  MFI.setObjectOffset(LocalB, 4);

  int SpillSlot = MFI.CreateStackObject(4, 4);
  MFI.setObjectOffset(SpillSlot, 12);

  int EmergencySlot = MFI.CreateStackObject(4, 4);
  MFI.setObjectOffset(EmergencySlot, 16);

  unsigned StackSize = MFI.getStackSize();
  std::cout << "  Total stack size: " << StackSize << "\n";

  // Create instructions with frame indices
  std::cout << "\n--- Instructions Before Frame Index Elimination ---\n";

  MachineInstr LoadInstr("LDR");
  LoadInstr.addOperand(MachineOperand::createReg("r0"));
  LoadInstr.addOperand(MachineOperand::createFrameIdx(LocalA));
  LoadInstr.addOperand(MachineOperand::createImm(0));

  MachineInstr StoreInstr("STR");
  StoreInstr.addOperand(MachineOperand::createReg("r1"));
  StoreInstr.addOperand(MachineOperand::createFrameIdx(SpillSlot));
  StoreInstr.addOperand(MachineOperand::createImm(0));

  std::cout << "Before:\n";
  LoadInstr.print();
  StoreInstr.print();

  // Eliminate frame indices (with FP)
  std::cout << "\n--- Eliminating Frame Indices (with Frame Pointer) ---\n";

  FrameIndexEliminator EliminatorWithFP(true, StackSize);

  std::cout << "Load:\n";
  EliminatorWithFP.eliminateFrameIndex(LoadInstr, 1, MFI);

  std::cout << "Store:\n";
  EliminatorWithFP.eliminateFrameIndex(StoreInstr, 1, MFI);

  std::cout << "\nAfter (FP-relative):\n";
  LoadInstr.print();
  StoreInstr.print();

  // Eliminate frame indices (without FP)
  std::cout << "\n--- Eliminating Frame Indices (without Frame Pointer) ---\n";

  // Fresh instructions
  MachineInstr Load2("LDR");
  Load2.addOperand(MachineOperand::createReg("r0"));
  Load2.addOperand(MachineOperand::createFrameIdx(LocalA));
  Load2.addOperand(MachineOperand::createImm(0));

  FrameIndexEliminator EliminatorNoFP(false, StackSize);
  EliminatorNoFP.eliminateFrameIndex(Load2, 1, MFI);

  std::cout << "After (SP-relative):\n";
  Load2.print();

  // Demonstrate register scavenging
  std::cout << "\n--- Register Scavenging ---\n";

  RegScavenger RS;
  RS.addScavengingFrameIndex(EmergencySlot);

  // Scenario: all registers are used
  std::set<std::string> AllUsed = {"r0", "r1", "r2", "r3", "r12"};
  RS.setUsedRegisters(AllUsed);

  std::string Scavenged = RS.scavengeRegister();
  std::cout << "  Scavenged register: " << Scavenged << "\n";

  // Scenario: r12 is free
  std::set<std::string> SomeUsed = {"r0", "r1", "r2", "r3"};
  RS.setUsedRegisters(SomeUsed);

  Scavenged = RS.scavengeRegister();
  std::cout << "  Scavenged register: " << Scavenged << "\n";

  // Large offset scenario
  std::cout << "\n--- Large Offset Handling ---\n";
  std::cout << "  When frame index offset exceeds immediate range:\n";
  std::cout << "    1. Scavenge a temporary register\n";
  std::cout << "    2. ADDri temp, SP, #offset_high\n";
  std::cout << "    3. LDR/STR dest, [temp, #offset_low]\n";
  std::cout << "  The emergency spill slot ensures at least one register is available.\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. eliminateFrameIndex replaces frameIdx with register+offset\n";
  std::cout << "  2. Base register is FP (if available) or SP\n";
  std::cout << "  3. Register scavenging finds free registers or spills to free one\n";
  std::cout << "  4. Emergency spill slot guarantees spill space for scavenging\n";
  std::cout << "  5. Large offsets may require address computation in a temp register\n";

  return 0;
}
