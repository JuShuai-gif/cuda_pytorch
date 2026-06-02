//===----------------------------------------------------------------------===//
// Chapter 19 - Register Allocation
// Example: Spill Code - storeRegToStackSlot and loadRegFromStackSlot
//===----------------------------------------------------------------------===//
//
// This example demonstrates the spill code infrastructure:
// - Implementing storeRegToStackSlot for saving registers to memory
// - Implementing loadRegFromStackSlot for restoring registers from memory
// - MachineMemOperand creation for proper memory dependency tracking
// - Rematerialization concepts and isReMaterializable
// - Spill slot allocation and frame index management
//
// NOTE: In real LLVM, these are virtual methods on TargetInstrInfo.
// This simulation captures the structural patterns.
//

#include <iostream>
#include <map>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated LLVM types
//------------------------------------------------------------------------------

struct MachineFunction {
  std::string Name;
  int NextFrameIdx = 0;

  MachineFunction(const std::string &N) : Name(N) {}
};

struct MachineBasicBlock {
  std::string Name;
  std::vector<std::string> Instructions;

  MachineBasicBlock(const std::string &N) : Name(N) {}

  void addInstruction(const std::string &Instr) {
    Instructions.push_back(Instr);
  }

  void print() const {
    std::cout << "  " << Name << ":\n";
    for (auto &I : Instructions) {
      std::cout << "    " << I << "\n";
    }
  }
};

// Simulated machine memory operand
struct MachineMemOperand {
  int FrameIdx;
  bool IsLoad;
  bool IsStore;
  unsigned Size;
  unsigned Align;

  MachineMemOperand(int FI, bool Load, bool Store, unsigned Sz, unsigned Al)
    : FrameIdx(FI), IsLoad(Load), IsStore(Store), Size(Sz), Align(Al) {}

  void print() const {
    std::cout << "    MMO: frameIdx=" << FrameIdx
              << (IsLoad ? " load" : "")
              << (IsStore ? " store" : "")
              << " size=" << Size << " align=" << Align;
  }
};

// Simulated target register info
struct TargetRegisterInfo {
  std::map<std::string, unsigned> RegSizes;

  TargetRegisterInfo() {
    // Register class name -> size in bytes
    RegSizes["GPR16"] = 2;
    RegSizes["GPR32"] = 4;
    RegSizes["FPR32"] = 4;
    RegSizes["FPR64"] = 8;
  }

  unsigned getSpillSize(const std::string &RegClass) const {
    auto It = RegSizes.find(RegClass);
    return (It != RegSizes.end()) ? It->second : 4;
  }
};

//------------------------------------------------------------------------------
// Simulated TargetInstrInfo with spill code hooks
//------------------------------------------------------------------------------
class SimInstrInfo {
private:
  const TargetRegisterInfo &TRI;

public:
  SimInstrInfo(const TargetRegisterInfo &R) : TRI(R) {}

  // Store a register to a stack slot
  // In real LLVM: void storeRegToStackSlot(MachineBasicBlock &MBB,
  //     MachineBasicBlock::iterator MBBI, Register SrcReg, bool isKill,
  //     int FI, const TargetRegisterClass *RC, const TargetRegisterInfo *TRI,
  //     Register VReg, MachineInstr::MIFlag Flags) const
  void storeRegToStackSlot(MachineBasicBlock &MBB, const std::string &SrcReg,
                           bool isKill, int FrameIdx,
                           const std::string &RegClass) {
    unsigned Size = TRI.getSpillSize(RegClass);

    // Choose opcode based on register size
    std::string Opcode = (Size == 2) ? "STR16" :
                         (Size == 4) ? "STR32" : "STR64";

    // Build the store instruction
    std::string Instr = Opcode + " " + SrcReg +
                        ", [sp, #" + std::to_string(FrameIdx) + "]" +
                        (isKill ? " ; kill" : "");

    MBB.addInstruction(Instr);

    std::cout << "  [Spill: Store] " << Instr
              << " (size=" << Size << ")\n";

    // In real LLVM, also create MachineMemOperand:
    // MachineMemOperand *MMO = MF.getMachineMemOperand(
    //     MachinePointerInfo::getFixedStack(MF, FI),
    //     MachineMemOperand::MOStore,
    //     MFI.getObjectSize(FI),
    //     MFI.getObjectAlign(FI));
  }

  // Load a register from a stack slot
  void loadRegFromStackSlot(MachineBasicBlock &MBB, const std::string &DestReg,
                            int FrameIdx, const std::string &RegClass) {
    unsigned Size = TRI.getSpillSize(RegClass);

    // Choose opcode based on register size
    std::string Opcode = (Size == 2) ? "LDR16" :
                         (Size == 4) ? "LDR32" : "LDR64";

    // Build the load instruction
    std::string Instr = Opcode + " " + DestReg +
                        ", [sp, #" + std::to_string(FrameIdx) + "] ; reload";

    MBB.addInstruction(Instr);

    std::cout << "  [Spill: Load]  " << Instr
              << " (size=" << Size << ")\n";

    // In real LLVM, also create MachineMemOperand for the load
  }
};

//------------------------------------------------------------------------------
// Simulated Rematerialization concept
//------------------------------------------------------------------------------
struct RematerializationInfo {
  std::string Opcode;
  bool isReMaterializable;
  int Cost; // Cost to recompute vs. cost to spill

  RematerializationInfo(const std::string &Opc, bool Remat, int C)
    : Opcode(Opc), isReMaterializable(Remat), Cost(C) {}
};

class RematerializationChecker {
private:
  std::map<std::string, RematerializationInfo> Info;

public:
  RematerializationChecker() {
    // Trivial rematerializable instructions (only constant inputs)
    Info.emplace("MOVimm", RematerializationInfo("MOVimm", true, 1));
    Info.emplace("MOVaddr", RematerializationInfo("MOVaddr", true, 1));
    Info.emplace("ADDri", RematerializationInfo("ADDri", false, -1));
    Info.emplace("LDR", RematerializationInfo("LDR", false, -1));
  }

  bool canRematerialize(const std::string &Opc) const {
    auto It = Info.find(Opc);
    return (It != Info.end()) ? It->second.isReMaterializable : false;
  }

  void shouldSpillOrRemat(const std::string &Opc, int SpillCost) const {
    auto It = Info.find(Opc);
    if (It != Info.end() && It->second.isReMaterializable) {
      int RematCost = It->second.Cost;
      std::cout << "  [Decision] " << Opc << ": "
                << "rematerialize cost=" << RematCost
                << " vs spill cost=" << SpillCost;
      if (RematCost < SpillCost) {
        std::cout << " -> REMATERIALIZE (cheaper)\n";
      } else {
        std::cout << " -> SPILL (remat is more expensive)\n";
      }
    } else {
      std::cout << "  [Decision] " << Opc << ": not rematerializable -> SPILL\n";
    }
  }
};

//------------------------------------------------------------------------------
// Simulated Frame Info management for spill slots
//------------------------------------------------------------------------------
class SimFrameInfo {
private:
  struct StackObject {
    int Idx;
    unsigned Size;
    unsigned Align;
    bool IsSpillSlot;

    StackObject(int I, unsigned S, unsigned A, bool Spill)
      : Idx(I), Size(S), Align(A), IsSpillSlot(Spill) {}
  };

  std::vector<StackObject> Objects;

public:
  int CreateStackObject(unsigned Size, unsigned Align, bool isSpill = false) {
    int Idx = Objects.size();
    Objects.emplace_back(Idx, Size, Align, isSpill);
    std::cout << "  [FrameInfo] Created stack object #" << Idx
              << ": size=" << Size << ", align=" << Align
              << (isSpill ? " (spill slot)" : "") << "\n";
    return Idx;
  }

  unsigned getObjectSize(int Idx) const {
    return Objects[Idx].Size;
  }

  unsigned getObjectAlign(int Idx) const {
    return Objects[Idx].Align;
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 19: Spill Code and Register Allocation Hooks ===\n";

  TargetRegisterInfo TRI;
  SimInstrInfo TII(TRI);
  SimFrameInfo MFI;
  RematerializationChecker RematCheck;

  // Simulate a function where register allocation needs to spill
  std::cout << "\n--- Simulating Register Spilling ---\n";

  MachineBasicBlock MBB("bb.0");

  // Create spill slots
  int SpillSlotA = MFI.CreateStackObject(4, 4, true);  // 4-byte spill slot
  int SpillSlotB = MFI.CreateStackObject(8, 8, true);  // 8-byte spill slot

  std::cout << "\n--- Spill: Store before call ---\n";
  // Before a function call, spill live registers
  TII.storeRegToStackSlot(MBB, "%r0", true, SpillSlotA, "GPR32");
  TII.storeRegToStackSlot(MBB, "%r1", true, SpillSlotA + 1, "GPR32");
  TII.storeRegToStackSlot(MBB, "%d0", true, SpillSlotB, "FPR64");

  std::cout << "\n--- Function call happens here ---\n";
  MBB.addInstruction("CALL foo");

  std::cout << "\n--- Reload after call ---\n";
  // After the call, reload the spilled registers
  TII.loadRegFromStackSlot(MBB, "%r0", SpillSlotA, "GPR32");
  TII.loadRegFromStackSlot(MBB, "%r1", SpillSlotA + 1, "GPR32");
  TII.loadRegFromStackSlot(MBB, "%d0", SpillSlotB, "FPR64");

  std::cout << "\n--- Resulting Basic Block ---\n";
  MBB.print();

  // Rematerialization decision making
  std::cout << "\n--- Rematerialization Decisions ---\n";
  RematCheck.shouldSpillOrRemat("MOVimm", 5);  // Cheaper to recompute
  RematCheck.shouldSpillOrRemat("LDR", 3);     // Must spill (not rematerializable)
  RematCheck.shouldSpillOrRemat("ADDri", 5);   // Must spill

  // Demonstrate register allocation hints
  std::cout << "\n--- Register Allocation Hooks ---\n";
  std::cout << "  Key overridable methods:\n";
  std::cout << "    1. storeRegToStackSlot() - generate spill store\n";
  std::cout << "    2. loadRegFromStackSlot() - generate spill load\n";
  std::cout << "    3. TargetRegisterInfo::shouldCoalesce() - control coalescing\n";
  std::cout << "    4. TargetRegisterInfo::getRegAllocationHints() - preferred regs\n";
  std::cout << "    5. TargetSubtargetInfo::enableSubRegLiveness() - finer tracking\n";
  std::cout << "    6. isReMaterializable field in TableGen - enable rematerialization\n";

  std::cout << "\n  Rematerialization requirements:\n";
  std::cout << "    - Set isReMaterializable = true in TableGen\n";
  std::cout << "    - Instruction must have only trivial (constant) input operands\n";
  std::cout << "    - Override TargetInstrInfo rematerialization methods for custom behavior\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Spilling = store after def, reload before use\n";
  std::cout << "  2. Implement storeRegToStackSlot and loadRegFromStackSlot\n";
  std::cout << "  3. Choose opcode based on register class size\n";
  std::cout << "  4. Create MachineMemOperand for proper aliasing info\n";
  std::cout << "  5. Rematerialization avoids spilling for cheap-to-compute values\n";
  std::cout << "  6. Emergency spill slots provide guaranteed spill space\n";

  return 0;
}
