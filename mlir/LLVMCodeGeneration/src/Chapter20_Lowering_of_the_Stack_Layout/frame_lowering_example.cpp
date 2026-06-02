//===----------------------------------------------------------------------===//
// Chapter 20 - Lowering of the Stack Layout
// Example: Frame Lowering Hooks - Prologue, Epilogue, and Frame Management
//===----------------------------------------------------------------------===//
//
// This example demonstrates the frame lowering infrastructure:
// - emitPrologue: Save frame pointer, adjust stack pointer, save callee-saved regs
// - emitEpilogue: Restore callee-saved regs, restore SP, restore FP
// - hasFP / hasReservedCallFrame decisions
// - Stack frame layout and object allocation
//
// NOTE: In real LLVM, this code lives in TargetFrameLowering subclasses.
// This simulation captures the key patterns.
//

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated frame objects and layout
//------------------------------------------------------------------------------
struct StackObject {
  int Idx;
  unsigned Size;
  unsigned Align;
  std::string Purpose; // "local", "spill", "callee-saved", "argument"
  int Offset;          // Computed offset from SP/FP

  StackObject(int I, unsigned S, unsigned A, const std::string &P)
    : Idx(I), Size(S), Align(A), Purpose(P), Offset(0) {}
};

struct CalleeSavedInfo {
  std::string RegName;
  int FrameIdx; // Which stack slot holds this register
  unsigned Size;

  CalleeSavedInfo(const std::string &R, int FI, unsigned S)
    : RegName(R), FrameIdx(FI), Size(S) {}
};

//------------------------------------------------------------------------------
// Simulated MachineFrameInfo
//------------------------------------------------------------------------------
class SimMachineFrameInfo {
private:
  std::vector<StackObject> Objects;
  std::vector<CalleeSavedInfo> CSI;
  bool HasVarSizedObjects;
  unsigned StackSize;
  bool AdjustsStack;

public:
  SimMachineFrameInfo()
    : HasVarSizedObjects(false), StackSize(0), AdjustsStack(true) {}

  // Create a fixed stack object at a known offset
  int CreateFixedObject(unsigned Size, int Offset, bool IsImmutable) {
    int Idx = Objects.size();
    Objects.emplace_back(Idx, Size, 0, "fixed");
    Objects.back().Offset = Offset;
    std::cout << "  [MFI] Created fixed object #" << Idx
              << ": size=" << Size << ", offset=" << Offset << "\n";
    return Idx;
  }

  // Create a stack object for spills/locals
  int CreateStackObject(unsigned Size, unsigned Align, bool isSS) {
    int Idx = Objects.size();
    Objects.emplace_back(Idx, Size, Align, isSS ? "spill" : "local");
    std::cout << "  [MFI] Created stack object #" << Idx
              << ": size=" << Size << ", align=" << Align
              << (isSS ? " (spill)" : " (local)") << "\n";
    return Idx;
  }

  // Mark variable-sized objects exist (alloca/VLA)
  void setHasVarSizedObjects(bool V) { HasVarSizedObjects = V; }
  bool hasVarSizedObjects() const { return HasVarSizedObjects; }

  // Calculated stack size
  void setStackSize(unsigned S) { StackSize = S; }
  unsigned getStackSize() const { return StackSize; }

  // Callee-saved register tracking
  void setCalleeSavedInfo(const std::vector<CalleeSavedInfo> &C) { CSI = C; }
  const std::vector<CalleeSavedInfo> &getCalleeSavedInfo() const { return CSI; }

  // Object access
  unsigned getObjectSize(int Idx) const { return Objects[Idx].Size; }
  unsigned getObjectAlign(int Idx) const { return Objects[Idx].Align; }
  int getObjectOffset(int Idx) const { return Objects[Idx].Offset; }

  bool adjustsStack() const { return AdjustsStack; }

  void print() const {
    std::cout << "  Frame Info:\n";
    std::cout << "    StackSize: " << StackSize << "\n";
    std::cout << "    VarSizedObjects: " << (HasVarSizedObjects ? "yes" : "no") << "\n";
    std::cout << "    Objects:\n";
    for (auto &Obj : Objects) {
      std::cout << "      #" << Obj.Idx << ": " << Obj.Purpose
                << " size=" << Obj.Size << " align=" << Obj.Align
                << " offset=" << Obj.Offset << "\n";
    }
    std::cout << "    Callee-Saved:\n";
    for (auto &CS : CSI) {
      std::cout << "      " << CS.RegName << " -> frameIdx="
                << CS.FrameIdx << " (size=" << CS.Size << ")\n";
    }
  }
};

//------------------------------------------------------------------------------
// Simulated TargetRegisterInfo for frame lowering
//------------------------------------------------------------------------------
class SimTargetRegisterInfo {
public:
  std::string getFrameRegister() const { return "FP"; }
  std::string getStackPointer() const { return "SP"; }
  std::string getReturnAddressReg() const { return "LR"; }

  // Which registers are callee-saved
  std::vector<std::string> getCalleeSavedRegs() const {
    return {"r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "FP", "LR"};
  }
};

//------------------------------------------------------------------------------
// Simulated TargetInstrInfo for building prologue/epilogue
//------------------------------------------------------------------------------
class SimInstrInfo {
public:
  std::string getPUSH(const std::string &Reg) const {
    return "PUSH {" + Reg + "}";
  }
  std::string getPOP(const std::string &Reg) const {
    return "POP {" + Reg + "}";
  }
  std::string getMOV(const std::string &Dst, const std::string &Src) const {
    return "MOV " + Dst + ", " + Src;
  }
  std::string getSUBri(const std::string &Dst, int Imm) const {
    return "SUB " + Dst + ", #" + std::to_string(Imm);
  }
  std::string getADDri(const std::string &Dst, int Imm) const {
    return "ADD " + Dst + ", #" + std::to_string(Imm);
  }
  std::string getSTR(const std::string &Reg, int Offset) const {
    return "STR " + Reg + ", [SP, #" + std::to_string(Offset) + "]";
  }
  std::string getLDR(const std::string &Reg, int Offset) const {
    return "LDR " + Reg + ", [SP, #" + std::to_string(Offset) + "]";
  }
};

//------------------------------------------------------------------------------
// Frame Lowering class (simulated TargetFrameLowering)
//------------------------------------------------------------------------------
class SimFrameLowering {
private:
  SimTargetRegisterInfo TRI;
  SimInstrInfo TII;

public:
  // Determine if a frame pointer is needed
  bool hasFP(const SimMachineFrameInfo &MFI) const {
    // Need FP if: variable-sized objects, dynamic stack realignment,
    // or frame is too large for immediate offsets
    if (MFI.hasVarSizedObjects()) return true;
    if (MFI.getStackSize() > 1024) return true;
    return false; // Small, fixed-size frames can use SP-relative addressing
  }

  // Determine if the call frame is reserved
  bool hasReservedCallFrame(const SimMachineFrameInfo &MFI) const {
    // Reserve call frame if no variable-sized objects
    // With reserved call frame: allocate arg space once in prologue
    // Without: use ADJCALLSTACKDOWN/UP around each call
    return !MFI.hasVarSizedObjects();
  }

  // Emit function prologue
  void emitPrologue(SimMachineFrameInfo &MFI,
                    std::vector<std::string> &Prologue) {
    std::cout << "\n=== Emitting Prologue ===\n";

    unsigned StackSize = MFI.getStackSize();
    bool NeedsFP = hasFP(MFI);

    // Step 1: Save return address
    Prologue.push_back(TII.getPUSH(TRI.getReturnAddressReg()));
    std::cout << "  1. Save LR: " << Prologue.back() << "\n";

    // Step 2: Save frame pointer and set up new frame
    if (NeedsFP) {
      Prologue.push_back(TII.getPUSH(TRI.getFrameRegister()));
      Prologue.push_back(TII.getMOV(TRI.getFrameRegister(),
                                     TRI.getStackPointer()));
      std::cout << "  2. Save FP and set FP = SP\n";
    }

    // Step 3: Adjust stack pointer for local frame
    if (StackSize > 0) {
      Prologue.push_back(TII.getSUBri(TRI.getStackPointer(), StackSize));
      std::cout << "  3. Adjust SP -= " << StackSize << "\n";
    } else {
      std::cout << "  3. No stack adjustment needed\n";
    }

    // Step 4: Save callee-saved registers
    std::cout << "  4. Saving callee-saved registers:\n";
    for (auto &CS : MFI.getCalleeSavedInfo()) {
      Prologue.push_back(TII.getSTR(CS.RegName, MFI.getObjectOffset(CS.FrameIdx)));
      std::cout << "     " << Prologue.back() << "\n";
    }

    std::cout << "\n  Prologue generated (" << Prologue.size() << " instructions)\n";
  }

  // Emit function epilogue
  void emitEpilogue(SimMachineFrameInfo &MFI,
                    std::vector<std::string> &Epilogue) {
    std::cout << "\n=== Emitting Epilogue ===\n";

    unsigned StackSize = MFI.getStackSize();
    bool NeedsFP = hasFP(MFI);

    // Step 1: Restore callee-saved registers (reverse order)
    std::cout << "  1. Restoring callee-saved registers:\n";
    const auto &CSI = MFI.getCalleeSavedInfo();
    for (auto It = CSI.rbegin(); It != CSI.rend(); ++It) {
      Epilogue.push_back(TII.getLDR(It->RegName, MFI.getObjectOffset(It->FrameIdx)));
      std::cout << "     " << Epilogue.back() << "\n";
    }

    // Step 2: Restore stack pointer
    if (NeedsFP) {
      Epilogue.push_back(TII.getMOV(TRI.getStackPointer(),
                                     TRI.getFrameRegister()));
      std::cout << "  2. Restore SP = FP\n";
    } else if (StackSize > 0) {
      Epilogue.push_back(TII.getADDri(TRI.getStackPointer(), StackSize));
      std::cout << "  2. Restore SP += " << StackSize << "\n";
    }

    // Step 3: Restore frame pointer (if used)
    if (NeedsFP) {
      Epilogue.push_back(TII.getPOP(TRI.getFrameRegister()));
      std::cout << "  3. Restore FP\n";
    }

    // Step 4: Restore return address and return
    Epilogue.push_back(TII.getPOP("PC")); // POP {PC} = return
    std::cout << "  4. Return (POP PC)\n";

    std::cout << "\n  Epilogue generated (" << Epilogue.size() << " instructions)\n";
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 20: Stack Frame Lowering ===\n";

  SimFrameLowering TFL;
  SimMachineFrameInfo MFI;

  // Set up a typical stack frame
  std::cout << "\n--- Setting Up Stack Frame ---\n";

  // Fixed objects: return address, frame pointer save slots
  int RASlot = MFI.CreateFixedObject(4, 0, true);  // LR save at offset 0
  int FPSlot = MFI.CreateFixedObject(4, -4, true); // FP save at offset -4

  // Callee-saved register spill slots
  int r4Slot = MFI.CreateStackObject(4, 4, false);
  int r5Slot = MFI.CreateStackObject(4, 4, false);
  int r6Slot = MFI.CreateStackObject(4, 4, false);

  // Local variables
  int LocalA = MFI.CreateStackObject(8, 8, false); // 8-byte local
  int LocalB = MFI.CreateStackObject(16, 16, false); // 16-byte aligned local

  // Spill slot
  int SpillSlot = MFI.CreateStackObject(4, 4, true);

  // Register callee-saved info
  std::vector<CalleeSavedInfo> CSI = {
    {"r4", r4Slot, 4},
    {"r5", r5Slot, 4},
    {"r6", r6Slot, 4},
  };
  MFI.setCalleeSavedInfo(CSI);

  // Compute layout (simplified)
  int Offset = 0;
  // In real LLVM, stack grows downward, so offsets are negative
  // For simplicity, we track absolute offsets
  MFI.setStackSize(64); // Total frame size

  // Print frame info
  MFI.print();

  // Frame decisions
  std::cout << "\n--- Frame Lowering Decisions ---\n";
  std::cout << "  hasFP: " << (TFL.hasFP(MFI) ? "yes" : "no") << "\n";
  std::cout << "  hasReservedCallFrame: "
            << (TFL.hasReservedCallFrame(MFI) ? "yes" : "no") << "\n";

  // Emit prologue
  std::vector<std::string> Prologue;
  TFL.emitPrologue(MFI, Prologue);

  std::cout << "\n--- Final Prologue ---\n";
  for (auto &Instr : Prologue) {
    std::cout << "  " << Instr << "\n";
  }

  // Emit epilogue
  std::vector<std::string> Epilogue;
  TFL.emitEpilogue(MFI, Epilogue);

  std::cout << "\n--- Final Epilogue ---\n";
  for (auto &Instr : Epilogue) {
    std::cout << "  " << Instr << "\n";
  }

  // Frame pointer vs. stack pointer addressing
  std::cout << "\n--- SP-relative vs. FP-relative Addressing ---\n";
  std::cout << "  SP-relative: offset from current SP (changes during function)\n";
  std::cout << "  FP-relative: offset from FP (fixed throughout function)\n";
  std::cout << "  FP-relative offset = object_offset + stack_size (for negative offsets)\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Prologue saves LR, FP, adjusts SP, saves callee-saved regs\n";
  std::cout << "  2. Epilogue reverses prologue: restore regs, SP, FP, return\n";
  std::cout << "  3. hasFP() decides if frame pointer is needed\n";
  std::cout << "  4. hasReservedCallFrame() decides if arg space is pre-allocated\n";
  std::cout << "  5. MachineFrameInfo manages stack object layout\n";
  std::cout << "  6. ADJCALLSTACKDOWN/UP used when call frame is not reserved\n";

  return 0;
}
