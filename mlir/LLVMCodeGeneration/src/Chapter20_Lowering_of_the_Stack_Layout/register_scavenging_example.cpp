//===----------------------------------------------------------------------===//
// Chapter 20 - Lowering of the Stack Layout
// Example: Reserved Call Frame and Dynamic Stack Adjustment
//===----------------------------------------------------------------------===//
//
// This example demonstrates:
// - Reserved vs. non-reserved call frame
// - ADJCALLSTACKDOWN and ADJCALLSTACKUP pseudo-instructions
// - Dynamic stack adjustment around function calls
// - Callee-saved register save/restore in prologue/epilogue
//

#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Call frame management
//------------------------------------------------------------------------------
class CallFrameManager {
private:
  bool ReservedCallFrame;
  int MaxCallFrameSize;
  std::vector<std::string> Instructions;

public:
  CallFrameManager(bool Reserved, int MaxSize = 0)
    : ReservedCallFrame(Reserved), MaxCallFrameSize(MaxSize) {}

  // Called before a function call to allocate argument space
  void emitCallFrameDown(int ArgSize) {
    if (ReservedCallFrame) {
      // Space already allocated in prologue - nothing to do
      std::cout << "  [Reserved] No adjustment needed (space pre-allocated)\n";
    } else {
      // Dynamically adjust SP for this call's arguments
      std::string Instr = "ADJCALLSTACKDOWN " + std::to_string(ArgSize) +
                          ", 0, implicit-def dead $sp, implicit $sp";
      Instructions.push_back(Instr);
      std::cout << "  [Dynamic] " << Instr << "\n";
    }
  }

  // Called after a function call to deallocate argument space
  void emitCallFrameUp(int ArgSize) {
    if (!ReservedCallFrame) {
      // Restore SP after the call
      std::string Instr = "ADJCALLSTACKUP " + std::to_string(ArgSize) +
                         ", 0, implicit-def dead $sp, implicit $sp";
      Instructions.push_back(Instr);
      std::cout << "  [Dynamic] " << Instr << "\n";
    }
  }

  void printInstructions() const {
    for (auto &I : Instructions) {
      std::cout << "    " << I << "\n";
    }
  }
};

//------------------------------------------------------------------------------
// Callee-saved register management
//------------------------------------------------------------------------------
class CalleeSavedManager {
private:
  struct CalleeSave {
    std::string Reg;
    int FrameIdx;
    unsigned Size;
    CalleeSave(const std::string &R, int FI, unsigned S)
      : Reg(R), FrameIdx(FI), Size(S) {}
  };

  std::vector<CalleeSave> Registers;
  std::vector<std::string> PrologueInstructions;
  std::vector<std::string> EpilogueInstructions;

public:
  void addRegister(const std::string &Reg, int FrameIdx, unsigned Size) {
    Registers.emplace_back(Reg, FrameIdx, Size);
  }

  void emitPrologueSaves() {
    std::cout << "\n--- Callee-Saved Register Prologue ---\n";
    for (auto &CS : Registers) {
      std::string Instr = "STR " + CS.Reg + ", [SP, #" +
                          std::to_string(CS.FrameIdx) + "] ; save callee-saved";
      PrologueInstructions.push_back(Instr);
      std::cout << "  " << Instr << "\n";
    }
  }

  void emitEpilogueRestores() {
    std::cout << "\n--- Callee-Saved Register Epilogue ---\n";
    // Restore in reverse order
    for (auto It = Registers.rbegin(); It != Registers.rend(); ++It) {
      std::string Instr = "LDR " + It->Reg + ", [SP, #" +
                          std::to_string(It->FrameIdx) + "] ; restore callee-saved";
      EpilogueInstructions.push_back(Instr);
      std::cout << "  " << Instr << "\n";
    }
  }

  const std::vector<std::string> &getPrologue() const {
    return PrologueInstructions;
  }
  const std::vector<std::string> &getEpilogue() const {
    return EpilogueInstructions;
  }
};

//------------------------------------------------------------------------------
// Complete function lowering simulation
//------------------------------------------------------------------------------
void simulateFunctionLowering(bool hasVarSizedObjects, int maxCallArgsSize) {
  std::cout << "\n=== Function Lowering Simulation ===\n";
  std::cout << "  hasVarSizedObjects: " << (hasVarSizedObjects ? "yes" : "no")
            << "\n  maxCallArgsSize: " << maxCallArgsSize << "\n";

  // Decide frame strategy
  bool reservedCF = !hasVarSizedObjects;
  bool needsFP = hasVarSizedObjects || maxCallArgsSize > 256;

  std::cout << "  Strategy: callFrame="
            << (reservedCF ? "reserved" : "dynamic")
            << ", framePointer=" << (needsFP ? "yes" : "no") << "\n";

  // Set up call frame manager
  CallFrameManager CFM(reservedCF, maxCallArgsSize);

  // Set up callee-saved manager
  CalleeSavedManager CSM;
  CSM.addRegister("r4", 0, 4);
  CSM.addRegister("r5", 4, 4);
  CSM.addRegister("r6", 8, 4);

  // Emit prologue
  std::cout << "\n=== Prologue ===\n";
  std::cout << "  PUSH {LR} ; save return address\n";
  if (needsFP) {
    std::cout << "  PUSH {FP} ; save frame pointer\n";
    std::cout << "  MOV FP, SP ; set up frame pointer\n";
  }
  if (reservedCF && maxCallArgsSize > 0) {
    std::cout << "  SUB SP, #" << (16 + maxCallArgsSize)
              << " ; allocate locals + reserved call frame\n";
  } else {
    std::cout << "  SUB SP, #16 ; allocate locals only\n";
  }
  CSM.emitPrologueSaves();

  // Simulate function body with calls
  std::cout << "\n=== Function Body ===\n";

  // Call 1: 8 bytes of arguments
  std::cout << "\n  ; Call foo(1, 2)\n";
  std::cout << "  MOV r0, #1\n";
  std::cout << "  MOV r1, #2\n";
  CFM.emitCallFrameDown(8);
  std::cout << "  BL foo\n";
  CFM.emitCallFrameUp(8);

  // Call 2: 16 bytes of arguments
  std::cout << "\n  ; Call bar(a, b, c, d)\n";
  std::cout << "  MOV r0, r1\n";
  std::cout << "  MOV r1, r2\n";
  std::cout << "  MOV r2, r3\n";
  std::cout << "  STR r4, [SP, #0] ; 4th arg on stack\n";
  CFM.emitCallFrameDown(16);
  std::cout << "  BL bar\n";
  CFM.emitCallFrameUp(16);

  // Emit epilogue
  std::cout << "\n=== Epilogue ===\n";
  CSM.emitEpilogueRestores();
  if (needsFP) {
    std::cout << "  MOV SP, FP ; restore stack pointer\n";
    std::cout << "  POP {FP} ; restore frame pointer\n";
  } else {
    std::cout << "  ADD SP, #16 ; restore stack pointer\n";
  }
  std::cout << "  POP {PC} ; return\n";
}

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 20: Reserved Call Frame and Dynamic Stack ===\n\n";

  // Scenario 1: Fixed-size frame, reserved call frame
  std::cout << "--- Scenario 1: Fixed-Size Frame with Reserved Call Frame ---\n";
  simulateFunctionLowering(false, 64);

  // Scenario 2: Variable-sized objects, dynamic call frame
  std::cout << "\n\n--- Scenario 2: Variable-Sized Objects with Dynamic Call Frame ---\n";
  simulateFunctionLowering(true, 64);

  // Compare ADJCALLSTACKDOWN/UP
  std::cout << "\n\n--- ADJCALLSTACKDOWN/UP Pseudo-Instructions ---\n";
  std::cout << "  Purpose: Adjust SP around function calls when the call frame\n";
  std::cout << "           is NOT reserved (dynamic adjustment).\n";
  std::cout << "  ADJCALLSTACKDOWN: SUB SP, #args_size (before call)\n";
  std::cout << "  ADJCALLSTACKUP:   ADD SP, #args_size (after call)\n";
  std::cout << "  \n";
  std::cout << "  With reserved call frame:\n";
  std::cout << "    - Arg space allocated once in prologue\n";
  std::cout << "    - No per-call adjustment needed\n";
  std::cout << "    - ADJCALLSTACKDOWN/UP become no-ops\n";
  std::cout << "  \n";
  std::cout << "  Without reserved call frame:\n";
  std::cout << "    - Arg space allocated before each call\n";
  std::cout << "    - Deallocated after each call\n";
  std::cout << "    - Required when hasVarSizedObjects() is true\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Reserved call frame: allocate arg space once in prologue\n";
  std::cout << "  2. Dynamic call frame: ADJCALLSTACKDOWN/UP around each call\n";
  std::cout << "  3. hasReservedCallFrame() controls which strategy is used\n";
  std::cout << "  4. Variable-sized objects force dynamic call frame\n";
  std::cout << "  5. Callee-saved registers must be saved/restored in prologue/epilogue\n";
  std::cout << "  6. PrologueEpilogInserter pass orchestrates the entire process\n";

  return 0;
}
