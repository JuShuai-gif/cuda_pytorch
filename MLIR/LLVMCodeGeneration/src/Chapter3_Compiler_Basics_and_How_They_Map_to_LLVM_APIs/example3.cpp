// example3.cpp - Demonstrates Machine IR level basic operations
// Shows the relationship between LLVM IR and Machine IR structures.
// While creating MachineFunction directly requires a full backend setup,
// this demonstrates the conceptual mapping and key API entry points.
//
// Note: Creating MachineFunction from scratch requires a TargetMachine and
// other backend infrastructure. This example shows the API patterns used
// when working at the Machine IR level within a pass.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

// These headers are only available when building with the proper LLVM targets
// Uncomment if you have an X86 target built:
// #include "llvm/CodeGen/MachineFunction.h"
// #include "llvm/CodeGen/MachineBasicBlock.h"
// #include "llvm/CodeGen/MachineInstr.h"
// #include "llvm/CodeGen/MachineModuleInfo.h"
// #include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// This function demonstrates the conceptual relationship between LLVM IR
// constructs and their Machine IR counterparts. In a real codegen pass,
// MachineFunction would be created by the TargetMachine infrastructure.
// ---------------------------------------------------------------------------
void demonstrateMachineIRMapping(Module &M) {
  outs() << "=== LLVM IR -> Machine IR Conceptual Mapping ===\n\n";

  for (Function &F : M) {
    if (F.isDeclaration()) continue;

    outs() << "LLVM IR Function: " << F.getName() << "\n";
    outs() << "  Corresponding Machine IR class: MachineFunction\n";

    // In a real pass, you would access it via:
    // MachineFunction &MF = GetMachineFunction(F);
    // MF is created by the codegen pipeline infrastructure

    outs() << "  Machine IR properties (conceptual):\n";
    outs() << "    - MachineFunction::getFunction() returns &F\n";
    outs() << "    - MachineFunction has virtual register tracking (NoVRegs)\n";
    outs() << "    - No return type info at Machine IR level\n\n";

    for (BasicBlock &BB : F) {
      outs() << "  LLVM IR BB: " << BB.getName() << "\n";
      outs() << "    Corresponding Machine IR class: MachineBasicBlock\n";

      // MachineBasicBlock differences from BasicBlock:
      outs() << "    Key differences:\n";
      outs() << "      - Can have zero or multiple terminators\n";
      outs() << "      - Has direct predecessor/successor APIs (predXXX/succXXX)\n";
      outs() << "      - Fall-through possible if no terminator\n";

      for (Instruction &I : BB) {
        outs() << "    LLVM IR Instr: " << I.getOpcodeName();
        outs() << " (operands: " << I.getNumOperands() << ")\n";

        // In Machine IR, this would be a MachineInstr
        outs() << "      Corresponding Machine IR class: MachineInstr\n";
        outs() << "      Key differences:\n";
        outs() << "        - Definitions and operands accessed via getOperand(i)\n";
        outs() << "        - Must check MachineOperand::isDef() to distinguish\n";
        outs() << "        - Has explicit register types (virtual/physical)\n";
        outs() << "        - Uses MachineRegisterInfo for def-use chains\n";
        break; // Only show first instruction per block for brevity
      }
      outs() << "\n";
      break; // Only show first block for brevity
    }
    break; // Only show first function for brevity
  }
}

// ---------------------------------------------------------------------------
// Demonstrates how MachineRegisterInfo would be used in a Machine IR pass.
// This is pseudocode showing the API patterns described in Chapter 3.
// ---------------------------------------------------------------------------
void demonstrateRegisterInfoPattern() {
  outs() << "=== MachineRegisterInfo Usage Pattern ===\n\n";

  outs() << "// Accessing MachineRegisterInfo in a pass:\n";
  outs() << "MachineRegisterInfo &MRI = MF.getRegInfo();\n\n";

  outs() << "// Getting the unique definition of a register:\n";
  outs() << "MachineInstr *DefMI = MRI.getUniqueVRegDef(MyRegister);\n";
  outs() << "if (DefMI) {\n";
  outs() << "  // This is an SSA value - follow use-def chain\n";
  outs() << "}\n\n";

  outs() << "// Iterating over all uses of a register:\n";
  outs() << "for (MachineOperand &MO : MRI.use_operands(MyRegister)) {\n";
  outs() << "  MachineInstr &UserMI = *MO.getParent();\n";
  outs() << "  // Process the using instruction\n";
  outs() << "}\n\n";

  outs() << "// Iterating over all instructions that use a register:\n";
  outs() << "for (MachineInstr &UseMI : MRI.use_instructions(MyRegister)) {\n";
  outs() << "  // UseMI is an instruction that uses MyRegister\n";
  outs() << "}\n";
}

// ---------------------------------------------------------------------------
// Main: Build a simple IR module and demonstrate the Machine IR mapping
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  // Initialize native target for potential codegen
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();

  LLVMContext Context;
  Module M("MachineIRDemo", Context);
  M.setTargetTriple("x86_64-unknown-linux-gnu");

  // Build a simple function to demonstrate
  Type *I32 = Type::getInt32Ty(Context);
  FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "simple", M);

  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  IRBuilder<> Builder(BB);
  auto Args = F->args().begin();
  Value *A = Args++;
  Value *B = Args;
  Value *Res = Builder.CreateAdd(A, B, "res");
  Builder.CreateRet(Res);

  // Print the LLVM IR
  outs() << "; LLVM IR representation:\n";
  M.print(outs(), nullptr);
  outs() << "\n";

  // Demonstrate the conceptual mapping
  demonstrateMachineIRMapping(M);
  outs() << "\n";

  // Demonstrate register info patterns
  demonstrateRegisterInfoPattern();

  return 0;
}
