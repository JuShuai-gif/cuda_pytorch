// example1.cpp - Machine IR Concepts: Building and inspecting MachineFunction, MachineBasicBlock,
// MachineInstr, and MachineOperand using LLVM's MIR builder APIs.
//
// This example demonstrates:
// - Creating a MachineFunction with MachineBasicBlocks
// - Using BuildMI to create MachineInstr objects
// - Adding register operands (defs, uses, implicit)
// - Inspecting SSA form properties
// - Iterating over instructions and operands
//
// Build with:
//   clang++ -o example1 example1.cpp $(llvm-config --cxxflags --ldflags --libs core codegen)
//
// Requires LLVM built with a target (e.g., X86, AArch64). This example uses the X86 target.

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

using namespace llvm;

int main() {
    // Initialize native target and its MC layer for codegen.
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    LLVMContext Context;
    auto M = std::make_unique<Module>("mir_demo", Context);

    // Create a dummy Function to anchor the MachineFunction.
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "demo_func", M.get());

    // Create a target machine for codegen (use host triple).
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Error looking up target: " << Error << "\n";
        return 1;
    }

    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "", Options,
                                       std::nullopt));

    // Create MachineModuleInfo and MachineFunction.
    MachineModuleInfo MMI(TM.get());
    MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);

    // Access target-specific instruction and register info.
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    MachineRegisterInfo &MRI = MF.getRegInfo();

    // Set SSA property: Machine IR starts in SSA form before register allocation.
    MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);

    std::cout << "=== MachineFunction created ===" << std::endl;
    std::cout << "Function name: " << MF.getName() << std::endl;
    std::cout << "Is SSA: " << (MF.getProperties().hasProperty(MachineFunctionProperties::Property::IsSSA) ? "yes" : "no") << std::endl;

    // Create MachineBasicBlock 0 (entry block).
    MachineBasicBlock *MBB0 = MF.CreateMachineBasicBlock();
    MF.push_back(MBB0);
    MBB0->setName("entry");

    // Create MachineBasicBlock 1 (exit block).
    MachineBasicBlock *MBB1 = MF.CreateMachineBasicBlock();
    MF.push_back(MBB1);
    MBB1->setName("exit");

    // Add successors to MBB0.
    MBB0->addSuccessor(MBB1);

    std::cout << "\n=== Basic Blocks ===" << std::endl;
    for (MachineBasicBlock &MBB : MF) {
        std::cout << "  BB#" << MBB.getNumber() << " (" << MBB.getName() << ")" << std::endl;
        if (!MBB.succ_empty()) {
            std::cout << "    -> successors: ";
            for (MachineBasicBlock *Succ : MBB.successors())
                std::cout << "BB#" << Succ->getNumber() << " ";
            std::cout << std::endl;
        }
    }

    // Create a virtual register of the general-purpose register class.
    // Use the first available GPR class from the target.
    const TargetRegisterClass *RC = nullptr;
    for (const TargetRegisterClass *RCI : TRI->regclasses()) {
        if (RCI->isAllocatable() && RCI->getSizeInBits() >= 32) {
            RC = RCI;
            break;
        }
    }
    if (!RC) {
        std::cerr << "No allocatable register class found!\n";
        return 1;
    }

    Register VReg1 = MRI.createVirtualRegister(RC);
    Register VReg2 = MRI.createVirtualRegister(RC);
    Register VReg3 = MRI.createVirtualRegister(RC);

    std::cout << "\n=== Virtual Registers Created ===" << std::endl;
    std::cout << "  %" << Register::virtReg2Index(VReg1) << " (class: " << TRI->getRegClassName(RC) << ")" << std::endl;
    std::cout << "  %" << Register::virtReg2Index(VReg2) << " (class: " << TRI->getRegClassName(RC) << ")" << std::endl;
    std::cout << "  %" << Register::virtReg2Index(VReg3) << " (class: " << TRI->getRegClassName(RC) << ")" << std::endl;
    std::cout << "  Total virtual registers: " << MRI.getNumVirtRegs() << std::endl;

    // Build instruction: %vreg2 = COPY %vreg1 (explicit def, explicit use).
    // Use TargetOpcode::COPY which is target-independent.
    BuildMI(MBB0, MBB0->end(), DebugLoc(), TII->get(TargetOpcode::COPY), VReg2)
        .addReg(VReg1);
    std::cout << "\n  Built: %" << Register::virtReg2Index(VReg2)
              << " = COPY %" << Register::virtReg2Index(VReg1) << std::endl;

    // Build instruction: %vreg3 = ADD %vreg1, %vreg2 (if target has ADD)
    // Since ADD is target-specific, we check if the target has a generic add.
    // For demonstration, we use another COPY as a stand-in.
    BuildMI(MBB0, MBB0->end(), DebugLoc(), TII->get(TargetOpcode::COPY), VReg3)
        .addReg(VReg2);
    std::cout << "  Built: %" << Register::virtReg2Index(VReg3)
              << " = COPY %" << Register::virtReg2Index(VReg2) << std::endl;

    // Build instruction with implicit use (e.g., using flags register).
    // We look for a physical register to use as implicit operand.
    // The stack pointer is usually available and can serve as an example.
    std::cout << "\n=== Demonstrating Implicit Operand ===" << std::endl;
    Register SP = TRI->getStackRegister();
    if (SP != 0) {
        std::cout << "  Stack pointer: " << TRI->getName(SP) << " (physreg $" << SP << ")" << std::endl;

        // Build a COPY that implicitly uses SP (modeling a use that is always there).
        MachineInstrBuilder MIB = BuildMI(MBB0, MBB0->end(), DebugLoc(),
                                          TII->get(TargetOpcode::COPY), VReg3)
                                     .addReg(VReg2)
                                     .addReg(SP, RegState::Implicit);
        std::cout << "  Added implicit use of $" << TRI->getName(SP) << std::endl;
    }

    // Iterate over all instructions in MBB0 and inspect operands.
    std::cout << "\n=== MachineInstr Inspection ===" << std::endl;
    for (MachineInstr &MI : *MBB0) {
        std::cout << "  Instruction: " << TII->getName(MI.getOpcode()) << std::endl;

        // Inspect the MCInstrDesc.
        const MCInstrDesc &Desc = MI.getDesc();
        std::cout << "    Num operands (static): " << Desc.getNumOperands() << std::endl;
        std::cout << "    Num defs (static): " << Desc.getNumDefs() << std::endl;
        std::cout << "    May load: " << (MI.mayLoad() ? "yes" : "no") << std::endl;
        std::cout << "    May store: " << (MI.mayStore() ? "yes" : "no") << std::endl;
        std::cout << "    Is commutable: " << (MI.isCommutable() ? "yes" : "no") << std::endl;

        // Iterate over MachineOperands.
        for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
            const MachineOperand &MO = MI.getOperand(i);
            std::cout << "    Operand[" << i << "]: ";

            if (MO.isReg()) {
                Register Reg = MO.getReg();
                std::cout << "Register ";
                if (Reg.isPhysical())
                    std::cout << "$" << TRI->getName(Reg);
                else if (Reg.isVirtual())
                    std::cout << "%" << Register::virtReg2Index(Reg);

                if (MO.isDef())  std::cout << " [def]";
                if (MO.isUse())   std::cout << " [use]";
                if (MO.isImplicit()) std::cout << " [implicit]";
                if (MO.isDead())  std::cout << " [dead]";
                if (MO.isKill())  std::cout << " [kill]";
                if (MO.isTied())  std::cout << " [tied]";
                if (MO.isEarlyClobber()) std::cout << " [early-clobber]";

                if (MO.getSubReg() != 0)
                    std::cout << " sub:" << TRI->getSubRegIndexName(MO.getSubReg());
            } else if (MO.isImm()) {
                std::cout << "Immediate " << MO.getImm();
            } else if (MO.isSymbol()) {
                std::cout << "Symbol";
            } else if (MO.isRegMask()) {
                std::cout << "RegisterMask";
            } else if (MO.isGlobal()) {
                std::cout << "Global";
            } else {
                std::cout << "Other(type=" << MO.getType() << ")";
            }
            std::cout << std::endl;
        }
    }

    // Demonstrate SSA check.
    std::cout << "\n=== SSA Verification ===" << std::endl;
    // Count definitions of each virtual register.
    std::map<Register, unsigned> DefCount;
    for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
            for (MachineOperand &MO : MI.operands()) {
                if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
                    DefCount[MO.getReg()]++;
                }
            }
        }
    }
    bool IsSSA = true;
    for (auto &KV : DefCount) {
        unsigned Idx = Register::virtReg2Index(KV.first);
        std::cout << "  %" << Idx << " defined " << KV.second << " time(s)";
        if (KV.second > 1) {
            std::cout << " (VIOLATES SSA!)";
            IsSSA = false;
        }
        std::cout << std::endl;
    }
    std::cout << "  SSA check passed: " << (IsSSA ? "yes" : "no") << std::endl;

    // Add a terminator to MBB0 to jump to MBB1 (using a generic branch if available).
    // Use a simple fallthrough (no explicit branch) for demonstration.
    // G_BR is a generic opcode available when GlobalISel is enabled; for simplicity,
    // we note that in real backends, target-specific branch opcodes are used.
    std::cout << "\n=== Successors ===" << std::endl;
    if (!MBB0->succ_empty()) {
        for (MachineBasicBlock *Succ : MBB0->successors())
            std::cout << "  MBB0 -> BB#" << Succ->getNumber() << std::endl;
    }

    std::cout << "\nMachine IR example completed successfully!" << std::endl;
    return 0;
}
