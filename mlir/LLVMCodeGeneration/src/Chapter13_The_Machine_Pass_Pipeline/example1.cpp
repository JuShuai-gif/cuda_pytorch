// example1.cpp - Machine Pass Pipeline: Creating and registering custom machine passes
//
// Demonstrates:
// - Writing a custom MachineFunctionPass
// - Registering it with the pass manager
// - Implementing getAnalysisUsage() for machine analyses
// - Iterating over MachineInstrs in a pass
// - Basic machine IR transformations
//
// Build with:
//   clang++ -o example1 example1.cpp $(llvm-config --cxxflags --ldflags --libs core codegen)
//
// This demonstrates the structure of machine passes. To actually run the pass,
// it would need to be registered with a TargetPassConfig.

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include <iostream>

using namespace llvm;

// ============================================================
// Example 1: A simple counting pass (inspects Machine IR)
// ============================================================

namespace {

class MachineInstrCounter : public MachineFunctionPass {
public:
    static char ID;

    MachineInstrCounter() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override {
        return "Machine Instruction Counter";
    }

    // Declare required analyses. Machine passes use machine-specific analyses.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll(); // This pass does not modify the IR.
        MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
        const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
        const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

        unsigned TotalInstrs = 0;
        unsigned CopyInstrs = 0;
        unsigned BranchInstrs = 0;
        unsigned LoadStoreInstrs = 0;
        unsigned DefCount = 0;
        unsigned ImplicitCount = 0;

        std::map<StringRef, unsigned> OpcodeFreq;

        for (MachineBasicBlock &MBB : MF) {
            for (MachineInstr &MI : MBB) {
                TotalInstrs++;

                // Track opcode frequency.
                OpcodeFreq[TII->getName(MI.getOpcode())]++;

                // Categorize instructions.
                if (MI.isCopy())  CopyInstrs++;
                if (MI.isBranch()) BranchInstrs++;
                if (MI.mayLoad() || MI.mayStore()) LoadStoreInstrs++;

                // Count definitions and implicit operands.
                for (const MachineOperand &MO : MI.operands()) {
                    if (MO.isReg() && MO.isDef()) {
                        DefCount++;
                        if (MO.isImplicit()) ImplicitCount++;
                    }
                }
            }
        }

        // Print statistics.
        std::cout << "\n--- " << getPassName() << " ---" << std::endl;
        std::cout << "Function: " << MF.getName() << std::endl;
        std::cout << "  Basic blocks:     " << MF.getNumBlockIDs() << std::endl;
        std::cout << "  Total instrs:     " << TotalInstrs << std::endl;
        std::cout << "  Copy instrs:      " << CopyInstrs << std::endl;
        std::cout << "  Branch instrs:    " << BranchInstrs << std::endl;
        std::cout << "  Load/Store instrs:" << LoadStoreInstrs << std::endl;
        std::cout << "  Register defs:    " << DefCount << std::endl;
        std::cout << "  Implicit defs:    " << ImplicitCount << std::endl;
        std::cout << "  Is SSA:           "
                  << (MF.getProperties().hasProperty(
                          MachineFunctionProperties::Property::IsSSA)
                          ? "yes" : "no")
                  << std::endl;

        // Print top 5 most frequent opcodes.
        std::cout << "  Top opcodes:" << std::endl;
        std::vector<std::pair<StringRef, unsigned>> Sorted(
            OpcodeFreq.begin(), OpcodeFreq.end());
        std::sort(Sorted.begin(), Sorted.end(),
                  [](const auto &A, const auto &B) { return A.second > B.second; });
        for (unsigned i = 0; i < std::min<size_t>(Sorted.size(), 5); ++i) {
            std::cout << "    " << Sorted[i].first << ": " << Sorted[i].second << std::endl;
        }

        // Also inspect the first basic block's live-ins.
        if (!MF.empty()) {
            MachineBasicBlock &EntryMBB = MF.front();
            if (!EntryMBB.livein_empty()) {
                std::cout << "  Entry block live-ins:" << std::endl;
                for (const MachineBasicBlock::RegisterMaskPair &LI : EntryMBB.liveins()) {
                    std::cout << "    $" << TRI->getName(LI.PhysReg) << std::endl;
                }
            }
        }

        std::cout << "-------------------------------------------" << std::endl;
        return false; // Did not modify the function.
    }
};

char MachineInstrCounter::ID = 0;

} // end anonymous namespace

// ============================================================
// Example 2: A simple optimization pass (folds redundant copies)
// ============================================================

namespace {

class SimpleCopyPropagator : public MachineFunctionPass {
public:
    static char ID;

    SimpleCopyPropagator() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override {
        return "Simple Copy Propagator";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesCFG(); // Does not change control flow.
        MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
        const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
        const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

        bool Changed = false;
        std::map<Register, Register> CopyMap; // src -> dst mapping

        for (MachineBasicBlock &MBB : MF) {
            // Maps are local to each basic block for simplicity.
            CopyMap.clear();

            for (auto MII = MBB.begin(); MII != MBB.end(); ) {
                MachineInstr &MI = *MII;
                ++MII; // Increment before potentially erasing.

                // Check if this is a register-to-register copy.
                if (MI.isCopy() && MI.getNumOperands() == 2) {
                    MachineOperand &DstMO = MI.getOperand(0);
                    MachineOperand &SrcMO = MI.getOperand(1);

                    if (DstMO.isReg() && DstMO.isDef() &&
                        SrcMO.isReg() && SrcMO.isUse() &&
                        DstMO.getReg().isVirtual() &&
                        SrcMO.getReg().isVirtual()) {

                        Register Dst = DstMO.getReg();
                        Register Src = SrcMO.getReg();

                        // Record the copy: any use of Dst can be replaced by Src.
                        CopyMap[Dst] = Src;
                        std::cout << "  [CopyProp] Found copy: %"
                                  << Register::virtReg2Index(Dst)
                                  << " = COPY %"
                                  << Register::virtReg2Index(Src) << std::endl;

                        // Don't remove the copy yet (in real passes, it can be
                        // removed if Dst has no other uses).
                        continue;
                    }
                }

                // Try to propagate: replace uses of copied registers.
                for (MachineOperand &MO : MI.operands()) {
                    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual()) {
                        Register Reg = MO.getReg();
                        auto It = CopyMap.find(Reg);
                        if (It != CopyMap.end()) {
                            std::cout << "  [CopyProp] Replacing %"
                                      << Register::virtReg2Index(Reg)
                                      << " with %"
                                      << Register::virtReg2Index(It->second)
                                      << " in " << TII->getName(MI.getOpcode())
                                      << std::endl;
                            MO.setReg(It->second);
                            Changed = true;
                        }
                    }
                }
            }
        }

        return Changed;
    }
};

char SimpleCopyPropagator::ID = 1;

} // end anonymous namespace

// ============================================================
// Example 3: A dead instruction eliminator (demonstrates analysis)
// ============================================================

namespace {

class SimpleDeadInstEliminator : public MachineFunctionPass {
public:
    static char ID;

    SimpleDeadInstEliminator() : MachineFunctionPass(ID) {}

    StringRef getPassName() const override {
        return "Simple Dead Instruction Eliminator";
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesCFG();
        MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
        const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
        MachineRegisterInfo &MRI = MF.getRegInfo();
        bool Changed = false;

        // Build a use set: for each virtual register, count uses.
        std::map<Register, unsigned> UseCount;

        for (MachineBasicBlock &MBB : MF) {
            for (MachineInstr &MI : MBB) {
                for (const MachineOperand &MO : MI.operands()) {
                    if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual()) {
                        UseCount[MO.getReg()]++;
                    }
                }
            }
        }

        // Find and report dead definitions (defined but never used, except for
        // instructions with side effects or that define physical registers).
        for (MachineBasicBlock &MBB : MF) {
            for (MachineInstr &MI : MBB) {
                // Skip instructions with side effects (stores, calls, branches).
                if (MI.mayStore() || MI.isCall() || MI.isBranch() ||
                    MI.isReturn() || MI.hasUnmodeledSideEffects())
                    continue;

                bool HasDeadDef = false;
                for (const MachineOperand &MO : MI.operands()) {
                    if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
                        Register Reg = MO.getReg();
                        if (MO.isDead() || UseCount[Reg] == 0) {
                            HasDeadDef = true;
                            std::cout << "  [DeadElim] Dead def: %"
                                      << Register::virtReg2Index(Reg)
                                      << " in " << TII->getName(MI.getOpcode())
                                      << " (uses=" << UseCount[Reg] << ")"
                                      << std::endl;
                        }
                    }
                }

                if (HasDeadDef) {
                    std::cout << "  [DeadElim] Would remove: "
                              << TII->getName(MI.getOpcode()) << std::endl;
                    Changed = true;
                    // In a real pass: MI.eraseFromParent();
                }
            }
        }

        return Changed;
    }
};

char SimpleDeadInstEliminator::ID = 2;

} // end anonymous namespace

// ============================================================
// Registration and demonstration
// ============================================================

// Register the passes with LLVM's pass registry.
INITIALIZE_PASS_BEGIN(MachineInstrCounter, "mir-counter",
                      "Machine Instruction Counter", false, true)
INITIALIZE_PASS_END(MachineInstrCounter, "mir-counter",
                    "Machine Instruction Counter", false, true)

INITIALIZE_PASS_BEGIN(SimpleCopyPropagator, "simple-copy-prop",
                      "Simple Copy Propagator", false, false)
INITIALIZE_PASS_END(SimpleCopyPropagator, "simple-copy-prop",
                    "Simple Copy Propagator", false, false)

INITIALIZE_PASS_BEGIN(SimpleDeadInstEliminator, "simple-dead-inst-elim",
                      "Simple Dead Instruction Eliminator", false, false)
INITIALIZE_PASS_END(SimpleDeadInstEliminator, "simple-dead-inst-elim",
                    "Simple Dead Instruction Eliminator", false, false)

// Factory functions for pass creation.
MachineFunctionPass *createMachineInstrCounter() {
    return new MachineInstrCounter();
}
MachineFunctionPass *createSimpleCopyPropagator() {
    return new SimpleCopyPropagator();
}
MachineFunctionPass *createSimpleDeadInstEliminator() {
    return new SimpleDeadInstEliminator();
}

// ============================================================
// Main: Demonstrate pass concepts without needing a full pipeline.
// ============================================================

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

    // Register our custom passes.
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeMachineInstrCounterPass(Registry);
    initializeSimpleCopyPropagatorPass(Registry);
    initializeSimpleDeadInstEliminatorPass(Registry);

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Error: " << Error << "\n";
        return 1;
    }

    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "",
                                       Options, std::nullopt));

    // Create a simple module and function with Machine IR.
    LLVMContext Context;
    auto M = std::make_unique<Module>("pass_demo", Context);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "demo_func", M.get());

    MachineModuleInfo MMI(TM.get());
    MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

    // Build a simple function with several basic blocks and instructions.
    MachineBasicBlock *EntryBB = MF.CreateMachineBasicBlock();
    MachineBasicBlock *LoopBB = MF.CreateMachineBasicBlock();
    MachineBasicBlock *ExitBB = MF.CreateMachineBasicBlock();
    MF.push_back(EntryBB);
    MF.push_back(LoopBB);
    MF.push_back(ExitBB);
    EntryBB->addSuccessor(LoopBB);
    LoopBB->addSuccessor(ExitBB);

    // Find a GPR register class.
    const TargetRegisterClass *RC = nullptr;
    for (const TargetRegisterClass *RCI : TRI->regclasses()) {
        if (RCI->isAllocatable() && RCI->getSizeInBits() >= 32) {
            RC = RCI;
            break;
        }
    }
    if (!RC) {
        std::cerr << "No register class found!\n";
        return 1;
    }

    // Create virtual registers.
    Register V1 = MRI.createVirtualRegister(RC);
    Register V2 = MRI.createVirtualRegister(RC);
    Register V3 = MRI.createVirtualRegister(RC);
    Register V4 = MRI.createVirtualRegister(RC);

    // Build instructions in entry block.
    BuildMI(EntryBB, EntryBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V1)
        .addReg(V2);  // Dead def demo
    BuildMI(EntryBB, EntryBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V2)
        .addReg(V3);  // Copy for propagation
    BuildMI(EntryBB, EntryBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V3)
        .addReg(V4);  // Another copy

    // Build instruction in loop block that uses V2.
    BuildMI(LoopBB, LoopBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V4)
        .addReg(V2);

    MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);

    std::cout << "========== Machine Pass Pipeline Demo ==========" << std::endl;
    std::cout << "Function: " << MF.getName() << std::endl;
    std::cout << "Basic blocks: " << MF.getNumBlockIDs() << std::endl;
    std::cout << "Virtual registers: " << MRI.getNumVirtRegs() << std::endl;
    std::cout << std::endl;

    // Run our custom passes on the MachineFunction.
    std::cout << "--- Running MachineInstrCounter ---" << std::endl;
    MachineInstrCounter Counter;
    Counter.runOnMachineFunction(MF);

    std::cout << "\n--- Running SimpleCopyPropagator ---" << std::endl;
    SimpleCopyPropagator Propagator;
    bool PropChanged = Propagator.runOnMachineFunction(MF);
    std::cout << "  Copy propagation changed IR: " << (PropChanged ? "yes" : "no") << std::endl;

    std::cout << "\n--- Running SimpleDeadInstEliminator ---" << std::endl;
    SimpleDeadInstEliminator Eliminator;
    bool ElimChanged = Eliminator.runOnMachineFunction(MF);
    std::cout << "  Dead inst elimination changed IR: " << (ElimChanged ? "yes" : "no") << std::endl;

    // Demonstrate how passes would be injected in TargetPassConfig.
    std::cout << "\n========== TargetPassConfig Injection (Conceptual) ==========" << std::endl;
    std::cout << R"(
// In your backend's TargetPassConfig subclass:

class MyTargetPassConfig : public TargetPassConfig {
public:
  bool addPreRegAlloc() override {
    // Inject our custom passes before register allocation.
    addPass(createMachineInstrCounter());
    addPass(createSimpleCopyPropagator());
    addPass(createSimpleDeadInstEliminator());

    // Then add the standard pre-RA passes.
    return TargetPassConfig::addPreRegAlloc();
  }

  bool addPreEmitPass() override {
    // Late passes run after register allocation.
    // Use these for pseudo-instruction expansion, branch relaxation, etc.
    return TargetPassConfig::addPreEmitPass();
  }
};
    )" << std::endl;

    std::cout << "\nMachine pass pipeline concepts demonstrated successfully!" << std::endl;
    return 0;
}
