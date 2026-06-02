// example3.cpp - Machine Pass Pipeline: Demonstrating pipeline control and pass injection
//
// Demonstrates:
// - How llc command-line options control the pass pipeline
// - How to run specific passes (-run-pass, -start-before/after, -stop-before/after)
// - Conceptual TargetPassConfig subclass
// - Pass dependencies and invalidation
// - Debugging the pipeline with -debug-pass=Structure
//
// Build with:
//   clang++ -o example3 example3.cpp $(llvm-config --cxxflags --ldflags --libs core codegen)
//
// This file is largely conceptual/documentation; run the actual llc tool to exercise
// the pipeline control options.

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
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

static void sep(const char *title) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

// ============================================================
// A minimal custom pass for demonstration
// ============================================================

namespace {

class DemoPipelinePass : public MachineFunctionPass {
public:
    static char ID;
    std::string PassName;

    DemoPipelinePass() : MachineFunctionPass(ID), PassName("DemoPipelinePass") {}
    explicit DemoPipelinePass(const std::string &Name)
        : MachineFunctionPass(ID), PassName(Name) {}

    StringRef getPassName() const override { return PassName; }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
        MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override {
        unsigned BBs = MF.getNumBlockIDs();
        unsigned Instrs = 0;
        for (MachineBasicBlock &MBB : MF)
            Instrs += MBB.size();

        std::cout << "  [" << PassName << "] Function: " << MF.getName()
                  << ", BBs: " << BBs << ", Instrs: " << Instrs
                  << ", SSA: " << (MF.getProperties().hasProperty(
                         MachineFunctionProperties::Property::IsSSA)
                         ? "yes" : "no")
                  << std::endl;
        return false;
    }
};

char DemoPipelinePass::ID = 0;

} // end anonymous namespace

// ============================================================
// Main
// ============================================================

int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

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

    const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const TargetInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();

    // ========================
    // Part 1: Build a test MachineFunction
    // ========================
    LLVMContext Context;
    auto M = std::make_unique<Module>("pipeline_demo", Context);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "pipeline_func", M.get());

    MachineModuleInfo MMI(TM.get());
    MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
    MachineRegisterInfo &MRI = MF.getRegInfo();

    // Create two basic blocks.
    MachineBasicBlock *BB0 = MF.CreateMachineBasicBlock();
    MachineBasicBlock *BB1 = MF.CreateMachineBasicBlock();
    MF.push_back(BB0);
    MF.push_back(BB1);
    BB0->addSuccessor(BB1);

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

    Register V1 = MRI.createVirtualRegister(RC);
    Register V2 = MRI.createVirtualRegister(RC);
    Register V3 = MRI.createVirtualRegister(RC);

    BuildMI(BB0, BB0->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V1).addReg(V2);
    BuildMI(BB0, BB0->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V2).addReg(V3);

    BuildMI(BB1, BB1->end(), DebugLoc(), TII->get(TargetOpcode::COPY), V3).addReg(V1);

    MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);

    // ========================
    // Part 2: Demonstrate running individual passes
    // ========================
    sep("Pass Pipeline Control with llc (Documentation)");

    std::cout << R"(
The llc tool provides fine-grained control over the machine pass pipeline:

  1. Run only specific passes:
     $ llc -run-pass=peephole-opt input.mir -o out.mir

  2. Start pipeline before/after a specific pass:
     $ llc -start-before=regalloc input.mir -o out.s
     $ llc -start-after=finalize-isel input.mir -o out.s

  3. Stop pipeline before/after a specific pass:
     $ llc -stop-before=peephole-opt input.ll -o out.mir
     $ llc -stop-after=regalloc input.ll -o out.mir

  4. View the pass pipeline structure:
     $ llc -debug-pass=Structure input.ll -o /dev/null

  5. Enable debug logs for specific passes:
     $ llc -debug-only=peephole-opt input.ll -o /dev/null

  6. Chain commands to test pass interactions:
     $ llc -run-pass=pass1 input.mir -o tmp.mir
     $ llc -run-pass=pass2 tmp.mir -o out.mir

NOTE: -run-pass completely bypasses the predefined pipeline.
      -start-before/after run the pipeline from that point.
      You can combine -start-* and -stop-* to run pipeline segments.
    )" << std::endl;

    // ========================
    // Part 3: Run demo passes on the test function
    // ========================
    sep("Running Demo Passes on Test MachineFunction");

    DemoPipelinePass Pass1("Pre-RA-Analysis-Pass");
    DemoPipelinePass Pass2("Post-RA-Cleanup-Pass");

    std::cout << "Simulating pass pipeline execution:" << std::endl;
    std::cout << "  [Phase 1: SSA Machine Optimizations]" << std::endl;
    Pass1.runOnMachineFunction(MF);

    std::cout << "  [Phase 2: Register Allocation (conceptual)]" << std::endl;
    std::cout << "    - Live interval analysis" << std::endl;
    std::cout << "    - Register coloring" << std::endl;
    std::cout << "    - Virtual register rewriting" << std::endl;
    MF.getProperties().reset(MachineFunctionProperties::Property::IsSSA);

    std::cout << "  [Phase 3: Post-RA Optimizations]" << std::endl;
    Pass2.runOnMachineFunction(MF);

    // ========================
    // Part 4: TargetPassConfig injection points (documentation)
    // ========================
    sep("TargetPassConfig Injection Points");

    std::cout << R"(
// Your backend's TargetPassConfig subclass controls pass injection:

class MyTargetPassConfig : public TargetPassConfig {
public:
  MyTargetPassConfig(MyTargetTargetMachine &TM, PassManagerBase &PM);

  // === Injection Points ===

  // 1. Before instruction selection (LLVM IR passes)
  bool addPreISel() override {
    // Add target-specific IR passes here.
    return false; // Don't stop pipeline.
  }

  // 2. Instruction selection itself
  bool addInstSelector() override {
    // Usually: addPass(createMyTargetISelDag());
    return false;
  }

  // 3. IR-level passes run during CodeGen
  bool addIRPasses() override {
    // CodeGenPrepare runs here by default.
    addPass(createCodeGenPreparePass());
    return false;
  }

  // 4. Before register allocation (SSA Machine IR)
  bool addPreRegAlloc() override {
    // Inject custom SSA machine passes.
    addPass(createMyTargetPreRAPass());
    return TargetPassConfig::addPreRegAlloc(); // Chain to standard passes.
  }

  // 5. Register allocation
  bool addOptimizedRegAlloc() override {
    // Override to use a custom allocator.
    addPass(createGreedyRegisterAllocator());
    return true;
  }

  // 6. After register allocation (non-SSA Machine IR)
  bool addPostRegAlloc() override {
    // Inject custom post-RA passes.
    addPass(createMyTargetPostRAPass());
    return false;
  }

  // 7. Late, before code emission
  bool addPreEmitPass() override {
    // Expand pseudo-instructions, branch relaxation, etc.
    addPass(createMyTargetExpandPseudoPass());
    addPass(createMyTargetBranchRelaxationPass());
    return false;
  }

  // 8. Very late, after addPreEmitPass
  bool addPreEmitPass2() override {
    // Final cleanup passes.
    return false;
  }
};
    )" << std::endl;

    // ========================
    // Part 5: Pass Registration
    // ========================
    sep("Pass Registration Pattern");

    std::cout << R"(
To register a custom machine pass with LLVM:

  1. In your pass's .cpp file:

     char MyPass::ID = 0;

     INITIALIZE_PASS(MyPass, "my-pass-name",
                     "My Pass Description", false, false)

     MachineFunctionPass *createMyPass() {
       return new MyPass();
     }

  2. In your TargetPassConfig:

     addPass(createMyPass());

  3. The pass name ("my-pass-name") is what you use with:
     $ llc -run-pass=my-pass-name input.mir

  The INITIALIZE_PASS macro registers:
  - The pass name for CLI options
  - The pass with the PassRegistry
  - Dependencies and preservation info
    )" << std::endl;

    // ========================
    // Part 6: Analysis Pass vs Transform Pass
    // ========================
    sep("Machine Analysis vs Transform Passes");

    std::cout << "Machine passes come in two flavors:" << std::endl;
    std::cout << std::endl;
    std::cout << "  Analysis passes (MachineFunctionPass):" << std::endl;
    std::cout << "    - MachineLoopInfo:    Detect loops in machine IR" << std::endl;
    std::cout << "    - MachineDominatorTree: Dominance relationships" << std::endl;
    std::cout << "    - LiveIntervals:      Live range intervals for registers" << std::endl;
    std::cout << "    - LiveVariables:      Liveness at block boundaries" << std::endl;
    std::cout << "    - SlotIndexes:        Sequential numbering of instructions" << std::endl;
    std::cout << "    - MachineBlockFrequencyInfo: Execution frequency" << std::endl;
    std::cout << "    - MachineBranchProbabilityInfo: Branch probabilities" << std::endl;
    std::cout << std::endl;
    std::cout << "  Transform passes (MachineFunctionPass):" << std::endl;
    std::cout << "    - PeepholeOptimizer:  Local instruction optimization" << std::endl;
    std::cout << "    - MachineCombiner:    Instruction reassociation/merging" << std::endl;
    std::cout << "    - MachineCSE:         Common subexpression elimination" << std::endl;
    std::cout << "    - MachineLICM:        Loop-invariant code motion" << std::endl;
    std::cout << "    - MachineSink:        Sink instructions to successors" << std::endl;
    std::cout << "    - IfConverter:        Convert branches to predicated code" << std::endl;
    std::cout << "    - TailDuplicator:     Duplicate blocks to eliminate branches" << std::endl;
    std::cout << "    - BranchFolder:       Merge common tails of basic blocks" << std::endl;
    std::cout << "    - DeadMachineInstructionElim: Remove dead instructions" << std::endl;

    // ========================
    // Part 7: Pass invalidation
    // ========================
    sep("Pass Invalidation and Analysis Preservation");

    std::cout << R"(
When a transform pass modifies the Machine IR, it must invalidate
affected analyses:

  void MyPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<MachineLoopInfo>();     // Need loop info.
    AU.addPreserved<MachineDominatorTree>(); // Preserves dominator tree.
    AU.addPreserved<SlotIndexes>();         // Preserves slot indexes.
    // Not listing an analysis means it will be invalidated.
    MachineFunctionPass::getAnalysisUsage(AU);
  }

Consequences of not preserving analyses:
- Later passes will recompute them (compile-time cost)
- Incorrect preservation can lead to stale analysis results
- Use AU.setPreservesAll() only for read-only passes
- Use AU.setPreservesCFG() if control flow is unchanged
    )" << std::endl;

    std::cout << "Machine pass pipeline concepts demonstrated successfully!" << std::endl;
    return 0;
}
