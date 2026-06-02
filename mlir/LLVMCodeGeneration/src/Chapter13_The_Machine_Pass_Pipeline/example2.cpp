// example2.cpp - Machine Pass Pipeline: PeepholeOptimizer and MachineCombiner concepts
//
// Demonstrates:
// - Peephole optimization pattern (local window rewriting)
// - MachineCombiner pattern (instruction sequence reassociation)
// - CodeGenPrepare concepts
// - TargetInstrInfo hooks used by generic passes
//
// Build with:
//   clang++ -o example2 example2.cpp $(llvm-config --cxxflags --ldflags --libs core codegen)
//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
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
#include <set>
#include <vector>

using namespace llvm;

static void sep(const char *title) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

// ============================================================
// Conceptual Peephole Optimizer
// ============================================================
//
// The real PeepholeOptimizer scans small windows of instructions and applies
// TargetInstrInfo hooks to fold copies, remove redundant extensions, etc.
// This conceptual implementation demonstrates the pattern.

class ConceptualPeepholeOptimizer {
    const TargetInstrInfo &TII;
    const TargetRegisterInfo &TRI;

public:
    ConceptualPeepholeOptimizer(const TargetInstrInfo &TII, const TargetRegisterInfo &TRI)
        : TII(TII), TRI(TRI) {}

    struct Optimization {
        std::string Description;
        bool Applied;
    };

    // Scan a basic block for peephole optimization opportunities.
    std::vector<Optimization> scanBasicBlock(MachineBasicBlock &MBB) {
        std::vector<Optimization> Results;

        // Pattern 1: Redundant copy elimination.
        //   %a = COPY %b
        //   ... no use of %a except:
        //   %c = COPY %a
        //   =>  %c = COPY %b  (if %a has no other uses)
        for (auto MII = MBB.begin(); MII != MBB.end(); ++MII) {
            MachineInstr &MI = *MII;
            if (!MI.isCopy() || MI.getNumOperands() != 2)
                continue;

            MachineOperand &DstMO = MI.getOperand(0);
            MachineOperand &SrcMO = MI.getOperand(1);
            if (!DstMO.isReg() || !SrcMO.isReg()) continue;

            Register Dst = DstMO.getReg();
            Register Src = SrcMO.getReg();

            if (!Dst.isVirtual() || !Src.isVirtual()) continue;

            // Check the next instruction for a copy from Dst.
            auto NextII = std::next(MII);
            if (NextII != MBB.end() && NextII->isCopy() &&
                NextII->getNumOperands() == 2) {
                MachineOperand &NextSrcMO = NextII->getOperand(1);
                if (NextSrcMO.isReg() && NextSrcMO.getReg() == Dst) {
                    Optimization Opt;
                    Opt.Description = std::string("Redundant COPY: %") +
                                      std::to_string(Register::virtReg2Index(Dst)) +
                                      " = COPY %" +
                                      std::to_string(Register::virtReg2Index(Src)) +
                                      " then %c = COPY %" +
                                      std::to_string(Register::virtReg2Index(Dst));
                    Opt.Applied = true;
                    Results.push_back(Opt);
                }
            }
        }

        // Pattern 2: Find COPY chains.
        //   %a = COPY %b; %c = COPY %a; %d = COPY %c
        //   => can be shortened to %d = COPY %b
        for (auto MII = MBB.begin(); MII != MBB.end(); ++MII) {
            MachineInstr &MI = *MII;
            if (!MI.isCopy() || MI.getNumOperands() != 2) continue;

            MachineOperand &SrcMO = MI.getOperand(1);
            if (!SrcMO.isReg() || !SrcMO.getReg().isVirtual()) continue;

            Register SrcReg = SrcMO.getReg();

            // Look backward for the defining instruction of SrcReg.
            for (auto PrevII = std::make_reverse_iterator(MII);
                 PrevII != MBB.rend(); ++PrevII) {
                MachineInstr &PrevMI = *PrevII;
                for (MachineOperand &MO : PrevMI.operands()) {
                    if (MO.isReg() && MO.isDef() && MO.getReg() == SrcReg) {
                        if (PrevMI.isCopy() && PrevMI.getNumOperands() == 2) {
                            Optimization Opt;
                            Opt.Description = std::string("COPY chain: current COPY uses %") +
                                              std::to_string(Register::virtReg2Index(SrcReg)) +
                                              " which is defined by another COPY";
                            Opt.Applied = true;
                            Results.push_back(Opt);
                        }
                        goto next_instr;
                    }
                }
            }
            next_instr:;
        }

        // Pattern 3: Check for dead definitions (operand marked 'dead' or unused).
        // In the real PeepholeOptimizer, target hooks like optimizeLoadInstr()
        // and foldImmediate() are called for each instruction.
        for (MachineInstr &MI : MBB) {
            for (const MachineOperand &MO : MI.operands()) {
                if (MO.isReg() && MO.isDef() && MO.isDead()) {
                    Optimization Opt;
                    Opt.Description = std::string("Dead def found in ") +
                                      TII.getName(MI.getOpcode());
                    Opt.Applied = false; // Needs liveness to confirm.
                    Results.push_back(Opt);
                }
            }
        }

        return Results;
    }

    void printOpportunities(MachineBasicBlock &MBB) {
        auto Results = scanBasicBlock(MBB);
        std::cout << "Peephole analysis for BB#" << MBB.getNumber()
                  << " (" << MBB.size() << " instructions):" << std::endl;
        if (Results.empty()) {
            std::cout << "  No optimization opportunities found." << std::endl;
        } else {
            for (auto &R : Results) {
                std::cout << "  [" << (R.Applied ? "APPLIED" : "FOUND") << "] "
                          << R.Description << std::endl;
            }
        }
    }
};

// ============================================================
// Conceptual MachineCombiner
// ============================================================
//
// The real MachineCombiner looks at sequences of instructions and
// tries to reassociate them (e.g., (a+b)+c -> a+(b+c)) or combine
// them into more efficient instructions (e.g., mul+add -> fma).

class ConceptualMachineCombiner {
    const TargetInstrInfo &TII;

public:
    ConceptualMachineCombiner(const TargetInstrInfo &TII) : TII(TII) {}

    struct CombineOpportunity {
        std::string Pattern;
        std::string Replacement;
        unsigned Benefit; // Estimated improvement (higher = better).
    };

    // Analyze a basic block for combining opportunities.
    std::vector<CombineOpportunity> analyze(MachineBasicBlock &MBB) {
        std::vector<CombineOpportunity> Results;

        // Look for sequences of 2-3 instructions that could be combined.
        std::vector<MachineInstr *> Instrs;
        for (MachineInstr &MI : MBB)
            Instrs.push_back(&MI);

        for (size_t i = 0; i + 1 < Instrs.size(); ++i) {
            MachineInstr &First = *Instrs[i];
            MachineInstr &Second = *Instrs[i + 1];

            // Pattern 1: ineg + iadd -> isub (conceptual).
            // If Second uses the result of First and both are arithmetic:
            if (First.getNumOperands() >= 2 && Second.getNumOperands() >= 2) {
                // Check if Second uses First's definition.
                for (MachineOperand &UseMO : Second.operands()) {
                    if (!UseMO.isReg() || !UseMO.isUse()) continue;
                    for (MachineOperand &DefMO : First.operands()) {
                        if (DefMO.isReg() && DefMO.isDef() &&
                            DefMO.getReg() == UseMO.getReg()) {
                            CombineOpportunity Opp;
                            Opp.Pattern = TII.getName(First.getOpcode()) +
                                          std::string(" -> ") +
                                          TII.getName(Second.getOpcode());
                            Opp.Replacement = "Combined instruction (target-specific)";
                            Opp.Benefit = 1;
                            Results.push_back(Opp);
                        }
                    }
                }
            }

            // Pattern 2: Two identical commutative operations.
            if (First.isCommutable() && Second.isCommutable() &&
                First.getOpcode() == Second.getOpcode()) {
                // Check if they could be reassociated.
                for (MachineOperand &MO1 : First.operands()) {
                    if (!MO1.isReg() || MO1.isDef()) continue;
                    for (MachineOperand &MO2 : Second.operands()) {
                        if (!MO2.isReg() || MO2.isDef()) continue;
                        if (MO1.getReg() == MO2.getReg()) {
                            CombineOpportunity Opp;
                            Opp.Pattern = "Repeated operand in consecutive " +
                                          std::string(TII.getName(First.getOpcode())) +
                                          " instructions";
                            Opp.Replacement = "Reassociated instruction sequence";
                            Opp.Benefit = 2;
                            Results.push_back(Opp);
                        }
                    }
                }
            }

            // Pattern 3: mul + add in adjacent instructions.
            // Check for MUL-like instruction followed by ADD-like using its result.
            bool MulFound = (First.getOpcode() != 0); // Simplified check.
            // In real MachineCombiner, TargetInstrInfo::getMachineCombinerPatterns()
            // identifies specific patterns like MUL+ADD -> FMADD.

            if (MulFound && i + 2 < Instrs.size()) {
                MachineInstr &Third = *Instrs[i + 2];
                // Check if this is a 3-instruction sequence.
                CombineOpportunity Opp;
                Opp.Pattern = std::string("3-instr sequence: ") +
                              TII.getName(First.getOpcode()) + " + " +
                              TII.getName(Second.getOpcode()) + " + " +
                              TII.getName(Third.getOpcode());
                Opp.Replacement = "Potentially combinable into fewer instructions";
                Opp.Benefit = 3;
                Results.push_back(Opp);
            }
        }

        return Results;
    }

    void printOpportunities(MachineBasicBlock &MBB) {
        auto Results = analyze(MBB);
        std::cout << "MachineCombiner analysis for BB#" << MBB.getNumber()
                  << ":" << std::endl;
        if (Results.empty()) {
            std::cout << "  No combining opportunities found." << std::endl;
        } else {
            for (auto &R : Results) {
                std::cout << "  Pattern: " << R.Pattern << std::endl;
                std::cout << "    => " << R.Replacement << std::endl;
                std::cout << "    Benefit: " << R.Benefit << std::endl;
            }
        }
    }
};

// ============================================================
// Main
// ============================================================

int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();

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
    // Part 1: Build a MachineFunction with interesting patterns.
    // ========================
    LLVMContext Context;
    auto M = std::make_unique<Module>("opt_demo", Context);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Context), false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "opt_func", M.get());

    MachineModuleInfo MMI(TM.get());
    MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
    MachineRegisterInfo &MRI = MF.getRegInfo();

    MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);

    // Find register class.
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
    Register A = MRI.createVirtualRegister(RC);
    Register B = MRI.createVirtualRegister(RC);
    Register C = MRI.createVirtualRegister(RC);
    Register D = MRI.createVirtualRegister(RC);
    Register E = MRI.createVirtualRegister(RC);

    // Build a sequence with COPY chains and potential peephole opportunities.
    // %A = COPY %B       (copy 1)
    // %C = COPY %A       (copy 2 - could be folded)
    // %D = COPY %C       (copy 3 - chain)
    // %E = COPY %B       (direct copy)
    BuildMI(MBB, MBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), A).addReg(B);
    BuildMI(MBB, MBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), C).addReg(A);
    BuildMI(MBB, MBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), D).addReg(C);
    BuildMI(MBB, MBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), E).addReg(B);

    MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);

    // Print the IR.
    sep("Machine IR Before Optimization");
    std::cout << "Function: " << MF.getName() << std::endl;
    for (MachineInstr &MI : *MBB) {
        std::cout << "  " << TII->getName(MI.getOpcode());
        for (MachineOperand &MO : MI.operands()) {
            if (MO.isReg() && MO.getReg().isVirtual()) {
                if (MO.isDef()) std::cout << " [def]";
                std::cout << " %" << Register::virtReg2Index(MO.getReg());
            }
        }
        std::cout << std::endl;
    }

    // ========================
    // Part 2: Peephole Optimizer demo.
    // ========================
    sep("Peephole Optimizer Analysis");
    ConceptualPeepholeOptimizer PeepOpt(*TII, *TRI);
    PeepOpt.printOpportunities(*MBB);

    // ========================
    // Part 3: MachineCombiner demo.
    // ========================
    sep("MachineCombiner Analysis");
    ConceptualMachineCombiner Combiner(*TII);
    Combiner.printOpportunities(*MBB);

    // ========================
    // Part 4: CodeGenPrepare concepts.
    // ========================
    sep("CodeGenPrepare Concepts");

    std::cout << "CodeGenPrepare runs on LLVM IR (not Machine IR). It transforms IR" << std::endl;
    std::cout << "to make it friendlier for instruction selection. Key transformations:" << std::endl;
    std::cout << std::endl;
    std::cout << "  1. Address computation sinking:" << std::endl;
    std::cout << "     Sink GEP computations close to their memory operations." << std::endl;
    std::cout << "     This helps ISEL match addressing modes." << std::endl;
    std::cout << std::endl;
    std::cout << "  2. Overflow intrinsic optimization:" << std::endl;
    std::cout << "     Transform llvm.sadd.with.overflow into separate add+cmp." << std::endl;
    std::cout << std::endl;
    std::cout << "  3. ByVal argument promotion:" << std::endl;
    std::cout << "     Copy byval arguments to local allocas for better optimization." << std::endl;
    std::cout << std::endl;
    std::cout << "  4. Switch-to-lookup-table conversion:" << std::endl;
    std::cout << "     Convert dense switch statements to lookup tables." << std::endl;
    std::cout << std::endl;
    std::cout << "It uses TargetLowering hooks to make target-aware decisions." << std::endl;

    // ========================
    // Part 5: TargetInstrInfo hooks used by generic passes.
    // ========================
    sep("TargetInstrInfo Hooks for Generic Machine Passes");

    std::cout << "Generic machine passes call target-specific hooks via TargetInstrInfo:" << std::endl;
    std::cout << std::endl;

    // Check which hooks the target supports.
    std::cout << "Instruction property queries:" << std::endl;
    std::cout << "  isCopyInstr(MI)        - Used by PeepholeOptimizer for copy folding" << std::endl;
    std::cout << "  isTriviallyReMaterializable(MI) - Used by register allocator" << std::endl;
    std::cout << "  isSchedulingBoundary(MI) - Used by scheduler" << std::endl;
    std::cout << std::endl;

    std::cout << "Optimization hooks:" << std::endl;
    std::cout << "  optimizeLoadInstr()    - Peephole: fold loads into users" << std::endl;
    std::cout << "  foldImmediate()        - Peephole: fold immediates into instrs" << std::endl;
    std::cout << "  getMachineCombinerPatterns() - MachineCombiner: get patterns" << std::endl;
    std::cout << "  genAlternativeCodeSequence() - MachineCombiner: generate replacement" << std::endl;
    std::cout << "  commuteInstruction()   - Make operands commutative" << std::endl;
    std::cout << std::endl;

    std::cout << "Analysis results:" << std::endl;
    std::cout << "  analyzeBranch()        - Analyze branch structure" << std::endl;
    std::cout << "  insertBranch()         - Insert branch instruction" << std::endl;
    std::cout << "  reverseBranchCondition() - Reverse branch condition" << std::endl;

    // ========================
    // Part 6: Pass pipeline structure visualization.
    // ========================
    sep("Machine Pass Pipeline Structure (Conceptual)");

    std::cout << R"(
Standard CodeGen pipeline (after instruction selection):

  [SSA Machine IR]
       |
       v
  Early Machine Optimizations:
    - Phi elimination (out of SSA for targets that need it)
    - Two-address instruction pass
    - MachineCSE
    - MachineSink
       |
       v
  Peephole Optimizer
       |
       v
  MachineCombiner
       |
       v
  [Target-specific pre-RA passes injected here via addPreRegAlloc()]
       |
       v
  Register Allocation:
    - Live interval analysis (SlotIndexes, LiveIntervals)
    - Register coalescing
    - Greedy register allocator (or Basic / PBQP)
    - Virtual register rewriting
       |
       v
  [Non-SSA Machine IR]
       |
       v
  Post-RA Optimizations:
    - Post-RA MachineLICM
    - Post-RA MachineCSE
    - Branch folding
    - If-conversion
       |
       v
  [Target-specific post-RA passes injected here via addPostRegAlloc()]
       |
       v
  Pre-Emit:
    - Prologue/epilogue insertion
    - Frame lowering (PEI)
    - Late machine CFG optimization
    - Branch relaxation
       |
       v
  [Target-specific pre-emit passes injected here via addPreEmitPass()/addPreEmitPass2()]
       |
       v
  Assembly / Object Emission

Note: The exact pipeline depends on the target, optimization level,
      and subtarget features. Use llc -debug-pass=Structure to see
      the actual pipeline for your target.
    )" << std::endl;

    std::cout << "Machine pass pipeline optimization concepts demonstrated!" << std::endl;
    return 0;
}
