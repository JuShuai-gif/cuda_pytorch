// example1.cpp - Chapter 10: Debug Logging and IR Printing Techniques
//
// Demonstrates:
//   - Using LLVM_DEBUG macro with DEBUG_TYPE for selective debug logging
//   - Using the -debug-only command-line option mechanism
//   - Printing IR at various points via raw_ostream
//   - Creating a simple Function pass with the new pass manager
//   - Running a pass pipeline and observing debug output
//
// Build with the repository baseline, LLVM 20.1.x:
//   clang++ example1.cpp $(llvm-config --cxxflags --ldflags --libs core passes irreader support) -o example1
//
// Usage:
//   ./example1                           # normal output
//   ./example1 -debug                    # all debug output
//   ./example1 -debug-only=ch10-pass     # only our pass's debug output

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

using namespace llvm;

// Define the debug type for our pass.
// This enables filtering via -debug-only=ch10-pass
#define DEBUG_TYPE "ch10-pass"

// ──────────────── A simple function pass with debug logging ────────────────
struct Ch10DemoPass : public PassInfoMixin<Ch10DemoPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    LLVM_DEBUG(dbgs() << "[ch10-pass] Entering pass for function: "
                       << F.getName() << "\n");

    bool Changed = false;
    for (BasicBlock &BB : F) {
      LLVM_DEBUG(dbgs() << "[ch10-pass]   BasicBlock: " << BB.getName()
                         << " (" << BB.size() << " instructions)\n");

      for (Instruction &I : make_early_inc_range(BB)) {
        LLVM_DEBUG(dbgs() << "[ch10-pass]     " << I.getOpcodeName()
                           << ": " << I << "\n");

        // Simple optimization: replace `add X, 0` with `X`
        if (I.getOpcode() == Instruction::Add) {
          Value *Op0 = I.getOperand(0);
          Value *Op1 = I.getOperand(1);
          if (auto *ConstOp1 = dyn_cast<ConstantInt>(Op1)) {
            if (ConstOp1->isZero()) {
              LLVM_DEBUG(dbgs() << "[ch10-pass]       -> Optimizing add with zero\n");
              I.replaceAllUsesWith(Op0);
              I.eraseFromParent();
              Changed = true;
            }
          }
        }
      }
    }

    LLVM_DEBUG(dbgs() << "[ch10-pass] Exiting pass for function: "
                       << F.getName() << " (Changed=" << Changed << ")\n");

    // Replacing and erasing an instruction invalidates value-sensitive
    // analyses, even though the CFG itself is unchanged.
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// ──────────────── Register the pass with opt's plugin system ────────────────
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "Ch10Demo", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "ch10-demo-pass") {
                    FPM.addPass(Ch10DemoPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// ──────────────── Build a test module ────────────────
static std::unique_ptr<Module> buildTestModule(LLVMContext &Context) {
  auto M = std::make_unique<Module>("ch10_debug_demo", Context);
  IRBuilder<> Builder(Context);

  Type *I32 = Type::getInt32Ty(Context);

  // Function: simple_add
  FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "simple_add", M.get());
  F->getArg(0)->setName("a");
  F->getArg(1)->setName("b");

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  Builder.SetInsertPoint(Entry);

  // %sum = add i32 %a, %b
  Value *Sum = Builder.CreateAdd(F->getArg(0), F->getArg(1), "sum");
  // %extra = add i32 %sum, 0   <-- will be optimized away by our pass
  Value *Extra = Builder.CreateAdd(Sum, ConstantInt::get(I32, 0), "extra_zero");
  Builder.CreateRet(Extra);

  return M;
}

// ──────────────── Main ────────────────
int main(int argc, char **argv) {
  // Parse command-line options (enables -debug and -debug-only)
  cl::ParseCommandLineOptions(argc, argv, "Chapter 10 Debug Demo\n");

  LLVMContext Context;
  auto M = buildTestModule(Context);

  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: input module is invalid\n";
    return 1;
  }

  outs() << "=== Before Pass ===\n";
  M->print(outs(), nullptr);
  outs() << "\n";

  // Set up pass managers
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create a module pass manager and add our pass
  ModulePassManager MPM;
  FunctionPassManager FPM;
  FPM.addPass(Ch10DemoPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  // Run the pipeline
  MPM.run(*M, MAM);

  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: pass produced invalid IR\n";
    return 1;
  }

  outs() << "=== After Pass ===\n";
  M->print(outs(), nullptr);

  return 0;
}
