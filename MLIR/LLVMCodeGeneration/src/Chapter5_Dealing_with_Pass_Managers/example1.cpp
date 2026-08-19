// example1.cpp - Legacy Pass Manager Demo
// Demonstrates: creating a legacy FunctionPass, INITIALIZE_PASS macros,
// getAnalysisUsage for dependencies, pass registration and pipeline execution.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

#include <memory>

using namespace llvm;

// ---------------------------------------------------------------------------
// Legacy Pass: CountInstructions
// A simple FunctionPass that counts instructions per basic block
// and prints statistics. Demonstrates the legacy pass pattern.
// ---------------------------------------------------------------------------
namespace {

class CountInstructions : public FunctionPass {
public:
  static char ID;

  CountInstructions() : FunctionPass(ID) {}

  // -----------------------------------------------------------------------
  // This is the main entry point for a legacy FunctionPass.
  // Returns true if the IR was modified, false otherwise.
  // -----------------------------------------------------------------------
  bool runOnFunction(Function &F) override {
    unsigned TotalInsts = 0;
    unsigned TotalBBs = 0;

    outs() << "  [LegacyPass::CountInstructions] Function: "
           << F.getName() << "\n";

    for (BasicBlock &BB : F) {
      unsigned BBCount = 0;
      for (Instruction &I : BB) {
        (void)I;
        BBCount++;
      }
      TotalInsts += BBCount;
      TotalBBs++;

      outs() << "    BB '" << BB.getName() << "': "
             << BBCount << " instructions\n";
    }

    outs() << "    Total: " << TotalBBs << " basic blocks, "
           << TotalInsts << " instructions\n";

    // This pass never modifies the IR
    return false;
  }

  // -----------------------------------------------------------------------
  // Declare that this pass preserves all analyses (it doesn't modify IR)
  // and doesn't require any specific analyses.
  // -----------------------------------------------------------------------
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // This pass doesn't need any analyses
    // AU.addRequired<LoopInfoWrapperPass>();  // Example: if we needed LoopInfo

    // Mark all analyses as preserved since we don't modify IR
    AU.setPreservesAll();
  }

  // -----------------------------------------------------------------------
  // Returns the pass name for debugging output.
  // -----------------------------------------------------------------------
  StringRef getPassName() const override {
    return "Count Instructions (Legacy PM)";
  }
};

// Static ID for the pass - required by the legacy pass manager
char CountInstructions::ID = 0;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Registration: This macro makes the pass discoverable by the pass manager
// and allows it to be invoked via opt's command-line interface.
// ---------------------------------------------------------------------------
// In a standalone tool, we call the initializer directly instead of using
// the macros. But we show the macro pattern for reference:
//
// INITIALIZE_PASS(CountInstructions, "count-insts",
//                 "Count Instructions per Function", false, false)
//
// For standalone usage, we just register manually:

// ---------------------------------------------------------------------------
// Legacy Pass: SimplifyConstantBranches
// A simple FunctionPass that simplifies conditional branches with
// constant conditions. Demonstrates an optimization pass pattern.
// ---------------------------------------------------------------------------
namespace {

class SimplifyConstantBranches : public FunctionPass {
public:
  static char ID;

  SimplifyConstantBranches() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool Changed = false;

    for (BasicBlock &BB : F) {
      // Look for conditional branches with constant conditions
      auto *BI = dyn_cast<BranchInst>(BB.getTerminator());
      if (!BI || !BI->isConditional())
        continue;

      // Check if the condition is a constant integer
      if (auto *CI = dyn_cast<ConstantInt>(BI->getCondition())) {
        outs() << "  [LegacyPass::SimplifyConstantBranches] "
               << "Simplifying constant branch in " << F.getName()
               << "::" << BB.getName() << "\n";

        // Use LLVM's CFG utility instead of erasing the branch by hand.  The
        // helper also removes the obsolete predecessor from successor PHIs
        // and deletes the now-dead condition when it is safe to do so.
        Changed |= ConstantFoldTerminator(&BB, /*DeleteDeadConditions=*/true);
      }
    }

    return Changed;
  }

  StringRef getPassName() const override {
    return "Simplify Constant Branches (Legacy PM)";
  }
};

char SimplifyConstantBranches::ID = 0;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Build a test module with constant conditional branches
// ---------------------------------------------------------------------------
std::unique_ptr<Module> buildModule(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("LegacyPMDemo", Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);

  // Function with a constant branch (always takes true path)
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "const_branch_true", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *TrueBB  = BasicBlock::Create(Ctx, "true_bb", F);
    BasicBlock *FalseBB = BasicBlock::Create(Ctx, "false_bb", F);
    BasicBlock *MergeBB = BasicBlock::Create(Ctx, "merge", F);

    IRBuilder<> B(EntryBB);
    // Always-true condition
    B.CreateCondBr(ConstantInt::getTrue(Ctx), TrueBB, FalseBB);

    B.SetInsertPoint(TrueBB);
    B.CreateBr(MergeBB);

    B.SetInsertPoint(FalseBB);
    B.CreateBr(MergeBB);

    B.SetInsertPoint(MergeBB);
    PHINode *Phi = B.CreatePHI(I32, 2, "res");
    Phi->addIncoming(ConstantInt::get(I32, 10), TrueBB);
    Phi->addIncoming(ConstantInt::get(I32, 20), FalseBB);
    B.CreateRet(Phi);
  }

  // Function with a constant branch (always takes false path)
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "const_branch_false", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *TrueBB  = BasicBlock::Create(Ctx, "true_bb", F);
    BasicBlock *FalseBB = BasicBlock::Create(Ctx, "false_bb", F);

    IRBuilder<> B(EntryBB);
    // Always-false condition
    B.CreateCondBr(ConstantInt::getFalse(Ctx), TrueBB, FalseBB);

    B.SetInsertPoint(TrueBB);
    B.CreateRet(ConstantInt::get(I32, 100));

    B.SetInsertPoint(FalseBB);
    B.CreateRet(ConstantInt::get(I32, 200));
  }

  return M;
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  auto M = buildModule(Context);

  outs() << "=== Before Legacy Pass Pipeline ===\n";
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Input module verification failed!\n";
    return 1;
  }
  M->print(outs(), nullptr);
  outs() << "\n";

  // -----------------------------------------------------------------------
  // Set up and run a legacy pass manager pipeline
  // -----------------------------------------------------------------------
  legacy::PassManager PM;

  // Register passes with the legacy pass manager
  // Note: We add our passes directly since they are not analysis passes
  // that need to be pre-registered for other passes to use.
  PM.add(new CountInstructions());
  PM.add(new SimplifyConstantBranches());
  PM.add(new CountInstructions());  // Run again to see the changes

  outs() << "=== Running Legacy Pass Pipeline ===\n";
  PM.run(*M);
  outs() << "\n";

  // Verify the module is still valid
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Module verification failed after passes!\n";
    return 1;
  }

  outs() << "=== After Legacy Pass Pipeline ===\n";
  M->print(outs(), nullptr);

  return 0;
}
