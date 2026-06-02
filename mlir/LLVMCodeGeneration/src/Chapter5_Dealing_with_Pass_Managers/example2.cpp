// example2.cpp - New Pass Manager Demo
// Demonstrates: creating passes with PassInfoMixin, analysis registration,
// PreservedAnalyses, and building a new-PM pipeline.
// Also shows pass pipeline inspection and the CRTP pattern.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Analysis: DeadBlockDetector
// An analysis that identifies basic blocks with no predecessors
// (other than the entry block). Demonstrates the AnalysisInfoMixin pattern.
// ---------------------------------------------------------------------------
namespace {

// This is an analysis result - what the analysis produces
struct DeadBlockInfo {
  SmallVector<BasicBlock *, 4> DeadBlocks;

  bool hasDeadBlocks() const { return !DeadBlocks.empty(); }
};

class DeadBlockDetector : public AnalysisInfoMixin<DeadBlockDetector> {
  friend AnalysisInfoMixin<DeadBlockDetector>;

  // Every analysis needs a static AnalysisKey for identification
  static AnalysisKey Key;

public:
  using Result = DeadBlockInfo;

  // The run method for an analysis: takes the IR and the analysis manager,
  // returns the analysis result.
  Result run(Function &F, FunctionAnalysisManager &) {
    Result Info;

    for (BasicBlock &BB : F) {
      // Entry block always has 0 predecessors but is not "dead"
      if (&BB == &F.getEntryBlock())
        continue;

      // Check if this block has any predecessors
      if (pred_size(&BB) == 0) {
        Info.DeadBlocks.push_back(&BB);
      }
    }

    return Info;
  }
};

AnalysisKey DeadBlockDetector::Key;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pass: PrintFunctionStats - new PM pass using PassInfoMixin<>
// Prints statistics about each function. Demonstrates the basic new PM pattern.
// ---------------------------------------------------------------------------
namespace {

class PrintFunctionStats : public PassInfoMixin<PrintFunctionStats> {
public:
  // The run method signature for a Function-scoped pass:
  // Takes the Function and the FunctionAnalysisManager
  // Returns PreservedAnalyses
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    unsigned NumBBs = 0;
    unsigned NumInsts = 0;
    unsigned NumCalls = 0;

    for (BasicBlock &BB : F) {
      NumBBs++;
      for (Instruction &I : BB) {
        NumInsts++;
        if (isa<CallInst>(I))
          NumCalls++;
      }
    }

    outs() << "  [NewPM::PrintFunctionStats] " << F.getName() << ":\n";
    outs() << "    Basic blocks: " << NumBBs << "\n";
    outs() << "    Instructions: " << NumInsts << "\n";
    outs() << "    Function calls: " << NumCalls << "\n";

    // This pass only reads IR, never modifies it
    return PreservedAnalyses::all();
  }

  // Optional: provide a name for the pass
  static StringRef name() { return "PrintFunctionStats"; }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pass: RemoveDeadBlocks - new PM pass that depends on an analysis
// Uses the DeadBlockDetector analysis to find and remove unreachable blocks.
// Demonstrates how to consume an analysis in a new-PM pass.
// ---------------------------------------------------------------------------
namespace {

class RemoveDeadBlocks : public PassInfoMixin<RemoveDeadBlocks> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    // Request the DeadBlockDetector analysis
    // The analysis manager will compute it if not already cached,
    // or return the cached result if available.
    DeadBlockInfo &Info = AM.getResult<DeadBlockDetector>(F);

    if (!Info.hasDeadBlocks()) {
      outs() << "  [NewPM::RemoveDeadBlocks] " << F.getName()
             << ": No dead blocks found\n";
      return PreservedAnalyses::all();
    }

    outs() << "  [NewPM::RemoveDeadBlocks] " << F.getName()
           << ": Removing " << Info.DeadBlocks.size() << " dead block(s)\n";

    for (BasicBlock *BB : Info.DeadBlocks) {
      outs() << "    Removing dead block: " << BB->getName() << "\n";
      // We must also remove this block from any phi node uses
      BB->replaceAllUsesWith(UndefValue::get(BB->getType()));
      BB->eraseFromParent();
    }

    // We modified the CFG, so analyses that depend on CFG structure
    // (like LoopInfo, DominatorTree) are not preserved.
    // But we preserve our own DeadBlockDetector analysis since we
    // removed all dead blocks.
    PreservedAnalyses PA;
    PA.preserve<DeadBlockDetector>();
    return PA;
  }

  static StringRef name() { return "RemoveDeadBlocks"; }
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Build a test module with some dead blocks (unreachable from entry)
// ---------------------------------------------------------------------------
std::unique_ptr<Module> buildModule(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("NewPMDemo", Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);

  // Function 1: has an unreachable block
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "has_dead_block", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *LiveBB  = BasicBlock::Create(Ctx, "live", F);
    BasicBlock *DeadBB  = BasicBlock::Create(Ctx, "dead", F); // no predecessors

    IRBuilder<> B(EntryBB);
    B.CreateBr(LiveBB);

    B.SetInsertPoint(LiveBB);
    B.CreateRet(F->arg_begin());

    // DeadBB has code but no predecessors - it's unreachable
    B.SetInsertPoint(DeadBB);
    B.CreateRet(ConstantInt::get(I32, 999));
  }

  // Function 2: all blocks reachable
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "all_reachable", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *OtherBB = BasicBlock::Create(Ctx, "other", F);

    IRBuilder<> B(EntryBB);
    Value *Cmp = B.CreateICmpSGT(F->arg_begin(),
                                  ConstantInt::get(I32, 0), "cmp");
    B.CreateCondBr(Cmp, OtherBB, OtherBB); // both paths go to OtherBB

    B.SetInsertPoint(OtherBB);
    B.CreateRet(F->arg_begin());
  }

  return M;
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  auto M = buildModule(Context);

  outs() << "=== Before New PM Pipeline ===\n";
  M->print(outs(), nullptr);
  outs() << "\n";

  // -----------------------------------------------------------------------
  // Set up the new pass manager infrastructure
  // -----------------------------------------------------------------------

  // Create the analysis managers
  FunctionAnalysisManager FAM;

  // Register our custom analysis with the FunctionAnalysisManager
  // This tells the manager how to construct the analysis when needed.
  FAM.registerPass([&] { return DeadBlockDetector(); });

  // Also register standard LLVM analyses our pass might use
  // (LoopAnalysis, DominatorTreeAnalysis, etc.)
  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);

  // Create the function pass manager and add passes
  FunctionPassManager FPM;

  // Add our custom passes to the pipeline
  FPM.addPass(PrintFunctionStats());
  FPM.addPass(RemoveDeadBlocks());
  FPM.addPass(PrintFunctionStats());  // Run again to see changes

  // -----------------------------------------------------------------------
  // Run the pipeline on each function in the module
  // -----------------------------------------------------------------------
  outs() << "=== Running New PM Pipeline ===\n";

  // ModulePassManager wrapper to run function passes on all functions
  ModulePassManager MPM;
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);

  MPM.run(*M, MAM);
  outs() << "\n";

  // Verify the module
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Module verification failed after passes!\n";
    return 1;
  }

  outs() << "=== After New PM Pipeline ===\n";
  M->print(outs(), nullptr);

  // -----------------------------------------------------------------------
  // Demonstrate pass pipeline inspection
  // -----------------------------------------------------------------------
  outs() << "\n=== Pipeline Inspection Notes ===\n";
  outs() << "To inspect the pass pipeline, use opt with these flags:\n";
  outs() << "  opt -passes='print<block-freq>' input.ll\n";
  outs() << "  opt -passes='print<cost-model>' -cost-kind=throughput input.ll\n";
  outs() << "  opt -debug-pass-manager -passes='default<O2>' input.ll\n";

  return 0;
}
