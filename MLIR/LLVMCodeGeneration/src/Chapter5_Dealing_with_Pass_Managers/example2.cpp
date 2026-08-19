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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <memory>
#include <utility>

using namespace llvm;

// ---------------------------------------------------------------------------
// Analysis: DeadBlockDetector
// An analysis that identifies every basic block not reachable from entry.
// Checking only pred_size(BB) == 0 is insufficient: an unreachable cycle, or
// a chain rooted at a dead block, can still give every member a predecessor.
// Demonstrates the AnalysisInfoMixin pattern.
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

    if (F.empty())
      return Info;

    SmallPtrSet<BasicBlock *, 16> Reachable;
    for (BasicBlock *BB : depth_first_ext(&F, Reachable))
      (void)BB;

    for (BasicBlock &BB : F) {
      if (!Reachable.contains(&BB))
        Info.DeadBlocks.push_back(&BB);
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

    for (const BasicBlock *BB : Info.DeadBlocks)
      outs() << "    Removing dead block: " << BB->getName() << "\n";

    // Delete the unreachable region as one set.  LLVM updates successor PHIs
    // and handles references such as blockaddress correctly; replacing block
    // operands with undef by hand can silently produce malformed IR.
    DeleteDeadBlocks(Info.DeadBlocks);

    // Do not preserve DeadBlockDetector: its cached result owns pointers to
    // the blocks just deleted.  Returning none prevents stale-pointer reuse.
    return PreservedAnalyses::none();
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

  // Function 1: covers an orphan block, an unreachable chain/cycle, and a
  // PHI edge that must be repaired while dead blocks are deleted.
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "has_dead_block", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *LiveBB  = BasicBlock::Create(Ctx, "live", F);
    BasicBlock *DeadToLive = BasicBlock::Create(Ctx, "dead.to.live", F);
    BasicBlock *DeadRoot = BasicBlock::Create(Ctx, "dead.root", F);
    BasicBlock *DeadCycleA = BasicBlock::Create(Ctx, "dead.cycle.a", F);
    BasicBlock *DeadCycleB = BasicBlock::Create(Ctx, "dead.cycle.b", F);

    IRBuilder<> B(EntryBB);
    B.CreateBr(LiveBB);

    B.SetInsertPoint(LiveBB);
    PHINode *Result = B.CreatePHI(I32, 2, "result");
    Result->addIncoming(F->getArg(0), EntryBB);
    Result->addIncoming(ConstantInt::get(I32, 999), DeadToLive);
    B.CreateRet(Result);

    // This dead block points into live code. DeleteDeadBlocks must remove its
    // incoming value from the PHI in LiveBB.
    B.SetInsertPoint(DeadToLive);
    B.CreateBr(LiveBB);

    // These blocks form an unreachable region in which the cycle members do
    // have predecessors; a pred_size(BB) == 0 detector would miss them.
    B.SetInsertPoint(DeadRoot);
    B.CreateBr(DeadCycleA);
    B.SetInsertPoint(DeadCycleA);
    B.CreateBr(DeadCycleB);
    B.SetInsertPoint(DeadCycleB);
    B.CreateBr(DeadCycleA);
  }

  // Function 2: all blocks reachable
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "all_reachable", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *OtherBB = BasicBlock::Create(Ctx, "other", F);

    IRBuilder<> B(EntryBB);
    Value *Cmp = B.CreateICmpSGT(F->getArg(0),
                                  ConstantInt::get(I32, 0), "cmp");
    B.CreateCondBr(Cmp, OtherBB, OtherBB); // both paths go to OtherBB

    B.SetInsertPoint(OtherBB);
    B.CreateRet(F->getArg(0));
  }

  return M;
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  auto M = buildModule(Context);

  outs() << "=== Before New PM Pipeline ===\n";
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Input module verification failed!\n";
    return 1;
  }
  M->print(outs(), nullptr);
  outs() << "\n";

  // -----------------------------------------------------------------------
  // Set up the new pass manager infrastructure
  // -----------------------------------------------------------------------

  // Create all four analysis managers. Adaptors rely on the proxy analyses
  // installed by crossRegisterProxies; registering only FAM/MAM is incomplete.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register our custom analysis with the FunctionAnalysisManager
  // This tells the manager how to construct the analysis when needed.
  FAM.registerPass([&] { return DeadBlockDetector(); });

  // Also register standard LLVM analyses our pass might use
  // (LoopAnalysis, DominatorTreeAnalysis, etc.)
  PassBuilder PB;
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

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
