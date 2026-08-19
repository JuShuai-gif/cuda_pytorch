// example1.cpp - Chapter 8: Running Common Analysis Passes and Printing Results
//
// Demonstrates:
//   - Setting up ModuleAnalysisManager and FunctionAnalysisManager
//   - Running DominatorTree analysis and printing results
//   - Running LoopInfo analysis and printing loop structure
//   - Using ValueTracking to compute known bits
//   - Building a small test module containing a loop
//
// Build with the repository baseline, LLVM 20.1.x:
//   clang++ example1.cpp $(llvm-config --cxxflags --ldflags --libs core analysis irreader) -o example1

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

/// Build a module containing a simple function with a loop.
/// The function sums numbers from 0 to N-1:
///
///   define i32 @sum_to_n(i32 %n) {
///   entry:
///     br label %loop
///   loop:
///     %i = phi i32 [0, %entry], [%i_next, %loop_body]
///     %sum = phi i32 [0, %entry], [%sum_next, %loop_body]
///     %cmp = icmp slt i32 %i, %n
///     br i1 %cmp, label %loop_body, label %exit
///   loop_body:
///     %i_next = add i32 %i, 1
///     %sum_next = add i32 %sum, %i
///     br label %loop
///   exit:
///     ret i32 %sum
///   }
static void buildLoopModule(Module &M, Function *&F) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);

  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(I32, {I32}, false);
  F = Function::Create(FT, Function::ExternalLinkage, "sum_to_n", &M);
  F->getArg(0)->setName("n");

  BasicBlock *Entry    = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *Loop     = BasicBlock::Create(Ctx, "loop", F);
  BasicBlock *LoopBody = BasicBlock::Create(Ctx, "loop_body", F);
  BasicBlock *Exit     = BasicBlock::Create(Ctx, "exit", F);

  // entry: jump to loop
  Builder.SetInsertPoint(Entry);
  Builder.CreateBr(Loop);

  // loop: phi + icmp + cond_br
  Builder.SetInsertPoint(Loop);
  PHINode *I   = Builder.CreatePHI(I32, 2, "i");
  PHINode *Sum = Builder.CreatePHI(I32, 2, "sum");
  I->addIncoming(ConstantInt::get(I32, 0), Entry);
  Sum->addIncoming(ConstantInt::get(I32, 0), Entry);
  Value *Cmp = Builder.CreateICmpSLT(I, F->getArg(0), "cmp");
  Builder.CreateCondBr(Cmp, LoopBody, Exit);

  // loop_body: i_next, sum_next, branch back
  Builder.SetInsertPoint(LoopBody);
  Value *INext = Builder.CreateAdd(I, ConstantInt::get(I32, 1), "i_next");
  Value *SumNext = Builder.CreateAdd(Sum, I, "sum_next");
  I->addIncoming(INext, LoopBody);
  Sum->addIncoming(SumNext, LoopBody);
  Builder.CreateBr(Loop);

  // exit: return sum
  Builder.SetInsertPoint(Exit);
  Builder.CreateRet(Sum);
}

int main() {
  LLVMContext Context;
  Module M("ch8_analysis", Context);
  M.setDataLayout("e-m:e-i64:64-n32:64");

  // ──────────────── Build test module with a loop ────────────────
  Function *F = nullptr;
  buildLoopModule(M, F);

  outs() << "=== Input IR ===\n";
  M.print(outs(), nullptr);
  outs() << "\n";

  // ──────────────── Set up analysis managers ────────────────
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  PB.registerFunctionAnalyses(FAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerModuleAnalyses(MAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // ──────────────── Run DominatorTree analysis ────────────────
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);

  outs() << "=== DominatorTree Results ===\n";
  for (BasicBlock &BB : *F) {
    outs() << "Block: " << BB.getName() << "\n";
    // Print immediate dominator
    if (BasicBlock *IDom = DT.getNode(&BB)->getIDom() ?
                            DT.getNode(&BB)->getIDom()->getBlock() : nullptr) {
      outs() << "  Immediate dominator: " << IDom->getName() << "\n";
    } else {
      outs() << "  No immediate dominator (entry block)\n";
    }
    // Print dominated blocks
    outs() << "  Dominates:";
    for (BasicBlock &Other : *F) {
      if (&BB != &Other && DT.dominates(&BB, &Other))
        outs() << " " << Other.getName();
    }
    outs() << "\n";
  }
  outs() << "\n";

  // ──────────────── Run LoopInfo analysis ────────────────
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);

  outs() << "=== LoopInfo Results ===\n";
  outs() << "Number of top-level loops: " << std::distance(LI.begin(), LI.end()) << "\n";
  for (Loop *L : LI) {
    outs() << "Loop header: " << L->getHeader()->getName() << "\n";
    outs() << "  Loop depth: " << L->getLoopDepth() << "\n";
    outs() << "  Number of blocks: " << L->getNumBlocks() << "\n";
    outs() << "  Is loop simplify form: " << (L->isLoopSimplifyForm() ? "yes" : "no") << "\n";
    // Print exiting blocks
    SmallVector<BasicBlock *, 4> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    outs() << "  Exiting blocks:";
    for (BasicBlock *BB : ExitingBlocks)
      outs() << " " << BB->getName();
    outs() << "\n";
  }
  outs() << "\n";

  // ──────────────── Demonstrate ValueTracking ────────────────
  // For a known instruction, compute known bits.
  // Example: find an `add` instruction and check its argument's known bits.
  const DataLayout &DL = M.getDataLayout();
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.getOpcode() == Instruction::Add) {
        KnownBits KB = computeKnownBits(&I, DL);
        outs() << "=== ValueTracking on: " << I << "\n";
        outs() << "  Known bits (Zero): ";
        KB.Zero.print(outs(), false);
        outs() << "\n";
        outs() << "  Known bits (One):  ";
        KB.One.print(outs(), false);
        outs() << "\n";
      }
    }
  }

  return 0;
}
