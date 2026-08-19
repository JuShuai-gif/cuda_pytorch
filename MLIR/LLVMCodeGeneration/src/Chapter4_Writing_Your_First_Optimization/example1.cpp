// example1.cpp - Simple Constant Propagation Pass
// Implements a basic constant propagation optimization at the LLVM IR level.
//
// Algorithm:
// 1. Iterate through all instructions in a function
// 2. Ask LLVM's ConstantFoldInstruction to preserve all IR semantics
// 3. Replace foldable instructions with the returned Constant/PoisonValue
// 4. Erase the obsolete instruction without invalidating traversal
//
// This demonstrates: SSA traversal, use-def chains, DataLayout-aware folding,
// replaceAllUsesWith, safe deletion, and verifier-driven development.

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace llvm;

// ---------------------------------------------------------------------------
// Simple constant propagation using LLVM's canonical folding implementation.
// Hand-written APInt folding is deliberately avoided: it is easy to mishandle
// nsw/nuw/exact flags, poison, oversize shifts, vectors, or target DataLayout.
//
// Returns true if the IR was modified, false otherwise.
// ---------------------------------------------------------------------------
bool myConstantPropagation(Function &F) {
  bool Changed = false;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Snapshot instructions so erasing one never invalidates traversal.
  SmallVector<Instruction *, 32> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      Worklist.push_back(&I);

  for (Instruction *I : Worklist) {
    if (I->isTerminator())
      continue;

    Constant *Folded = ConstantFoldInstruction(I, DL);
    if (!Folded)
      continue;

    I->replaceAllUsesWith(Folded);
    I->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

// ---------------------------------------------------------------------------
// Build a test module with constant-foldable operations
// ---------------------------------------------------------------------------
std::unique_ptr<Module> buildTestModule(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("ConstPropTest", Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I1  = Type::getInt1Ty(Ctx);
  Type *I8  = Type::getInt8Ty(Ctx);

  // Function 1: Simple arithmetic with constants
  // int f1(int x) { return (2 + 3) * x; }
  // After constant prop: return 5 * x;
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f1_arith", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<NoFolder> B(BB);

    Value *Two  = ConstantInt::get(I32, 2);
    Value *Three = ConstantInt::get(I32, 3);
    Value *Add = B.CreateAdd(Two, Three, "add");    // 2 + 3 = 5
    Value *X = F->arg_begin();
    Value *Mul = B.CreateMul(Add, X, "mul");         // 5 * x
    B.CreateRet(Mul);
  }

  // Function 2: Nested constant expressions
  // int f2(int a) { return (10 - 3) + (8 * 2); }
  // After constant prop: return 23;
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f2_nested", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<NoFolder> B(BB);

    Value *C10 = ConstantInt::get(I32, 10);
    Value *C3  = ConstantInt::get(I32, 3);
    Value *C8  = ConstantInt::get(I32, 8);
    Value *C2  = ConstantInt::get(I32, 2);

    Value *Sub1 = B.CreateSub(C10, C3, "sub");        // 10 - 3 = 7
    Value *Mul1 = B.CreateMul(C8, C2, "mul");          // 8 * 2 = 16
    Value *Add1 = B.CreateAdd(Sub1, Mul1, "add");      // 7 + 16 = 23
    B.CreateRet(Add1);
  }

  // Function 3: Constant comparison
  // bool f3() { return 5 > 3; }
  // After constant prop: return true;
  {
    FunctionType *FT = FunctionType::get(I1, {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f3_icmp", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<NoFolder> B(BB);

    Value *C5 = ConstantInt::get(I32, 5);
    Value *C3 = ConstantInt::get(I32, 3);
    Value *Cmp = B.CreateICmpSGT(C5, C3, "cmp");
    B.CreateRet(Cmp);
  }

  // Function 4: Bitwise operations on constants
  // int f4() { return (0xFF & 0x0F) | 0x10; }
  // After constant prop: return 0x1F (31);
  {
    FunctionType *FT = FunctionType::get(I32, {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f4_bitwise", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<NoFolder> B(BB);

    Value *FF = ConstantInt::get(I32, 0xFF);
    Value *OF = ConstantInt::get(I32, 0x0F);
    Value *TEN = ConstantInt::get(I32, 0x10);
    Value *And1 = B.CreateAnd(FF, OF, "and");     // 0xFF & 0x0F = 0x0F
    Value *Or1  = B.CreateOr(And1, TEN, "or");     // 0x0F | 0x10 = 0x1F
    B.CreateRet(Or1);
  }

  // Function 5: poison-aware folding.
  // `add nsw i8 127, 1` is poison, not the wrapped integer -128.
  {
    FunctionType *FT = FunctionType::get(I8, {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "f5_nsw_overflow", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<NoFolder> B(BB);

    Value *Max = ConstantInt::get(I8, 127);
    Value *One = ConstantInt::get(I8, 1);
    Value *Overflow = B.CreateNSWAdd(Max, One, "overflow");
    B.CreateRet(Overflow);
  }

  return M;
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;

  outs() << "=== Before Constant Propagation ===\n";
  auto M = buildTestModule(Context);
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Input module verification failed!\n";
    return 1;
  }
  M->print(outs(), nullptr);

  outs() << "\n=== Running Constant Propagation ===\n";
  bool Modified = false;
  for (Function &F : *M) {
    if (!F.isDeclaration()) {
      bool FuncChanged = myConstantPropagation(F);
      if (FuncChanged) {
        outs() << "  Modified: " << F.getName() << "\n";
        Modified = true;
      }
    }
  }

  if (!Modified) {
    outs() << "  (no changes made)\n";
  }

  // Verify the result
  if (verifyModule(*M, &errs())) {
    errs() << "ERROR: Module verification failed!\n";
    return 1;
  }

  outs() << "\n=== After Constant Propagation ===\n";
  M->print(outs(), nullptr);

  return 0;
}
