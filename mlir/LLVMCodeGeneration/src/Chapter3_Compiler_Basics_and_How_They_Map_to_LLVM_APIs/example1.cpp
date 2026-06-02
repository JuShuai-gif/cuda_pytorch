// example1.cpp - Building LLVM IR from scratch using the C++ API
// Demonstrates: LLVMContext, Module, Function, BasicBlock, Instructions
// Builds a complete module with multiple functions showing different IR patterns.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Helper: Build a simple add function
//   define i32 @simple_add(i32 %a, i32 %b) {
//   entry:
//     %res = add i32 %a, %b
//     ret i32 %res
//   }
// ---------------------------------------------------------------------------
void buildSimpleAdd(Module &M, LLVMContext &Ctx) {
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "simple_add", M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> Builder(BB);

  auto Args = F->args().begin();
  Value *A = Args++;
  Value *B = Args;
  Value *Res = Builder.CreateAdd(A, B, "res");
  Builder.CreateRet(Res);
}

// ---------------------------------------------------------------------------
// Helper: Build an if-else function with phi node
//   define i32 @max(i32 %a, i32 %b) {
//   entry:
//     %cmp = icmp sgt i32 %a, %b
//     br i1 %cmp, label %then, label %else
//   then:
//     br label %merge
//   else:
//     br label %merge
//   merge:
//     %res = phi i32 [ %a, %then ], [ %b, %else ]
//     ret i32 %res
//   }
// ---------------------------------------------------------------------------
void buildIfElseWithPhi(Module &M, LLVMContext &Ctx) {
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "max", M);

  // Create all basic blocks first
  BasicBlock *EntryBB  = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *ThenBB   = BasicBlock::Create(Ctx, "then", F);
  BasicBlock *ElseBB   = BasicBlock::Create(Ctx, "else", F);
  BasicBlock *MergeBB  = BasicBlock::Create(Ctx, "merge", F);

  // Entry block: compare and branch
  IRBuilder<> Builder(EntryBB);
  auto Args = F->args().begin();
  Value *A = Args++;
  Value *B = Args;
  Value *Cmp = Builder.CreateICmpSGT(A, B, "cmp");
  Builder.CreateCondBr(Cmp, ThenBB, ElseBB);

  // then block: jump to merge
  Builder.SetInsertPoint(ThenBB);
  Builder.CreateBr(MergeBB);

  // else block: jump to merge
  Builder.SetInsertPoint(ElseBB);
  Builder.CreateBr(MergeBB);

  // merge block: phi node and return
  Builder.SetInsertPoint(MergeBB);
  PHINode *Phi = Builder.CreatePHI(I32, 2, "res");
  Phi->addIncoming(A, ThenBB);
  Phi->addIncoming(B, ElseBB);
  Builder.CreateRet(Phi);
}

// ---------------------------------------------------------------------------
// Helper: Build a loop function (computes factorial)
//   define i32 @factorial(i32 %n) {
//   entry:
//     %cmp = icmp sle i32 %n, 1
//     br i1 %cmp, label %done, label %loop
//   loop:
//     %n_prev = phi i32 [ %n, %entry ], [ %n_next, %loop ]
//     %acc = phi i32 [ 1, %entry ], [ %acc_next, %loop ]
//     %n_next = sub i32 %n_prev, 1
//     %acc_next = mul i32 %acc, %n_prev
//     %cont = icmp sgt i32 %n_next, 1
//     br i1 %cont, label %loop, label %done
//   done:
//     %result = phi i32 [ 1, %entry ], [ %acc_next, %loop ]
//     ret i32 %result
//   }
// ---------------------------------------------------------------------------
void buildLoop(Module &M, LLVMContext &Ctx) {
  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(I32, {I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "factorial", M);

  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *LoopBB  = BasicBlock::Create(Ctx, "loop", F);
  BasicBlock *DoneBB  = BasicBlock::Create(Ctx, "done", F);

  Value *N = F->arg_begin();

  // Entry: compare and branch
  IRBuilder<> Builder(EntryBB);
  Value *CmpInit = Builder.CreateICmpSLE(N, ConstantInt::get(I32, 1), "cmp");
  Builder.CreateCondBr(CmpInit, DoneBB, LoopBB);

  // Loop body: phi nodes + computation + backedge
  Builder.SetInsertPoint(LoopBB);
  PHINode *NPhi = Builder.CreatePHI(I32, 2, "n_prev");
  PHINode *AccPhi = Builder.CreatePHI(I32, 2, "acc");
  NPhi->addIncoming(N, EntryBB);
  AccPhi->addIncoming(ConstantInt::get(I32, 1), EntryBB);

  Value *NNext = Builder.CreateSub(NPhi, ConstantInt::get(I32, 1), "n_next");
  Value *AccNext = Builder.CreateMul(AccPhi, NPhi, "acc_next");
  Value *Cont = Builder.CreateICmpSGT(NNext, ConstantInt::get(I32, 1), "cont");
  Builder.CreateCondBr(Cont, LoopBB, DoneBB);

  // Update phi incoming values for the backedge
  NPhi->addIncoming(NNext, LoopBB);
  AccPhi->addIncoming(AccNext, LoopBB);

  // Done block: phi to select first-iteration result vs loop result
  Builder.SetInsertPoint(DoneBB);
  PHINode *Result = Builder.CreatePHI(I32, 2, "result");
  Result->addIncoming(ConstantInt::get(I32, 1), EntryBB);
  Result->addIncoming(AccNext, LoopBB);
  Builder.CreateRet(Result);
}

// ---------------------------------------------------------------------------
// Helper: Build a function with a global variable
//   @counter = global i32 0
//   define i32 @increment() {
//   entry:
//     %val = load i32, ptr @counter
//     %new = add i32 %val, 1
//     store i32 %new, ptr @counter
//     ret i32 %new
//   }
// ---------------------------------------------------------------------------
void buildWithGlobal(Module &M, LLVMContext &Ctx) {
  Type *I32 = Type::getInt32Ty(Ctx);

  // Create a global variable
  GlobalVariable *Counter = new GlobalVariable(
      M, I32, false, GlobalValue::InternalLinkage,
      ConstantInt::get(I32, 0), "counter");

  // Create the function
  FunctionType *FT = FunctionType::get(I32, {}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "increment", M);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> Builder(BB);

  Value *Val = Builder.CreateLoad(I32, Counter, "val");
  Value *NewVal = Builder.CreateAdd(Val, ConstantInt::get(I32, 1), "new");
  Builder.CreateStore(NewVal, Counter);
  Builder.CreateRet(NewVal);
}

// ---------------------------------------------------------------------------
// Main: Build all functions and print the resulting module
// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  Module M("FromScratchIR", Context);

  // Set target information to make the module well-formed
  M.setTargetTriple("x86_64-unknown-linux-gnu");

  // Build all functions
  buildSimpleAdd(M, Context);
  buildIfElseWithPhi(M, Context);
  buildLoop(M, Context);
  buildWithGlobal(M, Context);

  // Verify the IR is correct before printing
  if (verifyModule(M, &errs())) {
    errs() << "Error: Module verification failed!\n";
    return 1;
  }

  outs() << "; Module built from scratch using LLVM C++ API\n";
  outs() << "; Contains examples of: basic arithmetic, branches, phi nodes,\n";
  outs() << "; loops, global variables, load/store instructions\n\n";
  M.print(outs(), nullptr);

  return 0;
}
