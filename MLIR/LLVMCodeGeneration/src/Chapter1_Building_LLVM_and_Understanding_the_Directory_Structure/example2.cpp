// example2.cpp - Demonstrates lit/FileCheck testing patterns
// This program generates LLVM IR output that can be verified using FileCheck.
// It mimics the typical lit test workflow: generate IR, then use FileCheck
// directives to validate the output.
//
// Usage: ./example2 | FileCheck check-file.txt

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
  LLVMContext Context;
  Module M("FileCheckDemo", Context);

  // ------------------------------------------------------------------
  // Function 1: check_constant_folding
  // Tests that simple arithmetic with constants can be verified
  // FileCheck directives (in a .ll test file):
  //   CHECK-LABEL: define i32 @check_constant_folding
  //   CHECK: ret i32 42
  // ------------------------------------------------------------------
  {
    FunctionType *FT = FunctionType::get(Type::getInt32Ty(Context), {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "check_constant_folding", M);
    BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
    IRBuilder<> Builder(BB);
    Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 42));
  }

  // ------------------------------------------------------------------
  // Function 2: check_add_instruction
  // Tests that the add instruction appears with the right operands
  // FileCheck directives:
  //   CHECK-LABEL: define i32 @check_add_instruction
  //   CHECK: %result = add i32 %a, %b
  //   CHECK-NEXT: ret i32 %result
  // ------------------------------------------------------------------
  {
    Type *I32 = Type::getInt32Ty(Context);
    FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "check_add_instruction", M);
    BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
    IRBuilder<> Builder(BB);

    auto Args = F->args().begin();
    Value *A = Args++;
    Value *B = Args;
    Value *Result = Builder.CreateAdd(A, B, "result");
    Builder.CreateRet(Result);
  }

  // ------------------------------------------------------------------
  // Function 3: check_branch_structure
  // Tests control flow structure with conditional branches
  // FileCheck directives:
  //   CHECK-LABEL: define i32 @check_branch_structure
  //   CHECK: br i1 %cmp, label %then, label %else
  //   CHECK: then:
  //   CHECK-NEXT: ret i32 1
  //   CHECK: else:
  //   CHECK-NEXT: ret i32 0
  // ------------------------------------------------------------------
  {
    Type *I32 = Type::getInt32Ty(Context);
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "check_branch_structure", M);
    // entry block
    BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
    BasicBlock *ThenBB = BasicBlock::Create(Context, "then", F);
    BasicBlock *ElseBB = BasicBlock::Create(Context, "else", F);

    IRBuilder<> Builder(EntryBB);
    Value *Arg = F->arg_begin();
    Value *Cmp = Builder.CreateICmpSGT(
        Arg, ConstantInt::get(I32, 0), "cmp");
    Builder.CreateCondBr(Cmp, ThenBB, ElseBB);

    Builder.SetInsertPoint(ThenBB);
    Builder.CreateRet(ConstantInt::get(I32, 1));

    Builder.SetInsertPoint(ElseBB);
    Builder.CreateRet(ConstantInt::get(I32, 0));
  }

  // ------------------------------------------------------------------
  // Function 4: check_phi_node
  // Tests phi node generation for demonstrating CHECK-DAG patterns
  // FileCheck directives:
  //   CHECK-LABEL: define i32 @check_phi_node
  //   CHECK-DAG: phi i32 [ 10, %left ], [ 20, %right ]
  //   CHECK-DAG: ret i32 %result
  // ------------------------------------------------------------------
  {
    Type *I32 = Type::getInt32Ty(Context);
    FunctionType *FT = FunctionType::get(I32, {Type::getInt1Ty(Context)}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "check_phi_node", M);
    BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
    BasicBlock *LeftBB = BasicBlock::Create(Context, "left", F);
    BasicBlock *RightBB = BasicBlock::Create(Context, "right", F);
    BasicBlock *MergeBB = BasicBlock::Create(Context, "merge", F);

    IRBuilder<> Builder(EntryBB);
    Builder.CreateCondBr(F->arg_begin(), LeftBB, RightBB);

    Builder.SetInsertPoint(LeftBB);
    Builder.CreateBr(MergeBB);

    Builder.SetInsertPoint(RightBB);
    Builder.CreateBr(MergeBB);

    Builder.SetInsertPoint(MergeBB);
    PHINode *Phi = Builder.CreatePHI(I32, 2, "result");
    Phi->addIncoming(ConstantInt::get(I32, 10), LeftBB);
    Phi->addIncoming(ConstantInt::get(I32, 20), RightBB);
    Builder.CreateRet(Phi);
  }

  // Print the entire module - this output can be piped to FileCheck
  M.print(outs(), nullptr);

  return 0;
}
