// example1.cpp - Chapter 7: Building LLVM IR Types, Functions, and Instructions
//
// Demonstrates constructing various LLVM IR elements programmatically:
//   - Single-value types: i32, float, ptr
//   - Aggregate types: struct, array
//   - Functions with parameters and return types
//   - Basic blocks with terminator instructions
//   - Common instructions: add, mul, icmp, br, ret, load, store, alloca
//   - Building a complete module
//
// Build with LLVM 17+:
//   clang++ example1.cpp $(llvm-config --cxxflags --ldflags --libs core irreader) -o example1

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
  LLVMContext Context;
  Module M("ch7_example", Context);
  IRBuilder<> Builder(Context);

  // ──────────────── Demonstrate single-value types ────────────────
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int1Ty  = Type::getInt1Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  Type *PtrTy   = PointerType::get(Context, 0);

  // Vector type: <4 x float>
  Type *Vec4FloatTy = FixedVectorType::get(FloatTy, 4);

  outs() << "=== Single-Value Types ===\n";
  outs() << "i32:        "; Int32Ty->print(outs());     outs() << "\n";
  outs() << "float:      "; FloatTy->print(outs());      outs() << "\n";
  outs() << "ptr:        "; PtrTy->print(outs());        outs() << "\n";
  outs() << "<4 x float>: "; Vec4FloatTy->print(outs()); outs() << "\n\n";

  // ──────────────── Demonstrate aggregate types ────────────────
  // Named struct type: { i32, float, ptr }
  StructType *NamedStructTy = StructType::create(Context, {Int32Ty, FloatTy, PtrTy}, "MyStruct");

  // Anonymous struct type: { i32, i1 }
  StructType *AnonStructTy = StructType::get(Context, {Int32Ty, Int1Ty});

  // Array type: [10 x i32]
  Type *ArrayTy = ArrayType::get(Int32Ty, 10);

  outs() << "=== Aggregate Types ===\n";
  outs() << "%MyStruct:       "; NamedStructTy->print(outs()); outs() << "\n";
  outs() << "Anonymous struct: "; AnonStructTy->print(outs());  outs() << "\n";
  outs() << "[10 x i32]:      "; ArrayTy->print(outs());       outs() << "\n\n";

  // ──────────────── Create a function ────────────────
  // Function type: i32 (i32, i32)
  FunctionType *FuncTy = FunctionType::get(Int32Ty, {Int32Ty, Int32Ty}, false);
  Function *F = Function::Create(FuncTy, Function::ExternalLinkage, "add_and_mul", &M);

  // Name the arguments
  F->getArg(0)->setName("lhs");
  F->getArg(1)->setName("rhs");

  // ──────────────── Build basic blocks ────────────────
  BasicBlock *EntryBB  = BasicBlock::Create(Context, "entry", F);
  BasicBlock *ThenBB   = BasicBlock::Create(Context, "then", F);
  BasicBlock *ElseBB   = BasicBlock::Create(Context, "else", F);
  BasicBlock *MergeBB  = BasicBlock::Create(Context, "merge", F);

  // ──────────────── Fill entry block ────────────────
  Builder.SetInsertPoint(EntryBB);

  // alloca for a local variable
  Value *LocalVar = Builder.CreateAlloca(Int32Ty, nullptr, "local_val");
  Builder.CreateStore(ConstantInt::get(Int32Ty, 0), LocalVar);

  // Add instruction: %sum = add i32 %lhs, %rhs
  Value *Sum = Builder.CreateAdd(F->getArg(0), F->getArg(1), "sum");

  // Mul instruction: %prod = mul i32 %sum, 2
  Value *Prod = Builder.CreateMul(Sum, ConstantInt::get(Int32Ty, 2), "prod");

  // Store the result back
  Builder.CreateStore(Prod, LocalVar);

  // Load it back
  Value *Loaded = Builder.CreateLoad(Int32Ty, LocalVar, "loaded");

  // Conditional: icmp sgt i32 %loaded, 10
  Value *Cond = Builder.CreateICmpSGT(Loaded, ConstantInt::get(Int32Ty, 10), "cmp");

  // Branch based on condition
  Builder.CreateCondBr(Cond, ThenBB, ElseBB);

  // ──────────────── Fill then block ────────────────
  Builder.SetInsertPoint(ThenBB);
  Value *ThenVal = Builder.CreateAdd(Loaded, ConstantInt::get(Int32Ty, 100), "then_result");
  Builder.CreateBr(MergeBB);

  // ──────────────── Fill else block ────────────────
  Builder.SetInsertPoint(ElseBB);
  Value *ElseVal = Builder.CreateMul(Loaded, ConstantInt::get(Int32Ty, 3), "else_result");
  Builder.CreateBr(MergeBB);

  // ──────────────── Fill merge block (phi node) ────────────────
  Builder.SetInsertPoint(MergeBB);
  PHINode *Phi = Builder.CreatePHI(Int32Ty, 2, "result");
  Phi->addIncoming(ThenVal, ThenBB);
  Phi->addIncoming(ElseVal, ElseBB);

  Builder.CreateRet(Phi);

  // ──────────────── Dump the module ────────────────
  outs() << "=== Generated LLVM IR ===\n";
  M.print(outs(), nullptr);

  // ──────────────── Bonus: Undef and Poison values ────────────────
  outs() << "\n=== Special Constants ===\n";
  Value *UndefVal  = UndefValue::get(Int32Ty);
  Value *PoisonVal = PoisonValue::get(FloatTy);
  outs() << "undef i32:   "; UndefVal->print(outs());  outs() << "\n";
  outs() << "poison float: "; PoisonVal->print(outs()); outs() << "\n";

  return 0;
}
