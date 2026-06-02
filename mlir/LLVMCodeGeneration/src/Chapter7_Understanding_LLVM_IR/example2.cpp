// example2.cpp - Chapter 7: Working with LLVM IR Intrinsics and Target metadata
//
// Demonstrates:
//   - Setting target triple and data layout on a module
//   - Creating a function with target-specific attributes (#0 syntax)
//   - Building a function that uses the @llvm.vector.reduce.add intrinsic
//   - Using getelementptr (GEP) for aggregate type field access
//
// Build with LLVM 17+:
//   clang++ example2.cpp $(llvm-config --cxxflags --ldflags --libs core irreader) -o example2

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int main() {
  LLVMContext Context;
  Module M("ch7_target_example", Context);

  // ──────────────── Set target triple ────────────────
  M.setTargetTriple("aarch64-apple-macosx14.0.0");

  // ──────────────── Set data layout ────────────────
  M.setDataLayout("e-m:o-i64:64-i128:128-n32:64-S128");

  outs() << "Target Triple: " << M.getTargetTriple() << "\n";
  outs() << "Data Layout:   " << M.getDataLayoutStr() << "\n\n";

  // ──────────────── Create a struct type for GEP demo ────────────────
  // %Point = type { float, float, i32 }
  Type *FloatTy = Type::getFloatTy(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  StructType *PointTy = StructType::create(Context, {FloatTy, FloatTy, Int32Ty}, "Point");

  // ──────────────── Function to access struct fields via GEP ────────────────
  FunctionType *GEPFuncTy = FunctionType::get(Int32Ty, {PointerType::get(Context, 0)}, false);
  Function *GEPFunc = Function::Create(GEPFuncTy, Function::ExternalLinkage, "get_point_id", &M);
  GEPFunc->getArg(0)->setName("ptr");

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", GEPFunc);
  IRBuilder<> Builder(Entry);

  // GEP to access the third field (index 2): the i32 "id"
  Value *IdPtr = Builder.CreateStructGEP(PointTy, GEPFunc->getArg(0), 2, "id_ptr");
  Value *IdVal = Builder.CreateLoad(Int32Ty, IdPtr, "id_val");
  Builder.CreateRet(IdVal);

  // ──────────────── Create a function using vector intrinsic ────────────────
  Type *Vec4I32Ty = FixedVectorType::get(Int32Ty, 4);

  // Declare @llvm.vector.reduce.add.v4i32
  FunctionType *ReduceTy = FunctionType::get(Int32Ty, {Vec4I32Ty}, false);
  FunctionCallee ReduceIntrinsic =
      M.getOrInsertFunction("llvm.vector.reduce.add.v4i32", ReduceTy);

  FunctionType *VecFuncTy = FunctionType::get(Int32Ty, {PointerType::get(Context, 0)}, false);
  Function *VecFunc = Function::Create(VecFuncTy, Function::ExternalLinkage, "reduce_vector", &M);
  VecFunc->getArg(0)->setName("src");

  BasicBlock *VecEntry = BasicBlock::Create(Context, "entry", VecFunc);
  Builder.SetInsertPoint(VecEntry);

  // Load a <4 x i32> vector from %src
  Value *Vec = Builder.CreateLoad(Vec4I32Ty, VecFunc->getArg(0), "vec");
  // Call the intrinsic
  Value *Reduced = Builder.CreateCall(ReduceIntrinsic, {Vec}, "hadd");
  Builder.CreateRet(Reduced);

  // ──────────────── Print the complete module ────────────────
  outs() << "=== Generated Module ===\n";
  M.print(outs(), nullptr);

  return 0;
}
