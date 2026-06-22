// example2.cpp - Demonstrates lit test creation patterns for LLVM
// This program generates IR that can be used as part of a lit test.
// It also shows how to create a simple FileCheck-based test driver.
//
// Typical lit test file (test.ll) would contain:
//   ; RUN: opt -load-pass-plugin=./libMyPlugin.so -passes=my-pass %s -S | FileCheck %s
//   ; CHECK-LABEL: define i32 @test_func
//   ; CHECK: ret i32 10

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// A minimal pass implementation that can be plugged into lit tests.
// This demonstrates the "new pass manager" plugin mechanism used in testing.
// The pass simply annotates the module with a comment and cleans up unused fns.
// ---------------------------------------------------------------------------
namespace {

class CleanupDeadFunctions : public PassInfoMixin<CleanupDeadFunctions> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = false;
    SmallVector<Function *, 8> ToDelete;

    for (Function &F : M) {
      if (F.isDeclaration() || !F.hasAddressTaken()) {
        if (F.user_empty() && &F != M.getFunction("main")) {
          ToDelete.push_back(&F);
          Changed = true;
        }
      }
    }

    for (Function *F : ToDelete) {
      F->eraseFromParent();
    }

    return Changed ? PreservedAnalyses::none()
                   : PreservedAnalyses::all();
  }
};

// Registration for opt -load-pass-plugin
llvm::PassPluginLibraryInfo getCleanupPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CleanupDeadFunctions", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "cleanup-dead-funcs") {
                    MPM.addPass(CleanupDeadFunctions());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getCleanupPluginInfo();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Main: Creates an LLVM IR module that demonstrates test patterns.
// Output can be used with FileCheck to verify the IR structure.
// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  Module M("LitTestDemo", Context);

  // -----------------------------------------------------------------------
  // Build a module with multiple functions to test dead function removal.
  //
  // FileCheck directives (in a .ll test file):
  //   CHECK-LABEL: define i32 @main(
  //   CHECK-NOT: define void @unused_helper(
  //   CHECK-NOT: define i32 @unused_const(
  // -----------------------------------------------------------------------
  {
    // main function - always kept
    FunctionType *MainFT = FunctionType::get(
        Type::getInt32Ty(Context), {}, false);
    Function *MainF = Function::Create(
        MainFT, Function::ExternalLinkage, "main", M);
    BasicBlock *BB = BasicBlock::Create(Context, "entry", MainF);
    IRBuilder<> Builder(BB);
    Builder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 0));

    // unused_helper - should be removed by our pass
    FunctionType *HelperFT = FunctionType::get(
        Type::getVoidTy(Context), {}, false);
    Function::Create(HelperFT, Function::InternalLinkage, "unused_helper", M);

    // unused_const function - should be removed
    FunctionType *ConstFT = FunctionType::get(
        Type::getInt32Ty(Context), {}, false);
    Function *ConstF = Function::Create(
        ConstFT, Function::InternalLinkage, "unused_const", M);
    BasicBlock *ConstBB = BasicBlock::Create(Context, "entry", ConstF);
    IRBuilder<> ConstBuilder(ConstBB);
    ConstBuilder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 42));

    // used_helper - referenced in main so should be kept
    FunctionType *UsedFT = FunctionType::get(
        Type::getInt32Ty(Context), {}, false);
    Function *UsedF = Function::Create(
        UsedFT, Function::InternalLinkage, "used_helper", M);
    BasicBlock *UsedBB = BasicBlock::Create(Context, "entry", UsedF);
    IRBuilder<> UsedBuilder(UsedBB);
    UsedBuilder.CreateRet(ConstantInt::get(Type::getInt32Ty(Context), 99));

    // Make main depend on used_helper
    Builder.SetInsertPoint(&MainF->getEntryBlock().front());
    Value *CallResult = Builder.CreateCall(UsedF, {}, "call_res");
    Builder.CreateRet(CallResult);
    // Remove the old ret
    MainF->getEntryBlock().getTerminator()->eraseFromParent();
    Builder.SetInsertPoint(&MainF->getEntryBlock());
  }

  // -----------------------------------------------------------------------
  // Run the pass and print the result.
  // The output IR should only contain main and used_helper.
  // -----------------------------------------------------------------------
  ModulePassManager MPM;
  MPM.addPass(CleanupDeadFunctions());

  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);

  outs() << "; Before cleanup:\n";
  M.print(outs(), nullptr);

  MPM.run(M, MAM);

  outs() << "\n; After cleanup:\n";
  M.print(outs(), nullptr);

  return 0;
}
