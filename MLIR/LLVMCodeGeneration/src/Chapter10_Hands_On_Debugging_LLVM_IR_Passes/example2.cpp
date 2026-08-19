// example2.cpp - Chapter 10: IR Printing and Verifier Techniques
//
// Demonstrates:
//   - Manually printing IR before and after transformations
//   - Running the verifier on a module / function
//   - Using print-before/after equivalent in C++ code
//   - Creating StandardInstrumentations for pass pipeline introspection
//   - Simulating the effect of -print-before-all and -verify-each
//
// Build with the repository baseline, LLVM 20.1.x:
//   clang++ example2.cpp $(llvm-config --cxxflags --ldflags --libs core passes irreader support) -o example2

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <utility>

using namespace llvm;

// ──────────────── A pass that deliberately transforms IR ────────────────
struct DemoTransformPass : public PassInfoMixin<DemoTransformPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool Changed = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : make_early_inc_range(BB)) {
        // Replace `sub X, C` with `add X, -C`
        if (I.getOpcode() == Instruction::Sub) {
          Value *Op0 = I.getOperand(0);
          if (auto *ConstOp1 = dyn_cast<ConstantInt>(I.getOperand(1))) {
            APInt NegVal = -ConstOp1->getValue();
            Value *NegConst = ConstantInt::get(I.getType(), NegVal);
            auto *NewInst = BinaryOperator::CreateAdd(Op0, NegConst, "", &I);
            NewInst->takeName(&I);
            I.replaceAllUsesWith(NewInst);
            I.eraseFromParent();
            Changed = true;
          }
        }
      }
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// ──────────────── Build test module ────────────────
static std::unique_ptr<Module> buildTestModule(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("ch10_verify_demo", Ctx);
  IRBuilder<> Builder(Ctx);

  Type *I32 = Type::getInt32Ty(Ctx);
  FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, "compute", M.get());
  F->getArg(0)->setName("x");
  F->getArg(1)->setName("y");

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  Builder.SetInsertPoint(Entry);

  // %diff = sub i32 %x, 5
  Value *Diff = Builder.CreateSub(F->getArg(0), ConstantInt::get(I32, 5), "diff");
  Builder.CreateRet(Diff);

  return M;
}

int main() {
  LLVMContext Context;
  auto M = buildTestModule(Context);

  // ──────────────── Manual IR verification ────────────────
  outs() << "=== Manual Verification Before Pass ===\n";
  if (verifyModule(*M, &outs())) {
    outs() << "ERROR: Module verification failed before pass!\n";
    return 1;
  }
  outs() << "Module verification passed.\n\n";

  // ──────────────── Manual IR printing ────────────────
  outs() << "=== IR Before Pass (manual print) ===\n";
  M->print(outs(), nullptr);
  outs() << "\n";

  // ──────────────── Set up StandardInstrumentations with verbose printing ────────────────
  PassInstrumentationCallbacks PIC;
  PrintPassOptions PrintOpts;
  PrintOpts.Verbose = true;
  PrintOpts.SkipAnalyses = true;
  PrintOpts.Indent = true;

  StandardInstrumentations SI(Context, /*DebugLogging=*/true,
                              /*VerifyEachPass=*/true, PrintOpts);

  // ──────────────── Set up pass managers with instrumentation ────────────────
  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register the instrumentation analysis
  MAM.registerPass([&] { return PassInstrumentationAnalysis(&PIC); });

  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  SI.registerCallbacks(PIC, &MAM);

  // ──────────────── Create and run pipeline ────────────────
  ModulePassManager MPM;
  FunctionPassManager FPM;
  FPM.addPass(DemoTransformPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  outs() << "=== Running Pass Pipeline (with PrintInstrumentation) ===\n";
  MPM.run(*M, MAM);
  outs() << "\n";

  // ──────────────── Post-pass verification and printing ────────────────
  outs() << "=== IR After Pass (manual print) ===\n";
  M->print(outs(), nullptr);
  outs() << "\n";

  outs() << "=== Manual Verification After Pass ===\n";
  if (verifyModule(*M, &outs())) {
    outs() << "ERROR: Module verification failed after pass!\n";
    return 1;
  }
  outs() << "Module verification passed.\n\n";

  outs() << "=== Summary of debugging techniques demonstrated ===\n";
  outs() << "1. Manual IR printing before/after pass\n";
  outs() << "2. Manual module verification (verifyModule)\n";
  outs() << "3. StandardInstrumentations with VerifyEachPass=true\n";
  outs() << "4. StandardInstrumentations with DebugLogging=true\n";
  outs() << "5. PassInstrumentationCallbacks for pass pipeline introspection\n";

  return 0;
}
