// example2.cpp - Def-Use Chains, Use-Def Chains, and Dominance Analysis
// Demonstrates: Value::useXXX/userXXX, User::getOperand, use-def traversal,
// dominance queries, and the relationship between dominance and SSA.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Build a test module with functions that demonstrate SSA and dominance
// ---------------------------------------------------------------------------
std::unique_ptr<Module> buildTestModule(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("DefUseDemo", Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);

  // Function with clear dominance relationships
  // int dom_example(int a, int b) {
  //   int x = a + b;          // x defined in entry, dominates all uses
  //   if (a > 0) {
  //     x = x * 2;            // redefined x in then block
  //   }
  //   return x;               // phi needed: x from entry or x from then
  // }
  {
    FunctionType *FT = FunctionType::get(I32, {I32, I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage,
                                   "dom_example", *M);
    BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", F);
    BasicBlock *ThenBB  = BasicBlock::Create(Ctx, "then", F);
    BasicBlock *MergeBB = BasicBlock::Create(Ctx, "merge", F);

    auto Args = F->args().begin();
    Value *A = Args++;
    Value *B = Args;

    IRBuilder<> B(EntryBB);
    Value *X = B.CreateAdd(A, B, "x");     // x = a + b
    Value *Cmp = B.CreateICmpSGT(A, ConstantInt::get(I32, 0), "cmp");
    B.CreateCondBr(Cmp, ThenBB, MergeBB);

    B.SetInsertPoint(ThenBB);
    Value *X2 = B.CreateMul(X, ConstantInt::get(I32, 2), "x2"); // x = x * 2
    B.CreateBr(MergeBB);

    B.SetInsertPoint(MergeBB);
    PHINode *Phi = B.CreatePHI(I32, 2, "x_phi");
    Phi->addIncoming(X, EntryBB);
    Phi->addIncoming(X2, ThenBB);
    B.CreateRet(Phi);
  }

  // Function demonstrating cross-function use-def chain issue
  // @global_var = global i32 42
  // int use_global() { return @global_var; }
  // int also_uses_global() { return @global_var + 1; }
  {
    GlobalVariable *GV = new GlobalVariable(
        *M, I32, false, GlobalValue::InternalLinkage,
        ConstantInt::get(I32, 42), "global_var");

    // First user function
    {
      FunctionType *FT = FunctionType::get(I32, {}, false);
      Function *F = Function::Create(FT, Function::ExternalLinkage,
                                     "use_global", *M);
      BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
      IRBuilder<> B(BB);
      Value *Load = B.CreateLoad(I32, GV, "val");
      B.CreateRet(Load);
    }

    // Second user function
    {
      FunctionType *FT = FunctionType::get(I32, {}, false);
      Function *F = Function::Create(FT, Function::ExternalLinkage,
                                     "also_uses_global", *M);
      BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
      IRBuilder<> B(BB);
      Value *Load = B.CreateLoad(I32, GV, "val");
      Value *Add  = B.CreateAdd(Load, ConstantInt::get(I32, 1), "result");
      B.CreateRet(Add);
    }
  }

  return M;
}

// ---------------------------------------------------------------------------
// Analyze def-use chains for all values in a function
// ---------------------------------------------------------------------------
void analyzeDefUseChains(Function &F) {
  outs() << "  === Def-Use Chain Analysis: " << F.getName() << " ===\n";

  // Walk all instructions and analyze their uses
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Skip terminators for cleaner output
      if (I.isTerminator() && !isa<ReturnInst>(I)) continue;

      outs() << "  Value defined by: ";
      I.printAsOperand(outs(), false);
      outs() << " (" << I.getOpcodeName() << ")\n";

      // Count uses and users
      unsigned UseCount = I.getNumUses();
      outs() << "    Use count: " << UseCount << "\n";

      if (UseCount > 0) {
        outs() << "    Users:\n";
        // Walk all uses using the Value::users() iterator
        for (User *U : I.users()) {
          outs() << "      User: ";
          if (auto *UserI = dyn_cast<Instruction>(U)) {
            outs() << UserI->getOpcodeName();
            outs() << " in " << UserI->getParent()->getParent()->getName();
            outs() << "::" << UserI->getParent()->getName() << "\n";

            // Show which operand position this value occupies
            for (unsigned OpIdx = 0; OpIdx < UserI->getNumOperands(); ++OpIdx) {
              if (UserI->getOperand(OpIdx) == &I) {
                outs() << "        (operand #" << OpIdx << ")\n";
              }
            }
          } else {
            // Non-instruction user (e.g., ConstantExpr, GlobalVariable init)
            outs() << "non-instruction user\n";
          }
        }
      }

      // Show use-def chain: for each operand, find its definition
      outs() << "    Operands:\n";
      for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
        Value *Op = I.getOperand(OpIdx);
        outs() << "      Op[" << OpIdx << "]: ";
        Op->printAsOperand(outs(), false);

        if (auto *OpI = dyn_cast<Instruction>(Op)) {
          outs() << " (defined in " << OpI->getParent()->getName() << ")";
        } else if (isa<Constant>(Op)) {
          outs() << " (constant)";
        } else if (isa<Argument>(Op)) {
          outs() << " (argument)";
        }
        outs() << "\n";
      }
      outs() << "\n";
    }
  }
}

// ---------------------------------------------------------------------------
// Analyze dominance relationships for all blocks in a function
// ---------------------------------------------------------------------------
void analyzeDominance(Function &F) {
  outs() << "  === Dominance Analysis: " << F.getName() << " ===\n";

  DominatorTree DT(F);

  outs() << "  Dominator Tree structure:\n";
  // Print the dominator tree
  for (BasicBlock &BB : F) {
    DomTreeNodeBase<BasicBlock> *Node = DT.getNode(&BB);
    if (!Node) continue;

    // Print the immediate dominator (idom)
    DomTreeNodeBase<BasicBlock> *IDomNode = Node->getIDom();
    outs() << "    " << BB.getName();
    if (IDomNode) {
      outs() << " -> idom: " << IDomNode->getBlock()->getName();
    } else {
      outs() << " -> idom: (none - entry block)";
    }
    outs() << "\n";

    // Print which blocks this block dominates
    outs() << "      dominates: ";
    bool First = true;
    for (BasicBlock &OtherBB : F) {
      if (&BB != &OtherBB && DT.dominates(&BB, &OtherBB)) {
        if (!First) outs() << ", ";
        outs() << OtherBB.getName();
        First = false;
      }
    }
    if (First) outs() << "(none)";
    outs() << "\n";
  }

  // Verify SSA dominance property: each definition must dominate its uses
  outs() << "\n  SSA Dominance Verification:\n";
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
        Value *Op = I.getOperand(OpIdx);
        if (auto *OpI = dyn_cast<Instruction>(Op)) {
          BasicBlock *DefBB = OpI->getParent();
          BasicBlock *UseBB = I.getParent();
          if (DT.dominates(DefBB, UseBB)) {
            // OK: definition dominates use
          } else if (DT.dominates(UseBB, DefBB)) {
            // Use dominates definition - this is normal for phi nodes
            // (phi in merge block references values from predecessors)
          } else {
            outs() << "    WARNING: Definition of ";
            OpI->printAsOperand(outs(), false);
            outs() << " in " << DefBB->getName();
            outs() << " does not dominate use in " << UseBB->getName() << "\n";
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Demonstrate cross-function use-def traversal issue (Chapter 4 Table 4.1)
// Walking uses of a global value will cross function boundaries.
// ---------------------------------------------------------------------------
void demonstrateCrossFunctionUseDef(Module &M) {
  outs() << "\n  === Cross-Function Use-Def Chain Demo ===\n";

  GlobalVariable *GV = M.getGlobalVariable("global_var");
  if (!GV) return;

  outs() << "  Global variable: " << GV->getName() << "\n";
  outs() << "  Walking all users of this global...\n";

  for (User *U : GV->users()) {
    if (auto *UserI = dyn_cast<Instruction>(U)) {
      Function *UserFunc = UserI->getParent()->getParent();
      outs() << "    Found use in function: " << UserFunc->getName();
      outs() << ", block: " << UserI->getParent()->getName();
      outs() << ", instruction: " << UserI->getOpcodeName() << "\n";
    } else {
      outs() << "    Found non-instruction user\n";
    }
  }

  outs() << "  TAKEAWAY: Use-def chain traversal can cross function boundaries!\n";
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;
  auto M = buildTestModule(Context);

  outs() << "; Test Module IR:\n";
  M->print(outs(), nullptr);
  outs() << "\n";

  // Analyze each function
  for (Function &F : *M) {
    if (F.isDeclaration()) continue;

    analyzeDefUseChains(F);
    outs() << "\n";
    analyzeDominance(F);
    outs() << "\n";
  }

  demonstrateCrossFunctionUseDef(*M);

  return 0;
}
