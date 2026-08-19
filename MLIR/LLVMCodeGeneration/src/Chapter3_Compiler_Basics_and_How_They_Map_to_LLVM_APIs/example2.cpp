// example2.cpp - Loading and traversing LLVM IR: Module, Function, BasicBlock, Instruction
// Demonstrates: parseAssemblyString, iteration of all IR levels, CFG traversal
// Shows how to walk the IR hierarchy and extract information at each level.

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace llvm;

// ---------------------------------------------------------------------------
// A sample LLVM IR program embedded as a string for parsing
// ---------------------------------------------------------------------------
static const char *SampleIR = R"(
define i32 @gcd(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %done, label %loop

loop:
  %a_phi = phi i32 [ %a, %entry ], [ %b_phi, %loop ]
  %b_phi = phi i32 [ %b, %entry ], [ %mod, %loop ]
  %mod = srem i32 %a_phi, %b_phi
  %cmp_loop = icmp eq i32 %mod, 0
  br i1 %cmp_loop, label %done, label %loop

done:
  %result = phi i32 [ %a, %entry ], [ %b_phi, %loop ]
  ret i32 %result
}

define i32 @main() {
entry:
  %val = call i32 @gcd(i32 48, i32 18)
  ret i32 %val
}
)";

int main() {
  LLVMContext Context;

  // Parse IR from string
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(SampleIR, Err, Context);
  if (!M) {
    Err.print("example2", errs());
    return 1;
  }
  if (verifyModule(*M, &errs())) {
    errs() << "example2: parsed module is not valid LLVM IR\n";
    return 1;
  }

  outs() << "=== Module: " << M->getName() << " ===\n\n";

  // -------------------------------------------------------------------
  // Level 1: Module inspection
  // -------------------------------------------------------------------
  outs() << "Module contains " << M->size() << " function(s)\n";
  outs() << "Target triple: " << M->getTargetTriple() << "\n\n";

  // -------------------------------------------------------------------
  // Level 2: Function iteration
  // -------------------------------------------------------------------
  for (Function &F : *M) {
    outs() << "--- Function: " << F.getName() << " ---\n";
    outs() << "  Return type: ";
    F.getReturnType()->print(outs());
    outs() << "\n";
    outs() << "  Arguments: " << F.arg_size() << "\n";
    outs() << "  Is declaration: " << (F.isDeclaration() ? "yes" : "no") << "\n";
    outs() << "  Linkage: ";
    switch (F.getLinkage()) {
      case GlobalValue::ExternalLinkage: outs() << "external"; break;
      case GlobalValue::InternalLinkage: outs() << "internal"; break;
      default: outs() << "other"; break;
    }
    outs() << "\n";

    // Print argument names and types
    unsigned ArgIdx = 0;
    for (Argument &Arg : F.args()) {
      outs() << "  Arg[" << ArgIdx++ << "]: " << Arg.getName() << " : ";
      Arg.getType()->print(outs());
      outs() << "\n";
    }

    outs() << "  Basic blocks: " << F.size() << "\n";

    // -----------------------------------------------------------------
    // Level 3: BasicBlock iteration and CFG information
    // -----------------------------------------------------------------
    for (BasicBlock &BB : F) {
      outs() << "\n    BasicBlock: " << BB.getName() << "\n";
      outs() << "      Instruction count: " << BB.size() << "\n";

      // Print predecessors (LLVM IR BB doesn't have direct pred list,
      // we use the pred_iterator from CFG.h)
      outs() << "      Predecessors: ";
      bool First = true;
      for (auto *Pred : predecessors(&BB)) {
        if (!First) outs() << ", ";
        outs() << Pred->getName();
        First = false;
      }
      if (First) outs() << "(none - entry block)";
      outs() << "\n";

      // Print successors (from terminator instruction)
      outs() << "      Successors: ";
      Instruction *Term = BB.getTerminator();
      First = true;
      for (unsigned i = 0; i < Term->getNumSuccessors(); ++i) {
        if (!First) outs() << ", ";
        outs() << Term->getSuccessor(i)->getName();
        First = false;
      }
      outs() << "\n";

      // Check for critical edges
      for (unsigned i = 0; i < Term->getNumSuccessors(); ++i) {
        BasicBlock *Succ = Term->getSuccessor(i);
        if (isCriticalEdge(Term, i)) {
          outs() << "      ** Critical edge: " << BB.getName()
                 << " -> " << Succ->getName() << " **\n";
        }
      }

      // -----------------------------------------------------------------
      // Level 4: Instruction iteration
      // -----------------------------------------------------------------
      for (Instruction &I : BB) {
        outs() << "      ";
        if (&I == BB.getFirstNonPHI() && I.getOpcode() != Instruction::PHI) {
          outs() << "[first non-PHI] ";
        }
        if (&I == BB.getTerminator()) {
          outs() << "[terminator] ";
        }

        // Print opcode name
        outs() << I.getOpcodeName();

        // Print number of operands
        outs() << " (operands: " << I.getNumOperands() << ")";

        // For PHI nodes, print incoming edges
        if (auto *Phi = dyn_cast<PHINode>(&I)) {
          outs() << " incoming: ";
          for (unsigned j = 0; j < Phi->getNumIncomingValues(); ++j) {
            if (j > 0) outs() << ", ";
            outs() << "[";
            Phi->getIncomingValue(j)->printAsOperand(outs(), false);
            outs() << " from " << Phi->getIncomingBlock(j)->getName() << "]";
          }
        }
        outs() << "\n";
      }
    }
    outs() << "\n";
  }

  // -------------------------------------------------------------------
  // Demonstrate RPO traversal
  // -------------------------------------------------------------------
  outs() << "=== RPO Traversal of 'gcd' function ===\n";
  if (Function *GCD = M->getFunction("gcd")) {
    ReversePostOrderTraversal<Function *> RPOT(GCD);
    unsigned Order = 0;
    for (BasicBlock *BB : RPOT) {
      outs() << "  RPO[" << Order++ << "]: " << BB->getName() << "\n";
    }
  }

  return 0;
}
