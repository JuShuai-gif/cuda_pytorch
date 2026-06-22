// example1.cpp - Simple Constant Propagation Pass
// Implements a basic constant propagation optimization at the LLVM IR level.
//
// Algorithm:
// 1. Iterate through all instructions in a function
// 2. For each instruction, check if all operands are constant
// 3. If so, compute the result at compile time using APInt
// 4. Replace the instruction with the computed constant
//
// This demonstrates: SSA traversal, use-def chains, APInt arithmetic,
// replaceAllUsesWith, and LLVM's RTTI (isa/dyn_cast).

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// ---------------------------------------------------------------------------
// Simple constant propagation: if all operands of an instruction are
// constant integers, compute the result at compile time and replace.
//
// Returns true if the IR was modified, false otherwise.
// ---------------------------------------------------------------------------
bool myConstantPropagation(Function &F) {
  bool Changed = false;

  // We iterate over basic blocks, then instructions.
  // Use a worklist approach: collect instructions to replace, then process.
  SmallVector<std::pair<Instruction *, APInt>, 8> Replacements;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Skip terminators (br, ret, switch, etc.) - not handled in this pass
      if (I.isTerminator())
        continue;

      // Skip phi nodes - they need special handling
      if (isa<PHINode>(I))
        continue;

      // Only handle binary operations and icmp for this example
      // These are the most common constant-foldable instructions
      APInt Result;
      bool Foldable = false;

      // Collect constant operands
      SmallVector<APInt, 2> ConstOps;
      for (unsigned i = 0; i < I.getNumOperands(); ++i) {
        if (auto *CI = dyn_cast<ConstantInt>(I.getOperand(i))) {
          ConstOps.push_back(CI->getValue());
        } else {
          // Non-constant operand - skip this instruction
          break;
        }
      }

      // All operands must be constant
      if (ConstOps.size() != I.getNumOperands())
        continue;

      // Perform constant folding based on opcode
      unsigned Opcode = I.getOpcode();
      switch (Opcode) {
        // Arithmetic operations
        case Instruction::Add:
          Result = ConstOps[0] + ConstOps[1];
          Foldable = true;
          break;
        case Instruction::Sub:
          Result = ConstOps[0] - ConstOps[1];
          Foldable = true;
          break;
        case Instruction::Mul:
          Result = ConstOps[0] * ConstOps[1];
          Foldable = true;
          break;
        case Instruction::SDiv:
          if (!ConstOps[1].isZero()) {
            Result = ConstOps[0].sdiv(ConstOps[1]);
            Foldable = true;
          }
          break;
        case Instruction::UDiv:
          if (!ConstOps[1].isZero()) {
            Result = ConstOps[0].udiv(ConstOps[1]);
            Foldable = true;
          }
          break;
        case Instruction::SRem:
          if (!ConstOps[1].isZero()) {
            Result = ConstOps[0].srem(ConstOps[1]);
            Foldable = true;
          }
          break;
        case Instruction::URem:
          if (!ConstOps[1].isZero()) {
            Result = ConstOps[0].urem(ConstOps[1]);
            Foldable = true;
          }
          break;
        // Bitwise operations
        case Instruction::And:
          Result = ConstOps[0] & ConstOps[1];
          Foldable = true;
          break;
        case Instruction::Or:
          Result = ConstOps[0] | ConstOps[1];
          Foldable = true;
          break;
        case Instruction::Xor:
          Result = ConstOps[0] ^ ConstOps[1];
          Foldable = true;
          break;
        case Instruction::Shl:
          Result = ConstOps[0].shl(ConstOps[1]);
          Foldable = true;
          break;
        case Instruction::LShr:
          Result = ConstOps[0].lshr(ConstOps[1]);
          Foldable = true;
          break;
        case Instruction::AShr:
          Result = ConstOps[0].ashr(ConstOps[1]);
          Foldable = true;
          break;
        // Integer comparisons
        case Instruction::ICmp: {
          auto *ICmpInst = cast<ICmpInst>(&I);
          bool CmpResult = false;
          switch (ICmpInst->getPredicate()) {
            case ICmpInst::ICMP_EQ:  CmpResult = ConstOps[0].eq(ConstOps[1]); break;
            case ICmpInst::ICMP_NE:  CmpResult = ConstOps[0].ne(ConstOps[1]); break;
            case ICmpInst::ICMP_UGT: CmpResult = ConstOps[0].ugt(ConstOps[1]); break;
            case ICmpInst::ICMP_UGE: CmpResult = ConstOps[0].uge(ConstOps[1]); break;
            case ICmpInst::ICMP_ULT: CmpResult = ConstOps[0].ult(ConstOps[1]); break;
            case ICmpInst::ICMP_ULE: CmpResult = ConstOps[0].ule(ConstOps[1]); break;
            case ICmpInst::ICMP_SGT: CmpResult = ConstOps[0].sgt(ConstOps[1]); break;
            case ICmpInst::ICMP_SGE: CmpResult = ConstOps[0].sge(ConstOps[1]); break;
            case ICmpInst::ICMP_SLT: CmpResult = ConstOps[0].slt(ConstOps[1]); break;
            case ICmpInst::ICMP_SLE: CmpResult = ConstOps[0].sle(ConstOps[1]); break;
            default: break;
          }
          // i1 result for comparisons
          Result = APInt(1, CmpResult ? 1 : 0);
          Foldable = true;
          break;
        }
        default:
          // Unhandled opcode - skip
          break;
      }

      if (Foldable) {
        // Create the new constant and replace the instruction
        LLVMContext &Ctx = F.getContext();
        Type *I32 = Type::getInt32Ty(Ctx);
        Type *I1  = Type::getInt1Ty(Ctx);

        // Get the appropriate type for the result
        Type *ResTy = I.getType();
        ConstantInt *NewConst = nullptr;

        if (ResTy->isIntegerTy(1)) {
          NewConst = ConstantInt::get(I1, Result);
        } else {
          // Truncate or extend APInt to match the result bitwidth
          unsigned BitWidth = ResTy->getIntegerBitWidth();
          APInt AdjustedResult = Result.sextOrTrunc(BitWidth);
          NewConst = ConstantInt::get(Ctx, AdjustedResult);
        }

        // Replace all uses of the instruction with the new constant
        I.replaceAllUsesWith(NewConst);

        // Mark for deletion
        Replacements.push_back({&I, Result});
        Changed = true;
      }
    }
  }

  // Erase the replaced instructions after finishing iteration
  // (We must not modify the instruction list while iterating over it)
  for (auto &Pair : Replacements) {
    Pair.first->eraseFromParent();
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

  // Function 1: Simple arithmetic with constants
  // int f1(int x) { return (2 + 3) * x; }
  // After constant prop: return 5 * x;
  {
    FunctionType *FT = FunctionType::get(I32, {I32}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f1_arith", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<> B(BB);

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
    IRBuilder<> B(BB);

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
    IRBuilder<> B(BB);

    Value *C5 = ConstantInt::get(I32, 5);
    Value *C3 = ConstantInt::get(I32, 3);
    Value *Cmp = B.CreateICmpSGT(C5, C3, "cmp");
    // ZExt to i32 for return
    Value *Ext = B.CreateZExt(Cmp, I32, "ext");
    B.CreateRet(Ext);
  }

  // Function 4: Bitwise operations on constants
  // int f4() { return (0xFF & 0x0F) | 0x10; }
  // After constant prop: return 0x1F (31);
  {
    FunctionType *FT = FunctionType::get(I32, {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "f4_bitwise", *M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
    IRBuilder<> B(BB);

    Value *FF = ConstantInt::get(I32, 0xFF);
    Value *OF = ConstantInt::get(I32, 0x0F);
    Value *TEN = ConstantInt::get(I32, 0x10);
    Value *And1 = B.CreateAnd(FF, OF, "and");     // 0xFF & 0x0F = 0x0F
    Value *Or1  = B.CreateOr(And1, TEN, "or");     // 0x0F | 0x10 = 0x1F
    B.CreateRet(Or1);
  }

  return M;
}

// ---------------------------------------------------------------------------
int main() {
  LLVMContext Context;

  outs() << "=== Before Constant Propagation ===\n";
  auto M = buildTestModule(Context);
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
