// example1.cpp - Demonstrates LLVM build system integration and basic IR creation
// This program shows how to set up a standalone project that links against LLVM
// libraries, creates an LLVM Module with IR constructs, and prints it.
//
// Build with CMake (example CMakeLists.txt included):

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"

using namespace llvm;

int main(int argc, char **argv) {
  // Create an LLVMContext - required for uniquing types and constants
  LLVMContext Context;

  // Method 1: Create a Module from scratch
  // The Module is the top-level container for all IR in a translation unit
  Module FromScratch("FromScratchModule", Context);

  // Create a simple function: int add(int a, int b) { return a + b; }
  FunctionType *FuncType = FunctionType::get(
      Type::getInt32Ty(Context),                          // return type
      {Type::getInt32Ty(Context), Type::getInt32Ty(Context)}, // params
      false                                               // not vararg
  );

  Function *AddFunc = Function::Create(
      FuncType,
      Function::ExternalLinkage,  // linkage type
      "add",                      // function name
      FromScratch                 // parent module
  );

  // Create a basic block and set the insertion point
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", AddFunc);
  IRBuilder<> Builder(EntryBB);

  // Get function arguments
  auto Args = AddFunc->args().begin();
  Value *ArgA = Args++;
  Value *ArgB = Args;

  // Build: %result = add i32 %a, %b
  Value *Result = Builder.CreateAdd(ArgA, ArgB, "result");
  // Build: ret i32 %result
  Builder.CreateRet(Result);

  // Print the module to stdout
  outs() << "=== Module created from scratch ===\n";
  FromScratch.print(outs(), nullptr);

  // Method 2: Load a Module from a string containing LLVM IR assembly
  // This demonstrates the AsmParser library capability
  const char *IRString = R"(
    define i32 @mul(i32 %a, i32 %b) {
    entry:
      %result = mul i32 %a, %b
      ret i32 %result
    }
  )";

  SMDiagnostic Err;
  std::unique_ptr<Module> ParsedMod = parseAssemblyString(IRString, Err, Context);
  if (!ParsedMod) {
    errs() << "Failed to parse IR: ";
    Err.print(argv[0], errs());
    return 1;
  }

  outs() << "\n=== Module parsed from IR string ===\n";
  ParsedMod->print(outs(), nullptr);

  // Demonstrate module inspection APIs
  outs() << "\n=== Module inspection ===\n";
  outs() << "Module name: " << ParsedMod->getName() << "\n";
  outs() << "Number of functions: " << ParsedMod->size() << "\n";

  // Iterate through functions using range-based for loop
  for (Function &F : *ParsedMod) {
    outs() << "  Function: " << F.getName() << "\n";
    outs() << "    Return type: ";
    F.getReturnType()->print(outs());
    outs() << "\n";
    outs() << "    Is declaration: " << (F.isDeclaration() ? "yes" : "no") << "\n";
    outs() << "    Argument count: " << F.arg_size() << "\n";

    // Iterate through basic blocks
    for (BasicBlock &BB : F) {
      outs() << "    BasicBlock: " << BB.getName() << "\n";
      outs() << "      Instruction count: " << BB.size() << "\n";
      // Demonstrate getTerminator and getFirstNonPHI
      outs() << "      First non-PHI: ";
      BB.getFirstNonPHI()->print(outs());
      outs() << "\n";
      outs() << "      Terminator: ";
      BB.getTerminator()->print(outs());
      outs() << "\n";
    }
  }

  return 0;
}
