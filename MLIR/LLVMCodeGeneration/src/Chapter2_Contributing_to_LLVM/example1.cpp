// example1.cpp - Demonstrates gtest-style unit testing patterns used in LLVM
// This file shows how LLVM's own unit tests are structured using the Google Test
// framework (gtest). It tests basic LLVM IR construction and manipulation APIs.
//
// Compile with: -lgtest -lgtest_main -lLLVMCore -lLLVMSupport

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>
#include <string>

// Test fixture - LLVM provides TestBase classes in the source tree
class LLVMBasicsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Each test gets a fresh LLVMContext
    Context = std::make_unique<llvm::LLVMContext>();
  }

  void TearDown() override {
    Context.reset();
  }

  std::unique_ptr<llvm::LLVMContext> Context;
};

// ------------------------------------------------------------------------
// TEST 1: Verify basic Module creation
// Pattern from unittests/IR/ in the LLVM source tree
// ------------------------------------------------------------------------
TEST_F(LLVMBasicsTest, CreateEmptyModule) {
  llvm::Module M("TestModule", *Context);

  EXPECT_EQ(M.getName(), "TestModule");
  EXPECT_EQ(M.size(), 0u);           // no functions yet
  EXPECT_TRUE(M.empty());            // convenience check
  EXPECT_EQ(M.global_size(), 0u);    // no global variables
}

// ------------------------------------------------------------------------
// TEST 2: Verify Function creation with proper linkage
// Pattern from unittests/IR/FunctionTest.cpp
// ------------------------------------------------------------------------
TEST_F(LLVMBasicsTest, CreateFunction) {
  llvm::Module M("TestModule", *Context);

  // Create: i32 @foo(i32, i32)
  auto *FT = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(*Context),
      {llvm::Type::getInt32Ty(*Context), llvm::Type::getInt32Ty(*Context)},
      false);

  llvm::Function *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, "foo", M);

  // Verify function properties
  EXPECT_EQ(F->getName(), "foo");
  EXPECT_FALSE(F->isDeclaration());   // has no body yet but is not marked decl
  EXPECT_EQ(F->arg_size(), 2u);
  EXPECT_EQ(F->getReturnType(), llvm::Type::getInt32Ty(*Context));
  EXPECT_EQ(F->getParent(), &M);

  // Verify module contains the function
  auto *Found = M.getFunction("foo");
  EXPECT_EQ(Found, F);
  EXPECT_EQ(M.size(), 1u);

  // Non-existent function returns nullptr
  EXPECT_EQ(M.getFunction("nonexistent"), nullptr);
}

// ------------------------------------------------------------------------
// TEST 3: Verify BasicBlock construction and instruction insertion
// Pattern from unittests/IR/BasicBlockTest.cpp
// ------------------------------------------------------------------------
TEST_F(LLVMBasicsTest, CreateBasicBlockAndInstructions) {
  llvm::Module M("TestModule", *Context);

  auto *FT = llvm::FunctionType::get(llvm::Type::getInt32Ty(*Context), {}, false);
  auto *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, "constantFunc", M);

  // Create a single basic block
  auto *BB = llvm::BasicBlock::Create(*Context, "entry", F);

  EXPECT_EQ(BB->getName(), "entry");
  EXPECT_EQ(BB->getParent(), F);
  EXPECT_TRUE(BB->empty());  // no instructions yet

  // Insert a return instruction
  llvm::IRBuilder<> Builder(BB);
  Builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Context), 42));

  // Verify basic block contents
  EXPECT_FALSE(BB->empty());
  EXPECT_EQ(BB->size(), 1u);
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(BB->getTerminator()));

  // getFirstNonPHI returns the first instruction (no PHI nodes here)
  EXPECT_EQ(BB->getFirstNonPHI(), BB->getTerminator());
}

// ------------------------------------------------------------------------
// TEST 4: Verify LLVM IR verifier catches errors
// Pattern from unittests/IR/VerifierTest.cpp
// ------------------------------------------------------------------------
TEST_F(LLVMBasicsTest, VerifierDetectsInvalidIR) {
  llvm::Module M("TestModule", *Context);

  // Create a function without a terminator - this is invalid IR
  auto *FT = llvm::FunctionType::get(llvm::Type::getInt32Ty(*Context), {}, false);
  auto *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, "badFunc", M);

  // Basic block with NO terminator
  auto *BB = llvm::BasicBlock::Create(*Context, "entry", F);

  // Insert a non-terminator instruction but no ret
  llvm::IRBuilder<> Builder(BB);
  Builder.CreateAdd(
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Context), 1),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Context), 2),
      "tmp");

  // The verifier should detect this is invalid
  std::string ErrorMsg;
  llvm::raw_string_ostream ErrorStream(ErrorMsg);
  bool HasError = llvm::verifyModule(M, &ErrorStream);
  EXPECT_TRUE(HasError);
  EXPECT_FALSE(ErrorMsg.empty());

  // Fix it by adding a terminator
  Builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(*Context), 0));

  ErrorMsg.clear();
  HasError = llvm::verifyModule(M, &ErrorStream);
  EXPECT_FALSE(HasError);
}

// ------------------------------------------------------------------------
// TEST 5: Verify use-def chain correctness
// Pattern from unittests/IR/UseTest.cpp
// ------------------------------------------------------------------------
TEST_F(LLVMBasicsTest, UseDefChains) {
  llvm::Module M("TestModule", *Context);

  auto *FT = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(*Context),
      {llvm::Type::getInt32Ty(*Context), llvm::Type::getInt32Ty(*Context)},
      false);
  auto *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, "add", M);

  auto *BB = llvm::BasicBlock::Create(*Context, "entry", F);
  llvm::IRBuilder<> Builder(BB);

  auto Args = F->args().begin();
  llvm::Value *A = Args++;
  llvm::Value *B = Args;
  llvm::Value *Result = Builder.CreateAdd(A, B, "result");
  Builder.CreateRet(Result);

  // Test: Result uses both A and B
  auto *AddInst = llvm::cast<llvm::Instruction>(Result);

  // Check operand count
  EXPECT_EQ(AddInst->getNumOperands(), 2u);

  // Check specific operands
  EXPECT_EQ(AddInst->getOperand(0), A);
  EXPECT_EQ(AddInst->getOperand(1), B);

  // Check use count: A and B are each used once
  EXPECT_EQ(A->getNumUses(), 1u);
  EXPECT_EQ(B->getNumUses(), 1u);

  // Verify user iteration
  unsigned UserCount = 0;
  for (auto *U : A->users()) {
    (void)U;
    UserCount++;
  }
  EXPECT_EQ(UserCount, 1u);
}

// ------------------------------------------------------------------------
// Main entry point for gtest
// ------------------------------------------------------------------------
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
