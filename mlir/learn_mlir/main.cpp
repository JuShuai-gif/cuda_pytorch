#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main(int argc, char **argv) {
    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect>();
    auto src = parseSourceFile<ModuleOp>(argv[1], &ctx);
    src->print(llvm::outs());
    src->dump();
    return 0;
}