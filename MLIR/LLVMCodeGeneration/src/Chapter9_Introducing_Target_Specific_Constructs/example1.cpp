// example1.cpp - Chapter 9: Target Registration and TargetMachine Setup
//
// Demonstrates:
//   - How targets are registered via the TargetRegistry mechanism
//   - Listing all registered targets using TargetRegistry::printRegisteredTargetsForVersion
//   - Looking up a specific target by Triple
//   - Creating a TargetMachine for a given triple
//   - Inspecting TargetMachine properties (data layout, triple, relocation model)
//   - Basic TargetTransformInfo usage through TargetMachine
//
// Build with the repository baseline, LLVM 20.1.x:
//   clang++ example1.cpp $(llvm-config --cxxflags --ldflags --libs all-targets core) -o example1

#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

int main() {
  // ──────────────── Initialize all registered targets ────────────────
  // This calls LLVMInitializeAllTargetInfos, LLVMInitializeAllTargets, etc.
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();

  outs() << "=== Chapter 9: Target Registration Demo ===\n\n";

  // ──────────────── List all registered targets ────────────────
  outs() << "=== Registered Targets ===\n";
  for (const Target &T : TargetRegistry::targets()) {
    outs() << "  Target: " << T.getName()
           << " - " << T.getShortDescription() << "\n";
  }
  outs() << "\n";

  // ──────────────── Create a Triple to look up a specific target ────────────────
  // Use the host triple or fall back to x86_64
  std::string HostTripleStr = sys::getDefaultTargetTriple();
  outs() << "Host Triple: " << HostTripleStr << "\n";

  Triple TheTriple(HostTripleStr);
  std::string ErrorMsg;

  const Target *TheTarget = TargetRegistry::lookupTarget(
      TheTriple.getArchName(), TheTriple, ErrorMsg);

  if (!TheTarget) {
    // Fallback: try x86_64-unknown-linux-gnu
    outs() << "Host target not found, trying x86_64...\n";
    Triple FallbackTriple("x86_64-unknown-linux-gnu");
    TheTarget = TargetRegistry::lookupTarget(
        FallbackTriple.getArchName(), FallbackTriple, ErrorMsg);
  }

  if (!TheTarget) {
    outs() << "Error: Could not find any target. " << ErrorMsg << "\n";
    return 1;
  }

  outs() << "\nFound target: " << TheTarget->getName()
         << " - " << TheTarget->getShortDescription() << "\n\n";

  // ──────────────── Create a TargetMachine ────────────────
  TargetOptions Options;
  // Default relocation model: static
  std::optional<Reloc::Model> RM = Reloc::Static;
  // Default code model: small
  std::optional<CodeModel::Model> CM = CodeModel::Small;
  // Optimization level
  CodeGenOptLevel OptLevel = CodeGenOptLevel::Default;

  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TheTriple.str(),          // Target triple string
      "generic",                // CPU name
      "",                       // Feature string
      Options,
      RM,
      CM,
      OptLevel,
      /*JIT=*/false));

  if (!TM) {
    outs() << "Error: Failed to create TargetMachine.\n";
    return 1;
  }

  // ──────────────── Inspect TargetMachine properties ────────────────
  outs() << "=== TargetMachine Properties ===\n";
  outs() << "Target Triple:     " << TM->getTargetTriple().str() << "\n";
  outs() << "Data Layout:       " << TM->createDataLayout().getStringRepresentation() << "\n";
  outs() << "CPU:               " << TM->getTargetCPU() << "\n";
  outs() << "Feature String:    " << TM->getTargetFeatureString() << "\n";
  outs() << "Relocation Model:  "
         << (TM->getRelocationModel() == Reloc::Static ? "Static" :
             TM->getRelocationModel() == Reloc::PIC_ ? "PIC" :
             TM->getRelocationModel() == Reloc::DynamicNoPIC ? "DynamicNoPIC" :
             TM->getRelocationModel() == Reloc::ROPI ? "ROPI" :
             TM->getRelocationModel() == Reloc::RWPI ? "RWPI" :
             TM->getRelocationModel() == Reloc::ROPI_RWPI ? "ROPI_RWPI" : "Unknown") << "\n";
  outs() << "Code Model:        "
         << (TM->getCodeModel() == CodeModel::Tiny ? "Tiny" :
             TM->getCodeModel() == CodeModel::Small ? "Small" :
             TM->getCodeModel() == CodeModel::Kernel ? "Kernel" :
             TM->getCodeModel() == CodeModel::Medium ? "Medium" :
             TM->getCodeModel() == CodeModel::Large ? "Large" : "Unknown") << "\n";
  outs() << "Opt Level:         "
         << (TM->getOptLevel() == CodeGenOptLevel::None ? "None" :
             TM->getOptLevel() == CodeGenOptLevel::Less ? "Less" :
             TM->getOptLevel() == CodeGenOptLevel::Default ? "Default" :
             TM->getOptLevel() == CodeGenOptLevel::Aggressive ? "Aggressive" : "Unknown") << "\n\n";

  // ──────────────── Examine Triple details ────────────────
  outs() << "=== Triple Breakdown ===\n";
  outs() << "Architecture:     " << Triple::getArchTypeName(TheTriple.getArch()) << "\n";
  outs() << "Vendor:           " << Triple::getVendorTypeName(TheTriple.getVendor()) << "\n";
  outs() << "OS:               " << Triple::getOSTypeName(TheTriple.getOS()) << "\n";
  outs() << "Environment:      " << Triple::getEnvironmentTypeName(TheTriple.getEnvironment()) << "\n";
  outs() << "Object Format:    "
         << (TheTriple.getObjectFormat() == Triple::ELF ? "ELF" :
             TheTriple.getObjectFormat() == Triple::COFF ? "COFF" :
             TheTriple.getObjectFormat() == Triple::MachO ? "MachO" :
             TheTriple.getObjectFormat() == Triple::Wasm ? "Wasm" :
             TheTriple.getObjectFormat() == Triple::XCOFF ? "XCOFF" :
             TheTriple.getObjectFormat() == Triple::GOFF ? "GOFF" :
             TheTriple.getObjectFormat() == Triple::SPIRV ? "SPIRV" : "Unknown") << "\n";
  outs() << "Is little-endian: " << (TheTriple.isLittleEndian() ? "yes" : "no") << "\n";
  outs() << "Arch pointer bit width: " << TheTriple.getArchPointerBitWidth() << "\n";

  return 0;
}
