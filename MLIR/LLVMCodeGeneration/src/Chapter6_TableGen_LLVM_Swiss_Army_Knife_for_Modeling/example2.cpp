// example2.cpp - Basic TableGen Backend
// Demonstrates how to implement a custom TableGen backend that reads
// TableGen records and generates formatted output.
//
// This backend takes a .td file as input and generates a Markdown
// documentation page describing the defined instructions and registers.
//
// Usage: llvm-tblgen example1.td -I/path/to/llvm/include --gen-my-backend
//
// Note: This requires building against LLVM's TableGen libraries.
// The minimal link dependencies are: LLVMTableGen, LLVMSupport.

#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

namespace {

// ---------------------------------------------------------------------------
// Command-line option to specify the output format
// ---------------------------------------------------------------------------
cl::OptionCategory MyBackendCat("My Backend Options");
cl::opt<bool> EmitJSON(
    "emit-json",
    cl::desc("Emit JSON instead of Markdown"),
    cl::init(false),
    cl::cat(MyBackendCat));

// ---------------------------------------------------------------------------
// Generate Markdown documentation from TableGen records
// ---------------------------------------------------------------------------
static bool GenerateMarkdownDocs(raw_ostream &OS, RecordKeeper &Records) {
  OS << "# Generated Instruction Set Documentation\n\n";
  OS << "> Auto-generated from TableGen definitions\n\n";

  // -------------------------------------------------------------------
  // Section 1: Register definitions
  // -------------------------------------------------------------------
  OS << "## Registers\n\n";
  OS << "| Name | Width | Encoding | Aliases |\n";
  OS << "|------|-------|----------|----------|\n";

  auto RegClasses = Records.getAllDerivedDefinitions("Register");
  for (Record *R : RegClasses) {
    std::string Name = R->getValueAsString("Name").str();
    int64_t Width = R->getValueAsInt("BitWidth");
    int64_t Encoding = R->getValueAsInt("Encoding");

    // Get aliases list
    std::string Aliases;
    ListInit *AliasList = R->getValueAsListInit("Aliases");
    for (size_t i = 0; i < AliasList->size(); ++i) {
      if (i > 0) Aliases += ", ";
      Aliases += AliasList->getElementAsString(i);
    }

    OS << "| `" << Name << "` | " << Width << " | 0x"
       << utohexstr(Encoding) << " | " << (Aliases.empty() ? "-" : Aliases)
       << " |\n";
  }

  // -------------------------------------------------------------------
  // Section 2: Instruction definitions
  // -------------------------------------------------------------------
  OS << "\n## Instructions\n\n";

  // Group regular instructions
  auto InstClasses = Records.getAllDerivedDefinitions("Instruction");
  if (!InstClasses.empty()) {
    OS << "### General Instructions\n\n";
    OS << "| Mnemonic | Opcode | Side Effects |\n";
    OS << "|----------|--------|-------------|\n";

    for (Record *R : InstClasses) {
      // Skip records from multiclass base templates (names starting with "")
      if (R->getName().empty() || R->isAnonymous())
        continue;

      std::string Mnemonic = R->getValueAsString("Mnemonic").str();
      int64_t Opcode = R->getValueAsInt("Opcode");
      bool HasSE = R->getValueAsBit("HasSideEffects");

      OS << "| `" << Mnemonic << "` | 0x"
         << utohexstr(Opcode, true) << " | "
         << (HasSE ? "Yes" : "No") << " |\n";
    }
  }

  // -------------------------------------------------------------------
  // Section 3: ALU Instructions (derived from ALUInst class)
  // -------------------------------------------------------------------
  auto ALUClasses = Records.getAllDerivedDefinitions("ALUInst");
  if (!ALUClasses.empty()) {
    OS << "\n### ALU Instructions\n\n";
    for (Record *R : ALUClasses) {
      auto Mnemonic = R->getValueAsString("Mnemonic");
      auto Opcode = R->getValueAsInt("Opcode");
      OS << "- **`" << Mnemonic << "`** (opcode: 0x"
         << utohexstr(Opcode, true) << ")\n";
    }
  }

  // -------------------------------------------------------------------
  // Section 4: Intrinsics
  // -------------------------------------------------------------------
  auto Intrinsics = Records.getAllDerivedDefinitions("Intrinsic");
  if (!Intrinsics.empty()) {
    OS << "\n## Intrinsics\n\n";
    for (Record *R : Intrinsics) {
      auto Name = R->getValueAsString("Name");
      bool HasSE = R->getValueAsBit("HasSideEffects");
      bool IsConv = R->getValueAsBit("IsConvergent");

      OS << "- **`" << Name << "`**";
      if (HasSE) OS << " (side effects)";
      if (IsConv) OS << " (convergent)";

      // Print parameter types
      ListInit *ParamTypes = R->getValueAsListInit("ParamTypes");
      if (ParamTypes && ParamTypes->size() > 0) {
        OS << " - params: ";
        for (size_t i = 0; i < ParamTypes->size(); ++i) {
          if (i > 0) OS << ", ";
          OS << ParamTypes->getElementAsString(i);
        }
      }
      OS << "\n";
    }
  }

  // -------------------------------------------------------------------
  // Section 5: Full Instruction descriptions
  // -------------------------------------------------------------------
  auto FullInsts = Records.getAllDerivedDefinitions("FullInstr");
  if (!FullInsts.empty()) {
    OS << "\n## Complete Instruction Descriptions\n\n";
    for (Record *R : FullInsts) {
      OS << "- **`" << R->getName() << "`**";

      if (auto AsmStr = R->getValueAsString("AsmString");
          !AsmStr.empty()) {
        OS << ": `" << AsmStr << "`";
      }
      OS << "\n";
    }
  }

  // -------------------------------------------------------------------
  // Section 6: Statistics
  // -------------------------------------------------------------------
  OS << "\n## Statistics\n\n";
  OS << "- Total records: " << Records.getDefs().size() << "\n";
  OS << "- Register definitions: " << RegClasses.size() << "\n";
  OS << "- Instruction definitions: " << InstClasses.size() << "\n";
  OS << "- Intrinsics: " << Intrinsics.size() << "\n";

  return false; // false = success in TableGen convention
}

// ---------------------------------------------------------------------------
// Generate JSON output from TableGen records
// ---------------------------------------------------------------------------
static bool GenerateJSON(raw_ostream &OS, RecordKeeper &Records) {
  OS << "{\n";
  OS << "  \"records\": [\n";

  bool First = true;
  for (const auto &Pair : Records.getDefs()) {
    Record *R = Pair.second.get();
    if (R->isAnonymous()) continue;

    if (!First) OS << ",\n";
    First = false;

    OS << "    {\n";
    OS << "      \"name\": \"" << R->getName() << "\",\n";
    OS << "      \"fields\": {\n";

    bool FirstField = true;
    for (const auto &ValPair : R->getValues()) {
      if (!FirstField) OS << ",\n";
      FirstField = false;

      std::string FieldName = ValPair.first.str();
      std::string FieldVal;
      raw_string_ostream FVS(FieldVal);
      ValPair.second->print(FVS);
      FVS.flush();

      OS << "        \"" << FieldName << "\": \"" << FieldVal << "\"";
    }
    OS << "\n      }\n";
    OS << "    }";
  }

  OS << "\n  ]\n";
  OS << "}\n";
  return false;
}

// ---------------------------------------------------------------------------
// Main backend entry point
// ---------------------------------------------------------------------------
bool MyBackendMain(raw_ostream &OS, RecordKeeper &Records) {
  if (EmitJSON) {
    return GenerateJSON(OS, Records);
  }
  return GenerateMarkdownDocs(OS, Records);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Registration with llvm-tblgen
// This allows: llvm-tblgen example.td --gen-my-backend
// ---------------------------------------------------------------------------
// In a real LLVM build, you would register this in TableGen.cpp:
//   RegisterTarget(name, description, function)
//
// For the standalone example, we provide a main() that directly invokes
// the backend given TableGen-parsed records.

// Standalone usage simulation:
// This section demonstrates what the backend does conceptually.
// In practice, you would link this against LLVMTableGen and register
// via the TableGen registration mechanism.

static cl::opt<std::string> InputFilename(
    cl::Positional, cl::desc("<input .td file>"), cl::init("-"));

// ---------------------------------------------------------------------------
// Note: To actually run this as a TableGen backend, you need to:
// 1. Build LLVM with LLVM_BUILD_LLVM_DYLIB or link to required libraries
// 2. Register this backend in the TableGen main or via dynamic plugin
// 3. Run: llvm-tblgen example1.td --gen-my-backend
//
// This main() shows the standalone invocation pattern.
// For a minimal test without full LLVM build, run:
//   llvm-tblgen example1.td -print-records
// to inspect records, then this backend conceptually processes them.
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
      "TableGen Backend Example - Generate ISA Documentation\n");

  outs() << "// This example demonstrates the TableGen backend pattern.\n";
  outs() << "// To see it in action, build against LLVM TableGen libraries:\n";
  outs() << "//   llvm-tblgen example1.td --gen-my-backend\n";
  outs() << "//   llvm-tblgen example1.td -print-records  (built-in)\n";
  outs() << "//   llvm-tblgen example1.td -dump-json        (built-in)\n";
  outs() << "\n";
  outs() << "// The TableGen file 'example1.td' at:\n";
  outs() << "//   src/Chapter6_TableGen_LLVM_Swiss_Army_Knife_for_Modeling/\n";
  outs() << "// contains sample definitions of:\n";
  outs() << "//   - Registers (R0-R2, SP, LR, PC, F0-F1)\n";
  outs() << "//   - Instructions (ADD, SUB, MUL, LOAD, STORE)\n";
  outs() << "//   - Branch instructions (Beq, Bne, Bgt, etc.)\n";
  outs() << "//   - ALU instructions (AND, OR, XOR)\n";
  outs() << "//   - Intrinsics (sqrt, memcpy, sadd.with.overflow)\n";
  outs() << "//   - Full instruction descriptions\n";
  outs() << "\n";
  outs() << "// The backend reads RecordKeeper and iterates over:\n";
  outs() << "// - Records::getAllDerivedDefinitions(\"ClassName\")\n";
  outs() << "// - R->getValueAsString(\"FieldName\")\n";
  outs() << "// - R->getValueAsInt(\"FieldName\")\n";
  outs() << "// - R->getValueAsBit(\"FieldName\")\n";
  outs() << "// - R->getValueAsListInit(\"FieldName\")\n";

  return 0;
}
