// example3.cpp - Simulating TableGen Record Processing in C++
// This demonstrates the conceptual flow of how TableGen records are
// processed without requiring the full LLVM TableGen library link.
//
// It builds a lightweight in-memory representation of TableGen records
// and shows how a backend would iterate over them to generate output.
// This mirrors the actual Record/RecordKeeper API from LLVM's TableGen.

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <memory>
#include <cassert>

// ---------------------------------------------------------------------------
// Minimal TableGen-like record system for demonstration
// ---------------------------------------------------------------------------

// Represents a value in a TableGen field
enum class ValueKind { Int, String, Bit, ListOfString };

struct FieldValue {
  ValueKind Kind;
  int64_t IntVal = 0;
  std::string StrVal;
  bool BitVal = false;
  std::vector<std::string> ListVal;

  // Factory methods
  static FieldValue makeInt(int64_t v) {
    FieldValue fv; fv.Kind = ValueKind::Int; fv.IntVal = v; return fv;
  }
  static FieldValue makeString(const std::string &s) {
    FieldValue fv; fv.Kind = ValueKind::String; fv.StrVal = s; return fv;
  }
  static FieldValue makeBit(bool b) {
    FieldValue fv; fv.Kind = ValueKind::Bit; fv.BitVal = b; return fv;
  }
  static FieldValue makeList(std::vector<std::string> l) {
    FieldValue fv; fv.Kind = ValueKind::ListOfString; fv.ListVal = std::move(l); return fv;
  }
};

// Represents a TableGen Record
class Record {
public:
  std::string Name;
  std::string ParentClass;
  std::map<std::string, FieldValue> Fields;

  Record(std::string name, std::string parent)
      : Name(std::move(name)), ParentClass(std::move(parent)) {}

  std::string getValueAsString(const std::string &field) const {
    auto it = Fields.find(field);
    if (it != Fields.end() && it->second.Kind == ValueKind::String)
      return it->second.StrVal;
    return "";
  }
  int64_t getValueAsInt(const std::string &field) const {
    auto it = Fields.find(field);
    if (it != Fields.end() && it->second.Kind == ValueKind::Int)
      return it->second.IntVal;
    return 0;
  }
  bool getValueAsBit(const std::string &field) const {
    auto it = Fields.find(field);
    if (it != Fields.end() && it->second.Kind == ValueKind::Bit)
      return it->second.BitVal;
    return false;
  }
};

// Represents the RecordKeeper that holds all definitions
class RecordKeeper {
public:
  std::vector<std::unique_ptr<Record>> Defs;

  // Get all records that inherit from a given class
  std::vector<Record *> getAllDerivedDefinitions(const std::string &className) {
    std::vector<Record *> result;
    for (auto &def : Defs) {
      if (def->ParentClass == className)
        result.push_back(def.get());
    }
    return result;
  }
};

// ---------------------------------------------------------------------------
// Emulate the .td file definitions from example1.td in C++ records
// ---------------------------------------------------------------------------
void populateRecords(RecordKeeper &RK) {
  // Registers (class: "Register")
  auto makeReg = [&](const char *name, int width, int enc,
                     std::vector<std::string> aliases = {}) {
    auto R = std::make_unique<Record>(name, "Register");
    R->Fields["Name"]     = FieldValue::makeString(name);
    R->Fields["BitWidth"] = FieldValue::makeInt(width);
    R->Fields["Encoding"] = FieldValue::makeInt(enc);
    R->Fields["Aliases"]  = FieldValue::makeList(std::move(aliases));
    RK.Defs.push_back(std::move(R));
  };
  makeReg("R0", 32, 0x0000);
  makeReg("R1", 32, 0x0001);
  makeReg("R2", 32, 0x0002);
  makeReg("SP", 32, 0x000D);
  makeReg("LR", 32, 0x000E, {"r14"});
  makeReg("PC", 32, 0x000F, {"r15"});

  // Instructions (class: "Instruction")
  auto makeInst = [&](const char *name, const char *mnem, int opc, bool se) {
    auto I = std::make_unique<Record>(name, "Instruction");
    I->Fields["Mnemonic"]       = FieldValue::makeString(mnem);
    I->Fields["Opcode"]         = FieldValue::makeInt(opc);
    I->Fields["HasSideEffects"] = FieldValue::makeBit(se);
    I->Fields["IsTwoAddress"]   = FieldValue::makeBit(false);
    RK.Defs.push_back(std::move(I));
  };
  makeInst("ADD_rr",  "add",  0x01, false);
  makeInst("ADD_ri",  "add",  0x01, false);
  makeInst("SUB_rr",  "sub",  0x02, false);
  makeInst("SUB_ri",  "sub",  0x02, false);
  makeInst("MUL_rr",  "mul",  0x03, false);
  makeInst("MUL_ri",  "mul",  0x03, false);
  makeInst("LOAD",    "ldr",  0x10, true);
  makeInst("STORE",   "str",  0x11, true);

  // Branch instructions (generated via foreach in TableGen)
  const char *bconds[] = {"eq", "ne", "gt", "lt", "ge", "le"};
  for (const char *cond : bconds) {
    std::string name = std::string("B") + cond;
    auto I = std::make_unique<Record>(name, "Instruction");
    I->Fields["Mnemonic"]       = FieldValue::makeString("b" + std::string(cond));
    I->Fields["Opcode"]         = FieldValue::makeInt(0x20);
    I->Fields["HasSideEffects"] = FieldValue::makeBit(false);
    RK.Defs.push_back(std::move(I));
  }

  // Intrinsics (class: "Intrinsic")
  auto makeIntrinsic = [&](const char *name, std::vector<int> params,
                           bool se, bool conv) {
    auto I = std::make_unique<Record>(name, "Intrinsic");
    I->Fields["Name"]          = FieldValue::makeString(name);
    I->Fields["HasSideEffects"] = FieldValue::makeBit(se);
    I->Fields["IsConvergent"]  = FieldValue::makeBit(conv);

    std::vector<std::string> paramStrs;
    for (int p : params) paramStrs.push_back(std::to_string(p));
    I->Fields["ParamTypes"] = FieldValue::makeList(paramStrs);
    RK.Defs.push_back(std::move(I));
  };
  makeIntrinsic("llvm.sqrt", {32}, false, true);
  makeIntrinsic("llvm.memcpy", {32, 32, 32}, true, false);
  makeIntrinsic("llvm.sadd.with.overflow", {32, 32}, false, false);
}

// ---------------------------------------------------------------------------
// Backend that generates documentation from records
// (mirrors the TableGen backend pattern)
// ---------------------------------------------------------------------------
void generateDocumentation(const RecordKeeper &RK) {
  std::cout << "# Generated Instruction Set Documentation\n\n";
  std::cout << "> Auto-generated from in-memory TableGen records\n\n";

  // Registers section
  auto Regs = RK.getAllDerivedDefinitions("Register");
  std::cout << "## Registers\n\n";
  std::cout << "| Name | Width | Encoding | Aliases |\n";
  std::cout << "|------|-------|----------|----------|\n";
  for (auto *R : Regs) {
    std::string name = R->getValueAsString("Name");
    int width = R->getValueAsInt("BitWidth");
    int enc = R->getValueAsInt("Encoding");
    std::cout << "| `" << name << "` | " << width << " | 0x"
              << std::hex << enc << std::dec << " | -\n";
  }

  // Instructions section
  auto Insts = RK.getAllDerivedDefinitions("Instruction");
  std::cout << "\n## Instructions\n\n";
  std::cout << "| Mnemonic | Opcode | Side Effects |\n";
  std::cout << "|----------|--------|-------------|\n";
  for (auto *I : Insts) {
    std::cout << "| `" << I->getValueAsString("Mnemonic") << "` | 0x"
              << std::hex << I->getValueAsInt("Opcode") << std::dec
              << " | " << (I->getValueAsBit("HasSideEffects") ? "Yes" : "No")
              << " |\n";
  }

  // Intrinsics section
  auto Intrs = RK.getAllDerivedDefinitions("Intrinsic");
  std::cout << "\n## Intrinsics\n\n";
  for (auto *I : Intrs) {
    std::cout << "- **`" << I->getValueAsString("Name") << "`**";
    if (I->getValueAsBit("HasSideEffects")) std::cout << " (side effects)";
    if (I->getValueAsBit("IsConvergent"))  std::cout << " (convergent)";
    std::cout << "\n";
  }

  std::cout << "\n## Statistics\n\n";
  std::cout << "- Total records: " << RK.Defs.size() << "\n";
  std::cout << "- Register definitions: " << Regs.size() << "\n";
  std::cout << "- Instruction definitions: " << Insts.size() << "\n";
  std::cout << "- Intrinsics: " << Intrs.size() << "\n";
}

// ---------------------------------------------------------------------------
int main() {
  RecordKeeper RK;
  populateRecords(RK);
  generateDocumentation(RK);
  return 0;
}
