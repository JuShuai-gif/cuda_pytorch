//===----------------------------------------------------------------------===//
// Chapter 21 - Getting Started with the Assembler
// Example: MCAsmBackend and MCObjectTargetWriter - Object File Emission
//===----------------------------------------------------------------------===//
//
// This example demonstrates:
// - MCAsmBackend: fixup application, instruction relaxation, alignment
// - MCObjectTargetWriter: writing object file sections, symbols, relocations
// - The complete assembly pipeline: MCInst → buffer → object file
// - TargetRegistry registration pattern
//
// NOTE: In real LLVM, these are full classes that handle platform-specific
// object file formats (ELF, MachO, COFF). This is a simulation.
//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated object file structures
//------------------------------------------------------------------------------
struct Section {
  std::string Name;
  std::vector<uint8_t> Data;
  unsigned Alignment;
  bool IsCode;
  bool IsReadOnly;

  Section(const std::string &N, unsigned Align, bool Code, bool RO)
    : Name(N), Alignment(Align), IsCode(Code), IsReadOnly(RO) {}

  void appendBytes(const std::vector<uint8_t> &Bytes) {
    Data.insert(Data.end(), Bytes.begin(), Bytes.end());
  }

  void print() const {
    std::cout << "  Section [" << Name << "]: "
              << Data.size() << " bytes, align=" << Alignment
              << (IsCode ? " code" : "")
              << (IsReadOnly ? " ro" : "") << "\n";
  }
};

struct ObjectSymbol {
  std::string Name;
  unsigned SectionIdx;
  uint64_t Offset;
  uint64_t Size;
  bool IsGlobal;
  bool IsDefined;

  void print() const {
    std::cout << "  Symbol [" << Name << "]: section=" << SectionIdx
              << " offset=" << Offset << " size=" << Size
              << (IsGlobal ? " global" : " local")
              << (IsDefined ? "" : " undefined") << "\n";
  }
};

struct ObjectRelocation {
  unsigned Offset;
  unsigned SymbolIdx;
  unsigned Type;
  int64_t Addend;

  void print() const {
    std::cout << "  Reloc: offset=" << Offset << " sym=" << SymbolIdx
              << " type=" << Type << " addend=" << Addend << "\n";
  }
};

//------------------------------------------------------------------------------
// MCObjectTargetWriter - writes target-specific object structures
//------------------------------------------------------------------------------
class MCObjectTargetWriter {
private:
  std::vector<Section> Sections;
  std::vector<ObjectSymbol> Symbols;
  std::vector<ObjectRelocation> Relocations;
  bool IsELF; // ELF vs. MachO vs. COFF

public:
  MCObjectTargetWriter(bool ELF = true) : IsELF(ELF) {}

  void addSection(const Section &S) { Sections.push_back(S); }
  void addSymbol(const ObjectSymbol &S) { Symbols.push_back(S); }
  void addRelocation(const ObjectRelocation &R) { Relocations.push_back(R); }

  unsigned getNumSections() const { return Sections.size(); }

  // Write the complete object file
  void writeObject() {
    std::cout << "\n=== Writing Object File ===";
    std::cout << "\n  Format: " << (IsELF ? "ELF" : "Other") << "\n\n";

    // ELF header
    std::cout << "  [ELF Header]\n";
    std::cout << "    Class: ELF32\n";
    std::cout << "    Data: 2's complement, little endian\n";
    std::cout << "    Type: REL (Relocatable)\n";

    // Section table
    std::cout << "\n  [Section Table]\n";
    for (auto &S : Sections) {
      S.print();
    }

    // Symbol table
    std::cout << "\n  [Symbol Table]\n";
    for (auto &S : Symbols) {
      S.print();
    }

    // Relocation sections
    if (!Relocations.empty()) {
      std::cout << "\n  [Relocations]\n";
      for (auto &R : Relocations) {
        R.print();
      }
    }
  }
};

//------------------------------------------------------------------------------
// MCAsmBackend - handles fixups, relaxation, instruction alignment
//------------------------------------------------------------------------------
class MCAsmBackend {
private:
  bool IsLittleEndian;
  unsigned PointerSize;    // 4 for 32-bit, 8 for 64-bit

public:
  MCAsmBackend(bool LE = true, unsigned PS = 4)
    : IsLittleEndian(LE), PointerSize(PS) {}

  // Apply a fixup to the data buffer
  void applyFixup(std::vector<uint8_t> &Data, unsigned Offset,
                  uint64_t Value, unsigned FixupKind) {
    std::cout << "  [applyFixup] offset=" << Offset
              << " kind=" << FixupKind << " value=0x"
              << std::hex << Value << std::dec << "\n";

    // Ensure buffer is large enough
    while (Data.size() <= Offset + 4) Data.push_back(0);

    switch (FixupKind) {
    case 0: // 32-bit absolute
      writeValue(Data, Offset, Value, 4);
      break;
    case 1: // 16-bit
      writeValue(Data, Offset, Value, 2);
      break;
    case 2: // PC-relative 32-bit (for branches: shifted by 2)
      // For branch instructions: value = (target - PC) >> 2
      {
        int32_t PCOffset = static_cast<int32_t>(Value);
        std::cout << "    PC-relative branch: offset=" << PCOffset
                  << " shifted=" << (PCOffset >> 2) << "\n";
        writeValue(Data, Offset, PCOffset >> 2, 3); // 24-bit shifted field
      }
      break;
    case 3: // HI16
      writeValue(Data, Offset, (Value >> 16) & 0xFFFF, 2);
      break;
    case 4: // LO16
      writeValue(Data, Offset, Value & 0xFFFF, 2);
      break;
    default:
      std::cout << "    Unknown fixup kind " << FixupKind << "\n";
      break;
    }
  }

  // Check if an instruction may need relaxation
  bool mayNeedRelaxation(const std::string &Opcode,
                         const std::vector<int64_t> &Operands) const {
    // Branches with large offsets may need relaxation to longer forms
    if (Opcode == "B" && Operands.size() > 0) {
      int64_t Offset = Operands[0];
      // If offset exceeds 24-bit signed range (16MB), need relaxation
      if (Offset < -(1 << 23) || Offset >= (1 << 23)) {
        std::cout << "  [Relaxation] Branch offset " << Offset
                  << " too large for B - needs BL or indirect jump\n";
        return true;
      }
    }
    return false;
  }

  // Relax an instruction (replace with longer form)
  bool relaxInstruction(const std::string &Opcode,
                        std::string &NewOpcode,
                        std::vector<int64_t> &NewOperands) {
    if (Opcode == "B") {
      NewOpcode = "BL_long";
      std::cout << "  [Relaxation] B -> BL_long (far branch)\n";
      return true;
    }
    return false;
  }

  // Handle section alignment
  void handleAlignment(std::vector<uint8_t> &Section, unsigned Align) {
    unsigned CurrentSize = Section.size();
    unsigned AlignMask = Align - 1;
    if (CurrentSize & AlignMask) {
      unsigned Padding = Align - (CurrentSize & AlignMask);
      std::cout << "  [Alignment] Padding " << Padding
                << " bytes for " << Align << "-byte alignment\n";
      for (unsigned i = 0; i < Padding; ++i) {
        Section.push_back(0); // NOP or zero padding
      }
    }
  }

  bool isLittleEndian() const { return IsLittleEndian; }
  unsigned getPointerSize() const { return PointerSize; }

private:
  void writeValue(std::vector<uint8_t> &Data, unsigned Offset,
                  uint64_t Value, unsigned Size) {
    for (unsigned i = 0; i < Size; ++i) {
      uint8_t Byte;
      if (IsLittleEndian) {
        Byte = static_cast<uint8_t>((Value >> (i * 8)) & 0xFF);
      } else {
        Byte = static_cast<uint8_t>((Value >> ((Size - 1 - i) * 8)) & 0xFF);
      }
      Data[Offset + i] = Byte;
    }
  }
};

//------------------------------------------------------------------------------
// Complete assembly pipeline simulation
//------------------------------------------------------------------------------
void simulateAssemblyPipeline() {
  std::cout << "\n=== Assembly Pipeline ===\n\n";

  // Create backend and object writer
  MCAsmBackend Backend(true, 4);
  MCObjectTargetWriter ObjWriter(true);

  // Create sections
  Section Text(".text", 4, true, true);
  Section Data(".data", 4, false, false);
  Section Rodata(".rodata", 4, false, true);

  // Simulate encoding instructions
  std::cout << "--- Encoding Instructions ---\n";

  // ADD r1, r2, #42 → 0x002A1011
  std::vector<uint8_t> AddInstr = {0x11, 0x10, 0x2A, 0x00};
  std::cout << "  ADD r1, r2, #42 -> ";
  for (auto B : AddInstr) std::cout << std::hex << "0x" << (int)B << " ";
  std::cout << std::dec << "\n";
  Text.appendBytes(AddInstr);

  // B loop_top - offset encoded as 0 (will be fixed via fixup)
  std::vector<uint8_t> BranchInstr = {0x00, 0x00, 0x00, 0x05};
  std::cout << "  B loop_top -> (with fixup)\n";
  Text.appendBytes(BranchInstr);

  // Apply fixups
  std::cout << "\n--- Applying Fixups ---\n";

  // Fix branch instruction at offset 4: target = 0x1008, fixup at 0x1004
  // PC value for branch = fixup_addr + 8 (ARM pipeline effect)
  // offset = (0x1008 - (0x1004 + 8)) >> 2 = (0x1008 - 0x100C) >> 2 = (-4) >> 2 = -1
  Backend.applyFixup(Text.Data, 4, 0x1008 - 0x100C, 2);

  // Fix data reference
  std::vector<uint8_t> LoadAddr = {0x00, 0x00, 0x00, 0x03};
  Text.appendBytes(LoadAddr);
  Backend.applyFixup(Text.Data, 8, 0x2000, 0); // 32-bit absolute

  // Handle alignment
  std::cout << "\n--- Alignment ---\n";
  Backend.handleAlignment(Text.Data, 4);

  // Create symbols
  std::cout << "\n--- Creating Symbols ---\n";
  ObjWriter.addSymbol({"main", 0, 0, 20, true, true});
  ObjWriter.addSymbol({"printf", 0, 0, 0, true, false}); // undefined
  ObjWriter.addSymbol({"loop_top", 0, 4, 0, false, true});
  ObjWriter.addSymbol({"data_start", 1, 0, 8, true, true});

  // Create relocations
  std::cout << "\n--- Creating Relocations ---\n";
  ObjWriter.addRelocation({4, 0, 2, 0}); // loop_top fixup
  ObjWriter.addRelocation({16, 1, 1, 0}); // printf reference

  // Add sections and write object
  ObjWriter.addSection(Text);
  ObjWriter.addSection(Data);
  ObjWriter.addSection(Rodata);

  ObjWriter.writeObject();

  // Instruction relaxation demo
  std::cout << "\n--- Instruction Relaxation ---\n";
  if (Backend.mayNeedRelaxation("B", {1 << 24})) {
    std::string NewOpc;
    std::vector<int64_t> NewOps;
    Backend.relaxInstruction("B", NewOpc, NewOps);
  }
}

//------------------------------------------------------------------------------
// TargetRegistry Registration Pattern
//------------------------------------------------------------------------------
void demonstrateTargetRegistry() {
  std::cout << "\n=== TargetRegistry Registration Pattern ===\n\n";

  std::cout << "  // In your target's initialization code:\n";
  std::cout << "  extern \"C\" LLVM_EXTERNAL_VISIBILITY\n";
  std::cout << "  void LLVMInitializeMyTarget() {\n";
  std::cout << "    RegisterTargetMachine<MyTargetMachine> X(getTheMyTarget());\n";
  std::cout << "    \n";
  std::cout << "    TargetRegistry::RegisterMCAsmBackend(\n";
  std::cout << "        getTheMyTarget(), createMyAsmBackend);\n";
  std::cout << "    TargetRegistry::RegisterMCCodeEmitter(\n";
  std::cout << "        getTheMyTarget(), createMyMCCodeEmitter);\n";
  std::cout << "    TargetRegistry::RegisterMCInstPrinter(\n";
  std::cout << "        getTheMyTarget(), createMyInstPrinter);\n";
  std::cout << "    TargetRegistry::RegisterMCAsmParser(\n";
  std::cout << "        getTheMyTarget(), createMyAsmParser);\n";
  std::cout << "    TargetRegistry::RegisterMCObjectWriter(\n";
  std::cout << "        getTheMyTarget(), createMyELFObjectWriter);\n";
  std::cout << "  }\n";

  std::cout << "\n  // TableGen backends used:\n";
  std::cout << "  //   gen-instr-info     -> MCInstrInfo\n";
  std::cout << "  //   gen-register-info  -> MCRegisterInfo\n";
  std::cout << "  //   gen-subtarget      -> MCSubtargetInfo\n";
  std::cout << "  //   gen-asm-writer     -> MCInstPrinter\n";
  std::cout << "  //   gen-asm-matcher    -> MCAsmParser tables\n";
  std::cout << "  //   gen-disassembler   -> Disassembler tables\n";
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 21: MCAsmBackend and MCObjectTargetWriter ===\n";

  // Simulate the complete assembly pipeline
  simulateAssemblyPipeline();

  // Show registration pattern
  demonstrateTargetRegistry();

  std::cout << "\n--- Assembly Pipeline Components ---\n";
  std::cout << "  MachineFunction → AsmPrinter (target-specific)\n";
  std::cout << "    ├── lowerMachineInstrToMCInst()\n";
  std::cout << "    ├── emitConstantPool()\n";
  std::cout << "    └── emitJumpTableInfo()\n";
  std::cout << "         ↓\n";
  std::cout << "  MCStreamer\n";
  std::cout << "    ├── EmitInstruction() → MCCodeEmitter → bytes + fixups\n";
  std::cout << "    ├── EmitLabel() → symbol definitions\n";
  std::cout << "    └── EmitBytes() → raw data\n";
  std::cout << "         ↓\n";
  std::cout << "  MCAssembler\n";
  std::cout << "    ├── Layout sections (MCAsmLayout)\n";
  std::cout << "    ├── Apply fixups (MCAsmBackend)\n";
  std::cout << "    └── Generate relocations\n";
  std::cout << "         ↓\n";
  std::cout << "  MCObjectWriter (MCObjectTargetWriter)\n";
  std::cout << "    ├── Write ELF/MachO/COFF header\n";
  std::cout << "    ├── Write section data\n";
  std::cout << "    ├── Write symbol table\n";
  std::cout << "    └── Write relocation tables\n";
  std::cout << "         ↓\n";
  std::cout << "  Object File (.o)\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. MCAsmBackend applies fixups and handles relaxation\n";
  std::cout << "  2. MCObjectTargetWriter produces platform-specific object files\n";
  std::cout << "  3. Instruction relaxation enlarges short forms when offsets overflow\n";
  std::cout << "  4. Alignment ensures sections and instructions are properly aligned\n";
  std::cout << "  5. TargetRegistry connects all components via factory functions\n";
  std::cout << "  6. TableGen generates most boilerplate for MC components\n";

  return 0;
}
