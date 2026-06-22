//===----------------------------------------------------------------------===//
// Chapter 21 - Getting Started with the Assembler
// Example: Fixups and Relocations - Unresolved References to Linker Directives
//===----------------------------------------------------------------------===//
//
// This example demonstrates how fixups and relocations work:
// - MCFixup: In-progress reference that may be resolved at assembly time
// - Relocation: Unresolved fixup recorded in the object file for the linker
// - Fixup kinds: absolute, PC-relative, hi/lo splits
// - Fixup application: writing resolved values into the instruction stream
// - ELF relocation entries and their processing
//

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Fixup Kind Definitions
//------------------------------------------------------------------------------
enum class FixupKind {
  // Absolute fixups
  FK_Data_1,         // 1-byte absolute
  FK_Data_2,         // 2-byte absolute
  FK_Data_4,         // 4-byte absolute
  FK_Data_8,         // 8-byte absolute

  // PC-relative fixups
  FK_PCRel_1,        // 1-byte PC-relative
  FK_PCRel_2,        // 2-byte PC-relative
  FK_PCRel_4,        // 4-byte PC-relative

  // Target-specific
  FK_HI16,           // Upper 16 bits of a 32-bit value
  FK_LO16,           // Lower 16 bits of a 32-bit value
  FK_GOTPCREL,       // GOT-relative (for position-independent code)
};

const char *fixupKindName(FixupKind K) {
  switch (K) {
  case FixupKind::FK_Data_1:    return "FK_Data_1";
  case FixupKind::FK_Data_2:    return "FK_Data_2";
  case FixupKind::FK_Data_4:    return "FK_Data_4";
  case FixupKind::FK_Data_8:    return "FK_Data_8";
  case FixupKind::FK_PCRel_1:   return "FK_PCRel_1";
  case FixupKind::FK_PCRel_2:   return "FK_PCRel_2";
  case FixupKind::FK_PCRel_4:   return "FK_PCRel_4";
  case FixupKind::FK_HI16:      return "FK_HI16";
  case FixupKind::FK_LO16:      return "FK_LO16";
  case FixupKind::FK_GOTPCREL:  return "FK_GOTPCREL";
  }
  return "?";
}

//------------------------------------------------------------------------------
// MCFixup
//------------------------------------------------------------------------------
struct MCFixup {
  FixupKind Kind;
  unsigned Offset;         // Byte offset within the fragment/section
  std::string Symbol;      // Target symbol name
  int64_t Addend;          // Constant addend
  bool IsResolved;         // Has the fixup been resolved already?

  MCFixup(FixupKind K, unsigned Off, const std::string &Sym, int64_t Add = 0)
    : Kind(K), Offset(Off), Symbol(Sym), Addend(Add), IsResolved(false) {}

  void print() const {
    std::cout << "    fixup offset=" << Offset
              << " kind=" << fixupKindName(Kind)
              << " symbol='" << Symbol << "'"
              << " addend=" << Addend
              << (IsResolved ? " [resolved]" : " [pending]") << "\n";
  }
};

//------------------------------------------------------------------------------
// ELF Relocation Entry
//------------------------------------------------------------------------------
struct ELFRelocation {
  unsigned Offset;         // Offset within the section
  unsigned SymIdx;         // Symbol table index
  unsigned Type;           // Relocation type (e.g., R_ARM_ABS32)
  int64_t Addend;          // Constant addend (for RELA)

  ELFRelocation(unsigned Off, unsigned Sym, unsigned T, int64_t Add = 0)
    : Offset(Off), SymIdx(Sym), Type(T), Addend(Add) {}

  void print() const {
    std::cout << "    relocation offset=" << Offset
              << " symIdx=" << SymIdx
              << " type=" << Type
              << " addend=" << Addend << "\n";
  }
};

//------------------------------------------------------------------------------
// Symbol Table (simulated)
//------------------------------------------------------------------------------
struct SymbolEntry {
  std::string Name;
  uint64_t Address;        // Address (0 if unresolved)
  bool IsDefined;          // Defined in this object?
  bool IsGlobal;

  SymbolEntry(const std::string &N, uint64_t Addr = 0, bool Def = false)
    : Name(N), Address(Addr), IsDefined(Def), IsGlobal(!Def) {}
};

//------------------------------------------------------------------------------
// Fixup Resolution / Relocation Generation
//------------------------------------------------------------------------------
class FixupResolver {
private:
  std::vector<SymbolEntry> SymbolTable;
  std::vector<ELFRelocation> Relocations;

  // Mock addresses for known symbols
  uint64_t SectionStartAddr; // Address of current section

public:
  FixupResolver(uint64_t SecAddr = 0x1000)
    : SectionStartAddr(SecAddr) {}

  void addSymbol(const std::string &Name, uint64_t Addr, bool Defined = true) {
    SymbolTable.emplace_back(Name, Addr, Defined);
  }

  unsigned findSymbol(const std::string &Name) const {
    for (unsigned i = 0; i < SymbolTable.size(); ++i) {
      if (SymbolTable[i].Name == Name) return i;
    }
    return ~0U;
  }

  // Attempt to resolve a fixup.
  // Returns true if resolved at assembly time; false if a relocation is needed.
  bool resolveFixup(MCFixup &Fixup, unsigned InstrOffset,
                    std::vector<uint8_t> &CodeBuffer) {
    unsigned SymIdx = findSymbol(Fixup.Symbol);

    if (SymIdx == ~0U) {
      std::cout << "    Symbol '" << Fixup.Symbol
                << "' not found - creating relocation\n";
      createRelocation(Fixup, InstrOffset, SymIdx);
      return false;
    }

    const SymbolEntry &Sym = SymbolTable[SymIdx];
    if (!Sym.IsDefined) {
      std::cout << "    Symbol '" << Fixup.Symbol
                << "' is undefined - creating relocation\n";
      createRelocation(Fixup, InstrOffset, SymIdx);
      return false;
    }

    // Symbol is defined in this object - can resolve at assembly time
    uint64_t TargetAddr = Sym.Address + Fixup.Addend;
    uint64_t FixupAddr = SectionStartAddr + InstrOffset;
    uint64_t ResolvedValue;

    std::cout << "    Resolving fixup at offset " << InstrOffset
              << ": target=" << TargetAddr << " fixup_addr=" << FixupAddr << "\n";

    switch (Fixup.Kind) {
    case FixupKind::FK_Data_4:
      // Absolute 32-bit: just the target address
      ResolvedValue = TargetAddr;
      writeToBuffer(CodeBuffer, InstrOffset, ResolvedValue, 4);
      std::cout << "      Absolute: value=0x" << std::hex << ResolvedValue
                << std::dec << "\n";
      break;

    case FixupKind::FK_PCRel_4:
      // PC-relative: target - (fixup_address + adjustment)
      // For many architectures, the PC value is fixup_address + 4 or + 8
      ResolvedValue = TargetAddr - (FixupAddr + 4);
      writeToBuffer(CodeBuffer, InstrOffset, ResolvedValue, 4);
      std::cout << "      PC-relative: value=0x" << std::hex << ResolvedValue
                << std::dec << " (target - (PC+4))\n";
      break;

    case FixupKind::FK_HI16:
      // Upper 16 bits: (value >> 16)
      ResolvedValue = (TargetAddr >> 16) & 0xFFFF;
      writeToBuffer(CodeBuffer, InstrOffset, ResolvedValue, 2);
      std::cout << "      HI16: value=0x" << std::hex << ResolvedValue
                << std::dec << "\n";
      break;

    case FixupKind::FK_LO16:
      // Lower 16 bits
      ResolvedValue = TargetAddr & 0xFFFF;
      writeToBuffer(CodeBuffer, InstrOffset, ResolvedValue, 2);
      std::cout << "      LO16: value=0x" << std::hex << ResolvedValue
                << std::dec << "\n";
      break;

    default:
      std::cout << "      Unhandled fixup kind - creating relocation\n";
      createRelocation(Fixup, InstrOffset, SymIdx);
      return false;
    }

    Fixup.IsResolved = true;
    return true;
  }

  void createRelocation(const MCFixup &Fixup, unsigned Offset,
                        unsigned SymIdx) {
    // Map fixup kind to ELF relocation type
    unsigned RelocType;
    switch (Fixup.Kind) {
    case FixupKind::FK_Data_4:    RelocType = 1; break; // R_XXX_ABS32
    case FixupKind::FK_PCRel_4:   RelocType = 2; break; // R_XXX_PC32
    case FixupKind::FK_HI16:      RelocType = 3; break; // R_XXX_HI16
    case FixupKind::FK_LO16:      RelocType = 4; break; // R_XXX_LO16
    case FixupKind::FK_GOTPCREL:  RelocType = 5; break; // R_XXX_GOTPCREL
    default:                      RelocType = 0; break;
    }

    Relocations.emplace_back(Offset, SymIdx, RelocType, Fixup.Addend);
  }

  void writeToBuffer(std::vector<uint8_t> &Buf, unsigned Offset,
                     uint64_t Value, unsigned Size) {
    while (Buf.size() <= Offset + Size) Buf.push_back(0);
    for (unsigned i = 0; i < Size; ++i) {
      Buf[Offset + i] = static_cast<uint8_t>((Value >> (i * 8)) & 0xFF);
    }
  }

  void printRelocations() const {
    std::cout << "\n  ELF Relocations:\n";
    for (auto &R : Relocations) {
      R.print();
    }
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 21: Fixups and Relocations ===\n";

  FixupResolver Resolver(0x1000);

  // Set up symbol table
  Resolver.addSymbol("main", 0x1000, true);       // Defined here
  Resolver.addSymbol("printf", 0, false);          // Undefined (external)
  Resolver.addSymbol("data_start", 0x2000, true); // Defined here
  Resolver.addSymbol("global_var", 0, false);      // Undefined (external)
  Resolver.addSymbol("loop_top", 0x10A0, true);   // Local label

  std::vector<uint8_t> CodeBuffer;
  std::vector<MCFixup> Fixups;

  // Create some fixups
  std::cout << "\n--- Creating Fixups ---\n";

  // PC-relative branch to local label (can be resolved)
  Fixups.emplace_back(FixupKind::FK_PCRel_4, 4, "loop_top", 0);
  Fixups.back().print();

  // Absolute address of external function (needs relocation)
  Fixups.emplace_back(FixupKind::FK_Data_4, 16, "printf", 0);
  Fixups.back().print();

  // HI16 of a known symbol
  Fixups.emplace_back(FixupKind::FK_HI16, 20, "data_start", 0);
  Fixups.back().print();

  // LO16 of a known symbol
  Fixups.emplace_back(FixupKind::FK_LO16, 22, "data_start", 0);
  Fixups.back().print();

  // Absolute address of external global (needs relocation)
  Fixups.emplace_back(FixupKind::FK_Data_4, 8, "global_var", 0);
  Fixups.back().print();

  // Resolve each fixup
  std::cout << "\n--- Resolving Fixups ---\n";
  for (auto &F : Fixups) {
    std::cout << "  Processing: " << fixupKindName(F.Kind)
              << " to '" << F.Symbol << "'\n";
    Resolver.resolveFixup(F, F.Offset, CodeBuffer);
  }

  // Print final fixup status
  std::cout << "\n--- Final Fixup Status ---\n";
  for (auto &F : Fixups) {
    F.print();
  }

  // Print relocations that were generated
  Resolver.printRelocations();

  // Fixup application demonstration
  std::cout << "\n--- Fixup Application Process ---\n";
  std::cout << "  1. MCCodeEmitter creates fixups for expression operands\n";
  std::cout << "  2. MCAssembler attempts to resolve fixups at assembly time\n";
  std::cout << "     - If the symbol is defined in the same section: resolve now\n";
  std::cout << "     - If the symbol is local but in a different section: may resolve\n";
  std::cout << "     - If the symbol is external/global: create relocation\n";
  std::cout << "  3. Unresolved fixups become relocations in the object file\n";
  std::cout << "  4. The linker processes relocations and patches the final binary\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Fixups represent references that need patching\n";
  std::cout << "  2. FixupKind determines how to compute the patch value\n";
  std::cout << "  3. PC-relative fixups: target - (fixup_addr + PC_bias)\n";
  std::cout << "  4. HI16/LO16: split 32-bit values for 16-bit immediate fields\n";
  std::cout << "  5. Resolved fixups → patched bytes; unresolved → relocations\n";
  std::cout << "  6. ELF relocations contain: offset, symbol, type, addend\n";

  return 0;
}
