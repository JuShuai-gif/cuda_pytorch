//===----------------------------------------------------------------------===//
// Chapter 21 - Getting Started with the Assembler
// Example: MCCodeEmitter - Encoding Instructions to Binary
//===----------------------------------------------------------------------===//
//
// This example demonstrates the MCCodeEmitter workflow:
// - Converting MCInst to binary machine code bytes
// - Encoding registers, immediates into instruction bitfields
// - Creating fixups for unresolved expressions
// - Endian-aware byte output
//
// NOTE: In real LLVM, MCCodeEmitter is generated from TableGen patterns
// and extended with target-specific C++ overrides.
//

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated MC-level types
//------------------------------------------------------------------------------

// Simple MCOperand: can be register, immediate, or expression
struct MCOperand {
  enum Kind { Register, Immediate, Expression };
  Kind OpKind;
  unsigned RegVal;
  int64_t ImmVal;
  std::string ExprName; // Simulated expression (e.g., symbol reference)

  static MCOperand createReg(unsigned Reg, const std::string &Name = "") {
    MCOperand MO;
    MO.OpKind = Register;
    MO.RegVal = Reg;
    return MO;
  }
  static MCOperand createImm(int64_t Val) {
    MCOperand MO;
    MO.OpKind = Immediate;
    MO.ImmVal = Val;
    return MO;
  }
  static MCOperand createExpr(const std::string &Name) {
    MCOperand MO;
    MO.OpKind = Expression;
    MO.ExprName = Name;
    return MO;
  }

  bool isReg() const { return OpKind == Register; }
  bool isImm() const { return OpKind == Immediate; }
  bool isExpr() const { return OpKind == Expression; }

  void print() const {
    switch (OpKind) {
    case Register:   std::cout << "r" << RegVal; break;
    case Immediate:  std::cout << "#" << ImmVal; break;
    case Expression: std::cout << "&" << ExprName; break;
    }
  }
};

// MCInst - a single instruction
struct MCInst {
  unsigned Opcode;
  std::string OpcodeName;
  std::vector<MCOperand> Operands;

  MCInst(unsigned Opc, const std::string &Name) : Opcode(Opc), OpcodeName(Name) {}

  void addOperand(const MCOperand &MO) { Operands.push_back(MO); }
  unsigned getNumOperands() const { return Operands.size(); }
  const MCOperand &getOperand(unsigned Idx) const { return Operands[Idx]; }

  void print() const {
    std::cout << "  " << OpcodeName << " ";
    for (size_t i = 0; i < Operands.size(); ++i) {
      if (i > 0) std::cout << ", ";
      Operands[i].print();
    }
    std::cout << "\n";
  }
};

// MCFixup: represents an unresolved reference
struct MCFixup {
  enum FixupKind {
    FK_32_Absolute,        // 32-bit absolute address
    FK_16_PCRel,           // 16-bit PC-relative with 2-bit shift
    FK_HI16,               // Upper 16 bits
    FK_LO16,               // Lower 16 bits
    FK_32_PCRel,           // 32-bit PC-relative
  };

  FixupKind Kind;
  unsigned Offset;          // Byte offset within the instruction
  std::string SymbolName;   // Target symbol
  int64_t Addend;           // Constant addend

  MCFixup(FixupKind K, unsigned Off, const std::string &Sym, int64_t Add = 0)
    : Kind(K), Offset(Off), SymbolName(Sym), Addend(Add) {}

  void print() const {
    std::cout << "    Fixup: offset=" << Offset << " kind=";
    switch (Kind) {
    case FK_32_Absolute: std::cout << "32-abs"; break;
    case FK_16_PCRel:   std::cout << "16-pcrel"; break;
    case FK_HI16:        std::cout << "HI16"; break;
    case FK_LO16:        std::cout << "LO16"; break;
    case FK_32_PCRel:   std::cout << "32-pcrel"; break;
    }
    std::cout << " symbol=" << SymbolName << " addend=" << Addend << "\n";
  }
};

//------------------------------------------------------------------------------
// Simulated instruction encoding tables
//------------------------------------------------------------------------------
struct InstrEncoding {
  unsigned Opcode;
  std::string Name;
  uint32_t BaseBits;         // Base opcode bits (from TableGen)
  unsigned RegOffset;        // Bit position for register operand
  unsigned ImmOffset;        // Bit position for immediate operand
  unsigned ImmWidth;         // Width of immediate field
  bool IsPCRel;              // Whether the immediate is PC-relative
  bool IsLittleEndian;
  unsigned InstrSize;        // Instruction size in bytes

  InstrEncoding(unsigned Opc, const std::string &N, uint32_t Base,
                unsigned RegOff, unsigned ImmOff, unsigned ImmW,
                bool PCRel = false, unsigned Size = 4)
    : Opcode(Opc), Name(N), BaseBits(Base), RegOffset(RegOff),
      ImmOffset(ImmOff), ImmWidth(ImmW), IsPCRel(PCRel),
      IsLittleEndian(true), InstrSize(Size) {}
};

//------------------------------------------------------------------------------
// MCCodeEmitter - encodes MCInst to bytes
//------------------------------------------------------------------------------
class SimMCCodeEmitter {
private:
  std::vector<InstrEncoding> EncodingTable;

  unsigned encodeRegister(unsigned Reg) const {
    // Simple: register number maps directly to bitfield
    return Reg & 0xF; // 4-bit register encoding
  }

  uint32_t encodeImmediate(int64_t Imm, unsigned Width) const {
    // Mask to fit within the field width
    uint64_t Mask = (1ULL << Width) - 1;
    return static_cast<uint32_t>(Imm & Mask);
  }

public:
  SimMCCodeEmitter() {
    // Set up encoding table (simulated TableGen output)
    // ADD rd, rn, #imm:  opcode(8) | rd(4) | rn(4) | imm(16)
    EncodingTable.emplace_back(1, "ADDri", 0x01000000, 20, 0, 16);
    // SUB rd, rn, #imm:  same format
    EncodingTable.emplace_back(2, "SUBri", 0x02000000, 20, 0, 16);
    // LDR rt, [rn, #offset]:  opcode(8) | rt(4) | rn(4) | offset(16)
    EncodingTable.emplace_back(3, "LDR", 0x03000000, 20, 0, 16);
    // STR rt, [rn, #offset]:  same format
    EncodingTable.emplace_back(4, "STR", 0x04000000, 20, 0, 16);
    // B target: opcode(8) | imm24 (PC-relative, shifted by 2)
    EncodingTable.emplace_back(5, "B", 0x05000000, 0, 0, 24, true);
    // BL target: same as B
    EncodingTable.emplace_back(6, "BL", 0x06000000, 0, 0, 24, true);
  }

  // Find encoding info for an opcode
  const InstrEncoding *findEncoding(unsigned Opcode) const {
    for (auto &E : EncodingTable) {
      if (E.Opcode == Opcode) return &E;
    }
    return nullptr;
  }

  // Main encoding function
  void encodeInstruction(const MCInst &MI, std::vector<uint8_t> &CodeBytes,
                         std::vector<MCFixup> &Fixups) {
    const InstrEncoding *Enc = findEncoding(MI.Opcode);
    if (!Enc) {
      std::cerr << "  ERROR: Unknown opcode " << MI.Opcode << "\n";
      return;
    }

    uint32_t Bits = Enc->BaseBits;

    std::cout << "  Encoding " << MI.OpcodeName << ":\n";
    std::cout << "    Base bits: 0x" << std::hex << Enc->BaseBits
              << std::dec << "\n";

    // Encode each operand
    unsigned RegCount = 0;
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      const MCOperand &MO = MI.getOperand(i);

      if (MO.isReg()) {
        uint32_t RegBits = encodeRegister(MO.RegVal);
        Bits |= RegBits << (Enc->RegOffset - RegCount * 4);
        RegCount++;
        std::cout << "    Reg r" << MO.RegVal << " -> bits["
                  << (Enc->RegOffset - (RegCount-1)*4) << ":"
                  << (Enc->RegOffset - (RegCount-1)*4 + 3) << "] = 0x"
                  << std::hex << RegBits << std::dec << "\n";
      } else if (MO.isImm()) {
        uint32_t ImmBits = encodeImmediate(MO.ImmVal, Enc->ImmWidth);
        Bits |= ImmBits << Enc->ImmOffset;
        std::cout << "    Imm " << MO.ImmVal << " -> bits["
                  << Enc->ImmOffset << ":" << (Enc->ImmOffset + Enc->ImmWidth - 1)
                  << "] = 0x" << std::hex << ImmBits << std::dec << "\n";
      } else if (MO.isExpr()) {
        // Expressions become fixups (references to symbols)
        MCFixup::FixupKind FK;
        if (Enc->IsPCRel) {
          FK = (Enc->InstrSize == 4) ? MCFixup::FK_32_PCRel
                                     : MCFixup::FK_16_PCRel;
        } else {
          FK = MCFixup::FK_32_Absolute;
        }
        Fixups.emplace_back(FK, 0, MO.ExprName, 0);
        std::cout << "    Expr " << MO.ExprName
                  << " -> fixup (will be resolved later)\n";
      }
    }

    // Write bits to byte buffer (little-endian)
    std::cout << "    Final 32-bit encoding: 0x"
              << std::hex << Bits << std::dec << "\n";

    CodeBytes.clear();
    for (unsigned b = 0; b < Enc->InstrSize; ++b) {
      CodeBytes.push_back(static_cast<uint8_t>((Bits >> (b * 8)) & 0xFF));
    }

    std::cout << "    Bytes (LE): ";
    for (auto B : CodeBytes) {
      std::cout << std::hex << "0x" << (int)B << " ";
    }
    std::cout << std::dec << "\n";
  }
};

//------------------------------------------------------------------------------
// Simulated MCStreamer for assembly text emission
//------------------------------------------------------------------------------
class SimMCInstPrinter {
public:
  void printInstruction(const MCInst &MI) {
    std::cout << "  " << MI.OpcodeName;
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      std::cout << (i == 0 ? " " : ", ");
      MI.getOperand(i).print();
    }
    std::cout << "\n";
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 21: MCCodeEmitter ===\n";

  SimMCCodeEmitter Emitter;
  SimMCInstPrinter Printer;

  // Create some MCInst instructions and encode them
  std::cout << "\n--- Encoding ADDri r1, r2, #42 ---\n";
  MCInst ADD(1, "ADDri");
  ADD.addOperand(MCOperand::createReg(1));
  ADD.addOperand(MCOperand::createReg(2));
  ADD.addOperand(MCOperand::createImm(42));

  std::cout << "Before:\n";
  Printer.printInstruction(ADD);

  std::vector<uint8_t> Bytes;
  std::vector<MCFixup> Fixups;
  Emitter.encodeInstruction(ADD, Bytes, Fixups);

  // Encode a branch with expression (creates fixup)
  std::cout << "\n--- Encoding B target (PC-relative, expression) ---\n";
  MCInst Branch(5, "B");
  Branch.addOperand(MCOperand::createExpr("loop_start"));

  std::cout << "Before:\n";
  Printer.printInstruction(Branch);

  Emitter.encodeInstruction(Branch, Bytes, Fixups);

  // Print all resulting fixups
  std::cout << "\n--- Resulting Fixups ---\n";
  for (auto &F : Fixups) {
    F.print();
  }

  // Demonstrate instruction bitfield layout
  std::cout << "\n--- Instruction Bitfield Layout ---\n";
  std::cout << "  ADDri (32-bit):\n";
  std::cout << "    [31:24] Opcode (8 bits)\n";
  std::cout << "    [23:20] Destination register Rd (4 bits)\n";
  std::cout << "    [19:16] Source register Rn (4 bits)\n";
  std::cout << "    [15:0]  Immediate value (16 bits)\n\n";

  std::cout << "  B (32-bit, PC-relative):\n";
  std::cout << "    [31:24] Opcode (8 bits)\n";
  std::cout << "    [23:0]  Signed offset / 4 (24 bits, 2-bit aligned)\n";

  // Encoding variants
  std::cout << "\n--- Encoding Variants ---\n";
  std::cout << "  Different instruction formats use different bitfield layouts:\n";
  std::cout << "    - R-type: opcode | rd | rs1 | rs2 | funct\n";
  std::cout << "    - I-type: opcode | rd | rs1 | immediate\n";
  std::cout << "    - B-type: opcode | offset | condition/predicate\n";
  std::cout << "    - J-type: opcode | jump target\n";

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. MCCodeEmitter::encodeInstruction converts MCInst to bytes\n";
  std::cout << "  2. Base bits come from TableGen (getBinaryCodeForInstr)\n";
  std::cout << "  3. Operand encoding (register/immediate) shifts bits into place\n";
  std::cout << "  4. Expression operands create MCFixup entries\n";
  std::cout << "  5. Endian-aware byte output via support::endian\n";
  std::cout << "  6. MCInstPrinter provides textual assembly output\n";

  return 0;
}
