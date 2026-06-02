// generic_mir_example.cpp - GlobalISel's Generic Machine IR concepts
//
// Demonstrates:
//   - Generic virtual registers (with LLT, register bank)
//   - G_ opcodes (G_ADD, G_LOAD, G_CONSTANT, etc.)
//   - Lowering constraints progression
//   - Key GlobalISel API patterns

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// 1. Low-Level Type (LLT) simulation
// ============================================================

struct LLT {
  enum Kind { Scalar, Pointer, Vector };
  Kind TheKind;
  unsigned SizeInBits;
  unsigned AddressSpace;     // For pointers
  unsigned NumElements;      // For vectors

  static LLT scalar(unsigned Size) { return {Scalar, Size, 0, 0}; }
  static LLT pointer(unsigned AS, unsigned Size) { return {Pointer, Size, AS, 0}; }
  static LLT vector(unsigned NumElem, LLT ElemTy) {
    return {Vector, ElemTy.SizeInBits * NumElem, 0, NumElem};
  }

  bool isScalar() const { return TheKind == Scalar; }
  bool isPointer() const { return TheKind == Pointer; }
  unsigned getSizeInBits() const { return SizeInBits; }
};

// ============================================================
// 2. Register bank and register class simulation
// ============================================================

struct RegisterBank {
  std::string Name;
  // Covered register classes (in real LLVM, determined from TableGen)
};

struct RegisterClass {
  std::string Name;
};

// ============================================================
// 3. Generic virtual register representation
// ============================================================

struct VirtualRegister {
  unsigned ID;
  LLT Type;
  const RegisterBank *Bank = nullptr;
  const RegisterClass *RC = nullptr;

  // A register is a "generic virtual register" if it has no register class.
  bool isGeneric() const { return RC == nullptr; }
};

struct MachineRegisterInfo {
  std::vector<VirtualRegister> VRegs;

  Register createGenericVirtualRegister(LLT Ty) {
    unsigned ID = static_cast<unsigned>(VRegs.size());
    VRegs.push_back({ID, Ty});
    return ID;
  }

  LLT getType(Register Reg) const { return VRegs[Reg].Type; }
  void setType(Register Reg, LLT Ty) { VRegs[Reg].Type = Ty; }

  const RegisterBank *getRegBank(Register Reg) const { return VRegs[Reg].Bank; }
  void setRegBank(Register Reg, const RegisterBank *B) { VRegs[Reg].Bank = B; }

  const RegisterClass *getRegClass(Register Reg) const { return VRegs[Reg].RC; }
  void setRegClass(Register Reg, const RegisterClass *RC) { VRegs[Reg].RC = RC; }
};

using Register = unsigned;

// ============================================================
// 4. Generic MachineInstr (G_MIR) representation
// ============================================================

struct MachineInstr {
  unsigned Opcode;
  std::vector<Register> Defs;  // Output register operands
  std::vector<Register> Uses;  // Input register operands

  bool isPreISelGenericOpcode() const {
    // In real LLVM, checks if opcode starts with G_ prefix
    return Opcode >= 1000; // Simulated: G_ opcodes are >= 1000
  }
};

// Generic opcode identifiers (from GenericOpcodes.td)
namespace TargetOpcode {
  enum Generic : unsigned {
    G_ADD       = 1000,
    G_SUB       = 1001,
    G_MUL       = 1002,
    G_LOAD      = 1003,
    G_STORE     = 1004,
    G_CONSTANT  = 1005,
    G_SEXT      = 1006,
    G_ZEXT      = 1007,
    G_TRUNC     = 1008,
    G_BRCOND    = 1009,
    G_PHI       = 1010,
    G_IMPLICIT_DEF = 1011,
    G_FADD      = 1012,
    G_FMUL      = 1013,
    G_BITCAST   = 1014,
    G_ANYEXT    = 1015,
  };
}

// ============================================================
// 5. Simulated MachineIRBuilder
// ============================================================

struct MachineIRBuilder {
  MachineRegisterInfo *MRI;

  // Build a G_ADD instruction
  MachineInstr buildAdd(LLT Ty, Register Src1, Register Src2) {
    Register Dst = MRI->createGenericVirtualRegister(Ty);
    return {TargetOpcode::G_ADD, {Dst}, {Src1, Src2}};
  }

  // Build a G_CONSTANT instruction (materialize immediate in vreg)
  MachineInstr buildConstant(LLT Ty, int64_t Val) {
    Register Dst = MRI->createGenericVirtualRegister(Ty);
    return {TargetOpcode::G_CONSTANT, {Dst}, {}};
  }

  // Build a G_LOAD instruction
  MachineInstr buildLoad(LLT ValTy, Register Ptr) {
    Register Dst = MRI->createGenericVirtualRegister(ValTy);
    return {TargetOpcode::G_LOAD, {Dst}, {Ptr}};
  }
};

// ============================================================
// 6. Lowering constraints progression
// ============================================================

void illustrateLoweringConstraints() {
  // Stage 1: After IRTranslator (pre-legalization)
  //   - Any G_ opcode on any type is allowed.
  //   - Virtual registers may not have a register bank, but MUST have LLT.
  //
  //   Example:
  //     %0:_(s32) = G_CONSTANT 42
  //     %1:_(s32) = G_ADD %0, %0
  //     %2:_(s3)  = G_ADD %0, %0   // Allowed! (s3 is 3-bit integer)
  //
  //   Textual representation:
  //     %vreg:_(s32)  = generic vreg, no bank, no class, 32-bit scalar type

  // Stage 2: After Legalizer (post-legalization)
  //   - Only legal G_ opcodes with legal types are allowed.
  //   - s3 G_ADD would be widened/replaced during legalization.
  //
  //   Example:
  //     %0:_(s32) = G_CONSTANT 42   // Legal
  //     %1:_(s32) = G_ADD %0, %0    // Legal
  //     // G_ADD s3 was expanded away
  //
  //   Textual representation:
  //     %vreg:gprb(s32)  = has register bank now

  // Stage 3: After RegBankSelect (post-regbank-select)
  //   - All alive generic virtual registers must have a register bank.
  //   - Same legalization constraints apply.
  //
  //   Example:
  //     %0:gprb(s32) = G_CONSTANT 42
  //     %1:gprb(s32) = G_ADD %0, %0
  //
  //   Textual representation:
  //     %vreg:gprb(s32)  = gprb is the register bank name

  // Stage 4: After InstructionSelect (post-isel)
  //   - NO generic opcodes remain.
  //   - NO generic virtual registers remain.
  //   - All virtual registers have a valid register class.
  //
  //   Example:
  //     %0:gpr32 = MYADD32 %0, %0
  //
  //   Textual representation:
  //     %vreg:gpr32  = gpr32 is a register class
}

// ============================================================
// 7. Key distinction: LLT scalar types disambiguation
// ============================================================

void disambiguateScalarTypes() {
  // In GlobalISel, scalar types of the same bit width share the same LLT:
  //   i32 (integer 32-bit)  -> s32
  //   f32 (float 32-bit)    -> s32
  //   bf16 (bfloat16)       -> s16
  //   f16 (half float)      -> s16
  //
  // Disambiguation is done by the OPCODE, not the type:
  //   Integer addition     -> G_ADD
  //   Float addition       -> G_FADD
  //
  // This means you can't have two different float32 types share the same
  // opcode space. This limitation is being reworked upstream.

  LLT s32 = LLT::scalar(32);
  LLT s16 = LLT::scalar(16);

  // G_ADD on s32 -> integer addition
  // G_FADD on s32 -> floating-point addition
  // G_ADD on s16 -> integer addition of 16-bit value
  // G_FADD on s16 -> unknown: is it f16 or bf16?
  //   (Need extra context/metadata to differentiate)
  (void)s32;
  (void)s16;
}

// ============================================================
// 8. Generic vs non-generic MachineInstr coexistence
// ============================================================

void coexistenceExample() {
  // It is perfectly valid to have both generic and regular instructions
  // in the same MachineBasicBlock during GlobalISel lowering.

  // For example, during InstructionSelect, some instructions may already
  // be selected while others still use G_ opcodes:

  MachineIRBuilder MIB;
  MIB.MRI = new MachineRegisterInfo();

  // Selected instruction (target-specific):
  //   %0:gpr32 = MYADD32 %1, %2

  // Still generic (waiting to be selected):
  //   %3:gprb(s32) = G_LOAD %4

  // In a real .mir dump:
  //   %0:gpr32 = ADD32 %1, %2           ; selected
  //   %3:gprb(s32) = G_LOAD %4          ; still generic
  //   %5:gpr32 = COPY %3                ; copy generic -> regular
}

// ============================================================
// 9. MachineFunction properties tracking lowering stage
// ============================================================

enum class MachineFunctionProperty {
  IsSSA,
  NoPHIs,
  TracksLiveness,
  Selected,        // Isel complete
  Legalized,       // Legalizer complete
  RegBankSelected, // RegBankSelect complete
};

// Passes query properties to know where they are in the pipeline:
bool checkStage(/*MachineFunction &MF*/) {
  // MF.getProperties().hasProperty(MachineFunctionProperty::Legalized);
  // This returns true only after the Legalizer pass has run.
  return true;
}
