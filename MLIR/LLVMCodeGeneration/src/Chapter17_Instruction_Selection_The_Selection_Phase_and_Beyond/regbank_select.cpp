// regbank_select.cpp - Register Bank Selection (GlobalISel)
//
// Demonstrates:
//   - RegisterBank TableGen description
//   - RegisterBankInfo implementation
//   - InstructionMapping, ValueMapping, PartialMapping classes
//   - Cross-register-bank copy optimization

#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ============================================================
// 1. RegisterBank TableGen Description
// ============================================================

// In .td file:
/*
def GPRBRegBank : RegisterBank<"GPRB", [GPR32, GPR16sp]>;
def VRBRegBank  : RegisterBank<"VRB",  [VR128]>;

// gen-register-bank TableGen backend produces:
//   - GET_REGBANK_DECLARATIONS (in header)
//   - GET_TARGET_REGBANK_IMPL    (in .cpp)
//   - GET_TARGET_REGBANK_CLASS   (in generated base class)
*/

// ============================================================
// 2. Simulated RegisterBank
// ============================================================

struct RegisterBank {
  std::string Name;
  unsigned ID;
  std::vector<unsigned> ContainedRegClasses; // Register class IDs

  bool covers(unsigned RegClassID) const {
    for (auto &ID : ContainedRegClasses)
      if (ID == RegClassID) return true;
    return false;
  }
};

// ============================================================
// 3. PartialMapping, ValueMapping, InstructionMapping
// ============================================================

// PartialMapping: maps a contiguous bit range of a value to a register bank.
//   {StartBit, NumBits, RegBank}
struct PartialMapping {
  unsigned StartIdx;    // First bit of the original value that is mapped
  unsigned Length;      // Number of bits mapped
  const RegisterBank *RegBank;

  PartialMapping(unsigned Start, unsigned Len, const RegisterBank *RB)
    : StartIdx(Start), Length(Len), RegBank(RB) {}
};

// ValueMapping: describes how a single value operand maps to registers.
// Can span multiple registers (e.g., <2 x i32> on 32-bit GPRs needs 2 regs).
struct ValueMapping {
  std::vector<PartialMapping> Parts;

  ValueMapping(std::initializer_list<PartialMapping> Pms) : Parts(Pms) {}

  size_t getNumParts() const { return Parts.size(); }
  const PartialMapping &getPart(size_t i) const { return Parts[i]; }
};

// InstructionMapping: describes register bank assignment for ALL operands
// of an instruction. Also carries a cost and an ID for the cost model.
struct InstructionMapping {
  unsigned ID;
  unsigned Cost;
  std::vector<const ValueMapping *> Operands; // One per instruction operand

  InstructionMapping(unsigned Id, unsigned C, std::vector<const ValueMapping *> Ops)
    : ID(Id), Cost(C), Operands(std::move(Ops)) {}
};

// ============================================================
// 4. RegisterBankInfo implementation
// ============================================================

// Each target creates its own RegisterBankInfo class.
// This is the class that drives the RegBankSelect pass.

// Generated base class (by TableGen):
class XXXGenRegisterBankInfo {
protected:
  // Contains the register bank definitions, enum values, etc.
  // Injected via:
  //   #define GET_TARGET_REGBANK_CLASS
  //   #include "XXXGenRegisterBank.inc"
};

class H2BLBRegisterBankInfo : public XXXGenRegisterBankInfo {
  // Define register banks
  RegisterBank GPRB{"GPRB", 0, {0, 1}};   // Covers GPR16, GPR32
  RegisterBank VRB{"VRB", 1, {2}};         // Covers VR128

  // Define PartialMappings
  PartialMapping PM_GPR_16{0, 16, &GPRB};
  PartialMapping PM_GPR_32{0, 32, &GPRB};
  PartialMapping PM_GPR_64{0, 64, &GPRB};
  PartialMapping PM_VR_128{0, 128, &VRB};

  // Define ValueMappings (common configurations)
  ValueMapping VM_GPR16{{PM_GPR_16}};
  ValueMapping VM_GPR32{{PM_GPR_32}};
  ValueMapping VM_2xGPR32{{PM_GPR_32}, {PM_GPR_32}};  // 64-bit on two 32-bit GPRs
  ValueMapping VM_VR128{{PM_VR_128}};

  // InstructionMapping IDs
  enum MappingID {
    DefaultMapping = 1,
    GPROperandMapping,
    VROperandMapping,
  };

  // ============================================================
  // 5. getInstrMapping - the core method
  // ============================================================
  //
  // For each instruction, return an InstructionMapping that describes
  // which register bank each operand should use.
  //
  // The RegBankSelect pass calls this for every instruction,
  // then uses the cost model to decide whether to rewrite the IR.

  const InstructionMapping &
  getInstrMapping(/* const MachineInstr &MI */) const {
    // // Get the opcode
    // unsigned Opcode = MI.getOpcode();
    //
    // // Determine mapping based on opcode and operand types
    // switch (Opcode) {
    // case TargetOpcode::G_ADD:
    // case TargetOpcode::G_SUB:
    // case TargetOpcode::G_AND:
    // case TargetOpcode::G_OR:
    //   // Integer arithmetic always uses GPR banks
    //   return getDefaultMappingGPR(MI);
    //
    // case TargetOpcode::G_FADD:
    // case TargetOpcode::G_FMUL:
    //   // Floating-point arithmetic on this target uses GPR (no FP regs)
    //   // On other targets (ARM NEON), these would use VR banks
    //   return getDefaultMappingGPR(MI);
    //
    // case TargetOpcode::G_LOAD:
    // case TargetOpcode::G_STORE:
    //   // Memory operations: address is GPR, data type determines bank
    //   // return getDefaultMappingLoadStore(MI);
    //   break;
    //
    // default:
    //   break;
    // }

    // For unrecognized instructions, return a default mapping
    // return getDefaultMapping(MI);
    static InstructionMapping dummy(0, 0, {});
    return dummy;
  }

  // Helper: create a mapping where all operands use GPR banks
  const InstructionMapping &getDefaultMappingGPR(/* MI */) const {
    // SmallVector<const ValueMapping *, 4> OpBanks;
    // for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
    //   LLT Ty = MRI.getType(MI.getOperand(i).getReg());
    //   OpBanks.push_back(getValueMapping(Ty));
    // }
    // return getInstructionMapping(DefaultMapping, 1, OpBanks, MI.getNumOperands());
    static InstructionMapping dummy(0, 0, {});
    return dummy;
  }

  // Choose ValueMapping based on LLT size
  const ValueMapping *getValueMapping(/* LLT Ty */) const {
    // unsigned Size = Ty.getSizeInBits();
    // if (Size <= 16) return &VM_GPR16;
    // if (Size <= 32) return &VM_GPR32;
    // if (Size <= 64) return &VM_2xGPR32;
    // return &VM_VR128;
    return &VM_GPR32;
  }

  // ============================================================
  // 6. getRegBankFromRegClass
  // ============================================================

  const RegisterBank &getRegBankFromRegClass(
      /* const TargetRegisterClass &RC */) const {
    // for (auto &RB : RegBanks) {
    //   if (RB.covers(RC.getID()))
    //     return RB;
    // }
    // llvm_unreachable("Register class not covered by any register bank");
    return GPRB;
  }
};

// ============================================================
// 7. Cross-Register-Bank Copy Optimization
// ============================================================

void optimizeCrossBankCopies() {
  // The RegBankSelect pass can rewrite instructions to avoid
  // copies between different register banks.

  // Example: inserting an integer into a vector
  //
  // Original G_MIR (assume GPR for integers, VR for vectors):
  //   %val:gprb(s32) = ...
  //   %vec:vr(<4 x s32>) = ...
  //   %idx:gprb(s32) = G_CONSTANT i32 2
  //   %newvec:vr(<4 x s32>) = G_INSERT_VECTOR_ELT %vec, %val, %idx
  //   // Problem: %val is on GPR, but G_INSERT_VECTOR_ELT expects VR
  //   // -> Need cross-bank copy: COPY %val -> %val_on_vr
  //
  // Optimized: scalarize the insert operation
  //   // The insert is decomposed into G_EXTRACT_VECTOR_ELT operations
  //   // that operate on GPRs, avoiding the cross-bank copy.
  //   // However, this creates more instructions -> cost model decides.

  // The cost model weighs:
  //   - Cost of cross-register-bank copies (usually expensive)
  //   - Cost of the alternative instruction sequence
  //   - Which bank the value is "naturally" on based on its uses

  // In simple (non-optimizing) mode:
  //   - Just assign register banks based on instruction type
  //   - G_ADD -> GPR, G_FADD -> VR, etc.
  //   - Cross-bank copies inserted as needed (coalesced later if possible)
}

// ============================================================
// 8. Connecting RegisterBankInfo to Subtarget
// ============================================================

void connectToSubtarget() {
  // In your XXXSubtarget class:
  //
  // class H2BLBSubtarget : public H2BLBGenSubtargetInfo {
  //   std::unique_ptr<H2BLBRegisterBankInfo> RegBankInfo;
  // public:
  //   H2BLBSubtarget() {
  //     RegBankInfo.reset(new H2BLBRegisterBankInfo(*this));
  //   }
  //   const RegisterBankInfo *getRegBankInfo() const override {
  //     return RegBankInfo.get();
  //   }
  // };
  //
  // The RegBankSelect pass automatically finds it via
  // MF.getSubtarget().getRegBankInfo().
}

// ============================================================
// 9. ValueMapping Examples (visual)
// ============================================================

void valueMappingExamples() {
  // Example 1: A 32-bit integer value on GPR
  //   ValueMapping: [{0, 32, GPR}]
  //   Value lives in one 32-bit GPR.

  // Example 2: A 64-bit value on 32-bit GPRs
  //   ValueMapping: [{0, 32, GPR}, {32, 32, GPR}]
  //   Low 32 bits in one GPR, high 32 bits in another GPR.

  // Example 3: A <2 x i32> vector on a 128-bit VR
  //   ValueMapping: [{0, 64, VR}]
  //   Both elements packed in one 64-bit VR.

  // Example 4: A <2 x i32> vector on two 32-bit GPRs
  //   ValueMapping: [{0, 32, GPR}, {32, 32, GPR}]
  //   First element in one GPR, second element in another GPR.

  // The getInstrMapping method is responsible for returning
  // the appropriate mapping for each instruction, considering
  // the type of each operand.
}
