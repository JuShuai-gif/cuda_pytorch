// legalization_globalisel.cpp - GlobalISel Legalization Examples
//
// Demonstrates:
//   - LegalizerInfo class and LegalizeRuleSet API
//   - Programmatic legalization rules (clampScalar, legalIf, lowerIf, etc.)
//   - Custom legalization (legalizeCustom)
//   - Key differences from SDISel

#include <cassert>
#include <cstdint>
#include <functional>
#include <vector>

// ============================================================
// 1. Simulated GlobalISel types
// ============================================================

struct LLT {
  enum Kind { Scalar, Pointer, Vector };
  Kind K;
  unsigned SizeInBits;
  unsigned NumElements;

  static LLT scalar(unsigned Size) { return {Scalar, Size, 0}; }
  static LLT pointer(unsigned /*AS*/, unsigned Size) { return {Pointer, Size, 0}; }
  static LLT vector(unsigned NumElem, LLT ElemTy) {
    return {Vector, ElemTy.SizeInBits * NumElem, NumElem};
  }

  bool isScalar() const { return K == Scalar; }
  bool isPointer() const { return K == Pointer; }
  bool isVector() const { return K == Vector; }
  unsigned getSizeInBits() const { return SizeInBits; }
};

// Predefined types for convenience
static const LLT s8  = LLT::scalar(8);
static const LLT s16 = LLT::scalar(16);
static const LLT s32 = LLT::scalar(32);
static const LLT p0  = LLT::pointer(0, 16);

// ============================================================
// 2. LegalityQuery - describes the instruction being legalized
// ============================================================

struct LegalityQuery {
  unsigned Opcode;
  std::vector<LLT> Types;             // Types at each type index
  struct MemDesc {
    LLT MemoryTy;      // The type stored in memory
    uint64_t Align;    // Alignment
  };
  std::vector<MemDesc> MMODescrs;     // Memory descriptors (for load/store)
};

// A LegalityPredicate is a function: (LegalityQuery) -> bool
using LegalityPredicate = std::function<bool(const LegalityQuery&)>;

// ============================================================
// 3. LegalizeRuleSet - the core API for defining rules
// ============================================================

class LegalizeRuleSet {
public:
  // Mark specific (type, addrType, memType, align) tuples as legal.
  // Used for G_LOAD, G_STORE, etc.
  struct TypePairAndMemDesc {
    LLT Type;       // The value type
    LLT AddrType;   // The address/pointer type
    LLT MemType;    // The type stored in memory
    uint64_t Align; // Alignment
  };

  LegalizeRuleSet& legalForTypesWithMemDesc(std::vector<TypePairAndMemDesc> Tuples) {
    // Each tuple describes a legal load/store variant.
    // Example: {s16, p0, s8, 8} means:
    //   - Load: load 8 bits from memory, get 16-bit register value (extending load)
    //   - Store: store 8 bits to memory from 16-bit register value (truncating store)
    //   - Address space 0, byte-aligned
    return *this;
  }

  // Allow the output type (typeIdx 0) to be in [Min, Max] range.
  // Anything outside this range will be widened or narrowed.
  LegalizeRuleSet& clampScalar(unsigned TypeIdx, const LLT &Min, const LLT &Max) {
    // Example: clampScalar(0, s16, s32) on G_ADD means:
    //   - s8  -> promote to s16 (widen)
    //   - s16 -> legal (in range)
    //   - s32 -> legal (in range)
    //   - s64 -> narrow to s32 (expand to 2x s32 ops)
    return *this;
  }

  // Lower the operation if the predicate returns true.
  LegalizeRuleSet& lowerIf(LegalityPredicate Pred) {
    // Lowering replaces the instruction with a sequence of simpler
    // instructions. For G_UREM, it becomes G_UDIV + G_MUL + G_SUB.
    // The exact expansion depends on the opcode.
    return *this;
  }

  // Mark the operation as legal if the predicate returns true.
  LegalizeRuleSet& legalIf(LegalityPredicate Pred) {
    // The operation passes through unchanged.
    return *this;
  }

  // Scalarize: break a vector operation into per-element scalar operations.
  // TypeIdx specifies which type index carries the vector type.
  LegalizeRuleSet& scalarize(unsigned TypeIdx) {
    // Example: G_ADD <3 x s16> with scalarize(0):
    //   Before: %vec = G_ADD <3 x s16> %a, %b
    //   After:
    //     %e0, %e1, %e2 = G_UNMERGE_VALUES %a
    //     %f0, %f1, %f2 = G_UNMERGE_VALUES %b
    //     %r0 = G_ADD s16 %e0, %f0
    //     %r1 = G_ADD s16 %e1, %f1
    //     %r2 = G_ADD s16 %e2, %f2
    //     %vec = G_BUILD_VECTOR %r0, %r1, %r2
    return *this;
  }

  // Unconditional lower.
  LegalizeRuleSet& lower() {
    return *this;
  }

  // Custom legalization predicate.
  LegalizeRuleSet& customIf(LegalityPredicate Pred) {
    return *this;
  }

  // Widen scalar to next power of 2, starting at a minimum size.
  LegalizeRuleSet& widenScalarToNextPow2(unsigned TypeIdx, unsigned MinSize) {
    // Example: widenScalarToNextPow2(0, 8) on G_ADD:
    //   - s1..s7  -> s8
    //   - s9..s15 -> s16
    //   - s17..s31 -> s32
    return *this;
  }

  // Reduce the number of vector elements to the next power of 2.
  LegalizeRuleSet& fewerElementsToNextPow2(unsigned TypeIdx, unsigned Min) {
    // Example: fewerElementsToNextPow2(0, 2) on G_ADD:
    //   - <3 x s16> -> <2 x s16> + s16
    return *this;
  }
};

// ============================================================
// 4. LegalizerInfo - the target's legalization rules
// ============================================================

class H2BLBLegalizerInfo {
public:
  // Called from the constructor: register all legalization rules.
  H2BLBLegalizerInfo() {

    // Pre-define common LLTs for readability
    const LLT s8  = LLT::scalar(8);
    const LLT s16 = LLT::scalar(16);
    const LLT s32 = LLT::scalar(32);
    const LLT p0  = LLT::pointer(0, 16);

    // --- Rule 1: Load/Store ---
    // getActionDefinitionsBuilder returns a LegalizeRuleSet for the given opcodes.
    getActionDefinitionsBuilder(/*G_LOAD, G_STORE*/)
      // Explicitly legal variants:
      .legalForTypesWithMemDesc({
        {s8,  p0, s8,  8},   // Load/store 8-bit, unaligned
        {s16, p0, s8,  8},   // Extending load / truncating store
        {s16, p0, s16, 8},   // Load/store 16-bit, unaligned
        {s32, p0, s32, 8},   // Load/store 32-bit, unaligned
      })
      // Allow scalar types between s16 and s32
      .clampScalar(0, s16, s32)
      // Lower extending/truncating loads/stores that don't match
      .lowerIf([](const LegalityQuery &Q) {
        return Q.Types[0].isScalar() &&
               Q.Types[0] != Q.MMODescrs[0].MemoryTy;
      })
      // Legal if 16 or 32 bits (any non-trunc/ext)
      .legalIf([](const LegalityQuery &Q) {
        return Q.Types[0].getSizeInBits() == 16 ||
               Q.Types[0].getSizeInBits() == 32;
      })
      // Scalarize vector loads/stores
      .scalarize(0)
      // Lower anything else
      .lower();

    // --- Rule 2: Integer Arithmetic (G_ADD, G_SUB, G_AND, G_OR, G_XOR) ---
    getActionDefinitionsBuilder(/*G_ADD, G_SUB, G_AND, G_OR, G_XOR, G_SHL*/)
      // Allow s16 and s32 natively
      .legalFor({{s16}, {s32}})
      // Clamp to [s16, s32]: narrower types widened, wider types narrowed
      .clampScalar(0, s16, s32)
      // Scalarize vector variants
      .scalarize(0)
      // Lower anything not handled
      .lower();

    // --- Rule 3: G_MUL (custom legalization for widening multiply) ---
    getActionDefinitionsBuilder(/*G_MUL*/)
      // Custom legalization: when destination is scalar 32-bit
      .customIf([](const LegalityQuery &Q) {
        const auto &DstTy = Q.Types[0];
        return !DstTy.isVector() && DstTy.getSizeInBits() == 32;
      })
      // Legal for s16 multiply (native 16-bit multiplier)
      .legalFor({{s16}})
      // Clamp to [s16, s16]: everything else is lowered
      .clampScalar(0, s16, s16);

    // --- Rule 4: G_SEXT (sign extension) ---
    getActionDefinitionsBuilder(/*G_SEXT*/)
      .legalFor({{s16, s8},  {s32, s8},   // sext i8 to i16/i32
                  {s32, s16}})             // sext i16 to i32
      .clampScalar(0, s16, s32)
      .clampScalar(1, s8, s16);

    // --- Finalization: REQUIRED ---
    // Build the internal decision tables for fast lookup.
    // computeTables();  // getLegacyLegalizerInfo().computeTables();
  }

  // --- Custom legalization implementation ---

  // Called for ops with custom rules.
  // Helper gives access to MIRBuilder, Observer, and utility methods.
  // Must return true on success, false on failure.

  bool legalizeCustom(/*LegalizerHelper &Helper, MachineInstr &MI,
                        LostDebugLocObserver &LocObserver*/) {
    // switch (MI.getOpcode()) {
    // case TargetOpcode::G_MUL:
    //   return legalizeMul(MI, MRI, MIRBuilder, Observer);
    // }
    return true;
  }

  bool legalizeMul(/*MachineInstr &MI, ...*/) {
    // --- Step 1: Pattern match the operands ---
    // We want to match: G_MUL s32 %a, %b
    // where %a is defined by G_SEXT (or G_ZEXT) from s16
    // and %b is defined by G_SEXT (or G_ZEXT) from s16
    //
    // bool isSigned = false;
    // if (mi_match(LHS, MRI, m_GSExt(m_Reg(PlainLHS))) &&
    //     mi_match(RHS, MRI, m_GSExt(m_Reg(PlainRHS)))) {
    //   isSigned = true;
    // }

    // --- Step 2: Choose the target instruction ---
    // unsigned Opcode = isSigned ? H2BLB::WIDENING_SMUL : H2BLB::WIDENING_UMUL;

    // --- Step 3: Morph the instruction ---
    // Observer.changingInstr(MI);         // Notify before change
    // MI.setDesc(TII.get(Opcode));        // Change opcode to target-specific
    // MI.RemoveOperand(...);              // Adjust operands
    // constrainSelectedInstRegOperands(...); // Set register classes
    // Observer.changedInstr(MI);          // Notify after change

    return true;
  }

private:
  // Simulated getActionDefinitionsBuilder (in real LLVM, this is on LegalizerInfo)
  LegalizeRuleSet& getActionDefinitionsBuilder(unsigned Opcode) {
    return ruleSet; // Simplified
  }
  LegalizeRuleSet& getActionDefinitionsBuilder(std::initializer_list<unsigned>) {
    return ruleSet;
  }
  LegalizeRuleSet ruleSet;
};

// ============================================================
// 5. Type index explanation
// ============================================================

void explainTypeIndex() {
  // When describing rules, you reference types by their index in
  // the instruction's type list. Each generic opcode has a defined
  // set of types (from OutOperandList / InOperandList in GenericOpcodes.td).

  // Examples:
  //
  // G_ADD: 1 type (type index 0 only)
  //   - Result and both operands must have the same type.
  //   - OutOperandList = (outs type0:$dst)
  //   - InOperandList  = (ins type0:$src1, type0:$src2)
  //
  // G_SEXT: 2 types (type indices 0 and 1)
  //   - type0 = destination type (the extended type)
  //   - type1 = source type (the original type)
  //   - OutOperandList = (outs type0:$dst)
  //   - InOperandList  = (ins type1:$src)
  //
  // G_LOAD: 1 type + memory descriptor
  //   - type0 = the loaded value type
  //   - The memory type comes from the MachineMemOperand
  //
  // When you write clampScalar(0, s16, s32), you're constraining
  // type at index 0 to be between s16 and s32 (inclusive).
}

// ============================================================
// 6. LegalizeMutation - customizing the lowering
// ============================================================

void legalizeMutations() {
  // LegalizeMutation instances let you customize HOW an action is applied.
  // They are passed to lowerIf, widenScalarIf, etc.

  // Example mutations:
  //   moreElementsToNextPow2(typeIdx, minElements)
  //     - When widening a vector, add elements to reach next power of 2.
  //     - <3 x s16> -> <4 x s16>
  //
  //   widenScalarOrEltToNextPow2(typeIdx, minSize)
  //     - Widen scalar or widen element type to next power of 2.
  //
  //   changeTo(typeIdx, LLT)
  //     - Change a type to a specific LLT.

  // Usage pattern:
  //   .widenScalarIf(predicate, typeIdx, mutation)
  // Where mutation describes HOW to widen.
}

// ============================================================
// 7. Key differences: SDISel vs GlobalISel legalization
// ============================================================

void compareLegalization() {
  // SDISel:
  // - Two-pass: type legalization first, then operation legalization.
  // - Legal types are declared via addRegisterClass().
  // - Operations on legal types are legal by default -> explicit actions needed.
  // - Fixed type set (MVT enum).
  // - Custom: LowerOperation() in TargetLowering.
  //
  // GlobalISel:
  // - Single pass: type and operation legalized together.
  // - No concept of "legal type" -> EVERY (opcode, type) must be covered.
  // - Programmatic rules via LegalizeRuleSet (powerful predicates).
  // - Infinite type space (any LLT::scalar(n)) handled by clampScalar etc.
  // - Custom: legalizeCustom() in LegalizerInfo.
  //
  // The GlobalISel approach is more flexible but requires more upfront
  // description. SDISel is simpler for common cases but rigid.
}

// ============================================================
// 8. LegalityPredicate helpers
// ============================================================

// The LegalityPredicates namespace provides utility functions to
// compose more complex predicates:

namespace LegalityPredicates {
  // Match if any of the given predicates return true (OR).
  auto any(std::initializer_list<LegalityPredicate> Preds) -> LegalityPredicate {
    return [Preds](const LegalityQuery &Q) {
      for (auto &P : Preds)
        if (P(Q)) return true;
      return false;
    };
  }

  // Match if all predicates return true (AND).
  auto all(std::initializer_list<LegalityPredicate> Preds) -> LegalityPredicate {
    return [Preds](const LegalityQuery &Q) {
      for (auto &P : Preds)
        if (!P(Q)) return false;
      return true;
    };
  }

  // Match a specific type at a given type index.
  auto typeIs(unsigned TypeIdx, LLT Ty) -> LegalityPredicate {
    return [TypeIdx, Ty](const LegalityQuery &Q) {
      return Q.Types[TypeIdx].getSizeInBits() == Ty.getSizeInBits();
    };
  }
}

// Example complex predicate:
void complexPredicateExample() {
  // lowerIf(any({
  //   // Lower if: (type is not s16 or s32) AND type is scalar
  //   all({ not(typeIs(0, s16)), not(typeIs(0, s32)), typeIsScalar(0) }),
  //   // Lower if: type is a vector
  //   typeIsVector(0),
  // }));
}
