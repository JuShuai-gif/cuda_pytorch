// legalization_sdisel.cpp - SDISel Legalization Examples
//
// Demonstrates:
//   - Declaring legal types (addRegisterClass / computeRegisterProperties)
//   - Setting legalization actions (setOperationAction, etc.)
//   - Custom legalization (LowerOperation)
//   - The two-pass legalization: Type legalization then Operation legalization

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// 1. SDISel Legalization Actions (LegalizeAction enum)
// ============================================================

enum class LegalizeAction {
  Legal,      // Operation is natively supported
  Promote,    // Use a larger type (widen scalar / more vector elements)
  Expand,     // Use a simpler sequence (narrow scalar / fewer elements / lower)
  LibCall,    // Replace with a library call
  Custom,     // Custom C++ implementation in LowerOperation()
};

// ============================================================
// 2. ISD Operation opcodes (simulated subset)
// ============================================================

namespace ISD {
  enum Opcode {
    ADD, SUB, MUL, SDIV, UDIV, SREM, UREM,
    FADD, FSUB, FMUL, FDIV, FREM,
    LOAD, STORE,
    SIGN_EXTEND, ZERO_EXTEND, ANY_EXTEND, TRUNCATE,
    FP_EXTEND, FP_ROUND,
    BUILD_VECTOR, EXTRACT_VECTOR_ELT,
    // ... many more
  };
}

// ============================================================
// 3. Simulated MVT / EVT types
// ============================================================

struct MVT {
  enum Kind {
    i8, i16, i32, i64,
    f16, f32, f64,
    v2i16, v4i16, v2i32, v4i32,
    Other, Glue, // Chain and glue
  };
  Kind V;
  bool operator==(const MVT &O) const { return V == O.V; }
};

// ============================================================
// 4. TargetLowering: describes legal types and legalization actions
// ============================================================

class H2BLBTargetLowering {
public:
  // Legal types: types that the target supports natively in registers
  std::vector<MVT> LegalTypes;

  // Legalization actions: map (opcode, type) -> action
  struct ActionEntry {
    unsigned Opcode;
    MVT Type;
    LegalizeAction Action;
  };
  std::vector<ActionEntry> OpActions;

  // Similar for special operations (truncstore, etc.)
  std::vector<ActionEntry> TruncStoreActions;

  // ============================================================
  // Constructor: declare legal types and actions
  // ============================================================

  H2BLBTargetLowering() {
    // --- Step 1: Register legal types ---
    // Each legal type must be associated with a register class.
    //
    // In real LLVM:
    //   addRegisterClass(MVT::i16, &H2BLB::GPR16RegClass);
    //   addRegisterClass(MVT::i32, &H2BLB::GPR32RegClass);
    //   computeRegisterProperties(Subtarget.getRegisterInfo());
    LegalTypes.push_back(MVT::i16);
    LegalTypes.push_back(MVT::i32);

    // --- Step 2: Describe legalization actions ---
    //
    // For each (opcode, type) pair where the operation is NOT natively
    // supported on a legal type, you must set a legalization action.
    //
    // Marking a type as legal means ALL operations are Legal by default.
    // This is why you must explicitly mark unsupported operations.

    // We don't support floating-point addition on i32 type.
    // Action: use a library call (e.g., __addsf3 from libgcc/compiler-rt).
    OpActions.push_back({ISD::FADD, MVT::f32, LegalizeAction::LibCall});

    // We don't support integer multiplication on i32 (our ALU doesn't have it).
    // Action: custom lowering to our widening multiply instruction.
    OpActions.push_back({ISD::MUL, MVT::i32, LegalizeAction::Custom});

    // Truncating store: store i32 to i16 memory location.
    // Action: expand into TRUNC + STORE.
    TruncStoreActions.push_back({ISD::STORE, MVT::i32, LegalizeAction::Expand});

    // 64-bit integer type is not legal at all.
    // Since it's not registered, type legalization will expand i64 to 2x i32.
    // Then operation legalization handles the i32 op sequences.

    // 8-bit integer: not legal.
    // Type legalization promotes it to i16 (or i32).
  }

  // ============================================================
  // Helper: query the action for a given (opcode, type)
  // ============================================================

  LegalizeAction getOperationAction(unsigned Opcode, MVT Type) const {
    for (auto &Entry : OpActions) {
      if (Entry.Opcode == Opcode && Entry.Type == Type)
        return Entry.Action;
    }
    // Default: legal (if type is legal, operation is legal)
    return LegalizeAction::Legal;
  }

  // ============================================================
  // 5. Custom legalization: LowerOperation()
  // ============================================================
  //
  // When an (opcode, type) pair has Custom action, SDISel calls
  // LowerOperation() to get the replacement SDValue sequence.
  //
  // Return SDValue() (empty) to indicate failure.
  // Failure chain: Custom fails -> Expand -> LibCall -> abort.

  // Simulated SDValue
  struct SDValue {
    bool isValid() const { return valid; }
    bool valid = false;
    unsigned Opcode = 0;
  };

  SDValue LowerOperation(SDValue Op, /*SelectionDAG &DAG*/) {
    switch (Op.Opcode) {
    case ISD::MUL:
      return lowerMUL(Op);
    case ISD::SDIV:
      return lowerSDIV(Op);
    // ... other custom-lowered operations
    default:
      return SDValue(); // Failed -> fall back
    }
  }

  // Custom lowering for MUL i32:
  // We try to match it as a widening multiply (16-bit inputs -> 32-bit result).
  SDValue lowerMUL(SDValue Op) {
    assert(Op.Opcode == ISD::MUL);

    // Check if both operands are sign-extended 16-bit values.
    // bool isSigned = isSignExtended(Op.getOperand(0)) &&
    //                 isSignExtended(Op.getOperand(1));
    //
    // if (isSigned) {
    //   // Replace with our target-specific widening multiply
    //   // return DAG.getNode(H2BLBISD::WIDENING_SMUL, ...);
    // }

    // If we can't match a widening pattern, return empty -> fallback to Expand
    return SDValue();
  }

  SDValue lowerSDIV(SDValue Op) {
    (void)Op;
    // Custom division lowering example:
    // SDIV i32 might be expanded into a sequence using our divider unit
    return SDValue();
  }

  // ============================================================
  // 6. Type legalization hooks
  // ============================================================

  // Override to control vector type legalization.
  // Example: when we encounter <3 x i16>, how should it be legalized?
  enum LegalizeAction2 {
    Widen,    // Widen to <4 x i16>
    Promote,  // Promote elements: <3 x i32>
    Split,    // Split into <2 x i16> + i16
  };

  LegalizeAction2 getPreferredVectorAction(MVT VT) {
    // Default: LLVM picks the action. Override to customize.
    // Typical choice for <3 x i16>: widen to <4 x i16>
    // Typical choice for <5 x i32>: split to <2 x i32> + <3 x i32> -> widen
    return Widen;
  }
};

// ============================================================
// 7. The two-pass legalization in action
// ============================================================

void illustrateTwoPassLegalization() {
  // Pass 1: TYPE legalization
  //
  // For each SDNode, if its result type is not legal:
  //   1. Mark the node for type legalization.
  //   2. Check what the legal equivalent type should be.
  //   3. Replace uses with the legalized version.
  //
  // Example: add i8 on a target where only i16 and i32 are legal.
  //   Before:  %res:i8 = add i8 %a, %b
  //   After:   %a_promoted:i16 = any_extend i16 %a:i8
  //            %b_promoted:i16 = any_extend i16 %b:i8
  //            %res_promoted:i16 = add i16 %a_promoted, %b_promoted
  //            %res:i8 = truncate i8 %res_promoted:i16
  //
  // The extension/truncation are "legalization artifacts".
  // DAGCombine later optimizes them away when possible.

  // Pass 2: OPERATION legalization
  //
  // After type legalization, all types are legal.
  // Now legalize operations that don't exist on legal types.
  //
  // Example: fadd f32 on a target without FPU (but f32 is a legal type).
  //   Action: LibCall
  //   Before:  %res:f32 = fadd f32 %a, %b
  //   After:   %res:f32 = call f32 @__addsf3(%a, %b)
  //
  // Example: urem i32 (unsigned remainder) not supported natively.
  //   Action: Expand
  //   Before:  %res:i32 = urem i32 %a, %b
  //   After:   %div:i32 = udiv i32 %a, %b
  //            %mul:i32 = mul i32 %div, %b
  //            %res:i32 = sub i32 %a, %mul
  //   (urem = a - (a / b) * b)

  (void)0; // suppress unused warning
}

// ============================================================
// 8. Type promotion vs widening (SDISel terminology)
// ============================================================

void promoteVsWiden() {
  // In SDISel, "Promote" means using a larger type.
  // This covers two cases:

  // Case 1: Scalar promotion (WidenScalar in GlobalISel)
  //   i8 -> i16 (widen the scalar to the next legal size)
  //   The value is sign/zero extended.

  // Case 2: Vector promotion (MoreElements in GlobalISel)
  //   <3 x i16> -> <4 x i16> (widen number of elements)
  //   New elements are filled with undef.

  // In SDISel, "Expand" means using a simpler sequence.
  // This covers three cases:

  // Case 1: Scalar expansion (NarrowScalar in GlobalISel)
  //   i64 -> i32 + i32 (split into smaller legal types)
  //   Carry propagation for arithmetic.

  // Case 2: Vector splitting (FewerElements in GlobalISel)
  //   <8 x i16> -> <4 x i16> + <4 x i16>

  // Case 3: Operation lowering (Lower in GlobalISel)
  //   urem -> udiv + mul + sub
}

// ============================================================
// 9. Summary: SDISel Legalization Strategy Checklist
// ============================================================

void legalizationChecklist() {
  // [] 1. In TargetLowering constructor, call addRegisterClass() for each
  //       native type (determined by your ISA).
  // [] 2. Call computeRegisterProperties().
  // [] 3. For each operation on each legal type that is NOT natively
  //       supported, call setOperationAction().
  // [] 4. For special operations (trunc stores, extending loads), use
  //       setTruncStoreAction(), setLoadExtAction(), etc.
  // [] 5. If using Custom actions, implement LowerOperation().
  // [] 6. If Custom lowering may produce illegal types with opaque nodes,
  //       implement ReplaceNodeResults() and/or LowerOperationWrapper().
  // [] 7. Optionally override getPreferredVectorAction().
  // [] 8. Test with .ll inputs and check DAG dumps:
  //       llc -debug-only=isel -view-legalize-dags input.ll
}
