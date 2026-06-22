// legalization_actions.cpp - Legalization Actions Reference
//
// Demonstrates all the legalization actions available in LLVM with
// concrete before/after examples for each action.

#include <string>
#include <vector>

// ============================================================
// Legalization Actions Illustrated
// ============================================================

struct ActionExample {
  std::string ActionName;
  std::string InputCode;       // Pseudo-IR before legalization
  std::string OutputCode;      // Pseudo-IR after legalization
  std::string Description;
};

std::vector<ActionExample> createExamples() {
  return {
    // ==========================================================
    // Legal
    // ==========================================================
    {
      "Legal",
      "%res:i32 = add i32 %a, %b",
      "%res:i32 = add i32 %a, %b  (unchanged)",
      "Operation is natively supported by the target. No transformation needed."
    },

    // ==========================================================
    // Widen Scalar
    // ==========================================================
    {
      "WidenScalar",
      "%res:i16 = urem i16 %a, %b",
      "%a_ext:i32 = anyext i32 %a\n"
      "%b_ext:i32 = anyext i32 %b\n"
      "%res_ext:i32 = urem i32 %a_ext, %b_ext\n"
      "%res:i16 = trunc i16 %res_ext",
      "Widen the operation to a larger supported type. "
      "Inputs are extended (sign/zero/any), output is truncated back."
    },

    // ==========================================================
    // Narrow Scalar
    // ==========================================================
    {
      "NarrowScalar",
      "%res:i48 = and i48 %a, %b",
      "%a_lo:i32, %a_hi:i16 = split i48 %a\n"
      "%b_lo:i32, %b_hi:i16 = split i48 %b\n"
      "%res_lo:i32 = and i32 %a_lo, %b_lo\n"
      "%res_hi:i16 = and i16 %a_hi, %b_hi\n"
      "%res:i48 = concat i32 %res_lo, i16 %res_hi",
      "Break down the operation into smaller supported types. "
      "Used for types larger than the widest legal type."
    },

    // ==========================================================
    // Fewer Elements (vector)
    // ==========================================================
    {
      "FewerElements",
      "%res:<5 x i8> = mul %a:<5 x i8>, %b:<5 x i8>",
      "%a0, %a1, %a2, %a3, %a4 = unmerge %a\n"
      "%b0, ... = unmerge %b\n"
      "%r0 = mul i8 %a0, %b0\n"
      "... 4 more scalar muls ...\n"
      "%res = build_vector %r0, %r1, %r2, %r3, %r4",
      "Break vector operation into smaller pieces. "
      "Can scalarize to individual elements or split into smaller vectors."
    },

    // ==========================================================
    // More Elements (vector)
    // ==========================================================
    {
      "MoreElements",
      "%res:<3 x i8> = and %a:<3 x i8>, %b:<3 x i8>",
      "%a0, %a1, %a2 = unmerge %a\n"
      "%a_wide = build_vector %a0, %a1, %a2, undef\n"
      "%b0, %b1, %b2 = unmerge %b\n"
      "%b_wide = build_vector %b0, %b1, %b2, undef\n"
      "%r_wide:<4 x i8> = and %a_wide, %b_wide\n"
      "%r0, %r1, %r2, _ = unmerge %r_wide\n"
      "%res = build_vector %r0, %r1, %r2",
      "Promote vector to a larger vector type (e.g., next power of 2). "
      "New elements filled with undef, then discarded after the op."
    },

    // ==========================================================
    // Bitcast
    // ==========================================================
    {
      "Bitcast",
      "%res:i16 = and <2 x i8> %a, %b",
      "%a_bitcast:i16 = bitcast <2 x i8> %a to i16\n"
      "%b_bitcast:i16 = bitcast <2 x i8> %b to i16\n"
      "%res_bitcast:i16 = and i16 %a_bitcast, %b_bitcast\n"
      "%res:<2 x i8> = bitcast i16 %res_bitcast to <2 x i8>",
      "Change the type of the operation while keeping the same bitwidth. "
      "Useful for targets that support the operation on one type but not another."
    },

    // ==========================================================
    // Lower
    // ==========================================================
    {
      "Lower",
      "%res:i32 = urem i32 %a, %b",
      "%div:i32 = udiv i32 %a, %b\n"
      "%mul:i32 = mul i32 %div, %b\n"
      "%res:i32 = sub i32 %a, %mul",
      "Break down the operation into a sequence of simpler instructions. "
      "Each target defines the exact expansion for each opcode."
    },

    // ==========================================================
    // LibCall
    // ==========================================================
    {
      "LibCall",
      "%res:f32 = fadd f32 %a, %b",
      "%res:f32 = call f32 @__addsf3(f32 %a, f32 %b)",
      "Replace the operation with a call to a runtime library function. "
      "Common for floating-point operations on targets without FPU. "
      "Function names follow compiler-rt naming conventions."
    },

    // ==========================================================
    // Custom
    // ==========================================================
    {
      "Custom",
      "%res:i32 = mul i32 %a, %b",
      "// Target-specific C++ code runs.\n"
      "// May produce: target-specific widening mul,\n"
      "// or custom instruction sequence.",
      "Hand-written C++ implementation for a specific (opcode, type) pair. "
      "Gives full control over the generated code. "
      "Falls back to Expand if custom lowering returns failure."
    },
  };
}

// ============================================================
// SDISel Action Name Mapping
// ============================================================

// SDISel groups actions into fewer categories:
//
//   SDISel Name    | GlobalISel Actions
//   ---------------+-------------------
//   Legal          | Legal
//   Promote        | WidenScalar, MoreElements
//   Expand         | NarrowScalar, FewerElements, Lower
//   LibCall        | LibCall
//   Custom         | Custom
//
// Additionally, SDISel has type-specific setter methods:
//   - setOperationAction(Opcode, Type, Action)
//   - setLoadExtAction(ExtType, ValType, MemType, Action)
//   - setTruncStoreAction(ValType, MemType, Action)
//   - setCondCodeAction(CondCode, Type, Action)
//   - setIndexedLoadAction(IdxMode, Type, Action)
//   - setIndexedStoreAction(IdxMode, Type, Action)

// ============================================================
// Legalization Artifacts Guide
// ============================================================

void legalizationArtifacts() {
  // These are the "glue" instructions that legalization creates
  // around the core operation. They must themselves be legalized.

  // Extension (widen scalar):
  //   SIGN_EXTEND, ZERO_EXTEND, ANY_EXTEND, FP_EXTEND
  //
  // Truncation (narrow scalar):
  //   TRUNCATE, FP_ROUND
  //
  // Bitcast:
  //   BITCAST
  //
  // Vector construction/deconstruction:
  //   BUILD_VECTOR, EXTRACT_VECTOR_ELT, INSERT_VECTOR_ELT
  //   CONCAT_VECTORS, EXTRACT_SUBVECTOR
  //
  // Scalar split/merge:
  //   EXTRACT_ELEMENT (from merged value)
  //   BUILD_PAIR (merge two values)
  //
  // Undef (for filling vector lanes):
  //   UNDEF

  // The key insight: artifacts often cancel out.
  //   trunc(anyext(x)) == x  (when types match)
  //   This cancellation happens in DAGCombiner/optimization passes.
  //
  // Focus on where the chain of artifacts originates and terminates:
  //   - ABI boundaries (arguments, return values, calls)
  //   - Loads and stores
  // These are where values enter/leave the register file,
  // and where truncation/extension must ultimately be resolved.
}
