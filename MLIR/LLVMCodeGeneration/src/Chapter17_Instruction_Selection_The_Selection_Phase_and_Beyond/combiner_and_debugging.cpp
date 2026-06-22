// combiner_and_debugging.cpp - DAGCombiner, GICombiner, and Debugging
//
// Demonstrates:
//   - SDISel DAGCombiner (PerformDAGCombine)
//   - GlobalISel GICombiner
//   - MachineFunction finalization
//   - Debugging techniques for instruction selectors

#include <cassert>
#include <string>
#include <vector>

// ============================================================
// 1. SDISel DAGCombiner
// ============================================================

// The DAGCombiner runs before AND after legalization in SDISel.
// It applies combining rules to optimize the DAG.

struct SDNode { unsigned Opcode; };
struct SDValue { SDNode *Node; };
struct SelectionDAG {};

class DAGCombinerInfo {
  // Holds context about the current combine:
  //   - Which DAGCombine pass is running (pre- or post-legalize)
  //   - Whether we're at the top level
  //   - Debug location
public:
  bool isBeforeLegalize() const { return true; }
  bool isAfterLegalize() const { return false; }
};

// TargetLowering::PerformDAGCombine - the target hook for custom combines.
class H2BLBTargetLowering {
public:
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI/*, SelectionDAG &DAG*/) {
    // The combiner traverses the DAG and for each node, calls this hook.
    // Return SDValue() if no combine applies; return a replacement node otherwise.

    // switch (N->getOpcode()) {
    // case ISD::ADD:
    //   return combineAdd(N, DAG, DCI);
    // case ISD::LOAD:
    //   return combineLoad(N, DAG, DCI);
    // case H2BLBISD::WIDENING_SMUL:
    //   return combineWideningMul(N, DAG, DCI);
    // default:
    //   break;
    // }
    // return SDValue();
    return SDValue();
  }

  // Example combine rules:

  // Combine 1: Fold (add x, 0) -> x
  SDValue combineAdd(SDNode *N/*, SelectionDAG &DAG, DAGCombinerInfo &DCI*/) {
    // if (isNullConstant(N->getOperand(1)))
    //   return N->getOperand(0);  // add x, 0 = x
    // if (isNullConstant(N->getOperand(0)))
    //   return N->getOperand(1);  // add 0, x = x
    return SDValue();
  }

  // Combine 2: Fold (load (frameindex)) into addressing mode
  SDValue combineLoad(SDNode *N/*, SelectionDAG &DAG, DAGCombinerInfo &DCI*/) {
    // If the address is a frame index and we have a target-specific
    // addressing mode that can fold the frame index, do it.
    return SDValue();
  }

  // Combine 3: Fold chain of operations after legalization
  SDValue combineWideningMul(SDNode *N/*, SelectionDAG &DAG, DAGCombinerInfo &DCI*/) {
    // if (DCI.isAfterLegalize()) {
    //   // Post-legalize: try to fold extensions with the widening mul
    // }
    return SDValue();
  }
};

// ============================================================
// 2. Generic DAGCombiner Rules (Built-in)
// ============================================================

void builtinDAGCombines() {
  // LLVM's DAGCombiner has many built-in target-independent combines:
  //
  // Arithmetic:
  //   - (add x, (add y, z)) -> (add (add x, y), z)  // reassociate
  //   - (sub x, x) -> 0
  //   - (mul x, 2) -> (shl x, 1)  // multiply by constant
  //
  // Logical:
  //   - (and x, x) -> x
  //   - (or x, -1) -> -1
  //   - (xor x, x) -> 0
  //
  // Memory:
  //   - (load (store x, addr)) -> x  // store-to-load forwarding
  //   - (load (load addr)) -> undef  // dead load
  //
  // Extensions:
  //   - (sext (zext x)) -> (sext x)  // redundant extensions
  //   - (trunc (sext x)) -> x        // cancel extension/truncation
  //
  // Vector:
  //   - (extractelt (build_vector x, y, ...), 0) -> x
  //

  // Target-specific combines are injected via PerformDAGCombine.
  // This is where you add your ISA-specific optimizations.
}

// ============================================================
// 3. GlobalISel GICombiner
// ============================================================

// The GICombiner is GlobalISel's optimization infrastructure.
// It runs after legalization (or can be inserted anywhere in the pipeline).

// TableGen-driven (newer approach):
/*
// In GICombiner.td:
def MyCombiner : GICombiner<"XXXGenGICombiner", [{
  // Match rules
  def match_add_zero : GICombineRule<
    (defs root:$root),
    (match (G_ADD $x, 0), reg:$root),
    (apply (COPY $root, $x))
  >;

  def match_sext_fold : GICombineRule<
    (defs root:$root),
    (match (G_SEXT (G_TRUNC $x)), reg:$root),
    (apply (COPY $root, $x))
  >;
}]> {
  let CombineAllMethodName = "tryCombineAllImpl";
}
*/

// C++-driven (manual approach):
class GICombinerExample {
public:
  bool tryCombine(/* MachineInstr &MI */) {
    // Switch on opcode and try various patterns:
    //
    // case TargetOpcode::G_ADD:
    //   // (G_ADD x, 0) -> x
    //   if (isNullConstant(MI.getOperand(2)))
    //     return replaceWithCopy(MI);
    //   // (G_ADD x, x) -> (G_SHL x, 1)
    //   if (isSameRegister(MI.getOperand(1), MI.getOperand(2)))
    //     return convertToShift(MI);
    //   break;
    //
    // case TargetOpcode::G_AND:
    //   // (G_AND x, -1) -> x
    //   if (isAllOnesConstant(MI.getOperand(2)))
    //     return replaceWithCopy(MI);
    //   break;

    return false;
  }
};

// ============================================================
// 4. MachineFunction Finalization
// ============================================================

// After selection, the MachineFunction must be finalized before
// later passes (register allocation, etc.) can consume it.

void finalizationSteps() {
  // In SDISel (happens automatically inside SelectionDAGISel pass):
  //   - PHI nodes are translated from CopyFromReg/CopyToReg mappings
  //   - Dead nodes are removed
  //   - The DAG is linearized (scheduled) into a MachineBasicBlock
  //
  // In GlobalISel (InstructionSelect pass handles this):
  //   - All generic opcodes (G_*) are replaced with target instructions
  //   - All generic virtual registers get register classes
  //   - MachineFunction properties are updated:
  //     - IsSSA property set/cleared
  //     - Selected property set
  //

  // After finalization, the MachineFunction is ready for:
  //   - PHI elimination
  //   - Two-address instruction pass
  //   - Register allocation
  //   - Prologue/epilogue insertion
  //   - Machine code emission
}

// ============================================================
// 5. Debugging Instruction Selectors
// ============================================================

void debuggingTechniques() {
  // --- SDISel Debugging ---

  // 1. View DAG dumps (generates .dot files, opened with Graphviz):
  //   llc -view-dag-combine1-dags input.ll   # Pre-legalize DAGCombine
  //   llc -view-legalize-dags input.ll       # During legalization
  //   llc -view-dag-combine2-dags input.ll   # Post-legalize DAGCombine
  //   llc -view-isel-dags input.ll           # Before instruction selection
  //   llc -view-sched-dags input.ll          # Before scheduling

  // 2. Text debug output:
  //   llc -debug-only=isel input.ll          # All ISel debug output
  //   llc -debug-only=legalize-types input.ll
  //   llc -debug-only=legalize-ops input.ll

  // 3. Print before/after specific passes:
  //   llc -print-before=isel -print-after=isel input.ll

  // 4. Force specific fallback:
  //   llc -fast-isel=0 input.ll              # Disable FastISel
  //   llc -global-isel=1 input.ll            # Force GlobalISel
  //   llc -global-isel-abort=2 input.ll      # Abort on GlobalISel fallback

  // --- GlobalISel Debugging ---

  // 1. Debug output per pass:
  //   llc -debug-only=mir-isel input.ll
  //   llc -debug-only=gisel-legalizer input.ll
  //   llc -debug-only=gisel-regbankselect input.ll
  //   llc -debug-only=gisel-instructionselect input.ll

  // 2. Print MIR before/after passes:
  //   llc -print-before=irtranslator -print-after=irtranslator input.ll
  //   llc -print-before=legalizer -print-after=legalizer input.ll
  //   llc -print-before=regbankselect -print-after=regbankselect input.ll
  //   llc -print-before=instruction-select -print-after=instruction-select input.ll

  // 3. Run individual GlobalISel passes with .mir input:
  //   llc -run-pass=legalizer input.mir -o -
  //   llc -run-pass=regbankselect input.mir -o -
  //   llc -run-pass=instruction-select input.mir -o -

  // 4. Coverage tracking:
  //   llc -global-isel -global-isel-abort=2 input.ll
  //   # Reports statistics on which patterns were matched

  // --- Pattern Debugging ---

  // 1. Check which SDISel patterns failed to import:
  //   # Build with -warn-on-skipped-patterns for GlobalISelEmitter
  //   # Outputs warnings for each pattern that couldn't be imported

  // 2. Dump matched patterns:
  //   llc -debug-only=isel -print-machineinstrs input.ll
}

// ============================================================
// 6. .mir Testing (GlobalISel unit testing)
// ============================================================

void mirTesting() {
  // Each GlobalISel pass can be tested with .mir files:
  //
  //   # RUN: llc -mtriple=h2blb -run-pass=legalizer %s -o - | FileCheck %s
  //
  //   ---
  //   name: test_add_legalization
  //   body: |
  //     bb.0:
  //       liveins: $r1, $r2
  //       %0:_(s16) = COPY $r1
  //       %1:_(s16) = COPY $r2
  //       %2:_(s16) = G_ADD %0, %1
  //       ; CHECK: %2:gpr16 = ADD16 %0, %1
  //   ...
  //
  // This is a major advantage of GlobalISel over SDISel:
  // you can test individual passes independently instead of
  // going through the entire LLVM IR -> MIR pipeline.

  // For SDISel, the only practical way to test is:
  //   llc input.ll | FileCheck input.ll
  // which requires the entire SDISel pipeline to work.
}

// ============================================================
// 7. Full Selection Pipeline Comparison
// ============================================================

void fullPipelineComparison() {
  // SDISel Pipeline (inside a single MachineFunctionPass):
  //
  //   LLVM IR (Function)
  //     |
  //   [CodeGenPrepare - optional LLVM IR to LLVM IR pass]
  //     |
  //   SelectionDAGBuilder  (IR Building)
  //     |  Translates LLVM IR -> SDNode DAG per basic block
  //   DAGCombiner 1         (Pre-legalize optimization)
  //     |
  //   LegalizeTypes         (Type legalization)
  //     |
  //   LegalizeOps           (Operation legalization)
  //     |
  //   DAGCombiner 2         (Post-legalize optimization)
  //     |
  //   InstructionSelect     (Pattern matching -> target MIR)
  //     |
  //   Schedule              (DAG linearization -> MachineBasicBlock)
  //     |
  //   Machine IR (MachineFunction)

  // GlobalISel Pipeline (separate MachineFunctionPass instances):
  //
  //   LLVM IR (Function)
  //     |
  //   IRTranslator          (IR Building)
  //     |  Translates LLVM IR -> G_MIR
  //   [Optional custom passes]
  //     |
  //   Legalizer             (Legalization)
  //     |  Legalizes G_MIR operations and types
  //   [Optional custom passes]
  //     |
  //   RegBankSelect         (Register bank selection)
  //     |  Assigns register banks, optionally rewrites to avoid copies
  //   [Optional custom passes / GICombiner]
  //     |
  //   InstructionSelect     (Selection)
  //     |  Translates G_MIR -> target MIR
  //   [Finalization]
  //     |
  //   Machine IR (MachineFunction)

  // FastISel (runs inside SDISel per basic block):
  //
  //   LLVM IR (Instruction)
  //     |
  //   fastSelectInstruction(I)  -> MachineInstr or fallback
  //     |
  //   If fallback: SDISel finishes the basic block
}
