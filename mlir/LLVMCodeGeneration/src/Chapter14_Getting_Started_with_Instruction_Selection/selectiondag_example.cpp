// selectiondag_example.cpp - SDISel DAG representation and manipulation
//
// Demonstrates the core concepts of SelectionDAG: SDNode, SDValue, chain/glue
// dependencies, node creation, the DAG structure, and how instructions flow
// through the SDISel framework.

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// 1. Simulated SDISel type system
// ============================================================

struct EVT {
  enum Kind { Simple, Extended };
  Kind TheKind;
  unsigned SizeInBits;

  static EVT getIntegerVT(unsigned BitWidth) { return {Simple, BitWidth}; }
  static EVT i16() { return {Simple, 16}; }
  static EVT i32() { return {Simple, 32}; }
  bool isSimple() const { return TheKind == Simple; }
  unsigned getSizeInBits() const { return SizeInBits; }
};

struct MVT {
  static constexpr unsigned i16 = 16;
  static constexpr unsigned i32 = 32;
  static constexpr unsigned Other = 0xFF;  // chain
  static constexpr unsigned Glue = 0xFE;   // glue
};

// ============================================================
// 2. SDNode and SDValue (simplified)
// ============================================================

// Forward declarations
struct SDNode;
struct SelectionDAG;

// SDValue wraps (SDNode*, result index)
// In real LLVM, this is lightweight and passed by value.
struct SDValue {
  SDNode *Node;
  unsigned ResNo;

  SDValue() : Node(nullptr), ResNo(0) {}
  SDValue(SDNode *N, unsigned R) : Node(N), ResNo(R) {}

  // Convenience: SDValue() == failed optimization/no replacement
  explicit operator bool() const { return Node != nullptr; }

  EVT getValueType() const;
};

// An SDNode represents a single operation in the DAG.
struct SDNode {
  unsigned Opcode;
  std::vector<EVT> ValueTypes;       // One per result value
  std::vector<SDValue> Operands;     // Input operands (SDValues)

  SDNode(unsigned Opc, std::vector<EVT> VTs, std::vector<SDValue> Ops)
    : Opcode(Opc), ValueTypes(std::move(VTs)), Operands(std::move(Ops)) {}

  unsigned getNumValues() const { return static_cast<unsigned>(ValueTypes.size()); }
  EVT getValueType(unsigned ResNo) const { return ValueTypes[ResNo]; }
  SDValue getOperand(unsigned i) const { return Operands[i]; }

  // Check if this node already carries a target machine opcode
  bool isMachineOpcode() const { return false; }
};

EVT SDValue::getValueType() const {
  return Node ? Node->getValueType(ResNo) : EVT();
}

// ============================================================
// 3. SelectionDAG - the node factory with CSE
// ============================================================

struct SelectionDAG {
  std::vector<SDNode*> AllNodes;

  // Core node creation (with CSE: returns existing node if identical)
  SDValue getNode(unsigned Opcode, std::vector<EVT> VTs,
                  std::vector<SDValue> Ops) {
    // In real LLVM, this checks FoldingSet for CSE.
    // For simplicity, always create a new node here.
    auto *N = new SDNode(Opcode, std::move(VTs), std::move(Ops));
    AllNodes.push_back(N);
    // Return value 0 of the new node
    return SDValue(N, 0);
  }

  // Convenience: single result type
  SDValue getNode(unsigned Opcode, EVT VT, std::vector<SDValue> Ops) {
    return getNode(Opcode, {VT}, std::move(Ops));
  }

  // Get the Nth value of an SDNode (for chain/glue)
  static SDValue getValue(SDNode *N, unsigned R) { return SDValue(N, R); }

  // === Special node factories ===

  // CopyFromReg: reads a register value at basic block entry
  SDValue getCopyFromReg(SDValue Chain, /*SDLoc*/ int Loc,
                         unsigned Reg, EVT VT) {
    // Node returns [value(VT), chain(MVT::Other)]
    return getNode(/*ISD::CopyFromReg*/ 0, {VT, EVT::getIntegerVT(0)},
                   {Chain, SDValue(/*Register node*/ nullptr, Reg)});
  }

  // CopyToReg: writes a value to a register at basic block exit
  SDValue getCopyToReg(SDValue Chain, /*SDLoc*/ int Loc,
                       unsigned Reg, SDValue Val) {
    return getNode(/*ISD::CopyToReg*/ 1, {EVT::getIntegerVT(0), EVT::getIntegerVT(1)},
                   {Chain, Val, SDValue(nullptr, Reg)});
  }
};

// ============================================================
// 4. Demonstrated SelectionDAG construction
// ============================================================

// Example: Build a DAG that represents:
//   %0 = add i16 %a, %b
//   store i16 %0 to address %ptr
//
// Textual dump of the DAG (simplified):
//   t0: ch,glue = EntryToken
//     t2: i16,ch = CopyFromReg t0, Register:i16 %a
//     t4: i16,ch = CopyFromReg t0, Register:i16 %b
//   t5: i16 = add t2, t4
//   t7: ch = store t0, t5, t6

void buildSimpleDAG() {
  SelectionDAG DAG;

  // --- EntryToken: always the first node ---
  SDValue EntryToken = DAG.getNode(/*ISD::EntryToken*/ 100,
    {EVT::getIntegerVT(0), EVT::getIntegerVT(1)}, {});
  // EntryToken produces chain (value 0) and glue (value 1)
  SDValue Chain = EntryToken;  // value 0 implicitly
  SDValue Glue = SDValue(EntryToken.Node, 1);

  // --- CopyFromReg for arguments ---
  SDValue CopyA = DAG.getCopyFromReg(Chain, 0, /*RegA=*/1, EVT::i16());
  SDValue ValA = CopyA;               // value 0 = the i16 value

  SDValue CopyB = DAG.getCopyFromReg(Chain, 0, /*RegB=*/2, EVT::i16());
  SDValue ValB = CopyB;

  // --- ADD node: data dependency on ValA and ValB ---
  SDValue Add = DAG.getNode(/*ISD::ADD*/ 200, EVT::i16(), {ValA, ValB});
  // Add only produces a data value (no chain/glue).

  // CopyToReg is used to pass a result out of the basic block.
  // In this example, we store to memory instead.

  // Note on dependencies:
  // - Data edges: Add depends on ValA and ValB (use-def chain).
  // - Chain edges: Copies and stores depend on the chain for ordering.
  // - Glue edges: Used to stick instructions together (e.g., ABI code).

  (void)Add; // suppress unused warning

  // Real DAG dumps look like:
  //   t0: ch,glue = EntryToken
  //   t2: i16,ch = CopyFromReg t0, Register:i16 %0
  //   t4: i16,ch = CopyFromReg t0, Register:i16 %1
  //   t5: i16 = add t2, t4
  //   t7: ch,glue = CopyToReg t0, Register:i16 $r1, t5
}

// ============================================================
// 5. DAG dependency types explained
// ============================================================

void explainDependencies() {
  // --- Data Dependency (use-def) ---
  // Source of the edge (parent) is the USE.
  // Destination of the edge (child) is the DEF.
  //
  //   %res = add %a, %b
  //
  // SDNode representation:
  //   add_node has operands [%a_node, %b_node]
  //   Edge direction: add_node -> %a_node  (parent uses child's def)
  // This is USE-DEF chain, reversed from typical def-use.

  // --- Chain Dependency ---
  // Enforces ordering between ops that may alias or have side effects.
  //   store %addr1, %val1    // must stay before later loads/stores
  //   load %res, %addr2
  //
  // The store produces a chain value. The load consumes it.
  // If we can't prove addr1 != addr2, the chain keeps them in order.

  // --- Glue Dependency ---
  // Forces two instructions to be adjacent in the final instruction sequence.
  // Used when values must be live simultaneously or during ABI lowering.
  // For example, call arguments and the call itself must be adjacent.

  // In textual dumps:
  //   tX: type1,type2 = opcode ...
  //   - tX is a temporal name (not stable across runs)
  //   - type1,type2 are the result types
  //   - ch = chain, glue = glue
  //   - tX:1 means value index 1 of node tX
  //   - Register:i16 $r1 means a constant (physical register)
}

// ============================================================
// 6. SDISel phases within a basic block
// ============================================================

void illustrateSDISelPhases() {
  // For each basic block, SDISel runs these phases sequentially:
  //
  // Phase 1: SelectionDAGBuilder (IR Building)
  //   - Walks LLVM IR instructions
  //   - Creates SDNodes for each instruction
  //   - Handles ABI lowering (CopyFromReg/CopyToReg, call sequences)
  //
  // Phase 2: DAGCombine 1 (Pre-legalize optimization)
  //   - Runs combining rules
  //   - Folds patterns, eliminates redundancies
  //   - Target can inject custom combines via PerformDAGCombine()
  //
  // Phase 3: Type Legalization
  //   - Converts illegal types to legal types
  //   - e.g., i3 -> i8, <3 x i16> -> <4 x i16>
  //   - Introduces extension/truncation artifacts
  //
  // Phase 4: Operation Legalization
  //   - Legalizes operations on legal types
  //   - e.g., fadd f32 -> libcall if no FPU
  //   - e.g., mul i64 -> expand to i32 ops if no 64-bit multiplier
  //
  // Phase 5: DAGCombine 2 (Post-legalize optimization)
  //   - Cleans up legalization artifacts
  //   - More pattern folding
  //
  // Phase 6: Instruction Selection
  //   - Matches SDNodes to target instructions via SelectCode()
  //   - Generated by TableGen (gen-dag-isel backend)
  //   - Replaces generic SDNodes with machine-specific ones
  //
  // Phase 7: Scheduling
  //   - Linearizes the DAG into a MachineBasicBlock
  //   - Orders instructions respecting all dependencies
  //   - Creates terminator instructions
}

// ============================================================
// 7. Key takeaway: the DAG is per basic block
// ============================================================

void perBasicBlockScope() {
  // SDISel cannot look across basic blocks.
  // All values in the DAG are local to the current basic block.
  //
  // Values flow between basic blocks via:
  //   - PHI nodes at Machine IR level (not visible in DAG)
  //   - CopyFromReg / CopyToReg at entry/exit
  //
  // This is a fundamental limitation: SDISel cannot fold a sext in BB1
  // with a mul in BB2. GlobalISel can (function scope).
  //
  // Workaround: CodeGenPrepare pass duplicates/sinks instructions before
  // SDISel runs to expose cross-BB patterns.
}
