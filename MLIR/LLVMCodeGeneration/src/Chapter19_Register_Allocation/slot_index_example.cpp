//===----------------------------------------------------------------------===//
// Chapter 19 - Register Allocation
// Example: Slot Indexes - Instruction Numbering and Execution Slots
//===----------------------------------------------------------------------===//
//
// This example demonstrates the SlotIndex concept in LLVM's register allocator:
// - Continuous numbering of instructions in a MachineFunction
// - Four execution slots per index: Block, Early-Clobber, Register, Dead
// - Using slot indexes to model precise liveness boundaries
// - Maintaining the SlotIndex-to-MachineInstr mapping
//
// NOTE: This is a simulation for educational purposes. In LLVM, SlotIndexes
// is a real analysis pass (SlotIndexesWrapperPass / SlotIndexesAnalysis).
//

#include <cassert>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated Slot Index and Execution Slots
//------------------------------------------------------------------------------

enum class Slot {
  Block,          // 'B' - Beginning of basic block (nothing happens)
  EarlyClobber,   // 'e' - Early-clobber definitions start here
  Register,       // 'r' - Regular execution: defs start, uses die here
  Dead            // 'd' - Dead definitions, kept within same index
};

char slotToChar(Slot S) {
  switch (S) {
  case Slot::Block:        return 'B';
  case Slot::EarlyClobber: return 'e';
  case Slot::Register:     return 'r';
  case Slot::Dead:         return 'd';
  }
  return '?';
}

// A SlotIndex represents both an instruction and an execution stage
struct SlotIndex {
  unsigned Index;      // The numeric index (monotonically increasing)
  Slot Stage;          // Which execution stage

  SlotIndex(unsigned Idx = 0, Slot S = Slot::Block) : Index(Idx), Stage(S) {}

  // Comparison operators
  bool operator<(const SlotIndex &Other) const {
    if (Index != Other.Index) return Index < Other.Index;
    return static_cast<int>(Stage) < static_cast<int>(Other.Stage);
  }
  bool operator==(const SlotIndex &Other) const {
    return Index == Other.Index && Stage == Other.Stage;
  }
  bool operator<=(const SlotIndex &Other) const {
    return *this < Other || *this == Other;
  }

  // Check slot type
  bool isBlock() const { return Stage == Slot::Block; }
  bool isEarlyClobber() const { return Stage == Slot::EarlyClobber; }
  bool isRegister() const { return Stage == Slot::Register; }
  bool isDead() const { return Stage == Slot::Dead; }

  // Print format: "32r" (index 32, register slot)
  std::string toString() const {
    return std::to_string(Index) + slotToChar(Stage);
  }
};

//------------------------------------------------------------------------------
// Simulated MachineInstr
//------------------------------------------------------------------------------
struct SimMachineInstr {
  unsigned SlotIdx;            // Base SlotIndex value
  std::string Opcode;
  bool isEarlyClobber;         // Def cannot share reg with inputs
  std::vector<std::string> Defs;
  std::vector<std::string> Uses;
  bool isTerminator;           // Ends a basic block

  SimMachineInstr(unsigned Idx, const std::string &Opc, bool EC = false)
    : SlotIdx(Idx), Opcode(Opc), isEarlyClobber(EC), isTerminator(false) {}

  void addDef(const std::string &Reg) { Defs.push_back(Reg); }
  void addUse(const std::string &Reg) { Uses.push_back(Reg); }

  void print() const {
    std::cout << std::setw(4) << SlotIdx << "  ";
    if (!Defs.empty()) {
      for (size_t i = 0; i < Defs.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << Defs[i];
      }
      std::cout << " = ";
    }
    std::cout << Opcode;
    if (!Uses.empty()) {
      for (size_t i = 0; i < Uses.size(); ++i) {
        std::cout << (i == 0 ? " " : ", ") << Uses[i];
      }
    }
    if (isEarlyClobber) std::cout << " [early-clobber]";
    std::cout << "\n";
  }
};

//------------------------------------------------------------------------------
// Simulated SlotIndexes class
//------------------------------------------------------------------------------
class SimSlotIndexes {
private:
  std::map<unsigned, SimMachineInstr*> IndexToInstr;
  unsigned CurrentIndex;
  static const unsigned IndexGap = 16; // Default gap between indexes

public:
  SimSlotIndexes() : CurrentIndex(0) {}

  // Allocate the next slot index
  unsigned nextIndex() {
    CurrentIndex += IndexGap;
    return CurrentIndex;
  }

  // Register an instruction at its slot index
  void insertMachineInstrInMaps(SimMachineInstr *MI) {
    IndexToInstr[MI->SlotIdx] = MI;
  }

  // Remove mapping when instruction is deleted
  void removeMachineInstrFromMaps(SimMachineInstr *MI) {
    auto It = IndexToInstr.find(MI->SlotIdx);
    if (It != IndexToInstr.end()) {
      IndexToInstr.erase(It);
    }
  }

  // Get the instruction at a given slot index
  SimMachineInstr *getInstructionFromIndex(unsigned Idx) const {
    auto It = IndexToInstr.find(Idx);
    return (It != IndexToInstr.end()) ? It->second : nullptr;
  }

  // Get the starting slot index for a given index (Block slot)
  SlotIndex getStartIdx(unsigned Idx) const {
    return SlotIndex(Idx, Slot::Block);
  }

  // Get the register slot index for a given index
  SlotIndex getRegSlot(unsigned Idx) const {
    return SlotIndex(Idx, Slot::Register);
  }

  // Print all slot indexes with instructions
  void print() const {
    std::cout << "\n  SlotIndex Mapping:\n";
    for (auto &Pair : IndexToInstr) {
      SimMachineInstr *MI = Pair.second;
      std::cout << "    ";
      MI->print();
    }
  }

  // Helper to get related slot indexes for liveness modeling
  static SlotIndex getDefSlot(unsigned Idx, bool isEC) {
    return SlotIndex(Idx, isEC ? Slot::EarlyClobber : Slot::Register);
  }

  static SlotIndex getUseSlot(unsigned Idx) {
    return SlotIndex(Idx, Slot::Register);
  }

  static SlotIndex getDeadSlot(unsigned Idx) {
    return SlotIndex(Idx, Slot::Dead);
  }
};

//------------------------------------------------------------------------------
// Live Range representation using SlotIndex
//------------------------------------------------------------------------------
struct LiveRange {
  SlotIndex Start;    // Inclusive: '['
  SlotIndex End;      // Exclusive: ')'

  LiveRange(SlotIndex S, SlotIndex E) : Start(S), End(E) {}

  // Check if two live ranges overlap
  bool overlaps(const LiveRange &Other) const {
    // Ranges overlap if neither is completely before the other
    return Start < Other.End && Other.Start < End;
  }

  // Check if a point is within this range
  bool contains(SlotIndex Point) const {
    return Start <= Point && Point < End;
  }

  std::string toString() const {
    return "[" + Start.toString() + "," + End.toString() + ")";
  }
};

//------------------------------------------------------------------------------
// Main demonstration
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 19: Slot Indexes ===\n";

  SimSlotIndexes SI;

  // Build a simulated basic block with instructions
  std::cout << "\n--- Building Instructions with Slot Indexes ---\n";

  auto *I0 = new SimMachineInstr(0, "bb.0", false);
  I0->isTerminator = false;
  I0->SlotIdx = 0; // Basic block header

  auto *I1 = new SimMachineInstr(SI.nextIndex(), "COPY");
  I1->addDef("%0");
  I1->addUse("$w0");
  SI.insertMachineInstrInMaps(I1);
  std::cout << "  Created: ";
  I1->print();

  auto *I2 = new SimMachineInstr(SI.nextIndex(), "ADDri");
  I2->addDef("%1");
  I2->addUse("%0");
  SI.insertMachineInstrInMaps(I2);
  std::cout << "  Created: ";
  I2->print();

  auto *I3 = new SimMachineInstr(SI.nextIndex(), "MULrr");
  I3->addDef("%2");
  I3->addUse("%1");
  I3->addUse("%0");
  SI.insertMachineInstrInMaps(I3);
  std::cout << "  Created: ";
  I3->print();

  auto *I4 = new SimMachineInstr(SI.nextIndex(), "STR", true); // early-clobber
  I4->addUse("%2");
  SI.insertMachineInstrInMaps(I4);
  std::cout << "  Created: ";
  I4->print();

  SI.print();

  // Demonstrate the four slots per index
  std::cout << "\n--- Four Slots per Index ---\n";
  unsigned exampleIdx = I2->SlotIdx;
  std::cout << "  Index " << exampleIdx << " has four slots:\n";
  std::cout << "    SlotIndex(" << exampleIdx << ", Block)        -> "
            << SlotIndex(exampleIdx, Slot::Block).toString() << "\n";
  std::cout << "    SlotIndex(" << exampleIdx << ", EarlyClobber) -> "
            << SlotIndex(exampleIdx, Slot::EarlyClobber).toString() << "\n";
  std::cout << "    SlotIndex(" << exampleIdx << ", Register)     -> "
            << SlotIndex(exampleIdx, Slot::Register).toString() << "\n";
  std::cout << "    SlotIndex(" << exampleIdx << ", Dead)         -> "
            << SlotIndex(exampleIdx, Slot::Dead).toString() << "\n";

  // Demonstrate liveness modeling using slots
  std::cout << "\n--- Liveness Modeling with Slots ---\n";
  std::cout << "  For ADDri at index " << I2->SlotIdx << ":\n";

  // Normal definition: live from [32r, ...)
  SlotIndex defStart = SimSlotIndexes::getDefSlot(I2->SlotIdx, false);
  std::cout << "    %1 definition starts at: " << defStart.toString()
            << " (regular - can share reg with uses at same index)\n";

  // Normal use: live until ...32r)
  SlotIndex useEnd = SimSlotIndexes::getUseSlot(I2->SlotIdx);
  std::cout << "    %0 use ends at:         " << useEnd.toString()
            << " (exclusive - use dies here)\n";

  // Early-clobber definition (e.g., STR at index 64)
  SlotIndex ecStart = SimSlotIndexes::getDefSlot(I4->SlotIdx, true);
  std::cout << "\n  For STR (early-clobber) at index " << I4->SlotIdx << ":\n";
  std::cout << "    Definition starts at:  " << ecStart.toString()
            << " (early-clobber - CANNOT share reg with inputs)\n";

  // Dead definition
  SlotIndex deadEnd = SimSlotIndexes::getDeadSlot(I4->SlotIdx);
  std::cout << "    Dead def range: [" << ecStart.toString()
            << "," << deadEnd.toString() << ")\n";

  // Demonstrate overlap checking
  std::cout << "\n--- Live Range Overlap Checking ---\n";

  // Live range of %0: [16r, 48r)  -- from COPY def to MUL use
  LiveRange range0(
      SimSlotIndexes::getDefSlot(I1->SlotIdx, false),
      SimSlotIndexes::getUseSlot(I3->SlotIdx));
  std::cout << "  %0 live range: " << range0.toString() << "\n";

  // Live range of %1: [32r, 48r)  -- from ADD def to MUL use
  LiveRange range1(
      SimSlotIndexes::getDefSlot(I2->SlotIdx, false),
      SimSlotIndexes::getUseSlot(I3->SlotIdx));
  std::cout << "  %1 live range: " << range1.toString() << "\n";

  // Live range of %2: [48r, ...)  -- from MUL def
  LiveRange range2(
      SimSlotIndexes::getDefSlot(I3->SlotIdx, false),
      SimSlotIndexes::getUseSlot(I4->SlotIdx));
  std::cout << "  %2 live range: " << range2.toString() << "\n";

  std::cout << "\n  Overlap check:\n";
  std::cout << "    %0 overlaps %1? "
            << (range0.overlaps(range1) ? "YES" : "no") << "\n";
  std::cout << "    %1 overlaps %2? "
            << (range1.overlaps(range2) ? "YES" : "no") << "\n";
  std::cout << "    %0 overlaps %2? "
            << (range0.overlaps(range2) ? "YES" : "no") << "\n";

  // Key insight: %1's def at 32r and %0's use at 32r don't overlap
  // because [32r is inclusive and 32r) is exclusive
  std::cout << "\n  Key insight: %1 def at 32r (" << range1.Start.toString()
            << ") and %0 use at " << useEnd.toString()
            << " do NOT overlap because [32r is inclusive and 32r) is exclusive.\n";
  std::cout << "  This means %0 and %1 CAN share the same physical register!\n";

  // Cleanup
  delete I0; delete I1; delete I2; delete I3; delete I4;

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. Slot indexes are monotonically increasing instruction numbers\n";
  std::cout << "  2. Each index has 4 slots: B, e, r, d (in order)\n";
  std::cout << "  3. Register slot (r): normal defs start, uses die here\n";
  std::cout << "  4. Early-clobber (e): prevents def and use from sharing a register\n";
  std::cout << "  5. Slot indexes enable precise liveness modeling\n";

  return 0;
}
