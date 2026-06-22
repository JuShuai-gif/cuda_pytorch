//===----------------------------------------------------------------------===//
// Chapter 19 - Register Allocation
// Example: Live Intervals - Segments, VNInfo, and SSA Reconstruction
//===----------------------------------------------------------------------===//
//
// This example demonstrates the LiveInterval concept:
// - How live intervals represent where a virtual register is alive
// - Segments (ranges) and Value Number Information (VNInfo)
// - How SSA form is maintained in non-SSA MachineIR
// - Using live intervals for reaching definition queries
//
// NOTE: In LLVM, LiveInterval and LiveIntervals are real classes used by
// the register allocator. This simulation captures their core concepts.
//

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//------------------------------------------------------------------------------
// Simulated SlotIndex (simplified from previous example)
//------------------------------------------------------------------------------
struct SlotIndex {
  unsigned Index;
  char Slot; // 'B', 'e', 'r', 'd'

  SlotIndex(unsigned Idx = 0, char S = 'B') : Index(Idx), Slot(S) {}

  bool operator<(const SlotIndex &Other) const {
    if (Index != Other.Index) return Index < Other.Index;
    return Slot < Other.Slot;
  }
  bool operator==(const SlotIndex &Other) const {
    return Index == Other.Index && Slot == Other.Slot;
  }

  std::string toString() const {
    return std::to_string(Index) + Slot;
  }
};

//------------------------------------------------------------------------------
// Value Number Information (VNInfo)
//------------------------------------------------------------------------------
struct VNInfo {
  unsigned Id;               // Unique value identifier
  SlotIndex Def;             // Where this value is defined
  bool IsPHI;                // Whether this is a phi definition

  VNInfo(unsigned ID, SlotIndex D, bool Phi = false)
    : Id(ID), Def(D), IsPHI(Phi) {}

  std::string toString() const {
    std::string Result = std::to_string(Id) + "@" + Def.toString();
    if (IsPHI) Result += "-phi";
    return Result;
  }
};

//------------------------------------------------------------------------------
// Live Range Segment
//------------------------------------------------------------------------------
struct Segment {
  SlotIndex Start;   // Inclusive
  SlotIndex End;     // Exclusive
  unsigned ValNo;    // Which VNInfo this segment holds

  Segment(SlotIndex S, SlotIndex E, unsigned V)
    : Start(S), End(E), ValNo(V) {}

  bool contains(SlotIndex Point) const {
    return Start <= Point && Point < End;
  }

  bool overlaps(const Segment &Other) const {
    return Start < Other.End && Other.Start < End;
  }

  std::string toString() const {
    return "[" + Start.toString() + "," + End.toString() + ":" +
           std::to_string(ValNo) + ")";
  }
};

//------------------------------------------------------------------------------
// LiveInterval - the core liveness representation
//------------------------------------------------------------------------------
class LiveInterval {
private:
  unsigned RegNum;                    // Virtual register number
  std::vector<Segment> Segments;      // Live range segments
  std::vector<VNInfo> ValueNumbers;   // SSA value definitions
  std::vector<unsigned> PhiValNos;    // Which VNInfo IDs are phis

public:
  LiveInterval(unsigned Reg) : RegNum(Reg) {}

  unsigned getRegNum() const { return RegNum; }

  // Add a new segment
  void addSegment(SlotIndex Start, SlotIndex End, unsigned ValNo) {
    Segments.emplace_back(Start, End, ValNo);
    // Keep segments sorted by start
    std::sort(Segments.begin(), Segments.end(),
              [](const Segment &A, const Segment &B) {
                return A.Start < B.Start;
              });
  }

  // Add a VNInfo definition
  void addVNInfo(SlotIndex Def, bool IsPhi = false) {
    unsigned Id = ValueNumbers.size();
    ValueNumbers.emplace_back(Id, Def, IsPhi);
    if (IsPhi) PhiValNos.push_back(Id);
  }

  // Get the VNInfo at a specific program point
  const VNInfo *getVNInfoAt(SlotIndex Point) const {
    for (auto &Seg : Segments) {
      if (Seg.contains(Point)) {
        for (auto &VNI : ValueNumbers) {
          if (VNI.Id == Seg.ValNo) return &VNI;
        }
      }
    }
    return nullptr;
  }

  // Check if register is live at a point
  bool liveAt(SlotIndex Point) const {
    for (auto &Seg : Segments) {
      if (Seg.contains(Point)) return true;
    }
    return false;
  }

  // Check if this interval overlaps with another (interference)
  bool overlaps(const LiveInterval &Other) const {
    for (auto &SA : Segments) {
      for (auto &SB : Other.Segments) {
        if (SA.overlaps(SB)) return true;
      }
    }
    return false;
  }

  // Print in the familiar LLVM debug format
  void print() const {
    std::cout << "%%" << RegNum << "\n";
    std::cout << "  Segments:\n";
    for (auto &Seg : Segments) {
      std::cout << "    " << Seg.toString() << "\n";
    }
    std::cout << "  VNInfo:\n";
    for (auto &VNI : ValueNumbers) {
      std::cout << "    " << VNI.toString() << "\n";
    }
  }

  // Reconstruct SSA form for this virtual register
  void printSSAReconstruction() const {
    std::cout << "  SSA Form Reconstruction for %%"<< RegNum << ":\n";
    for (auto &Seg : Segments) {
      // Find the VNInfo for this segment
      for (auto &VNI : ValueNumbers) {
        if (VNI.Id == Seg.ValNo) {
          std::cout << "    ";
          if (VNI.IsPHI) {
            std::cout << "phi_use(";
          }
          std::cout << "%%"<< RegNum << "." << VNI.Id
                    << " (def at " << VNI.Def.toString() << ")";
          std::cout << " live in " << Seg.toString() << "\n";
          break;
        }
      }
    }
  }
};

//------------------------------------------------------------------------------
// LiveIntervals - manages live intervals for all virtual registers
//------------------------------------------------------------------------------
class LiveIntervals {
private:
  std::map<unsigned, LiveInterval> Intervals;

public:
  // Create an empty live interval for a new virtual register
  void createEmptyInterval(unsigned Reg) {
    Intervals.emplace(Reg, LiveInterval(Reg));
  }

  // Get the live interval for a virtual register
  LiveInterval &getInterval(unsigned Reg) {
    return Intervals.at(Reg);
  }

  // Shrink to only cover actual uses (called after removing uses)
  void shrinkToUses(unsigned Reg) {
    std::cout << "  [LiveIntervals] Shrinking interval for %%" << Reg << "\n";
  }

  // Extend to cover new uses (called after inserting uses)
  void extendToIndices(unsigned Reg) {
    std::cout << "  [LiveIntervals] Extending interval for %%" << Reg << "\n";
  }

  void print() const {
    for (auto &Pair : Intervals) {
      Pair.second.print();
      std::cout << "\n";
    }
  }
};

//------------------------------------------------------------------------------
// Main demonstration: Reconstruct reaching definitions
//------------------------------------------------------------------------------
SlotIndex findReachingDef(const LiveInterval &LI, SlotIndex UsePoint) {
  std::cout << "\n  Finding reaching definition for use at "
            << UsePoint.toString() << ":\n";

  const VNInfo *VNI = LI.getVNInfoAt(UsePoint);

  if (!VNI) {
    std::cout << "    Not live at this point!\n";
    return SlotIndex();
  }

  std::cout << "    VNInfo: " << VNI->toString() << "\n";

  if (VNI->IsPHI) {
    std::cout << "    This is a PHI - need to check predecessor blocks\n";
    std::cout << "    (In real LLVM, iterate predecessors and query\n";
    std::cout << "     getVNInfoAt at each predecessor's end)\n";
    return SlotIndex();
  } else {
    std::cout << "    Reaching definition is at slot: "
              << VNI->Def.toString() << "\n";
    return VNI->Def;
  }
}

//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------
int main() {
  std::cout << "=== Chapter 19: Live Intervals ===\n";

  // Build the example from Table 19.1 in the book:
  //
  //  0B  bb.0
  //  ...
  // 80B   %10 = COPY $w0           (def of value 0)
  // ...
  // 208B  Bcc ...                  (branch)
  // 224B  B %bb.1
  // 240B bb.1:
  // ...
  // 320B   %10 = ... %10           (def of value 1, use of value 0)
  // 368B bb.2:
  // 400B   $w0 = COPY %10          (use of value 2)

  LiveInterval LI10(10);

  // Add VNInfo definitions
  LI10.addVNInfo(SlotIndex(80, 'r'), false);   // 0@80r - real definition
  LI10.addVNInfo(SlotIndex(320, 'r'), false);  // 1@320r - real definition
  LI10.addVNInfo(SlotIndex(368, 'B'), true);   // 2@368B-phi - phi definition

  // Add segments
  LI10.addSegment(SlotIndex(80, 'r'), SlotIndex(320, 'r'), 0);
  LI10.addSegment(SlotIndex(320, 'r'), SlotIndex(368, 'B'), 1);
  LI10.addSegment(SlotIndex(368, 'B'), SlotIndex(400, 'r'), 2);

  // Print the live interval
  std::cout << "\n--- Live Interval for %10 ---\n";
  LI10.print();

  // Show SSA reconstruction
  std::cout << "\n--- SSA Reconstruction ---\n";
  LI10.printSSAReconstruction();

  // Demonstrate liveness queries
  std::cout << "\n--- Liveness Queries ---\n";
  std::cout << "  liveAt 100r? " << (LI10.liveAt(SlotIndex(100, 'r')) ? "YES" : "no") << "\n";
  std::cout << "  liveAt 350r? " << (LI10.liveAt(SlotIndex(350, 'r')) ? "YES" : "no") << "\n";
  std::cout << "  liveAt 70r?  " << (LI10.liveAt(SlotIndex(70, 'r')) ? "YES" : "no") << "\n";

  // Demonstrate reaching definition
  std::cout << "\n--- Reaching Definition Query ---\n";

  // At use in bb.2: $w0 = COPY %10 at 400r
  findReachingDef(LI10, SlotIndex(400, 'r'));

  // At the second definition: %10 = ... %10 at 320r
  findReachingDef(LI10, SlotIndex(320, 'r'));

  // At a point in the phi segment
  findReachingDef(LI10, SlotIndex(380, 'r'));

  // Demonstrate interference checking
  std::cout << "\n--- Interference Check ---\n";

  LiveInterval LI11(11);
  LI11.addVNInfo(SlotIndex(100, 'r'), false);
  LI11.addSegment(SlotIndex(100, 'r'), SlotIndex(200, 'r'), 0);

  std::cout << "  %10:\n";
  LI10.print();
  std::cout << "  %11:\n";
  LI11.print();
  std::cout << "  %10 overlaps %11? "
            << (LI10.overlaps(LI11) ? "YES (cannot share register)" : "no (can share)") << "\n";

  // Demonstrate LiveIntervals management
  std::cout << "\n--- LiveIntervals Management ---\n";
  LiveIntervals LIS;
  LIS.createEmptyInterval(20);
  LIS.shrinkToUses(20);
  LIS.extendToIndices(20);

  std::cout << "\n--- Summary ---\n";
  std::cout << "  1. LiveInterval = segments (ranges) + VNInfo (SSA definitions)\n";
  std::cout << "  2. Segments tell WHERE a register is alive\n";
  std::cout << "  3. VNInfo tells WHICH SSA value is held in each segment\n";
  std::cout << "  4. SSA form is preserved even in non-SSA MachineIR\n";
  std::cout << "  5. Use getVNInfoAt() for reaching definition queries\n";
  std::cout << "  6. Use overlaps() for interference (register sharing) queries\n";
  std::cout << "  7. Maintain via shrinkToUses / extendToIndices when modifying IR\n";

  return 0;
}
