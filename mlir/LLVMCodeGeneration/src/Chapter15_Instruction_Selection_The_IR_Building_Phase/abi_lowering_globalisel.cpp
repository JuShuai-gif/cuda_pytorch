// abi_lowering_globalisel.cpp - GlobalISel ABI Lowering Example
//
// Demonstrates the CallLowering class and the assigner/handler pattern
// used by GlobalISel for ABI lowering.

#include <cassert>
#include <functional>
#include <memory>
#include <vector>

// ============================================================
// 1. Common types (simulated)
// ============================================================

struct LLT {
  unsigned SizeInBits;
  bool IsPointer = false;
  static LLT scalar(unsigned Size) { return {Size, false}; }
  static LLT pointer(unsigned /*AS*/, unsigned Size) { return {Size, true}; }
  unsigned getSizeInBits() const { return SizeInBits; }
};

using Register = unsigned;
static constexpr Register NoRegister = 0;

struct MachinePointerInfo {
  // Describes a memory location (frame index, global, etc.)
};

struct MachineMemOperand {
  // Describes a memory access: size, alignment, flags (load/store/invariant)
};

struct MachineIRBuilder {
  // Build MIR instructions
  Register buildCopy(Register Src) { return Src; }
};

struct MachineRegisterInfo {
  Register createGenericVirtualRegister(LLT Ty) { return 0; }
  LLT getType(Register Reg) const { return LLT::scalar(32); }
};

struct MachineFunction {};

// ArgFlags from LLVM IR
struct ArgFlagsTy {
  bool SExt = false, ZExt = false;
};

struct CCValAssign {
  enum LocInfo { Full, SExt, ZExt };
  bool IsRegister = true;
  unsigned LocReg = 0;
  unsigned LocMemOffset = 0;
  unsigned ValSize = 0, LocSize = 0;
  LocInfo Info = Full;
};

using CCAssignFn = std::function<bool(unsigned, unsigned, unsigned,
                                       CCValAssign::LocInfo, ArgFlagsTy, void*)>;

// ============================================================
// 2. ValueAssigner (Incoming / Outgoing)
// ============================================================

// IncomingValueAssigner: for formal arguments (values coming into a function)
// OutgoingValueAssigner: for call arguments (values going out to another function)
// and for return values (values going out of the current function)

class IncomingValueAssigner {
  CCAssignFn AssignFn;
  unsigned StackSize = 0;
public:
  explicit IncomingValueAssigner(CCAssignFn Fn) : AssignFn(std::move(Fn)) {}

  // Called by determineAssignments for each argument.
  // Fills in CCState with assigned locations.
  bool assignArg(unsigned ValNo, unsigned ValVT, unsigned LocVT,
                 CCValAssign::LocInfo LocInfo, ArgFlagsTy Flags,
                 void *State) {
    bool Res = AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
    // StackSize = State.getStackSize();
    return Res;
  }

  unsigned getStackSize() const { return StackSize; }
};

class OutgoingValueAssigner {
  CCAssignFn AssignFn;
  unsigned StackSize = 0;
public:
  explicit OutgoingValueAssigner(CCAssignFn Fn) : AssignFn(std::move(Fn)) {}

  bool assignArg(unsigned ValNo, unsigned ValVT, unsigned LocVT,
                 CCValAssign::LocInfo LocInfo, ArgFlagsTy Flags,
                 void *State) {
    return AssignFn(ValNo, ValVT, LocVT, LocInfo, Flags, State);
  }
};

// ============================================================
// 3. ValueHandler (Incoming / Outgoing)
// ============================================================

// IncomingValueHandler: materializes locations for values coming in
// (formal arguments, call return values from the caller's perspective)
class IncomingValueHandler {
protected:
  MachineIRBuilder &MIRBuilder;
  MachineRegisterInfo &MRI;

public:
  IncomingValueHandler(MachineIRBuilder &MIB, MachineRegisterInfo &MRI)
    : MIRBuilder(MIB), MRI(MRI) {}

  // Called when the assigner says "this value is in a register"
  virtual void assignValueToReg(Register ValVReg, Register PhysReg,
                                 const CCValAssign &VA) {
    // Update live-in registers for the MachineFunction
    // MF.addLiveIn(PhysReg, ...)

    // Build a COPY from PhysReg to ValVReg
    // MIRBuilder.buildCopy(ValVReg, PhysReg);

    // If LocInfo != Full, also build extension instruction:
    if (VA.Info == CCValAssign::SExt) {
      // MIRBuilder.buildSExt(ValVReg, ...);
    }
  }

  // Called when the assigner says "this value is in memory"
  virtual void assignValueToAddress(Register ValVReg, Register Addr,
                                     LLT MemTy, const MachinePointerInfo &MPO,
                                     const CCValAssign &VA) {
    // Create a MachineMemOperand
    // auto MMO = MF.getMachineMemOperand(
    //     MPO, MachineMemOperand::MOLoad | MachineMemOperand::MOInvariant,
    //     MemTy, inferAlignFromPtrInfo(MF, MPO));

    // Build a LOAD from the address
    // MIRBuilder.buildLoad(ValVReg, Addr, *MMO);
  }

  // Returns the address of a stack slot for this argument
  virtual Register getStackAddress(unsigned StackOffset, LLT Ty,
                                    MachinePointerInfo &MPO) {
    // Create a fixed stack object (frame index)
    // int FrameIdx = MFI.CreateFixedObject(Ty.getSizeInBits()/8, StackOffset, true);

    // Create a G_FRAME_INDEX instruction
    // Register Addr = MRI.createGenericVirtualRegister(LLT::pointer(0, 16));
    // MIRBuilder.buildFrameIndex(Addr, FrameIdx);

    // Set MPO for the memory operand
    // MPO = MachinePointerInfo::getFixedStack(MF, FrameIdx);
    return NoRegister;
  }

  // Optional: the LLT for the stack memory access
  virtual LLT getStackValueStoreType(const CCValAssign &VA) {
    // Default: use the value type as-is
    return LLT::scalar(VA.ValSize);
  }
};

// OutgoingValueHandler: for call arguments and return values
class OutgoingValueHandler {
protected:
  MachineIRBuilder &MIRBuilder;
  MachineRegisterInfo &MRI;

public:
  OutgoingValueHandler(MachineIRBuilder &MIB, MachineRegisterInfo &MRI)
    : MIRBuilder(MIB), MRI(MRI) {}

  virtual void assignValueToReg(Register ValVReg, Register PhysReg,
                                 const CCValAssign &VA) {
    // Build COPY from ValVReg to PhysReg
    // MIRBuilder.buildCopy(PhysReg, ValVReg);
  }

  virtual void assignValueToAddress(Register ValVReg, Register Addr,
                                     LLT MemTy, const MachinePointerInfo &MPO,
                                     const CCValAssign &VA) {
    // Build a STORE from ValVReg to the address
    // MIRBuilder.buildStore(ValVReg, Addr, *MMO);
  }
};

// ============================================================
// 4. CallLowering class (target-specific)
// ============================================================

// This is the GlobalISel equivalent of TargetLowering for ABI handling.
// It must be implemented for each target.

class H2BLBCallLowering {
public:
  // ============================================================
  // lowerFormalArguments: read incoming function arguments
  // ============================================================
  //
  // Parameters:
  //   MIRBuilder - builder for creating MIR instructions
  //   F          - the LLVM IR function being lowered
  //   VRegs      - [OUT] array of virtual registers for each argument.
  //                Outer index = argument number.
  //                Inner array = vregs for split components (e.g.,
  //                a 64-bit value on 32-bit target uses two vregs).
  //   FLI        - FunctionLoweringInfo with metadata about the function
  //
  // Returns true on success, false to trigger fallback to SDISel.

  bool lowerFormalArguments(
      MachineIRBuilder &MIRBuilder,
      /* const Function &F, */
      /* ArrayRef<ArrayRef<Register>> VRegs, */
      /* FunctionLoweringInfo &FLI */) {

    // --- Step 1: Handle sret demotion if needed ---
    // if (!FLI.CanLowerReturn) {
    //   // Add the sret pointer as an extra formal argument at position 0
    //   insertSRetIncomingArgument(F, SplitArgs, FLI.DemoteRegister, MRI, DL);
    // }

    // --- Step 2: Build the list of low-level arguments ---
    // For each formal argument:
    //   - Skip zero-sized arguments
    //   - Create ArgInfo with vreg mapping
    //   - Call setArgFlags() to compute properties
    //   - Call splitToValueTypes() to split large values
    //
    // SmallVector<ArgInfo, 8> SplitArgs;
    // for (auto &Arg : F.args()) {
    //   ArgInfo OrigArg{VRegs[i], Arg, i};
    //   splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    // }

    // --- Step 3: Create assigner + handler + CCState ---
    // CCAssignFn *AssignFn = CC_H2BLB_Common;
    // H2BLBIncomingValueAssigner Assigner(AssignFn, AssignFn);
    // IncomingArgHandler Handler(MIRBuilder, MRI);
    // SmallVector<CCValAssign, 16> ArgLocs;
    // CCState CCInfo(F.getCallingConv(), F.isVarArg(), MF, ArgLocs,
    //                F.getContext());

    // --- Step 4: Determine assignments and materialize ---
    // if (!determineAssignments(Assigner, SplitArgs, CCInfo) ||
    //     !handleAssignments(Handler, SplitArgs, CCInfo, ArgLocs, MIRBuilder))
    //   return false;

    return true;
  }

  // ============================================================
  // lowerReturn: write return values
  // ============================================================
  bool lowerReturn(
      MachineIRBuilder &MIRBuilder,
      /* const Value *Val, ArrayRef<Register> VRegs,
         FunctionLoweringInfo &FLI */) {

    // Create OutgoingValueAssigner and OutgoingValueHandler
    // Use CCInfo.AnalyzeReturn(...)
    // Pack return values into ABI locations
    // Generate the return instruction

    return true;
  }

  // ============================================================
  // lowerCall: prepare arguments, call, unpack results
  // ============================================================
  bool lowerCall(
      MachineIRBuilder &MIRBuilder,
      /* CallLoweringInfo &Info */) {

    // 1. Create OutgoingValueAssigner + OutgoingValueHandler
    // 2. Pack arguments into ABI locations
    // 3. Build the actual call instruction
    // 4. Create IncomingValueAssigner + IncomingValueHandler
    // 5. Unpack return values

    return true;
  }

  // ============================================================
  // canLowerReturn: check if sret demotion is needed
  // ============================================================
  bool canLowerReturn(
      /* CallingConv::ID CC, bool IsVarArg,
         ArrayRef<LLT> ValTys, ... */) const {
    // Use CCInfo.CheckReturn(...) with the return CCAssignFn
    // Returns false if the return value is too large for registers
    return true;
  }
};

// ============================================================
// 5. Key differences from SDISel ABI lowering
// ============================================================

void differencesFromSDISel() {
  // 1. GlobalISel works directly on Machine IR, not SDNodes.
  //    - No need for CopyFromReg/CopyToReg indirection.
  //    - Built with MachineIRBuilder instead of SelectionDAG::getNode().

  // 2. The assigner/handler pattern separates concerns:
  //    - Assigner: WHERE should the value go? (register? stack? split?)
  //    - Handler: HOW to materialize that decision? (COPY, LOAD, STORE)
  //    - This separation makes the code more modular/testable.

  // 3. GlobalISel handles argument splitting explicitly:
  //    - splitToValueTypes() breaks large values into registers.
  //    - SDISel does this implicitly during type legalization.

  // 4. Return value indicates pass success/failure:
  //    - Returning false from lowerFormalArguments/lowerCall/lowerReturn
  //      causes IRTranslator to fail -> fallback to SDISel.
  //    - In SDISel, the monolithic pass handles everything internally.

  // 5. Use the provided MIRBuilder, don't create a local one:
  //    - It may have observers attached (for CSE, debug info).
  //    - Creating a local builder breaks those invariants.
}
