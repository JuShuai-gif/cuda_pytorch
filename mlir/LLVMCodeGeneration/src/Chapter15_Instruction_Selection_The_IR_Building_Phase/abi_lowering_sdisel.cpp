// abi_lowering_sdisel.cpp - SDISel ABI Lowering Example
//
// Demonstrates the target hooks that SDISel requires for ABI lowering:
//   - LowerFormalArguments  (callee: read incoming args)
//   - LowerReturn           (callee: write return values)
//   - LowerCall             (caller: prepare args, call, unpack results)
//   - CanLowerReturn        (check if sret demotion is needed)

#include <cassert>
#include <functional>
#include <vector>

// --- Reuse simplified types from callingconv_description.cpp concept ---

struct EVT {
  unsigned Size;
  static EVT i16() { return {16}; }
  static EVT i32() { return {32}; }
  static EVT Other() { return {0}; }
  unsigned getSizeInBits() const { return Size; }
};

using MVT = EVT;

struct ArgFlagsTy { bool SExt = false; };

// CCValAssign (simplified)
struct CCValAssign {
  enum LocInfo { Full, SExt, ZExt };
  bool IsRegister = true;
  unsigned LocReg = 0;
  unsigned LocMemOffset = 0;
  MVT ValVT{16};
  MVT LocVT{16};
  LocInfo Info = Full;
  bool isRegLoc() const { return IsRegister; }
  bool isMemLoc() const { return !IsRegister; }
  unsigned getLocReg() const { return LocReg; }
  unsigned getLocMemOffset() const { return LocMemOffset; }
  MVT getValVT() const { return ValVT; }
  MVT getLocVT() const { return LocVT; }
  LocInfo getLocInfo() const { return Info; }
};

using CCAssignFn = std::function<bool(unsigned, MVT, MVT, CCValAssign::LocInfo, ArgFlagsTy, void*)>;

// --- TargetLowering class with the four ABI hooks ---

class H2BLBTargetLowering {
public:
  // ============================================================
  // Hook 1: LowerFormalArguments (callee side, incoming args)
  // ============================================================
  //
  // This is called for each function being compiled.
  // It must create DAG nodes that read the function's arguments
  // from ABI-specified locations (registers or stack).
  //
  // Parameters:
  //   Chain   - The incoming chain from EntryToken
  //   CallConv - The calling convention ID
  //   IsVarArg - Whether this is a variadic function
  //   Ins      - List of input argument descriptions
  //   DL       - Debug location
  //   DAG      - The SelectionDAG for the current basic block
  //   InVals   - [OUT] Filled with SDValues representing each argument
  //
  // Returns:
  //   The chain value (possibly updated with new side-effecting nodes).

  void LowerFormalArguments(
      /* SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
         SmallVectorImpl<ISD::InputArg> &Ins, SDLoc DL,
         SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals */) {

    // --- Step 1: Create CCState and analyze arguments ---
    // MachineFunction &MF = DAG.getMachineFunction();
    // CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
    std::vector<CCValAssign> ArgLocs;
    // CCInfo.AnalyzeFormalArguments(Ins, CC_H2BLB_Common);

    // --- Step 2: Iterate over CCValAssign results ---
    /*
    for (size_t i = 0; i < ArgLocs.size(); ++i) {
      CCValAssign &VA = ArgLocs[i];

      if (VA.isRegLoc()) {
        // --- Register location ---

        // 2a. Handle non-Full loc info (not yet implemented)
        if (VA.getLocInfo() != CCValAssign::Full)
          report_fatal_error("non-full passing, not yet implemented");

        // 2b. Choose the right register class based on LocVT
        EVT RegVT = VA.getLocVT();
        const TargetRegisterClass *DstRC = nullptr;
        // if (RegVT == MVT::i16) DstRC = &H2BLB::GPR16RegClass;
        // else if (RegVT == MVT::i32) DstRC = &H2BLB::GPR32RegClass;

        // 2c. Create a virtual register for the argument
        Register VReg = 0; // RegInfo.createVirtualRegister(DstRC);

        // 2d. Map the physical register to the virtual register
        // RegInfo.addLiveIn(VA.getLocReg(), VReg);

        // 2e. Create CopyFromReg node
        // SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, RegVT);
        // InVals.push_back(ArgValue);

      } else {
        // --- Stack location ---

        // 3a. Get the offset and size
        unsigned ArgOffset = VA.getLocMemOffset();
        unsigned ArgSize   = VA.getValVT().getSizeInBits() / 8;

        // 3b. Create a fixed stack object
        // int FrameIdx = MFI.CreateFixedObject(ArgSize, ArgOffset, true);

        // 3c. Create a FrameIndex node and pointer info
        // SDValue FrameIdxNode =
        //     DAG.getFrameIndex(FrameIdx, getPointerTy(DAG.getDataLayout()));
        // MachinePointerInfo PtrInfo =
        //     MachinePointerInfo::getFixedStack(MF, FrameIdx);

        // 3d. Create a load
        // SDValue Load = DAG.getLoad(VA.getValVT(), DL, Chain,
        //                             FrameIdxNode, PtrInfo);
        // InVals.push_back(Load);
        // Chain = Load.getValue(1); // Update chain
      }
    }
    */

    // --- Step 4: Return the chain ---
    // return Chain;
  }

  // ============================================================
  // Hook 2: LowerReturn (callee side, outgoing results)
  // ============================================================
  //
  // Called at the end of each function. Creates DAG nodes that
  // write the return value(s) to ABI-specified locations.

  void LowerReturn(
      /* SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
         SmallVectorImpl<ISD::OutputArg> &Outs, SDLoc DL,
         SelectionDAG &DAG */) {

    // --- Step 1: Analyze return locations ---
    // CCState CCInfo(CallConv, IsVarArg, MF, RetLocs, ...);
    // CCInfo.AnalyzeReturn(Outs, CC_H2BLB_Common_Ret);

    // --- Step 2: Create CopyToReg for register locations ---
    /*
    for (size_t i = 0; i < RetLocs.size(); ++i) {
      CCValAssign &VA = RetLocs[i];
      if (VA.isRegLoc()) {
        // Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i]);
      }
    }
    */

    // --- Step 3: Create the return node ---
    // SDValue Ret = DAG.getNode(H2BLBISD::RET_GLUE, DL,
    //                           MVT::Other,
    //                           {Chain /*, ... glue from CopyToReg */});
    // return Ret;
  }

  // ============================================================
  // Hook 3: LowerCall (caller side)
  // ============================================================
  //
  // Generates the code to call a function:
  //   1. Prepare arguments in ABI locations
  //   2. Emit CALLSEQ_START (call frame setup)
  //   3. Emit the CALL node
  //   4. Emit CALLSEQ_END
  //   5. Unpack return values

  void LowerCall(
      /* SDValue Chain, SDValue Callee, CallingConv::ID CallConv,
         bool IsVarArg, bool IsTailCall, ... */) {

    // --- Step 1: Analyze call operands ---
    // CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, ...);
    // CCInfo.AnalyzeCallOperands(Outs, CC_H2BLB_Common);

    // --- Step 2: Lower each argument (CopyToReg or STORE to stack) ---
    /*
    for (auto &VA : ArgLocs) {
      if (VA.isRegLoc()) {
        Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg);
      } else {
        // Create stack slot, store argument to it
        int FrameIdx = MFI.CreateFixedObject(Size, Offset, true);
        Chain = DAG.getStore(Chain, DL, Arg, FrameIdxNode, PtrInfo);
      }
    }
    */

    // --- Step 3: Emit call sequence delimiters ---
    // Chain = DAG.getCALLSEQ_START(Chain, StackSize, 0, DL);

    // --- Step 4: Emit the CALL node ---
    /*
    SDValue Call = DAG.getNode(H2BLBISD::CALL, DL,
                                {MVT::Other, MVT::Glue},
                                {Chain, Callee, ... args ...});
    Chain = Call.getValue(0);
    SDValue Glue = Call.getValue(1);
    */

    // --- Step 5: CALLSEQ_END ---
    // Chain = DAG.getCALLSEQ_END(Chain, StackSize, 0, Glue, DL);

    // --- Step 6: Unpack return values ---
    // CCInfo.AnalyzeCallResult(Ins, CC_H2BLB_Common_Ret);
    /*
    for (auto &VA : RetLocs) {
      if (VA.isRegLoc()) {
        SDValue Val = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT());
        InVals.push_back(Val);
      }
    }
    */
  }

  // ============================================================
  // Hook 4: CanLowerReturn
  // ============================================================
  //
  // Returns true if the return value fits in registers.
  // If false, the infrastructure applies sret demotion:
  //   - Caller allocates stack space
  //   - Pointer is passed as extra first argument
  //   - Callee writes return value through the pointer

  bool CanLowerReturn(
      /* CallingConv::ID CallConv, bool IsVarArg,
         const SmallVectorImpl<ISD::OutputArg> &Outs */) const {
    // CCState CCInfo(CallConv, IsVarArg, MF, RetLocs, ...);
    // return CCInfo.CheckReturn(Outs, CC_H2BLB_Common_Ret);
    return true;
  }
};

// ============================================================
// Demonstration
// ============================================================

void demonstrateABIHooks() {
  H2BLBTargetLowering TL;

  // All four hooks work together to implement ABI lowering:
  //
  //   Caller:                              Callee:
  //   ┌─────────────────────┐              ┌──────────────────────┐
  //   │ LowerCall:          │              │ LowerFormalArguments:│
  //   │  - put args in regs │  ──call──>   │  - read args from    │
  //   │  - setup stack      │              │    regs/stack        │
  //   │  - CALL node        │              │                      │
  //   │  - read return val  │  <──ret───   │ LowerReturn:         │
  //   │                     │              │  - put ret in regs   │
  //   └─────────────────────┘              │  - RET node          │
  //                                        └──────────────────────┘

  // CanLowerReturn guards the sret demotion:
  //   If CanLowerReturn() == false:
  //     - Callee signature changes: struct -> void(struct*)
  //     - Caller allocates stack slot, passes address as arg[0]
  //     - Callee writes return value through arg[0]

  // This is a foundational part of any LLVM backend:
  // without these hooks, functions cannot be called or return values.
}
