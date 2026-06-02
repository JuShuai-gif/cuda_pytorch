// example3.cpp - Target Description Concepts (TableGen .td files for registers and instructions)
//
// Demonstrates:
// - Inspecting MCInstrDesc (the C++ representation of TableGen Instruction records)
// - Exploring MCRegisterInfo and TargetRegisterInfo generated from TableGen
// - Understanding register class membership, sub/super register relationships
// - Checking instruction properties and operand constraints from TableGen definitions
//
// Build with:
//   clang++ -o example3 example3.cpp $(llvm-config --cxxflags --ldflags --libs core codegen)
//
// This example assumes the host target (e.g., X86) is available and its TableGen-generated
// descriptions are compiled into the LLVM build.

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iomanip>
#include <iostream>

using namespace llvm;

// Helper: print instruction descriptor details.
static void inspectMCInstrDesc(const MCInstrDesc &Desc, const TargetInstrInfo *TII,
                                const TargetRegisterInfo *TRI) {
    std::cout << "  Opcode name: " << TII->getName(Desc.getOpcode()) << std::endl;
    std::cout << "  Opcode number: " << Desc.getOpcode() << std::endl;
    std::cout << "  Num operands (static): " << Desc.getNumOperands() << std::endl;
    std::cout << "  Num definitions (static): " << Desc.getNumDefs() << std::endl;
    std::cout << "  Size: " << Desc.getSize() << " bytes" << std::endl;
    std::cout << "  Sched class: " << Desc.getSchedClass() << std::endl;

    // Properties.
    std::cout << "  Properties:";
    if (Desc.isReturn())      std::cout << " return";
    if (Desc.isCall())        std::cout << " call";
    if (Desc.isBarrier())     std::cout << " barrier";
    if (Desc.isTerminator())  std::cout << " terminator";
    if (Desc.isBranch())      std::cout << " branch";
    if (Desc.isIndirectBranch()) std::cout << " indirect-branch";
    if (Desc.isCompare())     std::cout << " compare";
    if (Desc.isMoveImm())     std::cout << " move-imm";
    if (Desc.isMoveReg())     std::cout << " move-reg";
    if (Desc.isBitcast())     std::cout << " bitcast";
    if (Desc.isSelect())      std::cout << " select";
    if (Desc.isPredicable())  std::cout << " predicable";
    if (Desc.hasDelaySlot())  std::cout << " delay-slot";
    if (Desc.mayLoad())       std::cout << " may-load";
    if (Desc.mayStore())      std::cout << " may-store";
    if (Desc.hasUnmodeledSideEffects()) std::cout << " side-effects";
    if (Desc.isCommutable())  std::cout << " commutable";
    if (Desc.isConvertibleTo3Addr()) std::cout << " 3-addr-convertible";
    if (Desc.isAdd())         std::cout << " is-add";
    std::cout << std::endl;

    // Operand constraints from TableGen.
    std::cout << "  Operand constraints (from TableGen definition):" << std::endl;
    for (unsigned i = 0, e = Desc.getNumOperands(); i != e; ++i) {
        const MCOperandInfo &OpInfo = Desc.operands()[i];
        std::cout << "    [" << i << "] ";
        if (OpInfo.isLookupPtrRegClass())
            std::cout << "(lookup-regclass) ";

        if (OpInfo.RegClass != -1) {
            // RegClass field is the register class enum index.
            std::cout << "RegClass=" << OpInfo.RegClass;
        }
        if (OpInfo.isOptionalDef())
            std::cout << " [optional-def]";

        // Check tied operand.
        int Tied = Desc.getOperandConstraint(i, MCOI::TIED_TO);
        if (Tied != -1)
            std::cout << " [tied-to-op" << Tied << "]";

        // Check early clobber.
        int EarlyClobber = Desc.getOperandConstraint(i, MCOI::EARLY_CLOBBER);
        if (EarlyClobber != -1)
            std::cout << " [early-clobber]";

        std::cout << std::endl;
    }

    // Implicit operands from TableGen.
    unsigned NumImplicitUses = Desc.getNumImplicitUses();
    unsigned NumImplicitDefs = Desc.getNumImplicitDefs();
    if (NumImplicitUses > 0 || NumImplicitDefs > 0) {
        std::cout << "  Implicit operands (from TableGen):" << std::endl;
        for (unsigned i = 0; i != NumImplicitUses; ++i) {
            MCPhysReg Reg = Desc.implicit_uses()[i];
            std::cout << "    implicit-use: $" << TRI->getName(Reg) << std::endl;
        }
        for (unsigned i = 0; i != NumImplicitDefs; ++i) {
            MCPhysReg Reg = Desc.implicit_defs()[i];
            std::cout << "    implicit-def: $" << TRI->getName(Reg) << std::endl;
        }
    }
}

int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // Setup target machine.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Error: " << Error << "\n";
        return 1;
    }

    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "", Options,
                                       std::nullopt));

    // Access TableGen-generated target description classes.
    const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const TargetInstrInfo *TII = TM->getSubtargetImpl()->getInstrInfo();
    const MCRegisterInfo *MRI = TM->getSubtargetImpl()->getMCRegisterInfo();

    std::cout << "Target: " << TheTarget->getName() << std::endl;
    std::cout << "Triple: " << sys::getDefaultTargetTriple() << std::endl;

    // ========================
    // Part 1: Register Hierarchy from TableGen
    // ========================
    std::cout << "\n========== Register Hierarchy (TableGen-generated) ==========" << std::endl;

    // Print the number of physical registers.
    std::cout << "\nTotal physical registers: " << MRI->getNumRegs() << std::endl;

    // Print register classes and their members (first 8).
    std::cout << "\n--- Register Classes (first 8) ---" << std::endl;
    unsigned RCCount = 0;
    for (const TargetRegisterClass *RC : TRI->regclasses()) {
        if (RCCount++ >= 8) break;
        std::cout << "Class: " << TRI->getRegClassName(RC) << std::endl;
        std::cout << "  Size: " << RC->getSizeInBits() << " bits" << std::endl;
        std::cout << "  Align: " << RC->getAlignment() << " bytes" << std::endl;
        std::cout << "  Num registers: " << RC->getNumRegs() << std::endl;
        std::cout << "  Allocatable: " << (RC->isAllocatable() ? "yes" : "no") << std::endl;

        // Print first few register members.
        std::cout << "  Members (first 5): ";
        for (unsigned i = 0; i < std::min(RC->getNumRegs(), 5u); ++i) {
            if (i > 0) std::cout << ", ";
            MCRegister Reg = RC->getRegister(i);
            std::cout << "$" << TRI->getName(Reg);
        }
        if (RC->getNumRegs() > 5) std::cout << ", ...";
        std::cout << std::endl;
    }

    // Register class membership check.
    std::cout << "\n--- Register Class Membership Demo ---" << std::endl;
    if (!TRI->regclasses().empty()) {
        const TargetRegisterClass *RC = TRI->regclasses().front();
        if (RC->getNumRegs() > 0) {
            MCRegister Reg = RC->getRegister(0);
            std::cout << "Register $" << TRI->getName(Reg) << " contains class "
                      << TRI->getRegClassName(RC) << "? "
                      << (RC->contains(Reg) ? "yes" : "no") << std::endl;

            // Check which register classes contain this register.
            std::cout << "  Register classes containing $" << TRI->getName(Reg) << ":" << std::endl;
            for (const TargetRegisterClass *CheckRC : TRI->regclasses()) {
                if (CheckRC->contains(Reg)) {
                    std::cout << "    - " << TRI->getRegClassName(CheckRC) << std::endl;
                    break; // Just print one for brevity.
                }
            }
        }
    }

    // Sub-register and super-register demo.
    std::cout << "\n--- Sub/Super Register Relationships ---" << std::endl;
    if (!TRI->regclasses().empty()) {
        // Find a register with sub-registers.
        for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->getNumRegs() == 0) continue;

            for (unsigned i = 0; i < std::min(RC->getNumRegs(), 32u); ++i) {
                MCRegister Reg = RC->getRegister(i);
                MCSubRegIterator SRI(Reg, TRI);
                if (!SRI.isValid()) continue;

                std::cout << "$" << TRI->getName(Reg) << " sub-registers:" << std::endl;
                for (; SRI.isValid(); ++SRI) {
                    std::cout << "  -> $" << TRI->getName(*SRI);

                    // Check which sub-register index maps to this sub-register.
                    for (unsigned SI = 1; SI < 64; ++SI) {
                        if (TRI->getSubReg(Reg, SI) == *SRI) {
                            std::cout << " (index: " << TRI->getSubRegIndexName(SI) << "=" << SI << ")";
                            LaneBitmask LM = TRI->getSubRegIndexLaneMask(SI);
                            std::cout << " lane:0x" << std::hex << LM.getAsInteger() << std::dec;
                            break;
                        }
                    }
                    std::cout << std::endl;
                }

                // Show all super-registers.
                std::cout << "$" << TRI->getName(Reg) << " super-registers:" << std::endl;
                for (MCSuperRegIterator SupI(Reg, TRI); SupI.isValid(); ++SupI) {
                    std::cout << "  -> $" << TRI->getName(*SupI) << std::endl;
                }

                break; // Just show one register with sub-registers.
            }
            break;
        }
    }

    // Register overlap check.
    std::cout << "\n--- Register Overlap (via Register Units) ---" << std::endl;
    if (!TRI->regclasses().empty()) {
        // Find two adjacent registers from the same class.
        for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->getNumRegs() < 2) continue;

            MCRegister R1 = RC->getRegister(0);
            MCRegister R2 = RC->getRegister(1);

            if (TRI->regsOverlap(R1, R2)) continue; // Skip overlapping registers.

            std::cout << "$" << TRI->getName(R1) << " units: { ";
            for (MCRegUnitIterator UI(R1, TRI); UI.isValid(); ++UI)
                std::cout << *UI << " ";
            std::cout << "}" << std::endl;

            std::cout << "$" << TRI->getName(R2) << " units: { ";
            for (MCRegUnitIterator UI(R2, TRI); UI.isValid(); ++UI)
                std::cout << *UI << " ";
            std::cout << "}" << std::endl;

            std::cout << "Overlap between $" << TRI->getName(R1) << " and $"
                      << TRI->getName(R2) << ": "
                      << (TRI->regsOverlap(R1, R2) ? "yes" : "no") << std::endl;
            break;
        }
    }

    // ========================
    // Part 2: Instruction Description from TableGen
    // ========================
    std::cout << "\n========== Instruction Description (TableGen-generated) ==========" << std::endl;

    // Get the number of target-specific instructions.
    unsigned NumOpcodes = TII->getNumOpcodes();
    std::cout << "\nTotal target-specific opcodes: " << NumOpcodes << std::endl;

    // Inspect specific instruction types: return, branch, add, copy.
    std::cout << "\n--- Instruction Type Exploration ---" << std::endl;

    int returnCount = 0, branchCount = 0, addCount = 0, loadCount = 0;
    for (unsigned Opc = 0; Opc < NumOpcodes && (returnCount + branchCount + addCount + loadCount) < 12; ++Opc) {
        const MCInstrDesc &Desc = TII->get(Opc);

        if (Desc.isReturn() && returnCount < 3) {
            std::cout << "\n[Return instruction #" << (returnCount + 1) << "]" << std::endl;
            inspectMCInstrDesc(Desc, TII, TRI);
            returnCount++;
        }
        if (Desc.isBranch() && !Desc.isReturn() && branchCount < 3) {
            std::cout << "\n[Branch instruction #" << (branchCount + 1) << "]" << std::endl;
            inspectMCInstrDesc(Desc, TII, TRI);
            branchCount++;
        }
        if (Desc.isAdd() && addCount < 3) {
            std::cout << "\n[Add instruction #" << (addCount + 1) << "]" << std::endl;
            inspectMCInstrDesc(Desc, TII, TRI);
            addCount++;
        }
        if (Desc.mayLoad() && !Desc.mayStore() && !Desc.isReturn() && loadCount < 3) {
            std::cout << "\n[Load instruction #" << (loadCount + 1) << "]" << std::endl;
            inspectMCInstrDesc(Desc, TII, TRI);
            loadCount++;
        }
    }

    // Commutable instructions.
    std::cout << "\n--- Commutable Instructions (first 5) ---" << std::endl;
    unsigned commCount = 0;
    for (unsigned Opc = 0; Opc < NumOpcodes && commCount < 5; ++Opc) {
        const MCInstrDesc &Desc = TII->get(Opc);
        if (Desc.isCommutable()) {
            std::cout << "  [" << (commCount + 1) << "] " << TII->getName(Opc);
            if (Desc.isAdd()) std::cout << " (add-like)";
            std::cout << std::endl;
            commCount++;
        }
    }

    // ========================
    // Part 3: Demonstrating what a TableGen .td file would look like
    // ========================
    std::cout << "\n========== Sample TableGen .td Structure (Conceptual) ==========" << std::endl;

    std::cout << R"(
// --- Conceptual TableGen for a simple backend (e.g., "MyTarget") ---
//
// File: MyTargetRegisterInfo.td

// Sub-register indices
def sub16_lo : SubRegIndex<16, 0>;
def sub16_hi : SubRegIndex<16, 16>;

// Registers
def R0  : Register<"r0">,  DwarfRegNum<[0]>;
def R0L : Register<"r0l">, DwarfRegNum<[0]>;  // low 16 bits
def R0H : Register<"r0h">, DwarfRegNum<[0]>;  // high 16 bits

// R0 is a 32-bit register composed of R0L and R0H
def R0 : Register<"r0"> {
  let SubRegIndices = [sub16_lo, sub16_hi];
  let SubRegs = [R0L, R0H];
  let CoveredBySubRegs = true;
}

// Register classes
def GPR16 : RegisterClass<"MyTarget", [i16], 16, (add R0L, R0H)>;
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

// --- Conceptual TableGen for instructions ---
// File: MyTargetInstrInfo.td

// A simple add instruction
def ADD32rr : Instruction {
  let OutOperandList = (outs GPR32:$rd);
  let InOperandList  = (ins GPR32:$rs1, GPR32:$rs2);
  let AsmString       = "add $rd, $rs1, $rs2";
  let isCommutable    = true;
  let hasSideEffects  = false;
}

// A move instruction
def MOV32rr : Instruction {
  let OutOperandList = (outs GPR32:$rd);
  let InOperandList  = (ins GPR32:$rs);
  let AsmString       = "mov $rd, $rs";
  let isMoveImm       = true;  // or isMoveReg for register moves
}
    )" << std::endl;

    std::cout << "\nTarget description concepts demonstrated successfully!" << std::endl;
    return 0;
}
