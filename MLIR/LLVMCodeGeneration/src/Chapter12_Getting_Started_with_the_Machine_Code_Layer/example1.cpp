// example1.cpp - MC Layer Concepts: MCInst, MCInstrDesc, MCInstPrinter
//
// Demonstrates:
// - Creating MCInst objects programmatically
// - Inspecting MCInstrDesc (instruction properties from TableGen)
// - Using MCInstPrinter to emit assembly text
// - MCRegisterInfo for physical register naming
// - Exploring instruction encoding information
//
// Build with:
//   clang++ -o example1 example1.cpp $(llvm-config --cxxflags --ldflags --libs core codegen mc)
//
// This example requires an LLVM build with a target (e.g., X86, AArch64).

#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

using namespace llvm;

// Helper: print a separator.
static void sep(const char *title) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

int main() {
    // Initialize all targets and their MC components.
    InitializeAllTargetInfos();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    // Lookup the default target.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Error looking up target: " << Error << "\n";
        return 1;
    }
    std::cout << "Target: " << TheTarget->getName()
              << " (" << sys::getDefaultTargetTriple() << ")" << std::endl;

    // Create target machine to access MC layer components.
    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "",
                                       Options, std::nullopt));

    // Access MC-layer classes.
    const MCInstrInfo *MII = TM->getSubtargetImpl()->getInstrInfo();
    const MCRegisterInfo *MRI = TM->getSubtargetImpl()->getMCRegisterInfo();
    const MCSubtargetInfo *STI = TM->getSubtargetImpl()->getMCSubtargetInfo();
    const MCAsmInfo *MAI = TM->getSubtargetImpl()->getMCAsmInfo();

    // Create an MCInstPrinter for textual assembly output.
    std::unique_ptr<MCInstPrinter> IP(
        TheTarget->createMCInstPrinter(Triple(sys::getDefaultTargetTriple()),
                                        MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
    if (!IP) {
        std::cerr << "Failed to create MCInstPrinter\n";
        return 1;
    }

    // ========================
    // Part 1: MCInstrDesc and Properties
    // ========================
    sep("MCInstrDesc: Instruction Properties from TableGen");

    unsigned NumOpcodes = MII->getNumOpcodes();
    std::cout << "Total opcodes defined: " << NumOpcodes << std::endl;

    // Categorize instructions by their properties.
    unsigned retCount = 0, branchCount = 0, loadCount = 0, storeCount = 0;
    unsigned addCount = 0, callCount = 0, commCount = 0;

    for (unsigned Opc = 0; Opc < NumOpcodes; ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);
        if (Desc.isReturn())   retCount++;
        if (Desc.isBranch())   branchCount++;
        if (Desc.mayLoad())    loadCount++;
        if (Desc.mayStore())   storeCount++;
        if (Desc.isAdd())      addCount++;
        if (Desc.isCall())     callCount++;
        if (Desc.isCommutable()) commCount++;
    }

    std::cout << "  Return instructions:   " << retCount << std::endl;
    std::cout << "  Branch instructions:   " << branchCount << std::endl;
    std::cout << "  Load instructions:     " << loadCount << std::endl;
    std::cout << "  Store instructions:    " << storeCount << std::endl;
    std::cout << "  Add instructions:      " << addCount << std::endl;
    std::cout << "  Call instructions:     " << callCount << std::endl;
    std::cout << "  Commutable:            " << commCount << std::endl;

    // Inspect a few specific instructions in detail.
    sep("Detailed MCInstrDesc Inspection (first return, first branch)");

    bool showedReturn = false, showedBranch = false;
    for (unsigned Opc = 0; Opc < NumOpcodes; ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);

        if (Desc.isReturn() && !showedReturn) {
            showedReturn = true;
            std::cout << "\n[RETURN] " << MII->getName(Opc) << " (opcode " << Opc << ")" << std::endl;
            std::cout << "  Num operands: " << Desc.getNumOperands() << std::endl;
            std::cout << "  Num defs:     " << Desc.getNumDefs() << std::endl;
            std::cout << "  Size:         " << Desc.getSize() << " bytes" << std::endl;
            std::cout << "  Implicit uses: ";
            for (unsigned i = 0; i < Desc.getNumImplicitUses(); ++i)
                std::cout << "$" << MRI->getName(Desc.implicit_uses()[i]) << " ";
            if (Desc.getNumImplicitUses() == 0) std::cout << "(none)";
            std::cout << std::endl;
            std::cout << "  Implicit defs: ";
            for (unsigned i = 0; i < Desc.getNumImplicitDefs(); ++i)
                std::cout << "$" << MRI->getName(Desc.implicit_defs()[i]) << " ";
            if (Desc.getNumImplicitDefs() == 0) std::cout << "(none)";
            std::cout << std::endl;
        }

        if (Desc.isBranch() && !Desc.isReturn() && !Desc.isCall() && !showedBranch) {
            showedBranch = true;
            std::cout << "\n[BRANCH] " << MII->getName(Opc) << " (opcode " << Opc << ")" << std::endl;
            std::cout << "  Num operands: " << Desc.getNumOperands() << std::endl;
            std::cout << "  Is conditional: " << (Desc.isConditionalBranch() ? "yes" : "no") << std::endl;
            std::cout << "  Is indirect: " << (Desc.isIndirectBranch() ? "yes" : "no") << std::endl;

            // Show operand constraints.
            for (unsigned i = 0; i < Desc.getNumOperands(); ++i) {
                int Tied = Desc.getOperandConstraint(i, MCOI::TIED_TO);
                int EC = Desc.getOperandConstraint(i, MCOI::EARLY_CLOBBER);
                std::cout << "    Op[" << i << "]";
                if (Tied != -1) std::cout << " tied-to-op" << Tied;
                if (EC != -1)   std::cout << " early-clobber";
                std::cout << std::endl;
            }
        }

        if (showedReturn && showedBranch) break;
    }

    // ========================
    // Part 2: Creating MCInst Objects
    // ========================
    sep("Creating MCInst Objects Programmatically");

    // Find a simple move/copy instruction to demonstrate MCInst creation.
    unsigned copyOpc = 0;
    for (unsigned Opc = 0; Opc < NumOpcodes; ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);
        // Look for a simple register-to-register move that's not a pseudo.
        if (Desc.isMoveReg() && Desc.getNumDefs() >= 1 && Desc.getNumOperands() == 2
            && !Desc.isPseudo()) {
            copyOpc = Opc;
            break;
        }
    }

    if (copyOpc != 0) {
        const MCInstrDesc &Desc = MII->get(copyOpc);
        std::cout << "Using opcode: " << MII->getName(copyOpc)
                  << " (" << Desc.getNumOperands() << " operands)" << std::endl;

        // Find two physical registers from an allocatable register class.
        // Access TargetRegisterInfo for class info.
        const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
        MCRegister Reg1 = 0, Reg2 = 0;
        for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->isAllocatable() && RC->getNumRegs() >= 2) {
                Reg1 = RC->getRegister(0);
                Reg2 = RC->getRegister(1);
                break;
            }
        }

        if (Reg1 != 0 && Reg2 != 0) {
            // Create an MCInst.
            MCInst Inst;
            Inst.setOpcode(copyOpc);
            // Add destination register (definition).
            Inst.addOperand(MCOperand::createReg(Reg1));
            // Add source register (use).
            Inst.addOperand(MCOperand::createReg(Reg2));

            std::cout << "Created MCInst: ";
            Inst.dump();
            std::cout << std::endl;

            // Print as assembly text using MCInstPrinter.
            SmallString<128> AsmStr;
            raw_svector_ostream OS(AsmStr);
            IP->printInst(&Inst, 0, "", *STI, OS);
            std::cout << "Assembly output: " << AsmStr << std::endl;
        } else {
            std::cout << "Could not find suitable registers for MCInst demo." << std::endl;
        }
    } else {
        std::cout << "No suitable move instruction found." << std::endl;
    }

    // ========================
    // Part 3: MCInst with Immediates
    // ========================
    sep("MCInst with Immediate Operands");

    // Find a load-immediate or add-immediate instruction.
    unsigned immOpc = 0;
    for (unsigned Opc = 0; Opc < NumOpcodes; ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);
        if (Desc.isMoveImm() && Desc.getNumDefs() >= 1 && Desc.getNumOperands() == 2
            && !Desc.isPseudo()) {
            immOpc = Opc;
            break;
        }
    }
    if (immOpc == 0) {
        // Fallback: find any add instruction.
        for (unsigned Opc = 0; Opc < NumOpcodes; ++Opc) {
            const MCInstrDesc &Desc = MII->get(Opc);
            if (Desc.isAdd() && Desc.getNumDefs() >= 1 && Desc.getNumOperands() >= 2
                && !Desc.isPseudo()) {
                immOpc = Opc;
                break;
            }
        }
    }

    if (immOpc != 0) {
        const MCInstrDesc &Desc = MII->get(immOpc);
        std::cout << "Using opcode: " << MII->getName(immOpc)
                  << " (" << Desc.getNumOperands() << " operands)" << std::endl;

        const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
        MCRegister RegDst = 0, RegSrc = 0;
        for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->isAllocatable() && RC->getNumRegs() >= 2) {
                RegDst = RC->getRegister(0);
                RegSrc = RC->getRegister(1);
                break;
            }
        }

        if (RegDst != 0) {
            MCInst Inst2;
            Inst2.setOpcode(immOpc);
            Inst2.addOperand(MCOperand::createReg(RegDst)); // dst
            if (Desc.isMoveImm()) {
                Inst2.addOperand(MCOperand::createImm(42)); // immediate value
            } else {
                Inst2.addOperand(MCOperand::createReg(RegSrc)); // src1
                Inst2.addOperand(MCOperand::createImm(7));      // immediate
            }

            std::cout << "Created MCInst: ";
            Inst2.dump();
            std::cout << std::endl;

            SmallString<128> AsmStr2;
            raw_svector_ostream OS2(AsmStr2);
            IP->printInst(&Inst2, 0, "", *STI, OS2);
            std::cout << "Assembly output: " << AsmStr2 << std::endl;
        }
    } else {
        std::cout << "No suitable immediate instruction found." << std::endl;
    }

    // ========================
    // Part 4: MCAsmInfo - Assembly Syntax Conventions
    // ========================
    sep("MCAsmInfo: Assembly Syntax Conventions");

    std::cout << "Comment string:       '" << MAI->getCommentString() << "'" << std::endl;
    std::cout << "Data16bitsDirective:  " << MAI->getData16bitsDirective() << std::endl;
    std::cout << "Data32bitsDirective:  " << MAI->getData32bitsDirective() << std::endl;
    std::cout << "Data64bitsDirective:  " << MAI->getData64bitsDirective() << std::endl;
    std::cout << "Align directive:      " << MAI->getAlignDirective() << std::endl;
    std::cout << "Text section:         " << MAI->getTextSection() << std::endl;
    std::cout << "Data section:         " << MAI->getDataSection() << std::endl;
    std::cout << "Global directive:     " << MAI->getGlobalDirective() << std::endl;
    std::cout << "Has single parameter .file: " << (MAI->hasSingleParameterDotFile() ? "yes" : "no") << std::endl;
    std::cout << "Has .type @function:  " << (MAI->hasDotTypeDotSizeDirective() ? "yes" : "no") << std::endl;

    // ========================
    // Part 5: MCRegisterInfo - Physical Register Names
    // ========================
    sep("MCRegisterInfo: Physical Register Summary");

    std::cout << "Total physical registers: " << MRI->getNumRegs() << std::endl;

    // Print some key registers if available.
    auto printReg = [&](const char *label, MCRegister Reg) {
        if (Reg != 0) {
            std::cout << "  " << label << ": $" << MRI->getName(Reg) << std::endl;
        }
    };
    printReg("Stack pointer", MRI->getStackRegister());
    // Note: frame pointer and return address are target-specific; may be 0.
    std::cout << "  (Frame pointer and link register vary by target)" << std::endl;

    // Print register class names from MCRegisterInfo (if available through MCRegisterClass iterator).
    sep("MCRegisterClass Listing (via TargetRegisterInfo)");
    const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    unsigned classCount = 0;
    for (const TargetRegisterClass *RC : TRI->regclasses()) {
        if (classCount++ >= 10) break;
        std::cout << "  Class: " << TRI->getRegClassName(RC)
                  << " (" << RC->getNumRegs() << " regs, "
                  << RC->getSizeInBits() << " bits)" << std::endl;
    }

    std::cout << "\nMC layer concepts demonstrated successfully!" << std::endl;
    return 0;
}
