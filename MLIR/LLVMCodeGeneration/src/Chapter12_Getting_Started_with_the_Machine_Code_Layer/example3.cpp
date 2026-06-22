// example3.cpp - MC Layer: AsmParser, Assembly Emission, and Connecting the MC Layer
//
// Demonstrates:
// - How the MC layer is connected to the target machine
// - Assembly emission pipeline concept
// - MCStreamer for generating assembly output
// - MCTargetOptions and assembling concepts
//
// Build with:
//   clang++ -o example3 example3.cpp $(llvm-config --cxxflags --ldflags --libs core codegen mc)
//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>
#include <memory>

using namespace llvm;

static void sep(const char *title) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << "============================================================" << std::endl;
}

int main() {
    InitializeAllTargetInfos();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Target lookup error: " << Error << "\n";
        return 1;
    }

    std::cout << "Target: " << TheTarget->getName()
              << " (" << sys::getDefaultTargetTriple() << ")" << std::endl;

    // ========================
    // Part 1: MC Layer Component Registration
    // ========================
    sep("MC Layer Component Registration (Conceptual)");

    // In a real backend, LLVMInitializeXXXTargetMC() registers:
    std::cout << R"(
The MC layer is connected by registering components:

  extern "C" void LLVMInitializeMyTargetTargetMC() {
    // Register MCAsmInfo
    RegisterMCAsmInfoFn X(getTheMyTarget(), createMyTargetMCAsmInfo);

    // Register MCRegisterInfo
    RegisterMCInstrInfoFn Y(getTheMyTarget(), createMyTargetMCInstrInfo);

    // Register MCInstrInfo
    RegisterMCRegInfoFn Z(getTheMyTarget(), createMyTargetMCRegisterInfo);

    // Register MCSubtargetInfo
    RegisterMCSubtargetInfoFn A(getTheMyTarget(), createMyTargetMCSubtargetInfoImpl);

    // Register MCInstPrinter
    RegisterMCInstPrinterFn B(getTheMyTarget(), createMyTargetMCInstPrinter);
  }
    )" << std::endl;

    // ========================
    // Part 2: Assembly Emission Pipeline
    // ========================
    sep("Assembly Emission Pipeline");

    // Create a TargetMachine.
    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "",
                                       Options, std::nullopt));

    // Access MC layer components.
    const MCAsmInfo *MAI = TM->getSubtargetImpl()->getMCAsmInfo();
    const MCInstrInfo *MII = TM->getSubtargetImpl()->getInstrInfo();
    const MCRegisterInfo *MRI = TM->getSubtargetImpl()->getMCRegisterInfo();
    const MCSubtargetInfo *STI = TM->getSubtargetImpl()->getMCSubtargetInfo();

    std::cout << "Assembly emission pipeline:" << std::endl;
    std::cout << std::endl;

    std::cout << "  1. MachineInstr (CodeGen layer)" << std::endl;
    std::cout << "     |" << std::endl;
    std::cout << "     v MCInstLower::Lower()" << std::endl;
    std::cout << "  2. MCInst (MC layer)" << std::endl;
    std::cout << "     |" << std::endl;
    std::cout << "     +---> MCStreamer::EmitInstruction()" << std::endl;
    std::cout << "     |       |" << std::endl;
    std::cout << "     |       +---> (assembly output path)" << std::endl;
    std::cout << "     |       |     MCInstPrinter::printInst() -> assembly text" << std::endl;
    std::cout << "     |       |" << std::endl;
    std::cout << "     |       +---> (object file path)" << std::endl;
    std::cout << "     |             MCCodeEmitter::encodeInstruction() -> bytes" << std::endl;
    std::cout << "     |             MCAsmBackend::applyFixup() -> resolve relocations" << std::endl;
    std::cout << "     v" << std::endl;
    std::cout << "  3. Output (.s or .o)" << std::endl;

    // ========================
    // Part 3: MCStreamer Concepts
    // ========================
    sep("MCStreamer Types");

    std::cout << "MCStreamer is the abstract interface for emitting MC-level output." << std::endl;
    std::cout << std::endl;
    std::cout << "Concrete implementations:" << std::endl;
    std::cout << "  - MCAsmStreamer:    Emits textual assembly (.s)" << std::endl;
    std::cout << "  - MCObjectStreamer: Emits binary object files (.o)" << std::endl;
    std::cout << "      - MCELFStreamer:     ELF format" << std::endl;
    std::cout << "      - MCMachOStreamer:   Mach-O format" << std::endl;
    std::cout << "      - MCWinCOFFStreamer: COFF format" << std::endl;
    std::cout << "  - MCNullStreamer:   Discards all output (for testing)" << std::endl;

    // ========================
    // Part 4: MCAsmInfo for Different Formats
    // ========================
    sep("MCAsmInfo Format-Specific Details");

    std::cout << "Assembly info for " << sys::getDefaultTargetTriple() << ":" << std::endl;
    std::cout << std::endl;

    // Print assembly directives.
    std::cout << "  Comment character:    '" << MAI->getCommentString() << "'" << std::endl;
    std::cout << "  Separator string:     '" << MAI->getSeparatorString() << "'" << std::endl;
    std::cout << "  Assembler dialect:    " << MAI->getAssemblerDialect() << std::endl;
    std::cout << "  Code pointer size:    " << MAI->getCodePointerSize() << " bytes" << std::endl;

    std::cout << "  Data directives:" << std::endl;
    std::cout << "    .byte:  " << MAI->getData8bitsDirective() << std::endl;
    std::cout << "    .short: " << MAI->getData16bitsDirective() << std::endl;
    std::cout << "    .long:  " << MAI->getData32bitsDirective() << std::endl;
    std::cout << "    .quad:  " << MAI->getData64bitsDirective() << std::endl;

    std::cout << "  Section directives:" << std::endl;
    std::cout << "    Text:   " << MAI->getTextSection() << std::endl;
    std::cout << "    Data:   " << MAI->getDataSection() << std::endl;
    std::cout << "    BSS:    " << MAI->getBSSSection() << std::endl;

    std::cout << "  Symbol directives:" << std::endl;
    std::cout << "    Global: " << MAI->getGlobalDirective() << std::endl;
    std::cout << "    Align:  " << MAI->getAlignDirective() << std::endl;
    std::cout << "    .type:  " << (MAI->hasDotTypeDotSizeDirective() ? "supported" : "unsupported") << std::endl;

    // ========================
    // Part 5: MCInstPrinter Demonstration
    // ========================
    sep("MCInstPrinter: Printing MCInst as Assembly Text");

    // Create an MCInstPrinter.
    std::unique_ptr<MCInstPrinter> IP(
        TheTarget->createMCInstPrinter(Triple(sys::getDefaultTargetTriple()),
                                       MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
    if (!IP) {
        std::cerr << "Failed to create MCInstPrinter\n";
        return 1;
    }

    std::cout << "MCInstPrinter created: " << typeid(*IP).name() << std::endl;

    // Print a few instructions using the printer.
    std::cout << "\nPrinting sample instructions via MCInstPrinter:" << std::endl;

    // Helper: find and print an instruction by property filter.
    auto printSampleInst = [&](const char *label,
                                const std::function<bool(const MCInstrDesc &)> &filter,
                                const std::function<void(MCInst &)> &populator) {
        for (unsigned Opc = 0; Opc < MII->getNumOpcodes(); ++Opc) {
            const MCInstrDesc &Desc = MII->get(Opc);
            if (filter(Desc) && !Desc.isPseudo()) {
                MCInst Inst;
                Inst.setOpcode(Opc);
                populator(Inst);

                SmallString<128> Output;
                raw_svector_ostream OS(Output);
                IP->printInst(&Inst, 0, "", *STI, OS);

                std::cout << "  " << label << ": " << Output;
                if (Output.empty())
                    std::cout << "(empty output - may need specific operands)";
                std::cout << std::endl;
                return true;
            }
        }
        std::cout << "  " << label << ": (no matching instruction found)" << std::endl;
        return false;
    };

    // Find two physical registers for demonstrations.
    const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    MCRegister R0 = 0, R1 = 0;
    for (const TargetRegisterClass *RC : TRI->regclasses()) {
        if (RC->isAllocatable() && RC->getNumRegs() >= 2 && RC->getSizeInBits() >= 32) {
            R0 = RC->getRegister(0);
            R1 = RC->getRegister(1);
            break;
        }
    }

    // Print a return instruction.
    printSampleInst("return", [](const MCInstrDesc &D) { return D.isReturn(); },
                    [](MCInst &I) { /* return typically has no extra operands */ });

    // Print a move instruction (reg to reg).
    if (R0 != 0 && R1 != 0) {
        printSampleInst("move", [](const MCInstrDesc &D) { return D.isMoveReg(); },
                        [R0, R1](MCInst &I) {
                            I.addOperand(MCOperand::createReg(R0));
                            I.addOperand(MCOperand::createReg(R1));
                        });
    }

    // Print an add instruction (reg + reg).
    if (R0 != 0 && R1 != 0) {
        printSampleInst("add", [](const MCInstrDesc &D) { return D.isAdd(); },
                        [R0, R1](MCInst &I) {
                            I.addOperand(MCOperand::createReg(R0));
                            I.addOperand(MCOperand::createReg(R0));
                            I.addOperand(MCOperand::createReg(R1));
                        });
    }

    // Print a load-immediate.
    if (R0 != 0) {
        printSampleInst("load imm", [](const MCInstrDesc &D) { return D.isMoveImm(); },
                        [R0](MCInst &I) {
                            I.addOperand(MCOperand::createReg(R0));
                            I.addOperand(MCOperand::createImm(42));
                        });
    }

    // ========================
    // Part 6: Target-Specific Assembly Features
    // ========================
    sep("Target-Specific Assembly Features");

    std::cout << "isa operands with one assembly syntax:" << std::endl;
    std::cout << "  Some targets (e.g., x86) support multiple assembly dialects" << std::endl;
    std::cout << "  (AT&T vs Intel syntax). MCAsmInfo controls this." << std::endl;
    std::cout << std::endl;

    std::cout << "Pseudo-instructions:" << std::endl;
    std::cout << "  Instructions marked as 'isPseudo' in TableGen are expanded" << std::endl;
    std::cout << "  before assembly emission. Examples:" << std::endl;
    unsigned pseudoCount = 0;
    for (unsigned Opc = 0; Opc < MII->getNumOpcodes() && pseudoCount < 5; ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);
        if (Desc.isPseudo()) {
            std::cout << "    - " << MII->getName(Opc) << std::endl;
            pseudoCount++;
        }
    }
    if (pseudoCount == 0)
        std::cout << "    (no pseudo-instructions found in first scan)" << std::endl;

    // ========================
    // Part 7: llvm-mc Tool Usage (Conceptual)
    // ========================
    sep("llvm-mc: Standalone Assembler/Disassembler");

    std::cout << R"(
The MC layer enables standalone tools that work without the full CodeGen pipeline:

  # Assemble a .s file to object
  $ llvm-mc -triple=x86_64 file.s -filetype=obj -o file.o

  # Disassemble an object file
  $ llvm-objdump -d file.o

  # Show instruction encoding
  $ llvm-mc -triple=x86_64 -show-encoding file.s

  # Show instruction scheduling info
  $ llvm-mca -mtriple=x86_64 file.s

These tools use only the MC layer (MCRegisterInfo, MCInstrInfo,
MCAsmParser, MCCodeEmitter, MCInstPrinter), not the full backend.
    )" << std::endl;

    std::cout << "MC layer assembly emission concepts demonstrated successfully!" << std::endl;
    return 0;
}
