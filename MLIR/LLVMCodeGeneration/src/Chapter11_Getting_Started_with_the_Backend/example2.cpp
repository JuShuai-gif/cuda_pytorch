// example2.cpp - MIR Parser and .mir File Concepts
//
// Demonstrates:
// - Parsing a .mir file using LLVM's MIR parser
// - Understanding the YAML structure of .mir
// - Register class, sub-register, and register unit concepts
// - MachineRegisterInfo inspection
//
// Build with:
//   clang++ -o example2 example2.cpp $(llvm-config --cxxflags --ldflags --libs core codegen mirparser)
//
// Usage: ./example2 <input.mir>
// A minimal .mir file can be created with:
//   echo 'int foo(int a) { return a + 1; }' | clang -x c - -O0 -S -emit-llvm -o - \
//     | llc -stop-after=finalize-isel -o test.mir -simplify-mir

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include <iostream>

using namespace llvm;

// Helper to print separator.
static void printSep(const char *title) {
    std::cout << "\n=== " << title << " ===" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.mir>\n";
        std::cerr << "Generate a .mir file with:\n";
        std::cerr << "  echo 'int f(int a){return a+1;}' | clang -x c - -O0 -S -emit-llvm "
                  << "| llc -stop-after=finalize-isel -o test.mir -simplify-mir\n";
        return 1;
    }

    // Initialize native target components.
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // Load the .mir file.
    auto BufferOrErr = MemoryBuffer::getFile(argv[1]);
    if (!BufferOrErr) {
        std::cerr << "Error reading file: " << BufferOrErr.getError().message() << "\n";
        return 1;
    }
    std::unique_ptr<MemoryBuffer> Buffer = std::move(*BufferOrErr);
    std::cout << "Loaded .mir file: " << argv[1] << " (" << Buffer->getBufferSize() << " bytes)" << std::endl;

    // Create LLVM context and parse MIR.
    LLVMContext Context;

    // Lookup target from the file's triple.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Target lookup error: " << Error << "\n";
        return 1;
    }
    std::cout << "Target: " << TheTarget->getName() << std::endl;

    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "", Options,
                                       std::nullopt));

    auto M = std::make_unique<Module>("mir_module", Context);
    M->setTargetTriple(sys::getDefaultTargetTriple());

    MachineModuleInfo MMI(TM.get());

    // Parse the MIR.
    SourceMgr SM;
    SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
    auto MP = createMIRParser(std::move(SM), Context, MMI, *M);
    if (!MP) {
        std::cerr << "Failed to create MIR parser\n";
        return 1;
    }

    // Parse machine functions.
    auto MaybeMFs = MP->parseMachineFunctions();
    if (!MaybeMFs) {
        std::cerr << "Error: " << toString(MaybeMFs.takeError()) << "\n";
        return 1;
    }

    std::vector<MachineFunction *> MFs = *MaybeMFs;
    std::cout << "Parsed " << MFs.size() << " machine function(s)" << std::endl;

    // Inspect each MachineFunction.
    for (MachineFunction *MF : MFs) {
        const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
        const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
        MachineRegisterInfo &MRI = MF->getRegInfo();

        printSep(("MachineFunction: " + MF->getName()).str().c_str());

        // Function properties.
        std::cout << "Name: " << MF->getName() << std::endl;
        std::cout << "Alignment: " << MF->getAlignment().value() << std::endl;
        std::cout << "Num basic blocks: " << MF->getNumBlockIDs() << std::endl;

        // Check SSA property.
        bool isSSA = MF->getProperties().hasProperty(
            MachineFunctionProperties::Property::IsSSA);
        std::cout << "Is SSA: " << (isSSA ? "true" : "false") << std::endl;

        // Inspect register classes used by virtual registers.
        printSep("Register Classes");
        for (unsigned i = 0, e = MRI.getNumVirtRegs(); i != e; ++i) {
            Register VReg = Register::index2VirtReg(i);
            if (const TargetRegisterClass *RC = MRI.getRegClass(VReg)) {
                std::cout << "  %" << i << " -> class: " << TRI->getRegClassName(RC)
                          << " (size: " << RC->getSizeInBits() << " bits, alignment: "
                          << RC->getAlignment() << ")" << std::endl;
            }
        }

        // Inspect sub-register indices from the target.
        printSep("Sub-Register Indices (first 10)");
        unsigned SRIcount = 0;
        for (unsigned i = 1; SRIcount < 10 && i < 100; ++i) {
            const char *Name = TRI->getSubRegIndexName(i);
            if (Name && Name[0] != '\0') {
                LaneBitmask LM = TRI->getSubRegIndexLaneMask(i);
                std::cout << "  Index " << i << ": " << Name
                          << " (lane mask: 0x" << std::hex << LM.getAsInteger()
                          << std::dec << ")" << std::endl;
                SRIcount++;
            }
        }

        // Inspect register classes defined by the target.
        printSep("Register Classes (first 10 allocatable)");
        unsigned RCcount = 0;
        for (const TargetRegisterClass *RC : TRI->regclasses()) {
            if (RC->isAllocatable() && RCcount < 10) {
                std::cout << "  " << TRI->getRegClassName(RC)
                          << " (size: " << RC->getSizeInBits() << " bits, "
                          << "num regs: " << RC->getNumRegs() << ", "
                          << "align: " << RC->getAlignment() << ")" << std::endl;
                RCcount++;
            }
        }

        // Inspect register units for some physical registers.
        printSep("Register Units (sample)");
        // Get a few physical registers from the first register class.
        if (!TRI->regclasses().empty()) {
            const TargetRegisterClass *FirstRC = TRI->regclasses().front();
            if (FirstRC->getNumRegs() > 0) {
                for (unsigned i = 0; i < std::min(FirstRC->getNumRegs(), 3u); ++i) {
                    MCRegister PhysReg = FirstRC->getRegister(i);
                    std::cout << "  Physical register $" << TRI->getName(PhysReg)
                              << " -> units: { ";
                    for (MCRegUnitIterator UI(PhysReg, TRI); UI.isValid(); ++UI) {
                        std::cout << *UI << " ";
                    }
                    std::cout << "}" << std::endl;

                    // Check sub-registers of this register.
                    std::cout << "    sub-registers: ";
                    bool hasSub = false;
                    for (MCSubRegIterator SRI(PhysReg, TRI); SRI.isValid(); ++SRI) {
                        std::cout << "$" << TRI->getName(*SRI) << " ";
                        hasSub = true;
                    }
                    if (!hasSub) std::cout << "(none)";
                    std::cout << std::endl;

                    // Check super-registers of this register.
                    std::cout << "    super-registers: ";
                    bool hasSuper = false;
                    for (MCSuperRegIterator SupI(PhysReg, TRI); SupI.isValid(); ++SupI) {
                        std::cout << "$" << TRI->getName(*SupI) << " ";
                        hasSuper = true;
                    }
                    if (!hasSuper) std::cout << "(none)";
                    std::cout << std::endl;
                }
            }
        }

        // Inspect instructions in the first basic block.
        printSep("Instructions in First Basic Block");
        if (MF->empty()) {
            std::cout << "  (no basic blocks)" << std::endl;
        } else {
            MachineBasicBlock &MBB = MF->front();
            std::cout << "  BB#" << MBB.getNumber();
            if (MBB.hasAddressTaken())
                std::cout << " [address-taken]";
            std::cout << ", " << MBB.size() << " instructions" << std::endl;

            if (!MBB.succ_empty()) {
                std::cout << "  Successors: ";
                for (MachineBasicBlock *Succ : MBB.successors())
                    std::cout << "BB#" << Succ->getNumber() << " ";
                std::cout << std::endl;
            }

            if (!MBB.livein_empty()) {
                std::cout << "  Live-ins: ";
                for (const MachineBasicBlock::RegisterMaskPair &LI : MBB.liveins())
                    std::cout << "$" << TRI->getName(LI.PhysReg) << " ";
                std::cout << std::endl;
            }

            unsigned InstrCount = 0;
            for (MachineInstr &MI : MBB) {
                if (InstrCount++ >= 8) {
                    std::cout << "  ... (truncated)" << std::endl;
                    break;
                }
                std::cout << "  " << TII->getName(MI.getOpcode());

                // Print operands.
                for (unsigned j = 0, ej = MI.getNumOperands(); j != ej; ++j) {
                    const MachineOperand &MO = MI.getOperand(j);
                    if (MO.isReg()) {
                        Register Reg = MO.getReg();
                        if (MO.isDef()) std::cout << " [def]";
                        if (MO.isImplicit()) std::cout << " [implicit]";
                        if (Reg.isPhysical())
                            std::cout << " $" << TRI->getName(Reg);
                        else
                            std::cout << " %" << Register::virtReg2Index(Reg);
                        if (MO.isDead()) std::cout << "[dead]";
                        if (MO.isKill()) std::cout << "[kill]";
                    } else if (MO.isImm()) {
                        std::cout << " #" << MO.getImm();
                    }
                }
                std::cout << std::endl;
            }
        }

        // Register overlap demonstration.
        printSep("Register Overlap Demo");
        // Pick two physical registers and check if they overlap.
        if (!TRI->regclasses().empty()) {
            const TargetRegisterClass *RC0 = TRI->regclasses().front();
            if (RC0->getNumRegs() >= 2) {
                MCRegister R1 = RC0->getRegister(0);
                MCRegister R2 = RC0->getRegister(1);
                bool overlap = TRI->regsOverlap(R1, R2);
                std::cout << "  $" << TRI->getName(R1) << " overlaps with $"
                          << TRI->getName(R2) << "? " << (overlap ? "yes" : "no") << std::endl;

                // If R1 has sub-registers, check overlap with its own sub-register.
                MCSubRegIterator SRI(R1, TRI);
                if (SRI.isValid()) {
                    bool subOverlap = TRI->regsOverlap(R1, *SRI);
                    std::cout << "  $" << TRI->getName(R1) << " overlaps with sub-register $"
                              << TRI->getName(*SRI) << "? "
                              << (subOverlap ? "yes (expected)" : "no") << std::endl;
                }
            }
        }
    }

    std::cout << "\nMIR parsing and inspection completed successfully!" << std::endl;
    return 0;
}
