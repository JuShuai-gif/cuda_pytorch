// example2.cpp - MC Layer: MCInstLower bridge and MCCodeEmitter concepts
//
// Demonstrates:
// - The conceptual bridge from MachineInstr to MCInst (MCInstLower)
// - Understanding fixups and relocations
// - MCCodeEmitter encoding concepts
// - MCContext and MCExpr for symbol/relocation handling
//
// Build with:
//   clang++ -o example2 example2.cpp $(llvm-config --cxxflags --ldflags --libs core codegen mc)
//

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
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

// Conceptual MCInstLower: demonstrates the pattern of lowering MachineInstr -> MCInst.
// In real LLVM backends, each target implements its own MCInstLower subclass.
class ConceptualMCInstLower {
    MCContext &Ctx;
    const TargetRegisterInfo &TRI;
    const MCInstrInfo &MII;

public:
    ConceptualMCInstLower(MCContext &C, const TargetRegisterInfo &T, const MCInstrInfo &I)
        : Ctx(C), TRI(T), MII(I) {}

    // Lower a MachineInstr to an MCInst.
    MCInst lower(const MachineInstr &MI) const {
        MCInst Out;
        Out.setOpcode(MI.getOpcode());

        // Convert each MachineOperand to MCOperand.
        for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
            const MachineOperand &MO = MI.getOperand(i);
            MCOperand MCOp = lowerOperand(MO);
            if (MCOp.isValid())
                Out.addOperand(MCOp);
        }

        return Out;
    }

private:
    // Lower a single MachineOperand to MCOperand.
    MCOperand lowerOperand(const MachineOperand &MO) const {
        switch (MO.getType()) {
        case MachineOperand::MO_Register: {
            Register Reg = MO.getReg();
            if (Reg.isVirtual()) {
                // In real lowering, virtual registers would have been replaced
                // by physical registers during register allocation.
                // Here we note this is a simplification.
                return MCOperand();
            }
            // Physical register: pass through.
            unsigned SubReg = MO.getSubReg();
            if (SubReg != 0) {
                Reg = TRI.getSubReg(Reg, SubReg);
            }
            return MCOperand::createReg(Reg);
        }

        case MachineOperand::MO_Immediate:
            return MCOperand::createImm(MO.getImm());

        case MachineOperand::MO_GlobalAddress: {
            // Create a symbol for the global and wrap it in an MCExpr.
            MCSymbol *Sym = Ctx.getOrCreateSymbol(
                MO.getGlobal()->getName());
            const MCExpr *Expr = MCSymbolRefExpr::create(
                Sym, MCSymbolRefExpr::VK_None, Ctx);

            // If there's an offset, wrap in a binary expression.
            if (MO.getOffset() != 0) {
                const MCExpr *OffsetExpr = MCConstantExpr::create(MO.getOffset(), Ctx);
                Expr = MCBinaryExpr::createAdd(Expr, OffsetExpr, Ctx);
            }
            return MCOperand::createExpr(Expr);
        }

        case MachineOperand::MO_ExternalSymbol: {
            MCSymbol *Sym = Ctx.getOrCreateSymbol(MO.getSymbolName());
            const MCExpr *Expr = MCSymbolRefExpr::create(
                Sym, MCSymbolRefExpr::VK_None, Ctx);
            if (MO.getOffset() != 0) {
                const MCExpr *OffsetExpr = MCConstantExpr::create(MO.getOffset(), Ctx);
                Expr = MCBinaryExpr::createAdd(Expr, OffsetExpr, Ctx);
            }
            return MCOperand::createExpr(Expr);
        }

        case MachineOperand::MO_MachineBasicBlock: {
            // Basic block reference: create a temporary label symbol.
            MCSymbol *Sym = Ctx.createTempSymbol("BB", true);
            const MCExpr *Expr = MCSymbolRefExpr::create(
                Sym, MCSymbolRefExpr::VK_None, Ctx);
            return MCOperand::createExpr(Expr);
        }

        case MachineOperand::MO_RegisterMask:
            // Register masks are not emitted as MC operands.
            return MCOperand();

        default:
            return MCOperand();
        }
    }
};

int main() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(sys::getDefaultTargetTriple(), Error);
    if (!TheTarget) {
        std::cerr << "Error: " << Error << "\n";
        return 1;
    }

    TargetOptions Options;
    auto TM = std::unique_ptr<TargetMachine>(
        TheTarget->createTargetMachine(sys::getDefaultTargetTriple(), "generic", "",
                                       Options, std::nullopt));

    const MCInstrInfo *MII = TM->getSubtargetImpl()->getInstrInfo();
    const MCRegisterInfo *MRI = TM->getSubtargetImpl()->getMCRegisterInfo();
    const TargetRegisterInfo *TRI = TM->getSubtargetImpl()->getRegisterInfo();
    const MCAsmInfo *MAI = TM->getSubtargetImpl()->getMCAsmInfo();

    // Create MCContext (requires MCRegisterInfo and MCAsmInfo).
    MCContext Ctx(Triple(sys::getDefaultTargetTriple()), MAI, MRI, nullptr);
    ConceptualMCInstLower Lower(Ctx, *TRI, *MII);

    // ========================
    // Part 1: Demonstrate the lowering concept
    // ========================
    sep("MachineInstr to MCInst Lowering (Conceptual)");

    // Build a simple MachineFunction with a MachineInstr to lower.
    LLVMContext LLVMCtx;
    auto M = std::make_unique<Module>("lower_demo", LLVMCtx);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(LLVMCtx), false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "demo", M.get());

    MachineModuleInfo MMI(TM.get());
    MachineFunction &MF = MMI.getOrCreateMachineFunction(*F);
    MachineRegisterInfo &MRI2 = MF.getRegInfo();
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

    MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);

    // Create virtual registers and a simple instruction.
    const TargetRegisterClass *RC = nullptr;
    for (const TargetRegisterClass *RCI : TRI->regclasses()) {
        if (RCI->isAllocatable() && RCI->getSizeInBits() >= 32) {
            RC = RCI;
            break;
        }
    }

    if (RC) {
        Register VReg1 = MRI2.createVirtualRegister(RC);
        Register VReg2 = MRI2.createVirtualRegister(RC);

        // Build a COPY instruction as our test MachineInstr.
        BuildMI(MBB, MBB->end(), DebugLoc(), TII->get(TargetOpcode::COPY), VReg1)
            .addReg(VReg2);

        MachineInstr &MI = MBB->back();
        std::cout << "MachineInstr: " << TII->getName(MI.getOpcode())
                  << " (" << MI.getNumOperands() << " operands)" << std::endl;

        // Attempt to lower it using our conceptual lowerer.
        MCInst LoweredInst = Lower.lower(MI);

        std::cout << "Lowered MCInst: ";
        LoweredInst.dump();
        std::cout << std::endl;

        std::cout << "MCInst operands: " << LoweredInst.size() << std::endl;
        for (unsigned i = 0; i < LoweredInst.size(); ++i) {
            const MCOperand &Op = LoweredInst.getOperand(i);
            std::cout << "  [" << i << "] ";
            if (Op.isReg())   std::cout << "Reg(" << Op.getReg() << ")";
            else if (Op.isImm()) std::cout << "Imm(" << Op.getImm() << ")";
            else if (Op.isExpr()) std::cout << "Expr";
            else              std::cout << "Other";
            std::cout << std::endl;
        }
    }

    // ========================
    // Part 2: MCExpr concepts
    // ========================
    sep("MCExpr: Symbol References and Relocations");

    MCSymbol *FooSym = Ctx.getOrCreateSymbol("foo");
    std::cout << "Created symbol: " << FooSym->getName() << std::endl;
    std::cout << "  Is temporary: " << (FooSym->isTemporary() ? "yes" : "no") << std::endl;
    std::cout << "  Is defined:   " << (FooSym->isDefined() ? "yes" : "no") << std::endl;

    // Create a symbol reference expression.
    const MCExpr *SymExpr = MCSymbolRefExpr::create(
        FooSym, MCSymbolRefExpr::VK_None, Ctx);
    std::cout << "SymbolRefExpr kind: " << (int)SymExpr->getKind() << " (SymbolRef)" << std::endl;

    // Create a constant expression.
    const MCExpr *ConstExpr = MCConstantExpr::create(42, Ctx);
    std::cout << "ConstantExpr value: 42" << std::endl;

    // Create a binary expression: foo + 8.
    const MCExpr *OffsetExpr = MCConstantExpr::create(8, Ctx);
    const MCExpr *BinaryExpr = MCBinaryExpr::createAdd(SymExpr, OffsetExpr, Ctx);
    std::cout << "BinaryExpr: foo + 8 (kind=" << (int)BinaryExpr->getKind() << ")" << std::endl;

    // ========================
    // Part 3: MCInst with Symbolic Operands
    // ========================
    sep("MCInst with Symbolic Operands (call example)");

    unsigned callOpc = 0;
    for (unsigned Opc = 0; Opc < MII->getNumOpcodes(); ++Opc) {
        const MCInstrDesc &Desc = MII->get(Opc);
        if (Desc.isCall() && !Desc.isPseudo()) {
            callOpc = Opc;
            break;
        }
    }

    if (callOpc != 0) {
        std::cout << "Found call instruction: " << MII->getName(callOpc)
                  << " (opcode " << callOpc << ")" << std::endl;

        MCInst CallInst;
        CallInst.setOpcode(callOpc);
        CallInst.addOperand(MCOperand::createExpr(SymExpr));

        std::unique_ptr<MCInstPrinter> IP(
            TheTarget->createMCInstPrinter(Triple(sys::getDefaultTargetTriple()),
                                           MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
        SmallString<128> AsmOut;
        raw_svector_ostream OS(AsmOut);
        IP->printInst(&CallInst, 0, "", *TM->getSubtargetImpl()->getMCSubtargetInfo(), OS);
        std::cout << "Assembly: " << AsmOut << std::endl;
    } else {
        std::cout << "No call instruction found." << std::endl;
    }

    // ========================
    // Part 4: Encoding concepts (fixups)
    // ========================
    sep("Fixup / Relocation Concepts");

    std::cout << "Fixups represent values that the assembler cannot resolve." << std::endl;
    std::cout << "They are later resolved by the linker." << std::endl;
    std::cout << std::endl;
    std::cout << "Common fixup types:" << std::endl;
    std::cout << "  - PC-relative branch offsets" << std::endl;
    std::cout << "  - Absolute addresses of external symbols" << std::endl;
    std::cout << "  - GOT/PLT entries for position-independent code" << std::endl;
    std::cout << std::endl;
    std::cout << "In the MC layer, fixups are represented by:" << std::endl;
    std::cout << "  - MCFixup: offset + value + kind" << std::endl;
    std::cout << "  - MCFixupKindInfo: target-specific fixup type descriptions" << std::endl;
    std::cout << "  - MCCodeEmitter::getFixupKind() maps relocation types" << std::endl;

    // ========================
    // Part 5: Register Encoding Information
    // ========================
    sep("Register Encoding Information (HwEncoding from TableGen)");

    const TargetRegisterClass *FirstRC = nullptr;
    for (const TargetRegisterClass *RCI : TRI->regclasses()) {
        if (RCI->isAllocatable() && RCI->getNumRegs() >= 4) {
            FirstRC = RCI;
            break;
        }
    }

    if (FirstRC) {
        std::cout << "Register class: " << TRI->getRegClassName(FirstRC) << std::endl;
        std::cout << "Register encodings (first 5):" << std::endl;
        for (unsigned i = 0; i < std::min(FirstRC->getNumRegs(), 5u); ++i) {
            MCRegister Reg = FirstRC->getRegister(i);
            unsigned Encoding = MRI->getEncodingValue(Reg);
            std::cout << "  $" << MRI->getName(Reg)
                      << " -> encoding=" << Encoding;
            if (Encoding != (unsigned)Reg) {
                std::cout << " (differs from reg number " << Reg << ")";
            }
            std::cout << std::endl;
        }
    }

    // ========================
    // Part 6: MCSubtargetInfo
    // ========================
    sep("MCSubtargetInfo: Feature Flags");

    const MCSubtargetInfo *STI = TM->getSubtargetImpl()->getMCSubtargetInfo();
    std::cout << "CPU: " << STI->getCPU() << std::endl;
    std::cout << "Feature string: " << STI->getFeatureString() << std::endl;

    SmallVector<StringRef> FeatureNames;
    STI->getFeatureNames(FeatureNames);
    unsigned featCount = 0;
    for (const StringRef &Name : FeatureNames) {
        if (featCount >= 10) {
            std::cout << "  ... (" << FeatureNames.size() << " total)" << std::endl;
            break;
        }
        std::cout << "  " << Name << std::endl;
        featCount++;
    }

    std::cout << "\nMC layer concepts demonstrated successfully!" << std::endl;
    return 0;
}
