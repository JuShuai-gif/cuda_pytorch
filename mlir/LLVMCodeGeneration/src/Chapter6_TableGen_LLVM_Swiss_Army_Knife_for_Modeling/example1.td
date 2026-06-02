//===-- example1.td - TableGen Language Demo ------------------------------===//
// This file demonstrates the TableGen programming language features
// described in Chapter 6: classes, defs, multiclasses, let, foreach, dag.
//
// Usage: llvm-tblgen example1.td -print-records
//        llvm-tblgen example1.td -dump-json
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// 1. Basic Types and Classes
// ---------------------------------------------------------------------------

// A simple class with various field types
class Register<string name, int width> {
  string Name = name;
  int BitWidth = width;
  bits<16> Encoding = 0;
  list<string> Aliases = [];
}

// Class inheritance: a register that is also usable for floating-point
class FPR<Register r> : Register<r.Name, r.BitWidth> {
  // Inherits Name and BitWidth from parent
  let Aliases = r.Aliases;
  bit IsFloat = 1;
}

// ---------------------------------------------------------------------------
// 2. Defs - creating concrete records
// ---------------------------------------------------------------------------

def R0  : Register<"r0",  32> { let Encoding = 0x0000; }
def R1  : Register<"r1",  32> { let Encoding = 0x0001; }
def R2  : Register<"r2",  32> { let Encoding = 0x0002; }
def SP  : Register<"sp",  32> { let Encoding = 0x000D; }
def LR  : Register<"lr",  32> { let Encoding = 0x000E; let Aliases = ["r14"]; }
def PC  : Register<"pc",  32> { let Encoding = 0x000F; let Aliases = ["r15"]; }

// Floating-point registers - using class inheritance
def F0 : FPR<R0> { let IsFloat = 1; }
def F1 : FPR<R1> { let IsFloat = 1; }

// ---------------------------------------------------------------------------
// 3. Multiclass - generating multiple related records at once
// ---------------------------------------------------------------------------

// Base instruction class
class Instruction<string mnemonic, bits<8> opcode> {
  string Mnemonic = mnemonic;
  bits<8> Opcode = opcode;
  bit IsTwoAddress = 0;   // Does this instruction have 2-address encoding?
  bit HasSideEffects = 0;
  dag Pattern = (ops);     // DAG type for selection patterns
}

// A multiclass for binary arithmetic operations that have
// both register-register and register-immediate variants
multiclass BinaryArith<string name, bits<8> baseOpcode> {
  // Register-register variant: ADD  r1, r2, r3
  def rr : Instruction<name, baseOpcode> {
    let Pattern = (ops R0, R1, R2);
    let IsTwoAddress = 0;
  }

  // Register-immediate variant: ADDI r1, r2, #42
  def ri : Instruction<name, baseOpcode> {
    let Pattern = (ops R0, R1);
    let IsTwoAddress = 1;
    let Opcode = baseOpcode;  // override if needed
  }
}

// ---------------------------------------------------------------------------
// 4. defm - instantiating multiclasses
// ---------------------------------------------------------------------------

// Creates ADD and ADD_rr, ADD_ri
defm ADD : BinaryArith<"add", 0x01>;

// Creates SUB and SUB_rr, SUB_ri
defm SUB : BinaryArith<"sub", 0x02>;

// Creates MUL and MUL_rr, MUL_ri
defm MUL : BinaryArith<"mul", 0x03>;

// ---------------------------------------------------------------------------
// 5. Let statement - setting fields in scope
// ---------------------------------------------------------------------------

// Use let to set default values for a group of definitions
let HasSideEffects = 1 in {
  // Memory operations have side effects
  def LOAD  : Instruction<"ldr", 0x10> {
    let Pattern = (ops R0, R1);
  }
  def STORE : Instruction<"str", 0x11> {
    let Pattern = (ops R0, R1);
  }
}

// ---------------------------------------------------------------------------
// 6. foreach loop - generating records from iteration
// ---------------------------------------------------------------------------

// Generate branch instructions for different conditions
foreach cond = ["eq", "ne", "gt", "lt", "ge", "le"] in {
  def B#cond : Instruction<"b"#cond, 0x20> {
    let Pattern = (ops R0);
  }
}

// This creates: Beq, Bne, Bgt, Blt, Bge, Ble

// ---------------------------------------------------------------------------
// 7. Nested multiclass and complex record relationships
// ---------------------------------------------------------------------------

// A class to describe an operand
class Operand<string name, bits<4> encoding> {
  string Name = name;
  bits<4> Encoding = encoding;
}

// Operand types
def imm8  : Operand<"imm8",  0x1>;
def imm16 : Operand<"imm16", 0x2>;
def reg   : Operand<"reg",   0x3>;
def mem   : Operand<"mem",   0x4>;

// ---------------------------------------------------------------------------
// 8. Instruction set description with DAG patterns
// ---------------------------------------------------------------------------

// A more complete description using DAGs
class ALUInst<string mnemonic, bits<8> opcode, dag pattern>
    : Instruction<mnemonic, opcode> {
  dag Pattern = pattern;
}

// Define some ALU instructions with selection patterns
def AND : ALUInst<"and", 0x04,
    (ops (reg R0), (reg R1), (reg R2))>;

def OR  : ALUInst<"orr", 0x05,
    (ops (reg R0), (reg R1), (reg R2))>;

def XOR : ALUInst<"eor", 0x06,
    (ops (reg R0), (reg R1), (reg R2))>;

// ---------------------------------------------------------------------------
// 9. Intrinsic-like pattern example
// ---------------------------------------------------------------------------

class Intrinsic<string name, list<int> param_types> {
  string Name = name;
  list<int> ParamTypes = param_types;
  bit HasSideEffects = 0;
  bit IsConvergent = 0;
}

def int_sqrt : Intrinsic<"llvm.sqrt", [32]> {
  // sqrt is convergent but has no side effects
  let IsConvergent = 1;
}

def int_memcpy : Intrinsic<"llvm.memcpy", [32, 32, 32]> {
  // memcpy has side effects (writes to memory)
  let HasSideEffects = 1;
}

def int_sadd_overflow : Intrinsic<"llvm.sadd.with.overflow", [32, 32]>;

// ---------------------------------------------------------------------------
// 10. Record composition example
// ---------------------------------------------------------------------------

// Describe a complete instruction by composing multiple records
class FullInstr<Instruction inst, list<Operand> operands, string asm> {
  Instruction Instr = inst;
  list<Operand> Operands = operands;
  string AsmString = asm;
}

// Add with carry flag
def ADD_rrr : FullInstr<ADD_rr, [reg, reg, reg], "$dst, $src1, $src2">;
// Load immediate
def MOV_ri  : FullInstr<ADD_ri, [reg, imm16], "$dst, #$imm">;
