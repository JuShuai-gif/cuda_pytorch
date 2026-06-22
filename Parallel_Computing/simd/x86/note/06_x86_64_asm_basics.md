# x86-64 SIMD 编程汇编基础

```
-------------------------------------------------------------------------------
Reference: Modern X86 Assembly Language Programming, 2nd Edition (Kusswurm, 2018)
Audience:  C/C++ programmers transitioning from intrinsics to raw assembly
Prerequisites: basic understanding of x86 ISA and SIMD concepts
-------------------------------------------------------------------------------
```

---

## 1. x86-64 寄存器组

理解完整的寄存器组至关重要，因为 SIMD 代码运行在寄存器资源紧张的环境中。
每次将寄存器溢出到栈上大约需要 5 个时钟周期。

### 1.1 通用寄存器 (GPR)

```
64-bit:  RAX  RBX  RCX  RDX  RSI  RDI  RBP  RSP  R8   R9   R10  R11  R12  R13  R14  R15
32-bit:  EAX  EBX  ECX  EDX  ESI  EDI  EBP  ESP  R8D  R9D  R10D R11D R12D R13D R14D R15D
16-bit:  AX   BX   CX   DX   SI   DI   BP   SP   R8W  R9W  R10W R11W R12W R13W R14W R15W
8-bit:   AL   BL   CL   DL   SIL  DIL  BPL  SPL  R8B  R9B  R10B R11B R12B R13B R14B R15B
          AH   BH   CH   DH   (AH-DH 不能与 REX 前缀一起编码)
```

**常规用途 (System V AMD64 ABI):**

| 寄存器 | 用途 | 调用者/被调用者保存 |
|----------|------|---------------------|
| RAX | 返回值, 累加器 | 调用者 |
| RBX | 被调用者保存的基址指针 | **被调用者** |
| RCX | 第4个整数参数 (MS) / 第4个参数 (SysV) | 调用者 |
| RDX | 第3个整数参数, 128位返回值的高64位 | 调用者 |
| RSI | 第2个整数参数 | 调用者 |
| RDI | 第1个整数参数 | 调用者 |
| RBP | 帧指针 (可选) | **被调用者** |
| RSP | 栈指针 | 特殊 |
| R8-R9 | 第5/第6个整数参数 | 调用者 |
| R10-R11 | 临时 | 调用者 |
| R12-R15 | 通用 | **被调用者** |

**关键规则**: 如果在你的函数中修改了 RBX、RBP 或 R12-R15，你**必须**
将它们保存到栈上，并在 `ret` 之前恢复它们。

### 1.2 SIMD/浮点寄存器

```
AVX-512: ZMM0-ZMM31  (512位, 32个寄存器)
AVX/AVX2: YMM0-YMM15  (256位, 16个寄存器)
SSE:      XMM0-XMM15  (128位, x86-64 模式下有 16 个寄存器)
x87 FPU:  ST(0)-ST(7) (80位, 遗留技术, 新代码中应避免使用)
```

**SIMD 寄存器的 System V ABI 约定:**
- XMM0-XMM7 / YMM0-YMM7 / ZMM0-ZMM7: 用于 float/double 参数
- XMM0-XMM7: **调用者保存**（任何函数调用都可能覆盖它们）
- XMM8-XMM15: System V 中为**调用者保存**（与通用寄存器 R12-R15 不同）
- XMM16-XMM31 (仅 AVX-512): **被调用者保存**

### 1.3 MXCSR - SIMD 控制/状态寄存器

```asm
; 读取 MXCSR
stmxcsr [mem32]

; 写入 MXCSR (设置舍入模式、冲零等)
ldmxcsr [mem32]

; 常用设置:
;   第15位 = 冲零 (FTZ) - 将非规格化数视为零
;   第12位 = 非规格化数为零 (DAZ) - 输入非规格化数 → 零
;   第13-14位 = 舍入控制 (00=最近舍入, 01=向下, 10=向上, 11=向零截断)

; 设置向零舍入 + FTZ + DAZ:
mov eax, 0x9F80     ; 0x1F80 = FTZ+DAZ, 0x8000 = 向零舍入
push rax
ldmxcsr [rsp]
add rsp, 8
```

---

## 2. 调用约定

x86-64 有**两种**主要的调用约定。在将 C/C++ 与手写汇编混合使用时，
理解这两种约定至关重要。

### 2.1 System V AMD64 ABI (Linux, macOS, FreeBSD, 所有 BSD 系统)

**整数/指针参数 (从左到右):**
```
RDI, RSI, RDX, RCX, R8, R9, 然后是栈
```

**浮点参数 (从左到右):**
```
XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, 然后是栈
```

**返回值:**
- 整数/指针: RAX
- Float/double: XMM0
- __m128: XMM0
- __m256: YMM0 (或 XMM0 低128位)
- __m512: ZMM0

**栈对齐:** 调用点处需对齐到 16 字节。`call` 指令会压入一个 8 字节的返回地址，
因此函数入口处栈会偏移 8 字节。如果函数要调用其他函数，必须重新对齐。

```c
// 示例: C 函数签名与寄存器分配
int func(int a, int b, int c, int d, int e, int f, int g);
//        RDI    RSI    RDX    RCX    R8     R9    [RSP+8]
```

**红色区域:** RSP 以下 128 字节的空间，信号处理程序不得触碰。叶子函数
（不调用其他函数的函数）可以使用这块空间而无需调整 RSP。这是 System V
独有的特性；MS x64 没有红色区域。

### 2.2 Microsoft x64 调用约定 (Windows)

**整数/指针参数 (从左到右):**
```
RCX, RDX, R8, R9, 然后是栈
```

**浮点参数:**
```
XMM0, XMM1, XMM2, XMM3, 然后是栈
```

**与 System V 的主要区别:**
- 整数和浮点均仅使用 4 个寄存器传参（vs 6 个整数 + 8 个浮点）
- **影子空间**: 调用者必须在栈上分配 32 字节，即使所有参数都放在寄存器中。
  被调用者可以使用这块空间来溢出寄存器参数。
- **无红色区域**: RSP 以下 128 字节可能被内核使用。
- RBX、RBP、RDI、RSI、R12-R15 为被调用者保存（需要 16 字节对齐）

### 2.3 函数序言/尾声

```asm
; 使用帧指针的最小函数 (System V)
my_func:
    push rbp
    mov  rbp, rsp
    ; ... 函数体 ...
    pop  rbp
    ret

; 使用被调用者保存寄存器和局部变量的函数
my_func_with_locals:
    push rbp
    mov  rbp, rsp
    push rbx          ; 保存被调用者保存寄存器
    push r12
    sub  rsp, 32      ; 分配 32 字节局部存储空间

    ; ... 函数体, push 后 rsp 对齐到 16 字节 ...

    add  rsp, 32
    pop  r12
    pop  rbx
    pop  rbp
    ret
```

**专业提示:** 在不调用其他函数的叶子函数中，你可以使用
红色区域 (System V) 并完全避免调整 RSP:

```asm
; 使用红色区域的叶子函数 (仅限 Linux/macOS)
my_leaf:
    ; RSP-8 到 RSP-128 的空间可以自由使用
    mov [rsp-8],  rdi    ; 将参数溢出到红色区域
    ; ... 不允许使用 call 指令 ...
    mov rax, rdi
    ret
```

---

## 3. x86-64 寻址模式

```asm
; 寄存器寻址
mov eax, ebx                ; eax = ebx

; 立即数寻址
mov eax, 42                 ; eax = 42

; 直接内存寻址
mov eax, [my_variable]      ; eax = *my_variable (64位模式下为 RIP 相对寻址)

; 寄存器间接寻址
mov eax, [rbx]              ; eax = *rbx

; 基址 + 偏移量
mov eax, [rbx + 16]         ; eax = *(rbx + 16)

; 基址 + 索引
mov eax, [rbx + rcx]        ; eax = *(rbx + rcx)

; 基址 + 索引 * 比例因子
mov eax, [rbx + rcx*4]      ; eax = *(rbx + rcx*4)   ; 比例 ∈ {1,2,4,8}

; 完整寻址模式
mov eax, [rbx + rcx*8 + 32] ; eax = *(rbx + rcx*8 + 32)

; RIP 相对寻址 (64位模式下默认, 用 'rel' 显式表示)
mov eax, [rel my_var]       ; eax = *(RIP + offset_to_my_var)
```

**比例因子是数组索引的秘技:**
```asm
; 等价于 C: for (i=0; i<n; i++) sum += arr[i];
; RSI = arr, ECX = i, EAX = sum
loop:
    add  eax, [rsi + rcx*4]  ; sum += arr[i], 每个元素 4 字节
    inc  ecx
    cmp  ecx, edx             ; 比较 i 与 n
    jl   loop
```

---

## 4. SIMD 汇编指令

### 4.1 SSE (128位 XMM)

```asm
; 加载/存储
movaps  xmm0, [rdi]      ; 16字节对齐加载 (未对齐会出错)
movups  xmm0, [rdi]      ; 非对齐加载 (旧 CPU 上较慢)
movdqa  xmm0, [rdi]      ; 整数对齐加载
movdqu  xmm0, [rdi]      ; 整数非对齐加载

; 算术运算 (打包单精度 = .ps = 4x f32)
addps   xmm0, xmm1       ; xmm0 = xmm0 + xmm1 (4个浮点数并行运算)
subps   xmm0, xmm1
mulps   xmm0, xmm1
divps   xmm0, xmm1       ; 开销大! 延迟约 14 个周期
sqrtps  xmm0, xmm1       ; 平方根
rcpps   xmm0, xmm1       ; 近似倒数 (12位精度)
rsqrtps xmm0, xmm1       ; 近似平方根倒数

; 按位运算
andps   xmm0, xmm1       ; 按位与
orps    xmm0, xmm1       ; 按位或
xorps   xmm0, xmm1       ; 按位异或 (用于清零寄存器: xorpd xmm, xmm)

; 比较 (结果为真=全1, 假=全0)
cmpeqps xmm0, xmm1       ; ==
cmpltps xmm0, xmm1       ; <
cmpleps xmm0, xmm1       ; <=
cmpgtps xmm0, xmm1       ; > (注意: 没有 cmpgeps, 交换参数用 cmpleps)

; 洗牌/混合/移动
shufps  xmm0, xmm1, imm8 ; 通道内洗牌
unpcklps xmm0, xmm1      ; 交织低半部分
unpckhps xmm0, xmm1      ; 交织高半部分
movhlps xmm0, xmm1       ; 高到低移动
movlhps xmm0, xmm1       ; 低到高移动
movss   xmm0, xmm1       ; 移动标量单精度 (仅最低元素)
movmskps eax, xmm0        ; 提取符号位 → 4位掩码

; 转换
cvttps2dq xmm0, xmm1     ; f32 → i32 (截断)
cvtps2dq  xmm0, xmm1     ; f32 → i32 (最近舍入)
cvtdq2ps  xmm0, xmm1     ; i32 → f32
```

### 4.2 AVX/AVX2 (256位 YMM, VEX 编码)

AVX 指令使用**三操作数 VEX 编码**，这意味着目标操作数可以
不同于源操作数（不像 SSE 的破坏性双操作数形式）:

```asm
; SSE (双操作数, 破坏性):         AVX (三操作数, 非破坏性):
addps   xmm0, xmm1    ; xmm0 += xmm1    vaddps  xmm2, xmm0, xmm1  ; xmm2 = xmm0 + xmm1

; AVX/AVX2 打包浮点 (8x f32 或 4x f64)
vaddps    ymm0, ymm1, ymm2     ; ymm0 = ymm1 + ymm2
vsubps    ymm0, ymm1, ymm2
vmulps    ymm0, ymm1, ymm2
vdivps    ymm0, ymm1, ymm2

; FMA3 (融合乘加) - 使用 AVX2 而非普通 AVX 的核心原因
vfmadd231ps  ymm0, ymm1, ymm2  ; ymm0 = ymm0 + ymm1 * ymm2
vfmadd132ps  ymm0, ymm1, ymm2  ; ymm0 = ymm0 * ymm2 + ymm1
vfmadd213ps  ymm0, ymm1, ymm2  ; ymm0 = ymm1 * ymm0 + ymm2

; 数字 132/213/231 指定操作数顺序:
;   vfmadd132ps: dst = dst*src2 + src1  (132 = src1*dst + src2*dst → 不对, 实际是:)
;   助记符: 132 表示: 操作数1乘以操作数3, 再加上操作数2
;   对于 intrinsic: _mm256_fmadd_ps(a, b, c) → 如果编译器选择, 则映射为 vfmadd213ps c, a, b

; 广播
vbroadcastss ymm0, [mem]       ; 加载1个浮点数, 复制到所有8个通道

; 置换
vpermilps   ymm0, ymm1, imm8   ; 在每个128位通道内洗牌
vperm2f128  ymm0, ymm1, ymm2, imm8  ; 跨通道组合两个128位半部分
vpermq      ymm0, ymm1, imm8   ; 通道内置换64位元素
```

### 4.3 AVX-512 (512位 ZMM, EVEX 编码)

```asm
; 512位操作 (16x f32)
vaddps    zmm0, zmm1, zmm2        ; zmm0 = zmm1 + zmm2
vfmadd231ps zmm0, zmm1, zmm2      ; zmm0 = zmm0 + zmm1 * zmm2

; 掩码操作 (k1-k7 = 掩码寄存器)
vaddps    zmm0 {k1}, zmm1, zmm2   ; 仅对 k1=1 的通道执行
vaddps    zmm0 {k1}{z}, zmm1, zmm2 ; k1=0 的通道 → 零 (零掩码)

; 嵌入式舍入 (SAE = 抑制所有异常)
vaddps    zmm0, zmm1, zmm2 {rz-sae}  ; 向零舍入, 抑制异常

; 压缩/展开
vcompressps [mem], zmm0, k1       ; 将掩码=1的元素压缩到内存
vexpandps  zmm0, k1, [mem]       ; 将压缩的内存展开到稀疏寄存器

; 收集/散布
vgatherdps zmm0, [rdi + zmm1*4], k1  ; 从索引地址收集
vpscatterdd [rdi + zmm1*4], k1, zmm2 ; 散布到索引地址
```

---

## 5. 分支与循环

### 5.1 条件跳转

```asm
; 比较和跳转模式
cmp  eax, ebx
je   equal_label         ; eax == ebx 时跳转
jne  not_equal_label     ; eax != ebx 时跳转
jg   greater_label       ; eax > ebx 时跳转 (有符号)
jl   less_label          ; eax < ebx 时跳转 (有符号)
jge  greater_equal_label ; eax >= ebx 时跳转 (有符号)
ja   above_label         ; eax > ebx 时跳转 (无符号)
jb   below_label         ; eax < ebx 时跳转 (无符号)

; 测试并跳转
test eax, eax            ; 若 eax == 0 则设置 ZF
jz   zero_label          ; eax == 0 时跳转
jnz  not_zero_label      ; eax != 0 时跳转
```

### 5.2 循环模式

```asm
; 模式 1: 递增循环 (for i=0; i<n; i++)
    xor  ecx, ecx          ; i = 0
loop_start:
    ; ... 循环体, 用 ecx 作索引 ...
    inc  ecx
    cmp  ecx, edx          ; 比较 i 与 n
    jl   loop_start        ; i < n 继续

; 模式 2: 递减循环 (for i=n; i>0; i--)
    mov  ecx, n
loop_down:
    ; ... 循环体 ...
    dec  ecx
    jnz  loop_down         ; ecx != 0 继续

; 模式 3: 带标量尾部的 SIMD 循环
    ; rdi = 数组指针, esi = n
    mov  ecx, esi
    shr  ecx, 3            ; ecx = n / 8 (256位迭代次数)
    jz   scalar_tail       ; n < 8 则跳过 SIMD
simd_loop:
    vmovups ymm0, [rdi]
    ; ... 8元素 SIMD 操作 ...
    add   rdi, 32          ; 指针前移 32 字节 (8个浮点数)
    dec   ecx
    jnz   simd_loop
scalar_tail:
    ; 处理剩余元素
```

### 5.3 SETcc 和 CMOVcc (无分支)

```asm
; SETcc: 条件设置字节为 0 或 1
cmp   eax, ebx
sete  al                  ; al = (eax == ebx) ? 1 : 0

; CMOVcc: 条件移动 (避免分支预测失败!)
cmp   eax, ebx
cmovg eax, ebx            ; if eax > ebx, eax = ebx (max 函数, 无分支!)
```

---

## 6. 汇编与 C/C++ 混合编程

### 6.1 外部汇编函数 (`.s` 或 `.asm` 文件)

**[汇编文件: `my_kernel.s`]**
```asm
    .intel_syntax noprefix
    .text
    .globl  my_vec_add

# void my_vec_add(const float* a, const float* b, float* c, int n)
# System V: RDI=a, RSI=b, RDX=c, ECX=n
my_vec_add:
    test ecx, ecx
    jle  .Ldone

.Lloop:
    vmovups ymm0, [rdi]
    vmovups ymm1, [rsi]
    vaddps  ymm0, ymm0, ymm1
    vmovups [rdx], ymm0
    add    rdi, 32
    add    rsi, 32
    add    rdx, 32
    sub    ecx, 8
    jg     .Lloop

.Ldone:
    ret
```

**[C/C++ 文件: `main.cpp`]**
```c
extern "C" void my_vec_add(const float* a, const float* b, float* c, int n);

int main() {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};
    float c[8];
    my_vec_add(a, b, c, 8);
    return 0;
}
```

**[编译命令]**
```bash
# 汇编为目标文件
as -o my_kernel.o my_kernel.s

# 或使用 GCC 作为汇编器前端
gcc -c -o my_kernel.o my_kernel.s

# 编译并链接
gcc -mavx2 -O2 -o prog main.cpp my_kernel.o
```

### 6.2 内联汇编 (GCC/Clang 扩展汇编)

```c
// 单个 SIMD 指令的内联汇编
__m256 vec_add_inline(__m256 a, __m256 b) {
    __m256 result;
    __asm__ __volatile__(
        "vaddps %1, %2, %0"
        : "=v"(result)           // 输出操作数
        : "v"(a), "v"(b)        // 输入操作数
        :                        // 无破坏寄存器
    );
    return result;
}

// 带内存破坏的内联汇编
void zero_array_inline(float* arr, int n) {
    __asm__ __volatile__(
        "xor %%eax, %%eax\n\t"
        "rep stosl"
        : "+D"(arr), "+c"(n)    // RDI=arr, ECX=n (两者均为读写)
        : "a"(0)                // EAX=0
        : "memory"              // 内存被修改
    );
}
```

**何时使用内联汇编 vs 外部 .s 文件:**
- **内联汇编**: 单指令包装器、CPUID、访问 MSR、需要编译器寄存器分配的短序列
  （少于 5 条指令）
- **外部 .s 文件**: 完整函数、循环、复杂的 SIMD 核心，需要
  完全控制寄存器分配和指令调度

### 6.3 MASM (Microsoft 宏汇编器) vs GAS (GNU 汇编器)

本书使用 MASM 语法。移植时需要注意的关键差异:

| 特性 | MASM (Windows) | GAS `.intel_syntax` (Linux) |
|---------|----------------|---------------------------|
| 注释 | `; 注释` | `# 注释` 或 `// 注释` |
| 标签 | `label:` 或 `@@:` (匿名) | `label:` 或 `.Llabel:` (局部) |
| 指令 | `.code`, `.data`, `.const` | `.text`, `.data`, `.rodata` |
| 过程 | `proc` / `endp` | `.globl name` + `name:` |
| 局部跳转 | `@B` (向后), `@F` (向前) | `.L0`, `.L1` 等 |
| 导出 | 自动 | `.globl name` |
| 段 | `.code` = `.text` | `.text` |

```asm
; === MASM ===                                # === GAS (.intel_syntax) ===
.code                                         .text
                                              .globl IntegerAddSub
IntegerAddSub_ proc                           IntegerAddSub:
    mov eax, ecx                                  mov eax, edi    # SysV: 第1个参数在 EDI
    add eax, edx                                  add eax, esi    # 第2个参数在 ESI
    add eax, r8d                                  add eax, edx    # 第3个参数在 EDX
    sub eax, r9d                                  sub eax, ecx    # 第4个参数在 ECX
    ret                                           ret
IntegerAddSub_ endp
```

---

## 7. 从 C 调用汇编: 完整示例

### 7.1 整数操作 (System V)

**[汇编文件: `calc_sum.s`]**
```asm
    .intel_syntax noprefix
    .text
    .globl  CalcArraySum

# int CalcArraySum(const int* arr, int n);
# RDI = arr 指针, ESI = n
CalcArraySum:
    xor    eax, eax        # sum = 0
    test   esi, esi
    jle    .Ldone

.Lloop:
    add    eax, [rdi]      # sum += *arr
    add    rdi, 4          # arr++ (每个 int 4 字节)
    dec    esi             # n--
    jnz    .Lloop

.Ldone:
    ret
```

**[C 文件: `test_sum.cpp`]**
```c
#include <cstdio>

extern "C" int CalcArraySum(const int* arr, int n);

int main() {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int sum = CalcArraySum(arr, 10);
    printf("Sum = %d (expected 55)\n", sum);
    return sum != 55;
}
```

**[构建命令]**
```bash
as -o calc_sum.o calc_sum.s
g++ -o test_sum test_sum.cpp calc_sum.o
./test_sum
# 输出: Sum = 55 (expected 55)
```

### 7.2 SIMD 操作 (System V)

**[汇编文件: `simd_add.s`]**
```asm
    .intel_syntax noprefix
    .text
    .globl  SimdAddF32

# void SimdAddF32(const float* a, const float* b, float* c, int n)
# RDI=a, RSI=b, RDX=c, ECX=n
SimdAddF32:
    test   ecx, ecx
    jle    .Ldone_avx

    # 对齐到 32 字节边界 (可选优化)
    # ... 对齐循环省略, 以求简洁 ...

    # 主 AVX2 循环: 每次迭代 8 个浮点数
    shr    ecx, 3             # n /= 8
    jz     .Lscalar_tail

.Lavx_loop:
    vmovups  ymm0, [rdi]      # 从 a 加载 8 个浮点数
    vmovups  ymm1, [rsi]      # 从 b 加载 8 个浮点数
    vaddps   ymm0, ymm0, ymm1 # 相加
    vmovups  [rdx], ymm0      # 存储结果
    add      rdi, 32
    add      rsi, 32
    add      rdx, 32
    dec      ecx
    jnz      .Lavx_loop

.Lscalar_tail:
    # (处理剩余的 n % 8 个元素)

.Ldone_avx:
    ret
```

---

## 8. 常见陷阱与最佳实践

### 8.1 栈对齐

System V ABI 要求在 `call` 指令之前，栈必须**对齐到 16 字节**。
这意味着在函数入口处，`RSP % 16 == 8`（因为 `call` 压入了一个
8 字节的返回地址）。如果你的函数调用其他函数，需要将 RSP 调整一个
8 的奇数倍来修正对齐:

```asm
my_func:
    push rbp
    mov  rbp, rsp
    push rbx              ; 现在 RSP % 16 == 0 (两次 8 字节 push = 16)
    
    ; 这里可以安全调用其他函数了
    
    pop  rbx
    pop  rbp
    ret
```

### 8.2 System V 的红色区域

RSP 以下 128 字节（"红色区域"）对叶子函数是安全的，但会被
**信号处理程序破坏**。实践中:
- **叶子函数**（无 `call`）: 自由使用红色区域
- **非叶子函数**: 不要使用红色区域；改为调整 RSP

### 8.3 MXCSR 状态

MXCSR 寄存器在每个线程中全局控制浮点行为。如果在汇编中修改了它
（用于 FTZ、DAZ 或非默认舍入），在将控制权返回给 C/C++ 代码之前
需要恢复:

```asm
stmxcsr [rsp - 4]       ; 保存当前 MXCSR
mov    eax, [rsp - 4]    ; 读取值
or     eax, 0x8040       ; 设置 FTZ + DAZ 位
mov    [rsp - 8], eax
ldmxcsr [rsp - 8]       ; 应用新 MXCSR

; ... 用 FTZ/DAZ 做计算 ...

ldmxcsr [rsp - 4]        ; 恢复原始 MXCSR
```

### 8.4 Intel 语法 vs AT&T 语法

本书使用 **Intel 语法** (MASM)。GCC/GAS 默认使用 **AT&T 语法**。

```asm
; AT&T 语法 (GAS 默认):                     ; Intel 语法 (GAS 配合 .intel_syntax):
movl %eax, %ebx                             mov eax, ebx
addl $4, (%rdi)                             add dword ptr [rdi], 4
vmovups (%rdi), %ymm0                       vmovups ymm0, [rdi]
vaddps %ymm1, %ymm0, %ymm2                 vaddps ymm2, ymm0, ymm1
```

要在 GAS 中使用 Intel 语法，在文件顶部添加 `.intel_syntax noprefix`。
`noprefix` 变体省略了寄存器名称的 `%` 前缀。

### 8.5 CPUID 寄存器破坏 (关键!)

**重要**: 在某些 CPU 实现上，`cpuid` 指令可能会破坏
EAX、EBX、ECX、EDX 之外的寄存器。具体来说，在 AMD CPU 上 **RDX
（高32位）可能被清零**，而在某些 Intel 微架构上，完整的 RDX 寄存器
也可能受到影响。务必保存跨越 `cpuid` 需要使用的寄存器:

```asm
; 正确做法: 在 cpuid 之前保存 rdx
asm_cpuid_raw:
    push   rbx
    push   rdx            # 保存输出指针; cpuid 可能修改 RDX!
    mov    eax, edi
    mov    ecx, esi
    cpuid
    pop    rcx            # 恢复保存的 rdx (通过 rcx 间接)
    mov    [rcx],     eax
    mov    [rcx + 4], ebx
    mov    [rcx + 8], ecx
    mov    [rcx + 12],edx
    pop    rbx
    ret
```

这是一个将 Windows MASM 代码（RDX 是第2个参数而非第3个）
移植到 Linux System V 时的常见 bug。

### 8.6 VZEROUPPER: AVX-SSE 状态转换开销

在同一函数中混合使用 AVX/AVX2 (YMM) 指令和 SSE (XMM) 指令时，
在写入 XMM 寄存器的 SSE 指令之前**必须**执行 `vzeroupper`。
否则，CPU 会将 YMM 寄存器的高128位维持在"脏"状态，导致每次转换
约 70 个周期的开销:

```asm
; 正确做法: 在 SSE 标量尾部之前清除 YMM 高位状态
    vmovups  ymm0, [rdi]     # AVX 加载 (写入完整的 256位 YMM0)
    vaddps   ymm0, ymm0, ymm1
    vmovups  [rdx], ymm0
    vzeroupper               # ← 必须: 清除所有 YMM 寄存器的高位
    movss    xmm0, [rdi]     # SSE 标量操作 (现在安全了)
    addss    xmm0, [rsi]
```

现代编译器会在函数返回前自动插入 `vzeroupper`，但
手写汇编必须显式执行。开销发生在以下情况:
- 任何 YMM/ZMM 寄存器的高位非零
- 且后续 SSE 指令写入 XMM 寄存器
- CPU 必须在内部保存/恢复完整的 256/512 位状态

### 8.7 NASM vs MASM vs GAS

| 特性 | NASM | MASM | GAS |
|---------|------|------|-----|
| 默认语法 | Intel | Intel | AT&T |
| `.intel_syntax` | 不适用 (原生) | 不适用 (原生) | 支持 |
| 跨平台 | Linux, macOS, Windows | 仅 Windows | 全部 |
| 宏系统 | 强大 | 非常强大 | 基础 |
| 段指令 | `SECTION .text` | `.code` | `.text` |
| 数据指令 | `dd 42` (int32), `dq 3.14` (float64) | `dd 42`, `real4 3.14` | `.long 42`, `.single 3.14` |
| 最适合 | 可移植汇编 | Windows 内核/系统 | Linux/Unix |

---

## 9. 调试汇编

### 9.1 汇编 GDB 命令

```bash
gdb ./my_program

# 设置 Intel 反汇编语法 (推荐)
(gdb) set disassembly-flavor intel

# 反汇编当前函数
(gdb) disas

# 逐条汇编指令单步执行
(gdb) stepi
(gdb) nexti

# 查看寄存器
(gdb) info registers
(gdb) info registers ymm0    # SIMD 寄存器

# 查看内存
(gdb) x/8fw $rdi             # RDI 处的 8 个浮点数

# 在汇编标签处设置断点
(gdb) b *my_func+0x20

# 以各种格式打印寄存器
(gdb) p $rax
(gdb) p/x $ymm0.v8_float     # 将 ymm0 打印为 8 个浮点数
```

### 9.2 用 Objdump 做静态分析

```bash
# 使用 Intel 语法和源代码交织来反汇编
objdump -d -M intel --no-show-raw-insn ./my_program | less

# 仅显示特定段
objdump -d -M intel -j .text ./my_program

# 显示动态符号表 (检查导出)
objdump -T ./my_program | grep Calc
```

---

## 10. 性能考量

### 10.1 µop 分解

现代 x86 CPU 将 CISC 指令解码为类似 RISC 的 µop。理解
µop 数量和端口使用情况对于编写快速汇编至关重要:

```bash
# 使用 llvm-mca 分析指令吞吐量
llvm-mca -mcpu=skylake -iterations=1000 my_kernel.s
```

### 10.2 延迟 vs 吞吐量

```asm
; 高延迟, 低吞吐量: "串行" 指令
vdivps ymm0, ymm0, ymm1      ; 延迟 ~11, 吞吐量 ~5 (每5个周期一条)

; 低延迟, 高吞吐量: "并行" 指令
vaddps ymm0, ymm0, ymm1      ; 延迟 ~4, 吞吐量 ~0.5 (每周期2条)
vfmadd231ps ymm0, ymm1, ymm2 ; 延迟 ~4, 吞吐量 ~0.5 (每周期2条)

; 规则: 用展开来隐藏延迟, 使用独立的累加器
;  1 个累加器:  v0 = v0 + vec   → 每两次迭代等待 4 个周期
;  4 个累加器: v0..v3 独立 → 4 个在流水线中, 无停顿
```

### 10.3 端口压力

Skylake 关键 SIMD 指令的端口映射:
- 端口 0: FMA、add、mul、shuffle、整数 ALU
- 端口 1: FMA、add、mul、shuffle、整数 ALU
- 端口 5: shuffle、permute、hadd、blend、整数 ALU
- 端口 2、3、4: 加载/存储地址生成
- 端口 7: 存储地址生成

**瓶颈诊断**: 如果你的循环很慢，检查哪个端口已饱和。

---

## 11. 扩展阅读

| 资源 | 描述 |
|----------|-------------|
| [System V ABI AMD64](https://gitlab.com/x86-psABIs/x86-64-ABI) | System V AMD64 ABI 官方规范 |
| [Intel SDM Vol.2](https://www.intel.com/sdm) | 指令集参考 (完整操作码列表) |
| [uops.info](https://uops.info/) | 指令延迟/吞吐量/端口使用数据库 |
| Agner Fog 优化手册 | 微架构深入分析, 指令表 |
| [Compiler Explorer](https://godbolt.org/) | 实时查看 C++ → 汇编的对应关系 |
| Kusswurm《Modern X86 Assembly Language Programming》第2版 | 本文档的参考书籍 |

(文件结束 - 共 834 行)
