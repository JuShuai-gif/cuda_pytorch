# x86_64_calling_convention.s
# System V AMD64 ABI (Linux/macOS/FreeBSD) 手写汇编函数。
#
# 演示内容:
#   1. 整数数组求和 (标量循环)
#   2. AVX2 浮点向量加法 (SIMD 循环)
#   3. CPUID 和 XGETBV 包装函数
#
# 构建方法:
#   as -o x86_64_calling_convention.o x86_64_calling_convention.s
#   g++ -mavx2 -O2 main.cpp x86_64_calling_convention.o -o prog
#
# 调用约定 (System V AMD64 ABI):
#   RDI = 参数1, RSI = 参数2, RDX = 参数3, RCX = 参数4, R8 = 参数5, R9 = 参数6
#   返回值: RAX (整数/指针), XMM0 (float/double), YMM0 (__m256), ZMM0 (__m512)
#   被调用者保存: RBX, RBP, R12-R15
#   调用者保存: RAX, RCX, RDX, RSI, RDI, R8-R11, XMM0-XMM15

    .intel_syntax noprefix
    .text

# ---------------------------------------------------------------------------
# int asm_array_sum(const int* arr, int n)
#   RDI = arr 指针, ESI = n
#   返回求和结果到 EAX
# ---------------------------------------------------------------------------
    .globl  asm_array_sum
    .type   asm_array_sum, @function

asm_array_sum:
    xor    eax, eax              # sum = 0
    test   esi, esi               # 如果 n <= 0
    jle    .Las_sum_done

.Las_sum_loop:
    add    eax, [rdi]             # sum += *arr
    add    rdi, 4                 # arr++ (每个 int 占 4 字节)
    dec    esi                    # n--
    jnz    .Las_sum_loop

.Las_sum_done:
    ret
    .size   asm_array_sum, .-asm_array_sum

# ---------------------------------------------------------------------------
# void asm_simd_add_f32(const float* a, const float* b, float* c, int n)
#   RDI = a, RSI = b, RDX = c, ECX = n
#   使用 AVX2 8 路 SIMD (链接时需要 -mavx2)
# ---------------------------------------------------------------------------
    .globl  asm_simd_add_f32
    .type   asm_simd_add_f32, @function

asm_simd_add_f32:
    test   ecx, ecx               # 如果 n <= 0
    jle    .Las_simd_done

    # 计算 8 元素迭代次数
    mov    r8d, ecx
    shr    r8d, 3                  # r8d = n / 8
    jz     .Las_simd_scalar_tail

.Las_simd_loop:
    vmovups  ymm0, [rdi]          # 从 a 加载 8 个浮点数
    vmovups  ymm1, [rsi]          # 从 b 加载 8 个浮点数
    vaddps   ymm0, ymm0, ymm1     # ymm0 = ymm0 + ymm1
    vmovups  [rdx], ymm0          # 存储结果
    add      rdi, 32              # 指针前进 32 字节 (8 个浮点数)
    add      rsi, 32
    add      rdx, 32
    dec      r8d
    jnz      .Las_simd_loop

.Las_simd_scalar_tail:
    vzeroupper                    # 在 SSE 标量操作前清除 YMM 高位状态
    # 计算剩余元素数 (n % 8)
    and    ecx, 7
    jz     .Las_simd_done

.Las_simd_tail_loop:
    movss  xmm0, [rdi]            # 从 a 加载 1 个浮点数
    addss  xmm0, [rsi]            # 与 b 的 1 个浮点数相加 (浮点加法!)
    movss  [rdx], xmm0            # 存储结果
    add    rdi, 4
    add    rsi, 4
    add    rdx, 4
    dec    ecx
    jnz    .Las_simd_tail_loop

.Las_simd_done:
    ret
    .size   asm_simd_add_f32, .-asm_simd_add_f32

# ---------------------------------------------------------------------------
# void asm_cpuid_raw(uint32_t leaf, uint32_t subleaf, CpuidRegs* out)
#   EDI = leaf, ESI = subleaf, RDX = out 指针
#   执行 CPUID 指令, 将结果存入 *out
# ---------------------------------------------------------------------------
    .globl  asm_cpuid_raw
    .type   asm_cpuid_raw, @function

asm_cpuid_raw:
    push   rbx                    # 保存被调用者保存寄存器
    push   rdx                    # 保存 out 指针; cpuid 在某些 CPU 上可能会破坏 RDX

    mov    eax, edi               # leaf → EAX
    mov    ecx, esi               # subleaf → ECX
    cpuid

    pop    rcx                    # 恢复 out 指针
    mov    [rcx + 0],  eax        # out->eax
    mov    [rcx + 4],  ebx        # out->ebx
    mov    [rcx + 8],  ecx        # out->ecx
    mov    [rcx + 12], edx        # out->edx

    pop    rbx
    ret
    .size   asm_cpuid_raw, .-asm_cpuid_raw

# ---------------------------------------------------------------------------
# uint64_t asm_xgetbv(uint32_t ecx)
#   ECX = 寄存器索引 (0 = XCR0)
#   将 64 位返回值合并到 RAX (低 32 位) + RDX (高 32 位) 后返回
# ---------------------------------------------------------------------------
    .globl  asm_xgetbv
    .type   asm_xgetbv, @function

asm_xgetbv:
    # ECX 中已有寄存器索引
    # 在 System V 下, ECX 作为第 4 个参数传入 (RCX 的低 32 位),
    # 但此函数仅接受一个参数, 所以它在 EDI 中。
    # 我们需要将其移入 ECX 以供 xgetbv 使用。
    mov    ecx, edi
    xgetbv                        # 结果: EDX:EAX
    shl    rdx, 32                # 将高位部分移至 RDX 高 32 位
    or     rax, rdx               # 合并为 64 位返回值
    ret
    .size   asm_xgetbv, .-asm_xgetbv

# ---------------------------------------------------------------------------
# GNU ELF 注释
# ---------------------------------------------------------------------------
    .section .note.GNU-stack,"",@progbits
