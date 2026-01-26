#pragma once

// 将通用地址转换为共享内存地址
// addr: 目标变量，用于存储转换后的共享内存地址
// smem_ptr: 要转换的原始指针
#define CVTA_TO_SHARED_PTX(addr, smem_ptr) \
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr));

// 带条件保护的全局内存加载（32位浮点）
// 仅当guard不为0时才执行加载
// reg: 目标寄存器
// ptr: 源地址指针
// guard: 条件保护值
#define LDG32_GUARD_PTX(reg, ptr, guard)               \
    {                                                  \
        asm volatile("{.reg .pred p;\n\t"              \
                     "setp.ne.u32 p, %2, 0;\n\t"       \
                     "@p ld.global.f32 %0, [%1];}\n\t" \
                     : "=f"(reg)                       \
                     : "l"(ptr), "r"(guard));          \
    }

// 带条件保护的全局内存加载（32位浮点），并在条件不满足时将寄存器置0
#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard)          \
    {                                                  \
        asm volatile("{.reg .pred p;\n\t"              \
                     "setp.ne.u32 p, %2, 0;\n\t"       \
                     "@!p mov.b32 %0, 0;\n\t"          \
                     "@p ld.global.f32 %0, [%1];}\n\t" \
                     : "=f"(reg)                       \
                     : "l"(ptr), "r"(guard));          \
    }

// 将4个32位浮点寄存器存储到共享内存（128位连续存储）
// reg0-reg3: 要存储的4个浮点寄存器
// addr: 目标共享内存地址
#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                               \
    {                                                                          \
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t"            \
                     :                                                         \
                     : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)); \
    }

// 从共享内存加载4个32位浮点到寄存器（128位连续加载）
// reg0-reg3: 用于接收加载数据的4个浮点寄存器
// addr: 源共享内存地址
#define LDS128_PTX(reg0, reg1, reg2, reg3, addr)                      \
    {                                                                 \
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n\t"   \
                     : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3) \
                     : "l"(addr));                                    \
    }

// 将单个32位浮点寄存器存储到共享内存
// reg: 要存储的浮点寄存器
// addr: 目标共享内存地址
#define STS32_PTX(reg, addr)                                               \
    {                                                                      \
        asm volatile("st.shared.f32 [%0], %1;\n" : : "l"(addr), "f"(reg)); \
    }

// 带条件保护的全局内存存储（32位浮点）
// 仅当guard不为0时才执行存储
// reg: 要存储的浮点寄存器
// ptr: 目标地址指针
// guard: 条件保护值
#define STG32_GUARD_PTX(reg, ptr, guard)                \
    {                                                   \
        asm volatile("{.reg .pred p;\n\t"               \
                     "setp.ne.u32 p, %2, 0;\n\t"        \
                     "@p st.global.f32 [%0], %1;}\n\t"  \
                     :                                  \
                     : "l"(ptr), "f"(reg), "r"(guard)); \
    }

// 提交异步操作组（用于异步内存拷贝）
#define COMMIT_GROUP_PTX asm volatile("cp.async.commit_group;");

// 等待指定数量的异步操作组完成
// N: 要等待的异步操作组数量
#define WAIT_GROUP_PTX(N) asm volatile("cp.async.wait_group %0;" : : "n"(N))

// 等待所有异步操作完成
#define WAIT_ALL_PTX asm volatile("cp.async.wait_all ;")

// 带条件保护的异步内存拷贝（从全局内存到共享内存）
// 仅当guard不为0时才执行拷贝
// addr: 目标共享内存地址
// ptr: 源全局内存地址
// guard: 条件保护值
#define CP_ASYNC_GUARD_PTX(addr, ptr, guard)                          \
    {                                                                 \
        asm volatile("{.reg .pred p;\n\t"                             \
                     "setp.ne.u32 p, %2, 0;\n\t"                      \
                     "@p cp.async.ca.shared.global [%0], [%1], 4;}\n" \
                     :                                                \
                     : "l"(addr), "l"(ptr), "r"(guard));              \
    }

// 带条件保护的异步内存拷贝（当guard为0时忽略源地址）
// 与CP_ASYNC_GUARD_PTX行为相反
#define CP_ASYNC_IGNORE_SRC_PTX(addr, ptr, guard)                     \
    {                                                                 \
        asm volatile("{.reg .pred p;\n\t"                             \
                     "setp.eq.u32 p, %2, 0;\n\t"                      \
                     "cp.async.ca.shared.global [%0], [%1], 4, p;}\n" \
                     :                                                \
                     : "l"(addr), "l"(ptr), "r"(guard));              \
    }
