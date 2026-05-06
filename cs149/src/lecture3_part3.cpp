// lecture3_part3.cpp - 内存带宽流水线 + 数据移动分析
// =============================================================================
// CS149 第3讲核心概念：
//   - 多线程核心带宽分析：加载指令 + 数学运算的吞吐量权衡
//     核心在等待数据时会暂停（stall），带宽决定了最大计算速率
//   - 内存带宽受限执行：当数据未就绪时核心暂停
//     处理器请求数据的速度超过内存提供给数据的速度
//   - 稳态下的数学运算速率受内存带宽限制
//     无论有多少个未完成的内存请求，吞吐量最终由带宽决定
//   - 稳态核心利用率仅取决于指令吞吐量和内存吞吐量的比值，
//     与内存延迟或未完成请求的数量无关
//     这是课程中最反直觉但最重要的结论之一
//   - "数学是免费的"（Math is Free）：
//     算术运算很便宜（1-20 pJ），数据移动非常昂贵（1200 pJ for DRAM）
//     这种能量成本的不对称性是现代计算架构设计的核心原则
//   - 程序应尽量减少内存访问频率，以高效利用处理器
//     当重算比从内存加载更省能量时，应该选择重算
//   - ISPC foreach：提升抽象层次，关注迭代而非实例
//     程序员写近似顺序的代码，编译器处理并行化细节
//   - 面向集合的编程（Collection-Oriented Programming）：
//     NumPy 风格，整个数组作为一等公民操作，程序员不写循环
//
// 编译命令：g++ -std=c++17 -O2 lecture3_part3.cpp -o lecture3_part3
// =============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// 模拟第3讲中的多线程核心带宽分析
//
// 处理器假设：
// - 每时钟周期1次数学运算
// - 数学运算与加载指令共发射（co-issue）
// - 每时钟周期从内存获取8字节
//
// 线程序列：加载64字节，加法 x+x，加法 x+y
// 当有足够多的线程来隐藏延迟时，稳态行为取决于
// 指令吞吐量 vs 内存带宽的比值。
//
// 关键结论：在稳态下，核心未充分利用的程度
// 仅取决于指令吞吐量和内存吞吐量，
// 与内存延迟或未完成请求的数量无关！
// ---------------------------------------------------------------------------
void demo_memory_bandwidth_pipeline() {
    std::cout << "[1] 内存带宽流水线分析" << std::endl;
    std::cout << "    （第3讲：多线程核心示例）" << std::endl;

    // 系统参数
    double math_ops_per_clock = 1.0;           // 每时钟周期1次数学运算
    double bytes_per_clock_from_mem = 8.0;     // 每时钟周期从内存获取8字节
    int load_size = 64;                         // 每次加载64字节（缓存行大小）
    int math_per_thread = 2;                    // 每个线程：add x+x, add x+y
    int clocks_to_transfer = load_size / (int)bytes_per_clock_from_mem; // 传输所需时钟

    std::cout << "    系统规格：" << std::endl;
    std::cout << "    - ALU：" << math_ops_per_clock << " 次数学运算/时钟" << std::endl;
    std::cout << "    - 内存：" << bytes_per_clock_from_mem << " 字节/时钟" << std::endl;
    std::cout << "    - 每次加载：" << load_size << " 字节（需要" 
              << clocks_to_transfer << " 个时钟）" << std::endl;
    std::cout << "    - 线程：加载 " << load_size 
              << "B，然后计算 add x+x, add x+y" << std::endl;

    // 内存带宽决定最大吞吐量：
    // 每个线程需要 load_size 字节，内存每时钟提供 bytes_per_clock 字节
    double max_threads_per_clock = bytes_per_clock_from_mem / load_size;
    double threads_per_100_clocks = max_threads_per_clock * 100;

    std::cout << std::endl;
    std::cout << "    最大线程完成速率：" << std::fixed 
              << std::setprecision(3) << max_threads_per_clock << " 个线程/时钟" << std::endl;
    std::cout << "    100个时钟内：" << std::setprecision(1) 
              << threads_per_100_clocks << " 个线程完成" << std::endl;
    std::cout << "    100个时钟内，ALU 可完成：" << 100 * math_ops_per_clock 
              << " 次数学运算" << std::endl;
    std::cout << "    但仅有 " << threads_per_100_clocks * math_per_thread 
              << " 次有用的数学运算完成（受内存限制）" << std::endl;

    double utilization = (threads_per_100_clocks * math_per_thread) 
                          / (100 * math_ops_per_clock) * 100;
    std::cout << "    核心利用率：" << std::setprecision(1) 
              << utilization << "%" << std::endl;

    std::cout << std::endl;
    std::cout << "    来自课程的关键洞察：在稳态下，核心未充分利用的程度" << std::endl;
    std::cout << "    仅取决于指令吞吐量和内存吞吐量的比值，" << std::endl;
    std::cout << "    与内存延迟或未完成请求的数量无关。" << std::endl;
    std::cout << std::endl;

    // 比较：每次内存访问做更多数学运算可提高利用率
    // 这展示了算术强度（Arithmetic Intensity）对利用率的影响
    std::cout << "    算术强度对利用率的影响：" << std::endl;
    std::cout << "    " << std::setw(16) << "每次加载的运算"
              << std::setw(18) << "内存带宽利用率"
              << std::setw(16) << "核心利用率" << std::endl;
    std::cout << "    " << std::string(50, '-') << std::endl;

    for (int mpl : {1, 2, 4, 8, 16}) {
        // mpl = 每次加载（64字节）后执行的数学运算次数
        int total_bytes = load_size;  // 固定的加载大小
        double threads_per_clock = bytes_per_clock_from_mem / total_bytes;
        double core_util = (threads_per_clock * mpl) / math_ops_per_clock * 100;
        double mem_util = 100.0; // 内存100%繁忙（它是瓶颈）

        std::cout << "    " << std::setw(16) << mpl
                  << std::setw(18) << std::fixed << std::setprecision(1) << mem_util << "%"
                  << std::setw(16) << std::setprecision(1) << std::min(core_util, 100.0) << "%"
                  << std::endl;
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// "数学是免费的"演示：比较算术运算 vs 内存访问的能量成本
//
// 数据来源于第1讲中关于数据移动能量的幻灯片。
// 能量单位：pJ（皮焦耳，10⁻¹² 焦耳）
//
// 关键洞察：
// - 一次整数运算：约1 pJ
// - 一次浮点运算：约20 pJ
// - 从SRAM读取64位：约26 pJ（片上1mm距离）
// - 从DRAM读取64位：约1200 pJ（移动LPDDR）
//
// 从DRAM加载一个64位值的能量成本相当于执行约60次浮点运算！
// 这个巨大的不对称性解释了为何现代架构如此强调数据局部性。
// ---------------------------------------------------------------------------
void demo_math_is_free() {
    std::cout << "[2] 数学是免费的 -- 算术 vs 数据移动的成本" << std::endl;

    // 课程中的估算数值（pJ = 皮焦耳）
    double int_op_cost = 1.0;        // 整数运算的能量成本：约1 pJ
    double fp_op_cost = 20.0;        // 浮点运算的能量成本：约20 pJ
    double sram_read_64b = 26.0;     // 从SRAM读取64位：约26 pJ（片上，1mm距离）
    double dram_read_64b = 1200.0;   // 从DRAM读取64位：约1200 pJ（移动LPDDR）

    std::cout << "    能量成本（估算值，单位：皮焦耳）：" << std::endl;
    std::cout << "    整数运算：          ~" << int_op_cost << " pJ" << std::endl;
    std::cout << "    浮点运算：          ~" << fp_op_cost << " pJ" << std::endl;
    std::cout << "    从SRAM读取64位：    ~" << sram_read_64b << " pJ" << std::endl;
    std::cout << "    从DRAM读取64位：    ~" << dram_read_64b << " pJ" << std::endl;
    std::cout << std::endl;

    // 示例：计算一个值 vs 从内存加载它
    std::cout << "    场景：需要一个值。计算它还是加载它？" << std::endl;
    std::cout << "    计算成本（100次浮点运算）：" << (100 * fp_op_cost) 
              << " pJ" << std::endl;
    std::cout << "    从DRAM加载64位成本：" << dram_read_64b 
              << " pJ" << std::endl;
    std::cout << "    从DRAM加载一个64位值相当于执行" 
              << std::fixed << std::setprecision(0) 
              << dram_read_64b / fp_op_cost << " 次浮点运算！" << std::endl;
    std::cout << std::endl;

    std::cout << "    含义：重新计算一个值通常比" << std::endl;
    std::cout << "    存储并重新加载它更节能。" << std::endl;
    std::cout << "    这就是为何现代程序应尽量使用算术运算" << std::endl;
    std::cout << "    而非内存访问。" << std::endl;
    std::cout << std::endl;

    // 带宽能量计算
    // 以10 GB/s的速度从DRAM读取数据所需的功耗
    double bw_gb_per_sec = 10.0;
    double reads_per_sec = bw_gb_per_sec * 1e9 / 8.0; // 10 GB/s转换为64位读取次数
    double power_watts = reads_per_sec * dram_read_64b * 1e-12; // 功耗（瓦特）

    std::cout << "    以 " << bw_gb_per_sec << " GB/s 从 DRAM 读取：" << std::endl;
    std::cout << "    功耗：" << std::setprecision(1) << power_watts 
              << " 瓦特" << std::endl;
    std::cout << "    iPhone 电池：约7瓦时" << std::endl;
    std::cout << "    在 " << bw_gb_per_sec << " GB/s 速率下：电池续航约" 
              << std::setprecision(1) << 7.0 / power_watts << " 小时" << std::endl;
    std::cout << "    这就是为何移动GPU目标总功耗约1W" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// 面向集合的编程模型（NumPy风格）
//
// 从课程中的概念："甚至不允许数组索引"
// - 程序员不写循环，不进行数据索引
// - 整个数组作为一等公民进行操作
// - 运行时/编译器自动处理并行化
//
// 这是 NumPy、PyTorch、TensorFlow 等框架背后的核心编程模型。
// 它实现了关注点分离：程序员专注于"做什么"，
// 运行时/编译器负责"如何并行执行"。
// ---------------------------------------------------------------------------
void demo_collection_programming() {
    std::cout << "[3] 面向集合的并行编程\n" << std::endl;

    // NumPy风格：X + Y 操作整个数组
    // map(f, collection) 将 f 应用到集合的每个元素

    const int N = 16;
    std::vector<int> X(N), Y(N), Z(N);

    // 初始化
    for (int i = 0; i < N; i++) {
        X[i] = i;
        Y[i] = i;
    }

    // 集合操作：Z = X + Y（逐元素，隐式并行）
    // 程序员不需要写循环，编译器/运行时处理并行化
    for (int i = 0; i < N; i++) {
        Z[i] = X[i] + Y[i];
    }

    std::cout << "    NumPy风格：Z = X + Y（逐元素向量加法）" << std::endl;
    std::cout << "    X：[";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << X[i];
    std::cout << " ]" << std::endl;
    std::cout << "    Y：[";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Y[i];
    std::cout << " ]" << std::endl;
    std::cout << "    Z：[";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Z[i];
    std::cout << " ]" << std::endl;
    std::cout << std::endl;

    // map(f, collection)：对集合中每个元素应用函数
    auto addOne = [](int x) { return x + 1; };
    std::vector<int> Zplus1(N);
    for (int i = 0; i < N; i++) Zplus1[i] = addOne(Z[i]);

    std::cout << "    map(addOne, Z)：" << std::endl;
    std::cout << "    Zplus1：[";
    for (int i = 0; i < N; i++) std::cout << std::setw(3) << Zplus1[i];
    std::cout << " ]" << std::endl;
    std::cout << std::endl;

    std::cout << "    核心抽象：程序员不写循环，" << std::endl;
    std::cout << "    不进行数据索引。运行时/编译器" << std::endl;
    std::cout << "    自动处理并行化。" << std::endl;
    std::cout << "    这正是 NumPy、PyTorch 等框架背后的模型。" << std::endl;
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// ISPC foreach 抽象：关注迭代而非实例
//
// 从课程中的概念：foreach 声明了并行循环迭代。
// 程序员只需说："这些是整个 gang 必须执行的迭代"。
// ISPC 实现负责将迭代分配给程序实例。
//
// foreach 的四种可能实现：
// 1. 单个实例执行所有迭代（顺序执行）
// 2. 交错分配：loop_i += programCount, idx = loop_i + programIndex
// 3. 块分配：每个实例获得 N/programCount 个连续元素
// 4. 动态分配：使用原子计数器按需分配迭代
//
// foreach 抽象实现了关注点分离：
// - 程序员关注正确性（"做什么"）
// - 编译器/运行时关注性能（"如何做"）
// ---------------------------------------------------------------------------
void demo_foreach_abstraction() {
    std::cout << "[4] foreach 抽象（ISPC 风格）" << std::endl;

    std::cout << "    ISPC foreach 语义：" << std::endl;
    std::cout << "    foreach (i = 0 ... N) {" << std::endl;
    std::cout << "        // 循环体 -- 程序员只关注第i次迭代" << std::endl;
    std::cout << "    }" << std::endl;
    std::cout << std::endl;

    std::cout << "    foreach 的四种可能实现：" << std::endl;
    std::cout << "    1. 单个实例运行所有迭代（顺序执行）" << std::endl;
    std::cout << "    2. 交错分配：loop_i += programCount, idx = loop_i + programIndex" << std::endl;
    std::cout << "    3. 块分配：每个实例获得 N/programCount 个连续元素" << std::endl;
    std::cout << "    4. 动态分配：使用原子计数器按需分配迭代" << std::endl;
    std::cout << std::endl;

    std::cout << "    foreach 抽象的优势：" << std::endl;
    std::cout << "    - 程序员可以编写近似顺序执行的代码" << std::endl;
    std::cout << "    - 编译器/运行时可以选择最佳调度策略" << std::endl;
    std::cout << "    - 清晰分离正确性（做什么）与性能（如何做）" << std::endl;
    std::cout << std::endl;
}

// =============================================================================
// 主函数：按顺序展示课程第3讲中带宽流水线和数据移动逻辑的核心概念
// =============================================================================
int main() {
    std::cout << "=== CS149 第3讲：带宽流水线 + 数据移动逻辑 ===" << std::endl;
    std::cout << std::endl;

    demo_memory_bandwidth_pipeline();
    demo_math_is_free();
    demo_collection_programming();
    demo_foreach_abstraction();

    // ---- 总结 ----
    std::cout << "[5] 第3讲关键要点" << std::endl;
    std::cout << "    - 内存带宽（而非延迟）是最关键的瓶颈资源" << std::endl;
    std::cout << "    - 带宽受限：ALU空闲等待数据到达（利用率低）" << std::endl;
    std::cout << "    - 稳态利用率：取决于数学/内存吞吐量的比值" << std::endl;
    std::cout << "    - 数学运算很便宜（1-20 pJ），数据移动很昂贵（1200 pJ）" << std::endl;
    std::cout << "    - 当算术成本低于内存成本时，重算优于存储+重载" << std::endl;
    std::cout << "    - 面向集合的编程：不写循环，不进行索引" << std::endl;
    std::cout << "    - foreach：关注迭代，让编译器处理实例分配" << std::endl;
    std::cout << "    - 抽象 ≠ 实现（贯穿 CS149 的核心主题）" << std::endl;

    return 0;
}
