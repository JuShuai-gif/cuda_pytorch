# LLVM 指令调度 vs CPU 乱序执行：区别、联系与协同

> 来源：知乎文章，约 26273 字，翻译自 Min-Yih Hsu 的 LLVM Scheduling Model 系列
> 核心问题：LLVM 在编译期做指令调度，CPU 在运行时做乱序执行，二者功能重叠，区别是什么？如何协同？

## 0. 先理解"调度"到底在干什么

### 为什么指令顺序很重要？

看一个简单例子：

```
指令1: r1 = 读内存 [addr]     ← 需要 200 个周期（cache miss）
指令2: r2 = r3 + r4           ← 只需要 1 个周期
指令3: r5 = r1 * 2            ← 依赖指令1的结果，必须等
```

如果老老实实按顺序执行：指令2 明明不依赖任何人，却要空等 199 个周期。

**指令调度的目标**：把指令2 提到指令1 和指令3 之间，让加法在等内存的时候顺便算了。这就是"隐藏延迟"。

### 谁来做这件事？

有两个角色：

| | LLVM 编译期调度 | CPU 运行时乱序执行 |
|---|---|---|
| **什么时候做** | 编译时（静态） | 运行时（动态） |
| **知道什么** | 指令延迟、资源数量等"已知信息" | 真实运行状态（cache miss、分支结果等） |
| **能调多大范围** | 大范围（跨基本块、循环展开后） | 小范围（ROB 窗口内，通常几百条指令） |
| **有什么局限** | 无法预知动态事件 | 窗口有限，多了超出窗口就没辙 |

---

## 1. 基本概念：LLVM 如何描述调度信息

### 1.1 一句话版

LLVM 不直接写"ADD 指令的延迟是 1"，而是用**读/写 token**来描述每条指令的调度特征，然后在调度模型里为每个 token 绑定延迟和资源。

### 1.2 三步法

```
第1步：在指令定义中给操作数分配 token
第2步：定义这些 token 使用哪些处理器资源
第3步：通过 WriteRes 绑定资源、延迟等信息
```

#### 举个例子：RISC-V 除法指令

```tablegen
// 第1步：指令定义中分配 token
def DIV : ALU_rr<0b0000001, 0b100, "div">,
             Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
             //     ↑ 写操作数    ↑ 读操作数1  ↑ 读操作数2

// 第3步：调度模型里绑定资源
def : WriteRes<WriteIDiv, [SiFive7PipeB, SiFive7IDiv]> {
    let Latency = 66;             // 结果 66 周期后可用
    let ReleaseAtCycles = [1, 65]; // 在 PipeB 上占用 1 周期，在 IDiv 上占用 65 周期
}
```

**关键理解**：写操作数（SchedWrite）决定整条指令的调度属性；读操作数（SchedRead）默认假设瞬时完成，不消耗周期。

### 1.3 为什么这么设计？

LLVM 不想为几万条 opcode 各写一张延迟表。很多指令调度特征一样（比如 add、sub、移位在现代架构上延迟都相同），用 token 抽象后，**只需要为 token 写一次，就能覆盖所有使用该 token 的指令**。

---

## 2. 建模微架构：处理器资源怎么表示

### 2.1 超标量处理器的基本画像

```
issue width = N（每周期最多发射 N 条指令）
decode → 生成 uops → dispatch → 分配到功能单元 → execute
                     ↑
              多个功能单元并行工作（整数/浮点/内存）
```

**注意**：超标量和乱序是两个独立概念：
- **超标量**关心"执行单元有多宽"（每周期能发几条）
- **乱序**关心"指令按什么顺序执行"（可以先执行后面的）

你可以有顺序但超标量的处理器，也可以有乱序但非超标量的（几乎没人做而已）。

### 2.2 LLVM 如何描述处理器资源

```tablegen
// SiFive P670：4 条整数流水线
def SiFiveP600IEXQ0 : ProcResource<1>;   // 每条流水线一个 ProcResource
def SiFiveP600IEXQ1 : ProcResource<1>;
def SiFiveP600IEXQ2 : ProcResource<1>;
def SiFiveP600IEXQ3 : ProcResource<1>;

// 资源组：任意整数指令可以跑在任意一条空闲流水线上
def SiFiveP600IntArith : ProcResGroup<[SiFiveP600IEXQ0, SiFiveP600IEXQ1,
                                        SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;

// 绑定：ADD 指令消耗 IntArith 资源组
def : WriteRes<WriteIALU, [SiFiveP600IntArith]>;
```

---

## 3. 处理器资源缓冲区的四种类型

这就涉及到指令是怎么被"暂存"的。缓冲区的作用是**存那些操作数还没就绪的指令，等操作数好了再发**。

| 类型 | BufferSize | 行为 | 适合场景 |
|------|-----------|------|----------|
| **解耦保留站** | 正值（如 16） | 每个功能单元有自己的调度队列，互不干扰 | AMD Zen2：4 个整数 ALU 各有 16 项 buffer |
| **统一保留站** | -1（默认） | 所有单元共用一个缓冲区。大小由 MicroOpBufferSize 决定 | 很多 ARM 核、RISC-V 核 |
| **顺序执行内核** | 0 | 没有缓冲区，dispatch = issue，老指令不释放新指令进不来 | 嵌入式核心、向量加速器 |
| **延迟设备** | 1 | 极小 buffer（1 项），连续使用该资源的 uop 被强制作生产者/消费者对 | 未流水化的除法单元 |

### 3.1 解耦保留站的例子

```tablegen
// AMD Zen2：4 个整数 ALU，各有独立调度器，每个 16 项
// 在 LLVM 里，用 ProcResGroup 统一标 BufferSize = 64（16×4）
def Zn2ALU : ProcResGroup<[Zn2ALU0, Zn2ALU1, Zn2ALU2, Zn2ALU3]> {
    let BufferSize = 64;
}
```

### 3.2 统一保留站和 MicroOpBufferSize

```tablegen
def SiFiveP600Model : SchedMachineModel {
    let IssueWidth = 4;           // 每周期最多发 4 条 uop
    let MicroOpBufferSize = 160;  // 统一缓冲区最多缓 160 条 uop
}
```

MicroOpBufferSize 应该 ≤ min(ROB 大小, 寄存器重命名池容量, 统一保留站真实容量)。

### 3.3 顺序执行内核（BufferSize = 0）

uop 会一直"占着"资源直到 ReleaseAtCycle 结束，后面的 uop 必须等。适合追求面积效率的嵌入式芯片（比如 SiFive X280/X390 用顺序设计换来 512/1024 位超宽向量能力）。

### 3.4 延迟设备（BufferSize = 1）

特殊之处：
- 有一个大小为 1 的极小 buffer，年轻 uop 在里面等老的
- **调度器强制把连续使用该资源的 uop 当成生产者/消费者对**：后面的 uop 总是要等前一条 issue 后再等 Latency 个周期
- 典型用途：建模乱序核中的未流水化除法单元

---

## 4. Latency vs ReleaseAtCycles：别搞混

这是调度模型里最容易搞混的一对概念。

| | Latency（延迟） | ReleaseAtCycles（资源占用） |
|---|---|---|
| **含义** | 从 uop issue 到结果对后续 use 可见的周期数 | uop 在每个资源上"占坑"的周期数 |
| **管什么** | 数据依赖距离：用了这个结果的下一条指令得隔多久 | 资源冲突：这个资源什么时候能接新 uop |
| **谁用** | 编译器算"下一条依赖指令什么时候能安全 issue" | 编译器算"这个资源现在有没有空位" |

### 四种典型场景

| 场景 | Latency | ReleaseAtCycles | 解释 |
|------|---------|-----------------|------|
| **不流水化单元（除法）** | ≈ sum(ReleaseAtCycles) | 加起来等于总时间 | 算完才放资源 |
| **完全流水化单元（乘法）** | 3-4 周期 | 每个资源 1 周期 | 资源很快就能接新指令，但结果还在流水线里跑 |
| **in-order 资源（BufferSize=0）** | 数据依赖用 Latency | 调度约束看 ReleaseAtCycles | 年轻 uop 等老 uop 的 ReleaseAtCycle 结束 |
| **latency device（BufferSize=1）** | 主导串行间隔 | 用于统计 | 后一条要等前一条 issue + Latency 周期 |

---

## 5. ProcResource\<N\> vs ProcResGroup vs Super Resource

### 5.1 ProcResource\<N\>

`<N>` 表示内部有多少个单元，直接影响吞吐量。

```
Temp = NumUnits / ReleaseAtCycle
```

等价于 N 个完全相同的 ProcResource\<1\>，可以并行派发最多 N 条同类指令。

### 5.2 ProcResGroup：表达异构执行单元

真实硬件里，不同种类的指令能跑的 pipe 经常不一样：

```
Pipe0: 能做 ADD、MUL     ← IEX0
Pipe1: 能做 ADD、DIV     ← IEX1
Pipe2: 能做 ADD、MUL     ← IEX2
```

用 ProcResGroup 就能精确表达：

```tablegen
def IEX0 : ProcResource<1>;
def IEX1 : ProcResource<1>;
def IEX2 : ProcResource<1>;

def IntegerArith : ProcResGroup<[IEX0, IEX1, IEX2]>;  // ADD 可以跑任意一条
def IntegerMul   : ProcResGroup<[IEX0, IEX2]>;        // MUL 只能跑 0 和 2
// DIV 只用 IEX1

def : WriteRes<WriteIALU, [IntegerArith]>;
def : WriteRes<WriteIMul, [IntegerMul]>;
def : WriteRes<WriteIDiv, [IEX1]>;
```

LLVM 会自动展开重叠资源——派发一条 DIV 时，不仅消耗 IEX1，还会隐式消耗 IntegerArith（因为 IEX1 属于 IntegerArith 的范围内）。

### 5.3 Super Resource：表达树形层次

当资源关系是**严格的树形结构**（比如 "LSU 有 3 条管线，其中 3 条能做 load、2 条能做 store"），用 Super Resource 更简洁：

```tablegen
def Zn3LSU : ProcResource<3>;        // LSU 总共 3 条管线

let Super = Zn3LSU in
def Zn3Load : ProcResource<3>;       // 3 条都能做 load

let Super = Zn3LSU in
def Zn3Store : ProcResource<2>;      // 最多 2 条做 store
```

### 5.4 Super Resource 的致命限制

**Super Resource 只能表达树形层次**，不能表达多重重叠的 DAG 结构。

比如你不能同时声明 IntegerDiv 是 IntegerMul 的子集、又是 IntegerArith 的子集——因为每个 ProcResource 最多只有一个 Super。

这时必须用 ProcResGroup。

### 5.5 选择指南

| 场景 | 推荐 |
|------|------|
| 只关心单元数量，不关心具体哪条 pipe | ProcResource\<N\> 或 Super Resource |
| 执行单元能力异构、部分重叠（DAG 结构） | ProcResGroup |
| 大规模对称资源 | Super Resource 更简洁 |

---

## 6. LLVM 编译期调度 vs CPU 运行时乱序：如何协同？

### 6.1 一个比喻

LLVM 编译期调度像是一个**导航软件**（基于已知路况做路径规划），CPU 运行时乱序像是一个**自适应巡航**（根据实时路况微调速度/车道）。

- 导航软件不知道前面会不会突然堵车（cache miss），但能告诉你"这条路通常比另一条快"
- 自适应巡航不知道 10 公里外的路况，但能根据前车距离实时调整

### 6.2 二者的分工

| LLVM 编译期调度 | CPU 运行时乱序 |
|---|---|
| 编译时重排指令，基于调度模型做静态分析 | 运行时通过保留站、ROB 动态调度 |
| 处理软件可见的延迟隐藏和资源冲突 | 基于真实运行时状态做决策 |
| 处理"宏观"重排：跨基本块、循环展开后的调度 | 处理"微观"动态调度：cache miss、分支误预测 |
| 输出对硬件友好的指令序列 | 在编译器打好的基础上做最优动态微调 |

### 6.3 协同的实质

1. 编译器通过调度模型"理解"微架构的约束（延迟、资源数量、缓冲区大小）
2. 编译器产生的指令序列尽量**减少硬件调度器的压力**（均匀分布到各 pipe、提前隐藏已知延迟）
3. 编译器做**宏观布局**，硬件做**微观微调**
4. 好的调度模型让编译器不犯蠢，硬件在此基础上锦上添花

---

## 7. 关键要点速查

| 概念 | 一句话 |
|------|--------|
| SchedWrite / SchedRead | 操作数级的调度属性 token |
| WriteRes | 把资源和 SchedWrite 绑定，标注延迟 |
| ProcResource\<N\> | 有 N 个相同单元的处理器资源 |
| ProcResGroup | 把多条 pipe 打包成集合，可表达异构能力 |
| Super Resource | 树形父子关系，子资源是父资源的子集 |
| BufferSize | 缓冲区大小：正=解耦保留站，-1=统一保留站，0=顺序，1=延迟设备 |
| Latency | 数据依赖距离（issue 到结果可用） |
| ReleaseAtCycles | 资源占用时间（占用资源多久） |
| MicroOpBufferSize | 统一保留站的全局缓冲区大小 |
