# Chapter 2: Contributing to LLVM

## 核心概念（详细展开）

### 开源贡献的完整图景

对 AI 编译器工程师来说，向 LLVM 社区贡献不仅仅是"修 bug"或"写代码"。贡献是建立专业声誉、获得代码审查反馈、以及深入了解编译器内部机制的最有效途径。LLVM 社区的开放文化意味着即使你是编译器新手，只要提交的是合理且有价值的贡献，都会被认真对待。

**工业界的现实**：Google、Apple、Meta 等大厂的编译器团队都鼓励工程师向上游 LLVM 贡献代码。维护下游 fork 的成本远比参与上游开发高——每次 LLVM 版本升级时，下游 fork 需要手动合并大量冲突。这就是为什么 Intel 的 SYCL 编译器、NVIDIA 的 CUDA 编译器、AMD 的 ROCm 编译器都积极向上游 LLVM 贡献。

**贡献形式矩阵**：

| 贡献形式 | 难度 | 影响力 | 对初学者的适合度 |
|----------|------|--------|-----------------|
| 报告 bug（含最小复现） | 低 | 高 | ★★★★★ |
| 修复文档/拼写错误 | 极低 | 中 | ★★★★★ |
| 审查 PR（review） | 中 | 高 | ★★★★ |
| 添加测试用例 | 低-中 | 中 | ★★★★ |
| 小功能实现 | 中 | 中 | ★★★ |
| 新建后端/大重构 | 高 | 极高 | ★★ |
| 参与 RFC 讨论 | 中 | 极高 | ★★★ |

### 报告问题的工业标准

在工业编译器团队中，有效的 bug 报告是工程素养的核心体现。一个糟糕的报告（如"Clang crashes when compiling my code"）需要工程师花费数小时甚至数天来复现和理解问题。而一个优秀的报告（附带最小化 IR 复现、环境信息、明确的复现步骤）可能只需要几分钟就能被定位和修复。

**Bug 报告的核心要素（扩展版）**：
1. **LLVM 版本和 commit hash**：`git rev-parse HEAD` 或 `clang --version`。这是最重要的信息——没有版本信息，根本无法确定问题是在哪次提交引入的。
2. **构建配置**：确切的 CMake 命令，包括 `CMAKE_BUILD_TYPE`、`LLVM_TARGETS_TO_BUILD` 等。
3. **操作系统和硬件**：`uname -a`、CPU 型号、内存大小。
4. **最小化复现**：使用 `llvm-reduce` 或 `bugpoint` 工具自动缩小输入。对于 miscompile 问题，使用 `creduce` 也是一大利器。
5. **期望行为 vs 实际行为**：明确指出你期望发生什么以及实际发生了什么。对比 alive2 的验证结果（如果可以的话）。
6. **回归信息**：如果可能，通过 `git bisect` 定位引入问题的 commit。

**AI 编译器领域的 Bug 报告特殊性**：
在 MLIR/Triton 生态中，bug 可能跨越多个抽象层：
- Python frontend → Triton IR → Triton GPU IR → LLVM IR → PTX → 硬件
- 报告时需要明确问题出现在哪个 IR 层
- 提供每一层的 IR dump 对调试至关重要

### 社区参与的多层次体系

LLVM 社区提供了多个层次的参与渠道：

1. **Discord 服务器**（实时交流）：最活跃的日常讨论平台。有专门的 `#mlir`、`#backend`、`#codegen` 等频道。对 AI 编译器工程师，`#mlir` 和 `#global-isel` 频道尤其有用。
2. **Discourse 论坛**（异步深度讨论）：用于 RFC（Request for Comments）、设计讨论、发布公告。MLIR 有自己的 Discourse 类别。
3. **GitHub Issues/PRs**：正式的问题追踪和代码审查。
4. **Office Hours**（办公时间）：每两周一次的视频会议，由资深 LLVM 开发者主持。这是直接向专家提问的绝佳机会。对初学者来说，即使只是旁听也能学到很多。
5. **LLVM Developers' Meeting**（年度会议）：包括全球和欧洲两场。有顶级公司的编译器团队分享最新进展。AI/ML 编译器是近年来的热门议题。

**MLIR 社区的对比**：
- MLIR 使用相同的 Discord 服务器（在 `#mlir` 频道）
- MLIR 有自己的 Discourse 类别：https://discourse.llvm.org/c/mlir/
- MLIR 的设计讨论非常活跃，几乎每天都有新的 RFC 或设计提案
- MLIR Open Design Meetings：每周的视频会议，公开讨论 MLIR 的设计决策

### 代码审查的工业价值

代码审查不仅是质量控制机制，更是知识传递的渠道。作为审查者：
- 你学习如何评估代码质量
- 你理解设计决策的来龙去脉
- 你建立起与社区其他开发者的信任关系

**从初学者到审查者的进阶路径**：
1. 第一阶段：只提问题（"为什么这样做？"、"这里有更好的方法吗？"）
2. 第二阶段：检查风格和约定（命名、头文件包含顺序、注释质量）
3. 第三阶段：验证逻辑正确性（SSA 属性是否保持？支配关系是否正确？）
4. 第四阶段：评估设计选择（API 设计是否合理？性能影响如何？）

**生产环境中的 Code Review 文化**：
在 Google 和 Apple，每个 LLVM 的 PR 都会被至少 1-2 名工程师审查。Meta 的编译器团队甚至有专门的"review rotation"制度。审查不是一个负担，而是确保编译器正确性的必要环节——一个编译器 bug 可能导致数百万行代码的错误编译。

### PR 流程的完整生命周期

```
RFC (仅大规模变更)
  │
  ▼
实现（在你的 fork 中）
  │
  ▼
创建 PR（GitHub）
  │
  ▼
CI 自动检查（pre-commit checks）
  │  - Build: 确保编译通过
  │  - Test: 运行相关测试套件
  │  - Format: clang-format 检查
  │  - Lint: clang-tidy 检查
  │
  ▼
代码审查（人工）
  │  ▼
  │  请求修改（Request Changes）
  │  ▼
  │  提交修改（amend commit, force push）
  │  ▼
  │  循环直到 Approved
  │
  ▼
合并（Squash and Merge）
  │
  ▼
监控 Buildbot（post-commit）
  │  ▼
  │  如果失败：commit 被 revert
  │  ▼
  │  修复后重新提交
  │
  ▼
完成！
```

**关键细节**：
- LLVM 使用 "squash and merge" 策略：PR 中的所有 commits 在合并时被压缩为一个
- 这意味着你可以在开发过程中随意 commit，最后只有一个干净的 commit message
- Commit message 格式：标题行 + 空行 + 详细描述。标题应概括变更内容（如 `[MLIR][GPU] Add support for...`）
- PR 必须包含测试：包括正面测试（验证新功能工作）和负面测试（验证错误被正确拒绝）

### 测试哲学的深度理解

"如果一个 PR 没有测试，那它就没有被 PR"——这是 LLVM 社区的一句名言。测试的重要性体现在：

1. **意图表达**：测试告诉审查者"这段代码应该做什么"
2. **回归防护**：测试确保未来的变更不会破坏你的功能
3. **文档价值**：测试是最新、最可靠的 API 用法文档
4. **信任建立**：有完善测试的 PR 更容易获得审查者的信任

**测试层次结构**：
```
端到端测试（test-suite）
  ↑ 验证编译器输出能通过程序本身的测试
回归测试（lit + FileCheck）
  ↑ 验证特定优化或分析产生正确的 IR 转换
单元测试（gtest）
  ↑ 验证单个 API 或数据结构的行为
```

---

## LLVM / MLIR 流程（深入）

### 从 Issue 到合并的端到端时间线

以添加一个新的 MLIR Canonicalization Pattern 为例：

```
Day 0: 发现优化机会，构思方案
Day 1-2: 实现 canonicalization pattern（~50 行代码）
Day 2: 添加 lit 测试（.mlir 测试文件）
Day 3: 本地验证：
  - ninja check-mlir 全部通过
  - clang-format 格式化代码
Day 3: 创建 PR，添加描述
Day 3-7: 等待 review（通常 2-5 个工作日）
Day 4: Reviewer 提出修改建议
Day 4-5: 修改代码，更新 PR
Day 6: Reviewer 批准 (LGTM - Looks Good To Me)
Day 6: 合并到 main 分支
Day 6-7: 监控 Buildbot 确认无误
```

**实际工业案例**：小的 bug fix 可能在 24 小时内合并。大的设计变更（如添加新的 Dialect）可能需要数周到数月的讨论和迭代。

### MLIR Discourse/论坛的参与

MLIR 社区在 Discourse 上非常活跃。对 AI 编译器工程师：
- 订阅 `mlir` 类别以获取最新设计讨论
- 搜索历史讨论以避免重复提问
- 在设计自己的 dialect 或 pass 之前，先在 Discourse 上发 RFC 获取反馈

**典型的 MLIR RFC 结构**：
1. 动机：为什么需要这个变更？
2. 背景：现有方案的问题
3. 设计：具体的 API/IR 设计
4. 替代方案：考虑过但放弃的方案
5. 实现计划：分阶段的时间线
6. 影响：对现有代码的影响

### 向上游 MLIR 贡献的流程

MLIR 的贡献流程与 LLVM 完全相同（共享相同的仓库和基础设施），但有一些 MLIR 特有的注意事项：

1. **Dialect 设计**：新增 Dialect 需要充分的动机说明和设计讨论。不能随意添加——每个 Dialect 都增加了维护负担。
2. **转换 passes**：Dialect 间的转换需要在`mlir/lib/Conversion/`中实现
3. **测试**：MLIR 测试使用 `.mlir` 文件和 FileCheck，与 LLVM `.ll` 测试的模式相同
4. **上下游协作**：许多 MLIR contributor 同时也在 Triton、IREE、XLA 等项目工作——这些项目也向上游 MLIR 贡献

---

## 关键机制解析（工业视角）

### GitHub 工作流的实际操作

LLVM 项目在 2023 年底从 Phabricator 迁移到了 GitHub。对熟悉 GitHub 的开发者来说，这大大降低了参与门槛。但 LLVM 的 GitHub 使用有一些特殊约定：

```bash
# 1. Clone 官方仓库
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# 2. 创建功能分支（总是从最新的 main 分支）
git checkout main
git pull --rebase
git checkout -b my-feature

# 3. 开发、测试、commit
# ... 进行修改 ...
ninja check-llvm  # 或 check-mlir, check-clang
git add ...
git commit -m "描述你的变更"

# 4. 保持与上游同步
git fetch origin
git rebase origin/main
# 解决可能的冲突

# 5. 推送并创建 PR
git push origin my-feature
# 访问输出的 URL 创建 PR
```

**LLVM 特有的 GitHub 实践**：
- PR 标题应该描述**变更内容**而非"修复了一个 bug"
- 使用标签（labels）帮助分类（如 `mlir`, `backend:X86`, `bug`）
- 在 PR 描述中使用 `Fixes #1234` 链接到相关 issue
- CI 检查运行时间可能需要 30-60 分钟，请耐心等待

### 寻找审阅者的策略

找到合适的审阅者是 PR 能否及时合并的关键。以下是系统化的寻找方法：

```bash
# 方法1：查看相关文件/目录的最近贡献者
git log --pretty=format:"%aN <%ae>" --since="6 months ago" -- llvm/lib/Target/X86/ | sort | uniq -c | sort -rn

# 方法2：查看 CODE_OWNERS.TXT
# 位于仓库根目录，列出各组件的负责人
less CODE_OWNERS.TXT

# 方法3：查看最近合并的相似 PR 的审阅者
# 在 GitHub 上搜索 merged PR，查看 approval 记录

# 方法4：在 PR 评论中 @ 相关贡献者
# 但不要 @ 过多——选择 1-2 位最相关的
```

**工业界的审阅者关系建立**：
- 在公司内部，审阅者通常是你的团队同事——他们对代码有上下文理解
- 在开源社区，审阅者是你通过持续贡献建立的信任关系
- Google 的编译器团队有内部的"代码所有者"系统，映射到上游 LLVM 的 CODE_OWNERS

### 大型变更的 RFC 流程

对于影响面广的变更（新后端、API 改变、架构调整），LLVM 要求先发 RFC（Request for Comments）：

**RFC 应当包含**：
1. **摘要**（3-5 句话概括提案）
2. **动机和背景**（为什么这很重要？不做的后果是什么？）
3. **设计方案**（具体的 API/IR/命令行接口设计）
4. **替代方案**（考虑过但拒绝的方案及原因）
5. **实现计划**（如何分解为可审查的 PR）
6. **影响评估**（对现有代码、性能、编译时间的影响）
7. **相关工作**（引用现有的相关讨论、论文或实现）

RFC 通常发送到 Discourse 论坛而非 GitHub Issues。LLVM 社区对 RFC 的讨论非常严肃和深入——一个 RFC 的讨论可能持续数周、产生上百条回复。

### Buildbot 监控

Buildbot 是 LLVM 的 CI 系统。URL：https://lab.llvm.org/buildbot/

**关键概念**：
- **Pre-commit checks**（GitHub Actions）：PR 合并前运行的检查（编译+基本测试）
- **Post-commit builders**（Buildbot）：合并后运行的更全面测试
- **Green Dragon**：Apple 维护的另一套 CI 系统

**Buildbot 的实战使用**：
1. 如果你的 PR 合并后 Buildbot 变红（失败），你需要：
   - 查看哪个 builder 失败了
   - 查看失败日志确定原因
   - 如果是你的变更导致的：提交修复 PR
   - 如果不是你的变更：有时是环境问题（"flaky test"），re-run 即可
2. 初学者常犯的错误：看到 Buildbot 失败就恐慌。很多失败是已有的环境问题，与你的变更无关。
3. 判断方法：检查失败是否也出现在附近的 commits 上。如果是，那是已有问题。

---

## AI 编译器关联

### 向 MLIR/Triton 社区贡献的特殊性

对于 AI 编译器工程师来说，向 MLIR 上游贡献是建立行业声誉的绝佳途径。MLIR 社区非常欢迎来自 AI 编译器背景的贡献者，因为：

1. **MLIR 的很多设计决策来源于 AI 编译器的需求**：例如 Linalg Dialect 的设计直接受 Tensor Comprehensions（Meta 的 AI 编译器项目）影响
2. **AI 编译器用例推动 MLIR 的发展**：量化（quantization）、混合精度、稀疏计算等 AI 特有需求的讨论非常活跃
3. **Triton 与 MLIR 的整合**：Triton 正在将更多代码贡献到上游 MLIR

**具体的 AI 编译器贡献方向**：
- 为 `gpu` dialect 添加新的操作（ops）或优化
- 改进 `linalg` dialect 的 tiling/fusion passes
- 为 `sparse_tensor` dialect 贡献稀疏计算支持
- 改进 MLIR 到 LLVM IR 的转换（`mlir/lib/Conversion/*ToLLVM`）
- 优化 MLIR 的 pass pipeline 调度

### 开源贡献与职业发展

在 AI 编译器领域，LLVM/MLIR 的开源贡献记录是面试和晋升的重要加分项：
- Google、Apple、NVIDIA、Intel、AMD 的编译器团队都看重候选人的上游贡献记录
- 开源贡献展示了你编写高质量代码、参与代码审查、接受反馈的能力
- 你的贡献记录是可验证的——任何人都能在 GitHub 上查看

---

## 示例说明

本章没有代码示例（内容聚焦于社区参与流程），但在实践中：

### 一个完整的贡献流程示例（以修复 MLIR bug 为例）

问题：MLIR 的某个 canonicalization pattern 在特定输入下产生错误的 IR。

```
1. 发现并最小化问题：
   $ cat reproducer.mlir
   func.func @test(%arg0: i32) -> i32 {
     %0 = arith.addi %arg0, %arg0 : i32
     return %0 : i32
   }

2. 确定期望行为：
   canonicalization 应该将 %arg0 + %arg0 优化为 %arg0 * 2

3. 验证 bug：
   $ mlir-opt --canonicalize reproducer.mlir
   # 观察输出，确认 bug 存在

4. 定位源码：
   $ grep -r "AddIOp" mlir/lib/Dialect/Arith/IR/
   # 找到 canonicalization pattern 的实现位置

5. 修复代码：
   在 ArithCanonicalization.cpp 中修改 pattern 逻辑

6. 添加测试：
   在 mlir/test/Dialect/Arith/canonicalize.mlir 中添加：
   // CHECK-LABEL: func @mul_by_two
   // CHECK:       arith.muli
   func.func @mul_by_two(%arg0: i32) -> i32 {
     %0 = arith.addi %arg0, %arg0 : i32
     return %0 : i32
   }

7. 本地验证：
   $ ninja check-mlir

8. 提交 PR：
   $ git add mlir/lib/Dialect/Arith/IR/ArithCanonicalization.cpp
   $ git add mlir/test/Dialect/Arith/canonicalize.mlir
   $ git commit -m "[MLIR][Arith] Fix canonicalization of addi with same operands"
   $ git push
```

---

## 总结

### 技术要点清单
- LLVM/MLIR 社区高度开放，贡献形式包括报告问题、审查代码、提交补丁、参与讨论
- 有效的 bug 报告需包含版本信息、构建配置、最小化复现输入、复现步骤、期望/实际行为对比
- Discord 用于实时交流（`#mlir` 频道对 AI 编译器工程师尤为重要），Discourse 用于设计讨论
- Office Hours 是向专家直接提问的绝佳机会，每两周一次
- 代码审查是知识传递的核心机制——初学者应从提问题开始参与
- PR 必须是小的、增量化的、附带测试的——大规模变更需要先发 RFC
- LLVM 使用 Squash and Merge 策略，所有 commits 被压缩为一条
- Buildbot 监控是 PR 合并后的最后一道防线——并非所有失败都与你的变更有关
- `CODE_OWNERS.TXT` 和 `git log` 是寻找合适审阅者的主要工具
- MLIR 的贡献流程与 LLVM 完全相同——学习和贡献的迁移成本极低

### 实践建议
1. **从修复文档或拼写错误开始**：这是最低风险的上手方式，让你熟悉 PR 流程
2. **订阅 Discourse 并定期阅读 RFC**：了解社区的最新设计方向
3. **参加至少一次 Office Hours**：即使只是旁听，也能学到很多
4. **为你的每个功能需求先写测试**：测试驱动开发既是好的工程实践，也是好的贡献实践
5. **不要害怕被拒绝或要求修改**：审查意见不是对个人的否定，而是提升代码质量的协作过程
6. **参与 Code Review**：即使只评论风格或提出疑问，也是宝贵的贡献
7. **利用 `git bisect` 定位回归**：这是报告 bug 时最有价值的信息之一

### 进一步学习方向
- LLVM 开发者政策：https://llvm.org/docs/DeveloperPolicy.html
- LLVM 行为准则：https://llvm.org/docs/CodeOfConduct.html
- MLIR Discourse 类别：https://discourse.llvm.org/c/mlir/
- LLVM Discord 服务器：https://discord.com/invite/xS7Z362
- GitHub Issues：https://github.com/llvm/llvm-project/issues
- 阅读 LLVM Weekly（每周新闻简报）：https://llvmweekly.org/

### 工业界的实际案例
- **Apple 的 LLVM 贡献文化**：Apple 的编译器团队每周有固定的"upstream day"，工程师专门向上游 LLVM 推送他们的改进。这确保 Apple 的下游变更能及时合并到上游，减少维护负担。
- **Google MLIR 团队的开源策略**：Google 的 MLIR 核心团队将几乎所有工作都直接做在开源仓库中。他们使用 RFC 机制公开讨论设计决策，社区可以充分参与。
- **Meta 的编译器贡献**：Meta 将许多内部开发的 LLVM 优化（如 ThinLTO 的增强、新的内联策略）贡献给上游。这是一项战略决策——Meta 依赖 LLVM，因此投资上游对 Meta 也是有利的。

### 对 AI 编译器工程师的核心建议
- MLIR 社区正在快速增长，现在是建立参与度和影响力的黄金时期
- 你不需要成为编译器专家后才开始贡献——AI 编译器领域有很多 MLIR 相关的工作，你的领域知识本身就很有价值
- 关注 MLIR 的 Open Design Meeting（每周视频会议），了解最新的设计方向
- 如果你在 AI 编译器（Triton、IREE、XLA）中实现了有用的功能，考虑将其贡献到上游 MLIR——这会让整个生态受益
