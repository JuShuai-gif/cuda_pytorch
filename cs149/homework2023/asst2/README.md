
# 作业2：从零构建任务执行库

**截止日期：10月16日周四，晚上11:59**

**总分100分**

## 概述

每个人都喜欢快速完成任务，在本作业中，我们正是要你做到这一点！你将实现一个 C++ 库，在多核 CPU 上尽可能高效地执行应用程序提交的任务。

在作业的第一部分，你将实现一个任务执行库，支持批量（数据并行）启动同一任务的多个实例。此功能类似于你在作业1中用于跨核心并行化代码的 [ISPC 任务启动行为](http://ispc.github.io/ispc.html#task-parallelism-launch-and-sync-statements)。

在作业的第二部分，你将扩展你的任务运行时系统，以执行更复杂的**任务图**，其中任务的执行可能依赖于其他任务产生的结果。这些依赖关系约束了哪些任务可以被你的任务调度系统安全地并行运行。在并行机器上调度数据并行任务图的执行是许多流行并行运行时系统的特性，从 [Thread Building Blocks](https://github.com/intel/tbb) 库，到 [Apache Spark](https://spark.apache.org/)，再到现代深度学习框架如 [PyTorch](https://pytorch.org/) 和 [TensorFlow](https://www.tensorflow.org/)。

本作业要求你：

* 使用线程池管理任务执行
* 使用互斥锁和条件变量等同步原语协调工作线程的执行
* 实现一个反映任务图依赖关系的任务调度器
* 理解工作负载特性以做出高效的任务调度决策

我们建议你复习 [C++ 同步教程](tutorial/README.md) 以获取更多关于 C++ 标准库中同步原语的信息。此外，查看[测试用例描述](tests/)以了解你的库将支持的工作负载类型可能会有所帮助。

### 等等，我是不是做过这个？

你可能在 CS107 或 CS111 等课程中已经创建过线程池和任务执行库。然而，本次作业是一个更好地理解这些系统的独特机会。你将实现多个任务执行库，有些不使用线程池，有些使用不同类型的线程池。通过实现多种任务调度策略并比较它们在不同工作负载上的性能，你将更好地理解创建并行系统时关键设计选择的含义。

## 环境设置

**我们将在 Amazon AWS `c7g.4xlarge` 实例上批改此作业 —— 我们提供了[此处](https://github.com/stanford-cs149/asst2/blob/master/cloud_readme.md)的 VM 设置说明。请确保你的代码在此 VM 上能正常工作，因为我们将使用它进行性能测试和评分。**

作业初始代码可在 [Github](https://github.com/stanford-cs149/asst2) 上获取。请从以下地址下载作业2的初始代码：

    https://github.com/stanford-cs149/asst2/archive/refs/heads/master.zip

**重要提示：** 不要修改提供的 `Makefile`。这样做可能会破坏我们的评分脚本。

## Part A：同步批量任务启动

在作业1中，你使用了 ISPC 的任务启动原语来启动 N 个 ISPC 任务实例（`launch[N] myISPCFunction()`）。在本作业的第一部分，你将在你的任务执行库中实现类似的功能。

首先，请熟悉 `itasksys.h` 中 `ITaskSystem` 的定义。这个[抽象类](https://www.tutorialspoint.com/cplusplus/cpp_interfaces.htm)定义了你任务执行系统的接口。该接口包含一个 `run()` 方法，其签名如下：

    virtual void run(IRunnable* runnable, int num_total_tasks) = 0;

`run()` 执行指定任务的 `num_total_tasks` 个实例。由于单次函数调用导致多个任务的执行，我们将每次 `run()` 调用称为一次**批量任务启动**。

`tasksys.cpp` 中的初始代码包含了一个正确但串行的 `TaskSystemSerial::run()` 实现，作为任务系统如何使用 `IRunnable` 接口执行批量任务启动的示例。（`IRunnable` 的定义在 `itasksys.h` 中。）注意在每次调用 `IRunnable::runTask()` 时，任务系统会为任务提供一个当前任务标识符（0 到 `num_total_tasks` 之间的整数），以及批量任务启动中的任务总数。任务的实现将使用这些参数来确定该任务应执行什么工作。

`run()` 的一个重要细节是它必须相对于调用线程**同步**执行任务。换句话说，当 `run()` 调用返回时，应用程序保证任务系统已经完成了批量任务启动中**所有任务**的执行。初始代码中提供的串行 `run()` 实现在调用线程上执行所有任务，因此满足此要求。

### 运行测试

初始代码包含一套使用你任务系统的测试应用程序。关于测试框架的描述，请参见 `tests/README.md`，测试定义本身请参见 `tests/tests.h`。要运行测试，请使用 `runtasks` 脚本。例如，要运行名为 `mandelbrot_chunked` 的测试（该测试使用批量任务启动来计算 Mandelbrot 分形图像，每个任务处理图像的一个连续块），请输入：

```bash
./runtasks -n 16 mandelbrot_chunked
```

不同的测试具有不同的性能特征 —— 有些每个任务的工作量很少，有些则执行大量处理。有些测试每次启动创建大量任务，有些则很少。有时一次启动中的所有任务具有相似的计算成本。在其他情况下，单次批量启动中任务的成本是可变的。我们已在 `tests/README.md` 中描述了大多数测试，但我们鼓励你检查 `tests/tests.h` 中的代码以更详细地理解所有测试的行为。

> [!TIP]
> 在实现解决方案时，`simple_test_sync` 测试可能有助于调试正确性，这是一个非常小的测试，不应用于衡量性能，但足够小，可以用打印语句或调试器进行调试。参见 `tests/tests.h` 中的 `simpleTest` 函数。

我们鼓励你创建自己的测试。查看 `tests/tests.h` 中的现有测试以获取灵感。我们还包含了一个由 `class YourTask` 和函数 `yourTest()` 组成的骨架测试，供你在此基础上构建。对于你创建的测试，请确保将它们添加到 `tests/main.cpp` 的测试列表和测试名称中，并相应调整变量 `n_tests`。请注意，虽然你可以用自己的解决方案运行自己的测试，但你将无法编译参考解决方案来运行你的测试。

`-n` 命令行选项指定任务系统实现可以使用的最大线程数。在上面的例子中，我们选择 `-n 16`，因为 AWS 实例中的 CPU 有十六个执行上下文。可运行测试的完整列表可通过命令行帮助（`-h` 命令行选项）获取。

`-i` 命令行选项指定性能测量期间运行测试的次数。为了获得准确的性能测量，`./runtasks` 多次运行测试并记录多次运行中的**最小**运行时间；一般来说，默认值就足够了 —— 更大的值可能产生更准确的测量结果，但代价是测试运行时间更长。

此外，我们还提供了将用于评分性能的测试框架：

```bash
>>> python3 ../tests/run_test_harness.py
```

该框架有以下命令行参数：

```bash
>>> python3 run_test_harness.py -h
usage: run_test_harness.py [-h] [-n NUM_THREADS]
                           [-t TEST_NAMES [TEST_NAMES ...]] [-a]

Run task system performance tests

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_THREADS, --num_threads NUM_THREADS
                        Max number of threads that the task system can use. (16
                        by default)
  -t TEST_NAMES [TEST_NAMES ...], --test_names TEST_NAMES [TEST_NAMES ...]
                        List of tests to run
  -a, --run_async       Run async tests
```

它会产生如下详尽的性能报告：

```bash
>>> python3 ../tests/run_test_harness.py -t super_light super_super_light
python3 ../tests/run_test_harness.py -t super_light super_super_light
================================================================================
Running task system grading harness... (2 total tests)
  - Detected CPU with 16 execution contexts
  - Task system configured to use at most 16 threads
================================================================================
================================================================================
Executing test: super_super_light...
Reference binary: ./runtasks_ref_linux
Results for: super_super_light
                                        STUDENT   REFERENCE   PERF?
[Serial]                                9.053     9.022       1.00  (OK)
[Parallel + Always Spawn]               8.982     33.953      0.26  (OK)
[Parallel + Thread Pool + Spin]         8.942     12.095      0.74  (OK)
[Parallel + Thread Pool + Sleep]        8.97      8.849       1.01  (OK)
================================================================================
Executing test: super_light...
Reference binary: ./runtasks_ref_linux
Results for: super_light
                                        STUDENT   REFERENCE   PERF?
[Serial]                                68.525    68.03       1.01  (OK)
[Parallel + Always Spawn]               68.178    40.677      1.68  (NOT OK)
[Parallel + Thread Pool + Spin]         67.676    25.244      2.68  (NOT OK)
[Parallel + Thread Pool + Sleep]        68.464    20.588      3.33  (NOT OK)
================================================================================
Overall performance results
[Serial]                                : All passed Perf
[Parallel + Always Spawn]               : Perf did not pass all tests
[Parallel + Thread Pool + Spin]         : Perf did not pass all tests
[Parallel + Thread Pool + Sleep]        : Perf did not pass all tests
```

在上述输出中，`PERF` 是你的实现运行时间与参考解决方案运行时间的比值。因此，小于1的值表示你的任务系统实现比参考实现更快。

> [!TIP]
> Mac 用户：我们提供了 Part A 和 Part B 的参考解决方案二进制文件，但我们将使用 Linux 二进制文件测试你的代码。因此，我们建议你在提交前在 AWS 实例上检查你的实现。如果你使用的是搭载 M1 芯片的新款 Mac，请在本地测试时使用 `runtasks_ref_osx_arm` 二进制文件。否则，请使用 `runtasks_ref_osx_x86` 二进制文件。

> [!IMPORTANT]
> 我们将在 AWS 上使用 `runtasks_ref_linux_arm` 版本的参考解决方案进行评分。请确保你的解决方案在 AWS ARM 实例上能正确工作。

### 你需要做什么

你的工作是实现一个能高效利用多核 CPU 的任务执行引擎。你将根据实现的正确性（必须正确运行所有任务）以及性能来评分。这应该是一个有趣的编程挑战，但也是一项不小的工作。为了帮助你保持正确方向，要完成作业的 Part A，我们将让你实现多个版本的任务系统，实现的复杂度和性能逐步提高。你的三个实现将在 `tasksys.cpp/.h` 中定义的类中：

* `TaskSystemParallelSpawn`
* `TaskSystemParallelThreadPoolSpinning`
* `TaskSystemParallelThreadPoolSleeping`

**在 `part_a/` 子目录中实现你的 Part A 部分，以便与正确的参考实现（`part_a/runtasks_ref_*`）进行比较。**

_专业提示：注意以下说明采取了"先尝试最简单的改进"的方法。每一步都增加了任务执行系统实现的复杂性，但在每一步中你都应该有一个能正常工作的（完全正确的）任务运行时系统。_

我们还希望你创建至少一个测试，可以测试正确性或性能。更多信息请参见上面的"运行测试"部分。

#### 步骤1：迁移到并行任务系统

**在此步骤中，请实现 `TaskSystemParallelSpawn` 类。**

初始代码在 `TaskSystemSerial` 中为你提供了一个能正常工作的串行任务系统实现。在本作业的此步骤中，你将扩展初始代码以并行执行批量任务启动。

* 你需要创建额外的控制线程来执行批量任务启动的工作。注意 `TaskSystem` 的构造函数接收一个参数 `num_threads`，这是你的实现可以用于运行任务的**最大工作线程数**。

* 本着"先做最简单的事情"的精神，我们建议你在 `run()` 开始时创建工作线程，并在 `run()` 返回之前从主线程 join 这些线程。这将是一个正确的实现，但会因为频繁创建线程而产生显著的额外开销。

* 你将如何将任务分配给工作线程？你应该考虑静态分配还是动态分配任务给线程？

* 是否有共享变量（任务执行系统的内部状态）需要保护以免被多个线程同时访问？

#### 步骤2：使用线程池避免频繁创建线程

**在此步骤中，请实现 `TaskSystemParallelThreadPoolSpinning` 类。**

你在步骤1中的实现会因为每次 `run()` 调用都创建线程而产生开销。当任务计算成本很低时，这种开销尤为明显。此时，我们建议你迁移到"线程池"实现，你的任务执行系统预先创建所有工作线程（例如，在 `TaskSystem` 构造期间，或在首次调用 `run()` 时）。

* 作为起始实现，我们建议你将工作线程设计为不断循环，始终检查是否有更多工作要执行。（一个线程进入 while 循环直到某个条件为真，通常被称为"自旋"（spinning）。）工作线程如何确定有工作要做？

* 现在要确保 `run()` 实现所需的同步行为就不是那么简单了。你需要如何改变 `run()` 的实现来确定批量任务启动中的所有任务已经完成？

#### 步骤3：当无事可做时让线程休眠

**在此步骤中，请实现 `TaskSystemParallelThreadPoolSleeping` 类。**

步骤2实现的一个缺点是，线程在"自旋"等待有事可做时会占用 CPU 核心的执行资源。例如，工作线程可能循环等待新任务到达。另一个例子是，主线程可能循环等待工作线程完成所有任务，以便从 `run()` 调用返回。这会损害性能，因为 CPU 资源被用来运行这些线程，即使线程并没有做有用的工作。

在作业的这一部分，我们希望你将线程置于休眠状态，直到它们等待的条件得到满足，从而提高任务系统的效率。

* 你的实现可以选择使用条件变量来实现此行为。条件变量是一种同步原语，使线程能够在等待某个条件存在时休眠（不占用 CPU 处理资源）。其他线程"通知"等待的线程醒来，检查它们等待的条件是否已满足。例如，如果没有工作要做，你的工作线程可以进入休眠（这样它们就不会从试图做有用工作的线程那里占用 CPU 资源）。另一个例子是，调用 `run()` 的主应用程序线程可能希望在等待工作线程完成批量任务启动中的所有任务时休眠。（否则，自旋的主线程会从工作线程那里占用 CPU 资源！）

* 你在这部分作业中的实现可能需要考虑棘手的竞态条件。你需要考虑线程行为的许多可能交错情况。

* 你可能想考虑编写额外的测试用例来测试你的系统。**作业初始代码包含了评分脚本将用于评分代码性能的工作负载，但我们还将使用初始代码中未提供的更广泛的工作负载集来测试你实现的正确性！**

## Part B：支持任务图的执行

在作业的 Part B 中，你将扩展 Part A 的任务系统实现，以支持可能依赖于先前任务的异步任务启动。这些任务间依赖关系创建了你的任务执行库必须遵守的调度约束。

`ITaskSystem` 接口有一个额外的方法：

    virtual TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                    const std::vector<TaskID>& deps) = 0;

`runAsyncWithDeps()` 类似于 `run()`，因为它也用于执行 `num_total_tasks` 个任务的批量启动。然而，它在以下几个方面与 `run()` 不同……

#### 异步任务启动

首先，使用 `runAsyncWithDeps()` 创建的任务由任务系统**异步**执行，相对于调用线程而言。这意味着 `runAsyncWithDeps()` 应该**立即**返回给调用者，即使任务尚未完成执行。该方法返回与此批量任务启动关联的唯一标识符。

调用线程可以通过调用 `sync()` 来确定批量任务启动何时实际完成。

    virtual void sync() = 0;

`sync()` 只有在**与之前所有批量任务启动关联的任务都已完成时**才返回给调用者。例如，考虑以下代码：

    // 假设 taskA 和 taskB 是 IRunnable 的有效实例...

    std::vector<TaskID> noDeps;  // 空向量

    ITaskSystem *t = new TaskSystem(num_threads);

    // 批量启动 4 个任务
    TaskID launchA = t->runAsyncWithDeps(taskA, 4, noDeps);

    // 批量启动 8 个任务
    TaskID launchB = t->runAsyncWithDeps(taskB, 8, noDeps);

    // 此时与 launchA 和 launchB 关联的任务
    // 可能仍在运行

    t->sync();

    // 此时与 launchA 和 launchB 关联的所有 12 个任务
    // 保证已经终止

如上面注释所述，在调用线程调用 `sync()` 之前，不保证之前 `runAsyncWithDeps()` 调用中的任务已经完成。准确地说，`runAsyncWithDeps()` 告诉你的任务系统执行一次新的批量任务启动，但你的实现可以在下次 `sync()` 调用之前的任何时间灵活地执行这些任务。注意，此规范意味着不保证你的实现在启动 `launchB` 的任务之前先执行来自 `launchA` 的任务！

#### 支持显式依赖

`runAsyncWithDeps()` 的第二个关键细节是其第三个参数：一个 TaskID 标识符向量，这些标识符必须引用之前使用 `runAsyncWithDeps()` 进行的批量任务启动。此向量指定当前批量任务启动中的任务依赖于哪些先前的任务。**因此，你的任务运行时不能开始执行当前批量任务启动中的任何任务，直到依赖向量中给出的启动中的所有任务都完成！** 例如，考虑以下示例：

    std::vector<TaskID> noDeps;  // 空向量
    std::vector<TaskID> depOnA;
    std::vector<TaskID> depOnBC;

    ITaskSystem *t = new TaskSystem(num_threads);

    TaskID launchA = t->runAsyncWithDeps(taskA, 128, noDeps);
    depOnA.push_back(launchA);

    TaskID launchB = t->runAsyncWithDeps(taskB, 2, depOnA);
    TaskID launchC = t->runAsyncWithDeps(taskC, 6, depOnA);
    depOnBC.push_back(launchB);
    depOnBC.push_back(launchC);

    TaskID launchD = t->runAsyncWithDeps(taskD, 32, depOnBC);
    t->sync();

上面的代码包含四次批量任务启动（taskA：128个任务，taskB：2个任务，taskC：6个任务，taskD：32个任务）。注意 `launchB` 和 `launchC` 的启动都依赖于 taskA。`launchD` 的批量启动依赖于 `launchB` 和 `launchC` 两者的结果。因此，虽然你的任务运行时允许以任意顺序（包括并行）处理与 `launchB` 和 `launchC` 关联的任务，但这些启动中的所有任务必须在 `launchA` 的任务完成后才能开始执行，并且它们必须在你的运行时可以开始执行 `launchD` 的任何任务之前完成。

我们可以将这些依赖关系可视化为一个**任务图**。任务图是一个有向无环图（DAG），图中的节点对应批量任务启动，从节点 X 到节点 Y 的边表示 Y 依赖于 X 的输出。上述代码的任务图为：

<p align="center">
    <img src="figs/task_graph.png" width=400>
</p>

注意，如果你在具有八个执行上下文的 Myth 机器上运行上述示例，能够并行调度来自 `launchB` 和 `launchC` 的任务可能非常有用，因为单独任何一个批量任务启动都不足以利用机器的所有执行资源。

### 测试
所有带有 `Async` 后缀的测试应用于测试 Part B。评分框架中包含的测试子集在 `tests/README.md` 中描述，所有测试可在 `tests/tests.h` 中找到，并在 `tests/main.cpp` 中列出。为了调试正确性，我们提供了一个小测试 `simple_test_async`。查看 `tests/tests.h` 中的 `simpleTest` 函数。`simple_test_async` 应该足够小，可以在 `simpleTest` 内使用打印语句或断点进行调试。

我们鼓励你创建自己的测试。查看 `tests/tests.h` 中的现有测试以获取灵感。我们还包含了一个由 `class YourTask` 和函数 `yourTest()` 组成的骨架测试，供你在此基础上构建。对于你创建的测试，请确保将它们添加到 `tests/main.cpp` 的测试列表和测试名称中，并相应调整变量 `n_tests`。请注意，虽然你可以用自己的解决方案运行自己的测试，但你将无法编译参考解决方案来运行你的测试。

### 你需要做什么

你必须扩展 Part A 中使用线程池（并休眠）的任务系统实现，以正确实现 `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()` 和 `TaskSystemParallelThreadPoolSleeping::sync()`。我们还希望你创建至少一个测试，可以测试正确性或性能。更多信息请参见上面的"测试"部分。需要说明的是，你**需要**在报告中描述你自己的测试，但自动评分器**不会**测试你的测试。
**你不需要在 Part B 中实现其他 `TaskSystem` 类。**

与 Part A 一样，我们提供以下提示帮助你开始：
* 将 `runAsyncWithDeps()` 的行为想象为将与批量任务启动对应的记录，或者也许是与批量启动中每个任务对应的记录推送到"工作队列"中。一旦待处理的工作记录在队列中，`runAsyncWithDeps()` 就可以返回给调用者。

* 本部分作业的技巧是执行适当的簿记来跟踪依赖关系。当批量任务启动中的所有任务完成时必须做什么？（这是新任务可能变得可运行的时刻。）

* 在你的实现中有两个数据结构可能会很有帮助：(1) 一个结构，表示已通过 `runAsyncWithDeps()` 调用添加到系统中但尚未准备好执行的任务，因为它们依赖于仍在运行的任务（这些任务正在"等待"其他任务完成）；(2) 一个"就绪队列"，其中包含不等待任何先前任务完成的任务，只要有工作线程可用就可以安全地运行它们。

* 你不需要担心生成唯一任务启动 ID 时的整数回绕。我们不会对你的任务系统进行超过 2^31 次批量任务启动。

* 你可以假设所有程序要么只调用 `run()`，要么只调用 `runAsyncWithDeps()`；也就是说，你不需要处理 `run()` 调用需要等待所有之前的 `runAsyncWithDeps()` 调用完成的情况。注意，这一假设意味着你可以使用对 `runAsyncWithDeps()` 和 `sync()` 的适当调用来实现 `run()`。

* 你可以假设唯一的多线程是你实现创建/使用的多个线程。也就是说，我们不会创建额外的线程并从这些线程调用你的实现。

**在 `part_b/` 子目录中实现你的 Part B 部分，以便与正确的参考实现（`part_b/runtasks_ref_*`）进行比较。**

## 评分

本作业的分数分配如下：

**Part A（50分）**
- `TaskSystemParallelSpawn::run()` 的正确性 5分 + 性能 5分。（共10分）
- `TaskSystemParallelThreadPoolSpinning::run()` 和 `TaskSystemParallelThreadPoolSleeping::run()` 的正确性各 10分 + 这些方法的性能各 10分。（共40分）

**Part B（40分）**
- `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()`、`TaskSystemParallelThreadPoolSleeping::run()` 和 `TaskSystemParallelThreadPoolSleeping::sync()` 的正确性 30分
- `TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps()`、`TaskSystemParallelThreadPoolSleeping::run()` 和 `TaskSystemParallelThreadPoolSleeping::sync()` 的性能 10分。对于 Part B，你可以忽略 `Parallel + Always Spawn` 和 `Parallel + Thread Pool + Spin` 的结果。也就是说，你只需要每个测试用例的 `Parallel + Thread Pool + Sleep` 通过。

**报告（10分）**
- 详情请参见"提交"部分。

对于每个测试，在提供的参考实现的 20%（Part A）和 50%（Part B）范围内的实现将获得满分性能分。性能分仅授予返回正确答案的实现。如前所述，我们还可能使用初始代码中未提供的更广泛的工作负载集来测试你实现的**正确性**。

## 提交

请使用 [Gradescope](https://www.gradescope.com/) 提交你的作业。你的提交应包括你的任务系统代码和一份描述你实现的报告。我们期望提交中包含以下五个文件：

 * part_a/tasksys.cpp
 * part_a/tasksys.h
 * part_b/tasksys.cpp
 * part_b/tasksys.h
 * 你的报告 PDF（提交到 Gradescope 的报告作业）

#### 代码提交

我们要求你将源文件 `part_a/tasksys.cpp|.h` 和 `part_b/tasksys.cpp|.h` 以压缩文件形式提交。你可以创建一个目录（例如命名为 `asst2_submission`），其中包含子目录 `part_a` 和 `part_b`，将相关文件放入，通过运行 `tar -czvf asst2.tar.gz asst2_submission` 压缩该目录，然后上传。请将**压缩文件** `asst2.tar.gz` 提交到 Gradescope 上的作业 *Assignment 2 (Code)*。

在提交源文件之前，请确保所有代码都可以编译和运行！我们应该能够将这些文件放入干净的初始代码树中，输入 `make`，然后无需手动干预即可执行你的程序。

我们的评分脚本将运行初始代码中提供给你的检查器代码来确定性能分数。_我们还将使用初始代码中未提供的其他应用程序来运行你的代码以进一步测试其正确性！_评分脚本将在作业截止**之后**运行。

#### 报告提交

请将一份简短的报告提交到 Gradescope 上的作业 *Assignment 2 (Write-up)*，涵盖以下内容：

 1. 描述你的任务系统实现（1页即可）。除了一般性地描述其工作原理外，请确保回答以下问题：
   * 你是如何决定管理线程的？（例如，你是否实现了线程池？）
   * 你的系统如何将任务分配给工作线程？你使用了静态分配还是动态分配？
   * 在 Part B 中，你是如何跟踪依赖关系以确保任务图的正确执行的？

 2. 在 Part A 中，你可能已经注意到，更简单的任务系统实现（例如，完全串行的实现，或每次启动都创建线程的实现）表现得与更高级的实现一样好，有时甚至更好。请解释为什么会出现这种情况，并以某些测试为例。例如，在什么情况下串行任务系统实现表现最佳？为什么？在什么情况下每次启动都创建线程的实现表现得与使用线程池的更高级并行实现一样好？什么时候不是？

 3. 描述你为本作业实现的一个测试。该测试做什么，旨在检查什么，你是如何验证你的作业解决方案在你的测试上表现良好的？你添加的测试结果是否导致你更改了作业实现？
