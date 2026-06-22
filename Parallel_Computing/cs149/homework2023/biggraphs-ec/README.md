
# 作业5：基于 OpenMP 的大规模图处理

**截止日期：12月6日周五，太平洋时间晚上11:59（不接受迟交）**

**总分84分**

如果你完成本作业，你将在之前的一个常规编程作业（PA1-PA4）上获得最多10分的加分。然而，此额外加分最多只能将某项作业提升至100分。

## 概述

在本作业中，你将实现[广度优先搜索](https://en.wikipedia.org/wiki/Breadth-first_search)（BFS）。本作业的优秀实现将能够在多核机器上仅用数秒时间对包含数亿条边的图运行此算法。

## 环境设置

本作业的最终评分将在 Myth 机器上进行。

作业初始代码可在 [Github](https://github.com/stanford-cs149/biggraphs-ec) 上获取。请使用以下命令克隆作业5的初始代码：

```
git clone https://github.com/stanford-cs149/biggraphs-ec.git
```

#### 背景：学习 OpenMP

在本作业中，我们希望你使用 [OpenMP](http://openmp.org/) 进行多核并行化。OpenMP 是一个 API 和一组 C 语言扩展，为并行性提供编译器支持。你还可以使用 OpenMP 告诉编译器并行化 `for` 循环的迭代，并管理互斥。它的在线文档很完善，但这里有一个并行化 `for` 循环并带有互斥的简短示例。
```c
/* 此 for 循环的迭代可由编译器并行化 */      
#pragma omp parallel for                                                      
for (int i = 0; i < 100; i++) {  

    /* 循环体这部分的不同迭代可能
        在不同的核心上并行运行 */

    #pragma omp critical                                                          
    {
    /* 此代码块一次最多由一个线程执行。 */
    printf("Thread %d got iteration %lu\n", omp_get_thread_num(), i);           
    }                                                                             
}
``` 
请参阅 OpenMP 文档了解如何使用不同形式的静态或动态调度的语法。（例如，`omp parallel for schedule(dynamic 100)` 使用块大小为100次迭代的动态调度将迭代分配给线程）。你可以将其实现视为一个动态工作队列，线程池中的线程每次取出100次迭代，就像[我们在这些讲座幻灯片中所讨论的](https://gfxcourses.stanford.edu/cs149/fall24/lecture/perfopt1/slide_11)。

以下是 OpenMP 中原子计数器更新的示例。
```c
int my_counter = 0;
#pragma omp parallel for                                                        
for (int i = 0; i < 100; i++) {                                                      
    if ( ... 某些条件 ...) {
        #pragma omp atomic
        my_counter++;
    }
}
```
我们期望你能够自行阅读 OpenMP 文档（Google 会非常有用），但这里有一些有用的链接帮助你开始：

 * OpenMP 3.0 规范：<http://www.openmp.org/mp-documents/spec30.pdf>。
 * OpenMP 速查表：<http://openmp.org/mp-documents/OpenMP3.1-CCard.pdf>。
 * OpenMP 支持对共享变量进行归约操作，以及声明变量的线程本地副本。
 * 这是一个关于 `omp parallel_for` 指令的不错指南：<http://www.inf.ufsc.br/~bosco.sobral/ensino/ine5645/OpenMP_Dynamic_Scheduling.pdf>

#### 背景：图的表示

初始代码操作的是有向图，其实现可在 `graph.h` 和 `graph_internal.h` 中找到。我们建议你从理解这些文件中的图表示开始。图由边数组表示（包括 `outgoing_edges` 和 `incoming_edges`），每条边由一个表示目标顶点 id 的整数表示。边在图中按源顶点排序存储，因此源顶点在表示中是隐式的。这使得图可以紧凑表示，并且可以在内存中连续存储。例如，要遍历图中所有节点的出边，你可以使用以下代码，它利用了 `graph.h` 中定义（并在 `graph_internal.h` 中实现）的便捷辅助函数：
```c
for (int i=0; i<num_nodes(g); i++) {
    // Vertex 被 typedef 为 int。Vertex* 指向 g.outgoing_edges[]
    const Vertex* start = outgoing_begin(g, i);
    const Vertex* end = outgoing_end(g, i);
    for (const Vertex* v=start; v!=end; v++)
    printf("Edge %u %u\n", i, *v);
}
```

#### 数据集

在本项目中，你将使用大规模图数据集来测试性能。数据集的位置取决于你的设置：

- 如果你在 Myth 机器上工作，图目录的路径为 `/afs/ir.stanford.edu/class/cs149/data/asst3_graphs/`
- 如果你在本地机器上工作，数据集可从 <http://cs149.stanford.edu/cs149asstdata/all_graphs.tgz> 下载。你可以使用 `wget http://cs149.stanford.edu/cs149asstdata/all_graphs.tgz` 下载数据集，然后使用 `tar -xzvf all_graphs.tgz` 解压。请注意，这是一个 3 GB 的下载。

一些有趣的真实世界图包括：

 * com-orkut_117m.graph 
 * oc-pokec_30m.graph
 * soc-livejournal1_68m.graph
 
有用的大规模合成图包括：

 * random_500m.graph
 * rmat_200m.graph

还有一些用于测试的非常小的图。如果你查看初始代码的 `/tools` 目录，你会注意到一个名为 `graphTools.cpp` 的有用程序，也可以用来创建你自己的图。

## Part 1：并行"自顶向下"广度优先搜索（20分）

广度优先搜索（BFS）是一种常见算法，你可能在之前的算法课中见过（参见[此处](https://www.hackerearth.com/practice/algorithms/graphs/breadth-first-search/tutorial/)和[此处](https://www.youtube.com/watch?v=oDqjPvD54Ss)获取有用的参考资料）。
请熟悉 `bfs/bfs.cpp` 中的函数 `bfs_top_down()`，它包含了 BFS 的串行实现。该代码使用 BFS 计算图中所有顶点到顶点 0 的距离。你可能还需要熟悉 `common/graph.h` 中定义的图结构以及简单的数组数据结构 `vertex_set`（`bfs/bfs.h`），它是一个用于表示 BFS 当前边界的顶点数组。

你可以使用以下命令运行 bfs：

    ./bfs <图目录路径>/rmat_200m.graph

其中 `<图目录路径>` 是包含图文件的目录路径（参见上面的"数据集"部分）。

运行 `bfs` 时，你会看到算法每一步的执行时间和边界大小。自顶向下版本的正确性将通过（因为我们给了你一个正确的串行实现），但速度会很慢。（注意，`bfs` 会报告"自底向上"和"混合"版本算法的失败，这些你将在本作业后面实现。）

在作业的这部分，你的工作是并行化自顶向下的 BFS。你需要专注于识别并行性，以及插入适当的同步以确保正确性。我们想提醒你，你**不应该**期望在这个问题上达到接近完美的加速比（我们留给你自己去思考为什么！）。

__提示/建议：__

* 始终从考虑哪些工作可以并行完成开始。
* 计算的某些部分可能需要同步，例如，通过使用 `#pragma omp critical` 或 `#pragma omp atomic` 将适当的代码包裹在临界区内。**然而，在这个问题中，你应该思考如何利用称为 `compare and swap` 的简单原子操作。** 你可以阅读 [GCC 的 compare and swap 实现](https://gcc.gnu.org/onlinedocs/gcc-9.4.0/gcc/_005f_005fsync-Builtins.html)，它以函数 `__sync_bool_compare_and_swap` 的形式暴露给 C 代码。如果你能弄清楚如何在这个问题中使用 compare-and-swap，你将获得比使用临界区高得多的性能。
* 更新共享计数器可以使用 `#pragma omp atomic` 在像 `counter++;` 这样的行之前高效完成。
* 是否存在可以避免使用 `compare_and_swap` 的条件？换句话说，当你*事先知道*比较会失败时？
* 有一个预处理器宏 `VERBOSE`，可以方便地在你的解决方案中禁用在每步打印有用计时信息（参见 `bfs/bfs.cpp` 顶部）。一般来说，这些 printf 发生的频率足够低（每个 BFS 步骤仅一次），不会显著影响性能，但如果你想在计时期间禁用 printf，可以使用这个 `#define`。

## Part 2："自底向上" BFS（25分）

思考什么行为可能导致 Part 1.2 的 BFS 实现出现性能问题。在这些情况下，广度优先搜索步骤的另一种实现可能更高效。不同于遍历边界中的所有顶点并标记所有与边界相邻的顶点，可以通过让*每个顶点检查自己是否应被添加到边界中*来实现 BFS！该算法的基本伪代码如下：

```
    for each vertex v in graph:
        if v has not been visited AND 
           v shares an incoming edge with a vertex u on the frontier:
              add vertex v to frontier;
```

该算法有时被称为"自底向上"的 BFS 实现，因为每个顶点"沿着 BFS 树向上"查找其祖先。（而不是像 Part 1.2 中以"自顶向下"的方式被其祖先发现。）

请实现一个自底向上的 BFS 来计算从根节点到图中所有顶点的最短路径（参见 `bfs/bfs.cpp` 中的 `bfs_bottom_up()`）。首先实现一个简单的串行版本。然后并行化你的实现。

__提示/建议：__

* 思考如何表示未访问节点的集合可能会很有用。自顶向下和自底向上版本的代码是否适合不同的实现？
* 自底向上 BFS 的同步要求有何变化？

## Part 3：混合 BFS（25分）

注意在 BFS 的某些步骤中，"自底向上"的 BFS 比自顶向下版本快得多。而在其他步骤中，自顶向下版本则快得多。这表明你的实现有一个重大的性能改进机会，**如果你能根据边界的大小或图的其他属性动态地在"自顶向下"和"自底向上"方案之间选择的话！** 如果你想要一个与参考方案有竞争力的解决方案，你的实现很可能必须实现这种动态优化。请在 `bfs/bfs.cpp` 中的 `bfs_hybrid()` 中提供你的解决方案。

__提示/建议：__

* 如果你在 Part 1.2 和 1.3 中使用了不同的边界表示，在混合解决方案中你可能需要在这些表示之间进行转换。你如何高效地在它们之间转换？这样做有开销吗？

你可以通过以下命令运行我们的评分脚本：`./bfs_grader <图目录路径>`，它将报告多个图的正确性和性能分数。

## 评分与提交

除了你的代码，我们希望你还提交一份清晰但简洁的高层描述，说明你的实现如何工作，以及你是如何得出你的解决方案的简要描述。具体描述你在此过程中尝试过的方法，以及你是如何确定如何优化代码的（例如，你进行了哪些测量来指导你的优化工作？）。

你应在报告中提及的工作方面包括：

1. 在报告顶部写上两位合作者的姓名。
2. 在 Myth 机器上运行 bfs_grader，并在你的解决方案中插入分数表的副本。**我们将使用 Myth 机器来评分你的代码。**
3. 描述优化代码的过程：
 * 在 Part 1（自顶向下）和 Part 2（自底向上）中，每个解决方案中的同步在何处？你是否做了什么来限制同步的开销？
 * 在 Part 3（混合）中，你是否决定动态地在自顶向下和自底向上的 BFS 实现之间切换？你如何决定使用哪种实现？
 * 为什么你认为你的代码（以及教师参考代码）无法达到完美的加速比？（是工作负载不均衡？通信/同步？数据移动？）

## 分数分配

本作业的84分分配如下：

* 70分：BFS 性能
* 14分：报告

如果你在本作业中获得了 `x` 分，我们将按 `(x/84) * 10` 分（四舍五入到十分位）提升你之前任何编程作业的成绩。

## 提交说明

请使用 Gradescope 提交你的作业。

1. __请将你的报告以 PDF 格式提交到 Gradescope 作业 Programming Assignment 5 (Writeup)。__
2. __要提交代码，请运行 `sh create_submission.h` 生成一个 `tar.gz` 文件，并将其提交到 Programming Assignment 5 (Code)。__ 我们只查看你的 `bfs/bfs.cpp` 和 `bfs/bfs.h` 文件，所以不要更改任何其他文件。在提交源文件之前，请确保所有代码都可以编译和运行！我们应该能够简单地 make，然后在 `/bfs` 目录中执行你的程序，无需手动干预。

我们的评分脚本将重新运行检查器代码，以验证你的分数与你在报告中提交的分数一致。我们还可能在其他数据集上运行你的代码以进一步检查其正确性。
