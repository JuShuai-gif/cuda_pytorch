### NCCL 中的集合通信操作

这是我在练习中使用的主要 NCCL 集合通信操作的集合。

1. `gather.py`
    - 每个 GPU 有自己的张量
    - 所有张量收集（拼接）到单个 GPU（root）上
    - 其他 GPU 不接收完整结果
2. `all_gather.py` - 这里不是单个 GPU 拥有所有数据，而是所有 GPU 都拥有彼此的所有副本。
3. `reduce.py`
    - 所有 GPU 都有一些张量
    - 对元素逐位应用 reduce 操作（sum、mean 等）
    - 结果存储在单个 GPU（root）上
4. `all_reduce.py` - 这里所有 GPU 都拥有 reduce 后的副本
5. `scatter.py`
    - 一个 GPU 持有被分成 N 块的大张量
    - 每个 GPU 接收一块
    - 某种程度上与 gather 相反
6. `reduce_scatter.py`
    - 所有 GPU 以完整张量开始
    - 跨 GPU 应用 reduce 操作
    - 结果被拆分，每个 GPU 只获得自己的一块
    - 等同于 reduce + scatter
