# 08_ir_dump 实验

> 目的：用 `TL_ENABLE_DUMP_IR` + `TL_DUMP_IR_DIR` 观察每个 pass 之间的 IR。
> 前置：已完成源码编译（`12_编译与安装指南.md`）。本目录文件为模板，需在编译环境运行。

## 运行

```bash
python run_ir_dump.py
ls /tmp/dump_ir | head -30   # 查看每个 pass 的 IR 文件
```

## 输出说明

`/tmp/dump_ir/` 下每两个 pass 之间会写一个 IR 文件（`DumpIR` instrument，`tilelang/jit/kernel.py:240-242`）。

- 对比相邻文件可看出 `MaterializeKernelLaunch`、`LayoutInference`、`LowerTileOp` 等 pass 的作用。
- 用 `diff` 相邻文件可定位"哪个 pass 改变了什么"。

## 自测问题

1. 哪个 pass 把 `T.copy` 变成 `cp.async`？dump 对比确认。
2. `T.gemm` 在哪个 pass 之后消失（被展开成 mma）？
3. `thread_extent` 从哪来？
