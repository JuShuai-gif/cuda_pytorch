from typing import List
import torch

compilation_count = 0


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    global compilation_count
    compilation_count += 1
    print(f">>> 编译 #{compilation_count}")
    gm.graph.print_tabular()
    return gm


@torch.compile(backend=my_compiler)
def foo(x, y):
    return (x + y) * x


print("=== 调用 1: x.shape=(10,), y.shape=(10,) ===")
foo(torch.randn(10), torch.ones(10))

print("\n=== 调用 2: 相同形状  Guard 通过，不重新编译 ===")
foo(torch.randn(10), torch.ones(10))

# --- 不同形状：Guard 失败，重新编译 ---
print("\n=== 调用 3: x.shape=(20,), y.shape=(20,)  形状 Guard 失败，重新编译 ===")
foo(torch.randn(20), torch.ones(20))

# --- 不同 dtype：Guard 失败，重新编译 ---
print("\n=== 调用 4: 相同形状但 x.dtype=float64  dtype Guard 失败，重新编译 ===")
foo(torch.randn(10, dtype=torch.float64), torch.ones(10, dtype=torch.float64))

# --- 不同设备（如果有 CUDA）：Guard 失败 ---
if torch.cuda.is_available():
    print("\n=== 调用 5: 相同形状但在 cuda 上  device Guard 失败，重新编译 ===")
    foo(torch.randn(10, device="cuda"), torch.ones(10, device="cuda"))
else:
    print("\n=== 调用 5: 跳过（无 CUDA） ===")

print(f"\n{'=' * 60}")
print(f"总编译次数: {compilation_count}（每个唯一的 Guard 组合一次）")
total_calls = 5 if torch.cuda.is_available() else 4
print(f"总调用次数: {total_calls}")
print(f"缓存命中（Guard 通过）: {total_calls - compilation_count}")
print(f"{'=' * 60}")


x = torch.randn(10)
y = torch.ones(10)
explanation = torch._dynamo.explain(foo, x, y)
print(f"\n=== torch._dynamo.explain() ===")
print(f"图断裂数: {explanation.graph_break_count}")
print(f"已编译图数: {explanation.graph_count}")
print(f"\n  Guard ({len(explanation.out_guards)} 个):")
for g in explanation.out_guards:
    if g.code_list:
        print(f"    [{g.source}] {g.guard_types}: {g.code_list}")
    else:
        print(f"    [{g.source}] {g.guard_types or g.name}")


# TORCH_LOGS=guards python3 dynamo/03_guards.py
# TORCH_LOGS=verbose_guards python3 dynamo/03_guards.py
# TORCH_LOGS=recompiles python3 dynamo/03_guards.py
