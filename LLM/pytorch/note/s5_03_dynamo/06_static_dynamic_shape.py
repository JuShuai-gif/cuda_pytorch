import torch

# 默认行为
comp1 = [0]


def compiler1(gm, example_inputs):
    comp1[0] += 1
    print(f">>> 编译 #{comp1[0]}")
    gm.graph.print_tabular()
    return gm


@torch.compile(backend=compiler1)
def foo(x):
    return x * 2


print("=== 默认：静态到动态 ===")
print("调用 1: shape (10,)  特化，Guard 包含 size=[10]")
foo(torch.randn(10))

print("\n调用 2: shape (10,)  Guard 通过，缓存命中")
foo(torch.randn(10))

print("\n调用 3: shape (20,)  Guard 失败，重新编译。现在动态 size=[None]")
foo(torch.randn(20))

print("\n调用 4: shape (5,)  Guard 通过（动态），不重新编译")
foo(torch.randn(5))

print(f"\n总计: 4 次调用，{comp1[0]} 次编译")


# 使用显式 mark_dynamic 会怎样
comp2 = [0]


def compiler2(gm, example_inputs):
    comp2[0] += 1
    print(f">>> 编译 #{comp2[0]}")
    return gm


@torch.compile(backend=compiler2)
def bar(x):
    return x * 2


print("\n\n=== 显式 mark_dynamic(x, 0) ===")
print("调用 1: shape (10,) 但被标记为动态  无特化")
x1 = torch.randn(10)
torch._dynamo.mark_dynamic(x1, 0)
bar(x1)

print("\n调用 2: shape (20,)  不重新编译，已经是动态")
bar(torch.randn(20))

print("\n调用 3: shape (5,)  不重新编译")
bar(torch.randn(5))

print(f"\n总计: 3 次调用，{comp2[0]} 次编译")


print(f"\n{'=' * 60}")
print(f"默认:      4 次调用 {comp1[0]} 次编译（重新编译一次，然后动态）")
print(f"mark_dynamic: 3 次调用 {comp2[0]} 次编译（从一开始就是动态）")
print(f"{'=' * 60}")
print("静态: 编译器知道确切大小，所以内核更快（展开、预取）")
print("动态: 一个内核适配所有大小，所以不重新编译，但优化程度较低")


# python3 dynamo/06_static_dynamic_shape.py
