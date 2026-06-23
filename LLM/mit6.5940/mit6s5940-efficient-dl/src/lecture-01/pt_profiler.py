import torch
from torch.profiler import profile, record_function, ProfilerActivity


# ## 最简单的 profiler 用法
# # 用 with 语句包裹要分析的代码块, activities 指定要采集 CPU 和 CUDA 上的算子耗时
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
#     for _ in range(10):
#         a = torch.square(torch.randn(10000, 10000).cuda())
#
# # 把采集到的 trace 导出成 chrome trace 格式, 可在 chrome://tracing 中打开查看
# prof.export_chrome_trace("trace.json")


## 带有预热(warmup)和跳过(skip)的高级用法
# 官方文档: https://pytorch.org/docs/stable/profiler.html


# 非默认的 profiler schedule 允许用户在训练循环的不同迭代上动态开关 profiler;
# 每当一段新的 trace 采集完成时, trace_handler 就会被调用一次
def trace_handler(prof):
    # key_averages() 把同名算子的多次调用聚合成统计结果;
    # .table() 打印成表格, 这里按 "self_cuda_time_total"(算子自身在 GPU 上的总耗时)排序;
    # row_limit=-1 表示不限制行数, 打印全部算子
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # 把这一轮的 trace 导出, 文件名带上 step_num 以区分不同的采集周期
    prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


with torch.profiler.profile(
    # 同时采集 CPU 端和 CUDA 端(GPU)的活动
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # 一个采集周期(cycle) = wait + warmup + active 个迭代, 然后重复 repeat 次。
    # 本例 wait=1, warmup=1, active=2, repeat=1, 循环跑 10 次时各迭代的行为如下:
    #
    #   iter | 阶段   | 说明
    #   -----+--------+--------------------------------
    #     0  | wait   | 跳过(既不采集也不预热)
    #     1  | warmup | 预热, 结果丢弃(排除首次启用开销)
    #     2  | active | 采集
    #     3  | active | 采集 -> 周期结束, 触发 on_trace_ready
    #    4~9 | (空闲) | repeat=1 已用完, profiler 不再工作
    #
    # 即: 虽然循环 10 次, 但只有 iter 2、3 的数据被真正记录。
    schedule=torch.profiler.schedule(
        wait=1,  # 起始跳过的迭代数
        warmup=1,  # 预热的迭代数(结果被丢弃)
        active=2,  # 实际采集数据的迭代数
        repeat=1,
    ),  # 整个 (wait+warmup+active) 周期重复的次数
    on_trace_ready=trace_handler,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # 若要输出供 TensorBoard 查看的结果, 改用上面这一行
) as p:
    # 这个 for 循环模拟训练循环, 它与上面 with 块里的设置紧密配合:
    #   - with 块定义"采集规则"(schedule / on_trace_ready / activities)
    #   - for 循环执行"实际计算"(torch.square 这一工作负载)
    #   - p.step() 是把两者连起来的纽带: 每调用一次就推进 schedule 状态机一格,
    #     从而决定当前迭代处于 wait / warmup / active 哪个阶段。
    # 约束: 循环次数(10)必须 >= wait+warmup+active(=4), 否则采集不完整;
    #       p.step() 必须放在循环里, 否则状态机无法推进, 采集逻辑失效。
    for iter in range(10):
        # 待分析的工作负载: 生成随机矩阵搬到 GPU 上, 再做逐元素平方
        torch.square(torch.randn(10000, 10000).cuda())
        # 通知 profiler 下一个迭代已经开始(用于推进 schedule 状态机)
        p.step()
