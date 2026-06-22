import matplotlib.pyplot as plt

# 读取数据
x, y = [], []
with open('/home/robot/WorkCode/sgemm.cu/benchmark_results/sgemm.cu.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        x.append(int(parts[0]))
        y.append(int(parts[1]))

plt.rc("font", size=18)
fig, ax = plt.subplots(figsize=(14, 10))

ax.set_xlabel("m=n=k", fontsize=24)
ax.set_ylabel("GFLOP/S", fontsize=24)
ax.set_title("sgemm.cu", fontsize=22)
ax.legend(fontsize=12, loc='lower right', prop={'size': 19})
ax.grid(axis='y')

# 设置 y 轴刻度间隔（例如，每 500 GFLOP/S 一个刻度）
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))  # 调整 500 为你的期望间隔

# 绘制图形
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'g-', linewidth=2, label='Performance')  # 绿色实线

# # 自定义样式
# plt.title('Performance Benchmark')
# plt.xlabel('Input Size')
# plt.ylabel('Time (ms)')
# plt.grid(True)
# plt.legend()

plt.show()


