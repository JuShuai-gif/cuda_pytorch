# 06｜A/B Test：模型实验不能只看 accuracy

## 本模块解决的问题

模型实验选版本时，只看 accuracy 是经典的错误决策。本章回答：

```text
为什么 accuracy 高的模型，业务指标（robot success rate）可能更差？
A/B test 应该看哪些指标？
技术指标和业务指标怎么挂钩？
```

配套代码：`src/serving/ab_test/`（模型变体 + 任务模拟 + 多指标对比）。

---

## 1. 核心洞察：accuracy 不是终点

master prompt 明确：model experiment 不能只看 accuracy，还要看：

```text
accuracy / error rate   单步正确率
latency                 推理延迟
failure rate            系统失败率（crash/OOM）
resource usage          资源消耗
robot success rate      任务成功率（业务指标）
user metric             用户体验指标
```

因为**这些指标会互相冲突，而业务目标才是最终裁判**。

---

## 2. 实测：慢而准 vs 快而略不准

机器人抓取任务，控制环 deadline = 50ms（20Hz），连续 10000 步：

```text
model                    accuracy   latency   robot success rate
A_slow_accurate          95.0%      80ms      0.0%
B_fast_less_accurate     90.0%      15ms      89.3%
```

```text
accuracy says     : ship A（95% > 90%）
robot success says: ship B（89.3% > 0%）
```

### 为什么 A 的 accuracy 高却任务失败？

机器人任务是**实时**的：每个动作必须在 deadline 内到达执行器。

```text
模型 A：每个动作 95% 正确，但 80ms > 50ms deadline → 每个动作都迟到 → 失败
模型 B：每个动作 90% 正确，15ms < 50ms deadline → 动作及时 → 大部分成功
```

**"迟到"和"错误"一样致命**。A 的 5% 精度优势，被 100% 的超时完全抵消（甚至反转）。这就是 Stage 14（实时性）在 A/B 决策里的体现：**在实时系统里，latency 是 accuracy 的一部分**。

---

## 3. 业务指标的层次

```text
第 1 层：模型指标      accuracy, loss（算法侧）
第 2 层：技术指标      latency, failure rate, resource usage（Infra 侧）
第 3 层：业务指标      robot success rate, user metric（业务侧）
```

**A/B test 的决策必须用第 3 层**。第 1、2 层是中间量，它们最终都要转化为业务指标：

```text
accuracy 高 + latency 高 → 动作迟到 → success rate 低
accuracy 中 + latency 低 → 动作及时 → success rate 高
failure rate 高        → 任务中断 → success rate 低
```

工业实践中，算法团队盯着 accuracy，Infra 团队盯着 latency/failure，但**最终拍板的是业务指标**。这就是为什么 A/B test 要贯穿三层指标，而不是只在一个团队内部闭环。

---

## 4. A/B test 的正确姿势

1. **定义业务指标**：先问"这个模型服务最终影响什么"（robot success rate？用户留存？）。
2. **多指标同时采集**：accuracy + latency + failure + resource，缺一不可。
3. **用业务指标决策**：技术指标是解释工具，业务指标是决策依据。
4. **足够的样本和时长**：小样本的指标差异可能是噪声（呼应 Stage 20 的统计显著性）。

---

## 5. 本模块闭环小结

```text
问题：模型实验怎么选版本
      ↓
误区：只看 accuracy → 选慢而准的 A → 实时任务里全失败
      ↓
正确：accuracy + latency + failure + resource → robot success rate 决策
      ↓
结论：实时系统里 latency 是 accuracy 的一部分，业务指标是最终裁判
      ↓
下一步：Stage 22 分布式系统基础（RPC/MQ/cache/一致性/幂等）
```

要继续就说「继续」。
