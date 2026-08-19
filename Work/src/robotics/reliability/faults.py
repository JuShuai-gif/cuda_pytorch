"""Fault profiles: the six-step incident loop for each failure mode.

For every fault a production system can hit, the incident playbook records
the same six fields, so diagnosis is a checklist rather than guesswork:

    symptom -> first evidence -> diagnosis -> root cause -> recovery -> fix

This module encodes the nine faults master prompt lists.  The point of fault
injection (chaos engineering) is to *practice* these loops before they happen
in production.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaultProfile:
    name: str
    symptom: str
    first_evidence: str
    diagnosis: str
    root_cause: str
    recovery: str
    fix: str


FAULTS: dict[str, FaultProfile] = {
    "process_crash": FaultProfile(
        "Process Crash", "推理进程消失，请求无响应",
        "watchdog 心跳超时 / 进程退出码", "进程是否被 OOM killer 杀掉",
        "段错误 / 未捕获异常 / OOM killer", "watchdog 自动重启进程",
        "修复崩溃根因（空指针/越界），加核心转储"),
    "gpu_oom": FaultProfile(
        "GPU OOM", "推理报 CUDA out of memory",
        "torch.cuda OOM 异常 / nvidia-smi 显存满", "哪张 tensor 吃满显存",
        "KV cache 无界增长 / 模型过大 / 泄漏", "清理缓存 / 减小 batch / 重启",
        "显存预算 + 分页 KV cache + 泄漏修复"),
    "cuda_error": FaultProfile(
        "CUDA Error", "CUDA runtime 报错，后续 kernel 全失败",
        "cudaGetErrorString 返回错误码", "错误码含义 + 发生位置",
        "非法访存 / 竞争 / driver 不匹配", "重置 CUDA context",
        "compute-sanitizer 定位非法访存"),
    "model_load_failure": FaultProfile(
        "Model Load Failure", "模型加载失败，服务起不来",
        "启动日志报加载异常", "是文件损坏还是版本不兼容",
        "OTA 下载损坏 / 版本不匹配", "回滚到上一个可用版本",
        "checksum 校验 + 健康检查（Stage 24 OTA）"),
    "network_failure": FaultProfile(
        "Network Failure", "请求超时/失败",
        "连接错误 / 超时日志", "哪一跳断了",
        "网线/路由/DNS/防火墙", "重试 + circuit breaker",
        "冗余链路 + 重试策略（Stage 22）"),
    "cloud_disconnect": FaultProfile(
        "Cloud Disconnect", "机器人无法连云端",
        "心跳丢失 / MQTT 断连", "是网络还是云端故障",
        "云端不可达 / 证书过期", "降级到本地自治模式",
        "edge gateway 本地自治 + 离线缓存（Stage 23）"),
    "disk_full": FaultProfile(
        "Disk Full", "写日志/数据失败",
        "磁盘使用率 100%", "什么占满磁盘",
        "日志无轮转 / 数据无清理", "清理日志 + 停写入",
        "日志轮转 + 数据生命周期管理"),
    "memory_leak": FaultProfile(
        "Memory Leak", "内存持续增长，最终 OOM",
        "内存曲线单调上升", "哪个对象没释放",
        "缓存无界 / 引用未释放", "定时重启（治标）",
        "泄漏检测工具定位 + 修复"),
    "thermal_throttling": FaultProfile(
        "Thermal Throttling", "延迟突然变慢且抖动",
        "温度超阈值 + 频率下降", "是散热还是负载问题",
        "散热不足 / 环境温度高 / 持续满载", "降负载 / 降频",
        "散热改进 + 功耗预算（Stage 15）"),
}
