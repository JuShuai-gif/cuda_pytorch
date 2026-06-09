#!/usr/bin/env python3
"""
TimerContext: 作用域计时上下文管理器及函数级性能分析装饰器。
所有计时使用 time.perf_counter() 进行高精度挂钟时间测量。
"""

import time
import functools
from typing import Callable


class TimerContext:
    """
    作为上下文管理器用于计时代码块：

        with TimerContext("preprocess") as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> "TimerContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_s = time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    @property
    def elapsed_us(self) -> float:
        return self.elapsed_s * 1_000_000.0


def profile_func(tracker_or_name=None):
    """
    函数级性能分析装饰器。

    配合共享 LatencyTracker 使用:
        tracker = LatencyTracker()
        @profile_func(tracker)
        def my_func(): ...

    使用字符串名称（自动创建 tracker）:
        @profile_func("my_stage")
        def my_func(): ...
    """
    from tracker import LatencyTracker  # 延迟导入以避免循环依赖

    if callable(tracker_or_name):
        # 作为无参数的 @profile_func 使用
        func = tracker_or_name
        name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TimerContext(name) as t:
                result = func(*args, **kwargs)
            return result

        return wrapper

    # 作为 @profile_func(tracker) 或 @profile_func("name") 使用
    tracker = tracker_or_name

    def decorator(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with TimerContext(name) as t:
                result = func(*args, **kwargs)
            if isinstance(tracker, LatencyTracker):
                tracker.record(name, t.elapsed_us)
            return result

        return wrapper

    return decorator
