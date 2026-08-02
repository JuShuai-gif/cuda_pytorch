import os
import random

from triton.runtime.cache import FileCacheManager


class LigerTritonFileCacheManager(FileCacheManager):
    """
    自定义 Triton 文件缓存管理器。

    重写 Triton 默认的 FileCacheManager.put()，改为先写入临时目录再原子替换，
    以避免 Triton 编译缓存并发写入时的瞬时 FileNotFoundError。

    Triton 编译一次 kernel 后，缓存目录 ~/.triton/cache/<hash>/ 会写入一组产物，
    每个文件都经由 put()/put_group() 落盘：

    1. 中间表示产物（每个代表编译流水线的一环，文本格式）：
       - <kernel>.source : AST 前端代码生成后的源码
       - <kernel>.ttir   : Triton IR（MLIR 形式）
       - <kernel>.ttgir  : GPU 优化后的 IR
       - <kernel>.llir   : LLVM IR
       - <kernel>.ptx    : PTX 汇编（NVIDIA；AMD 为 .amdgcn）
       - <kernel>.cubin  : 最终 GPU 二进制机器码（AMD 为 .hsaco）

    2. 元数据文件：
       - <kernel>.json : 该 kernel 的编译元数据（hash、target、编译选项、
                         env_vars、triton_version 等），以文本形式写入
       - __grp__<kernel>.json : 组文件，用 JSON 记录这一整组产物的路径映射
                         child_paths；缓存命中时 get_group() 据此一次取回整组
    """

    def put(self, data, filename, binary=True) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        # 根据 data 类型判断是否按二进制写入
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)
        assert self.lock_path is not None
        # 拼接最终缓存文件的完整路径
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        # 随机 ID，避免多个进程临时目录冲突
        rnd_id = random.randint(0, 1000000)
        # we use the PID incase a bunch of these around so we can see what PID made it
        # 带上进程 PID，便于排查是哪个进程写的临时文件
        pid = os.getpid()
        # use temp dir to be robust against program interruptions
        # 使用临时目录，避免程序中途崩溃留下半成品文件
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{pid}_{rnd_id}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)

        mode = "wb" if binary else "w"
        with open(temp_path, mode) as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        # POSIX 系统上 os.replace 是原子的，因此目标路径不会读到写入一半的内容
        os.replace(temp_path, filepath)
        # 清理已空出的临时目录
        os.removedirs(temp_dir)
        return filepath


def apply_liger_triton_cache_manager():
    """
    Experimental feature to get around transient FileNotFoundError in triton compilation.
    For more details please see https://github.com/triton-lang/triton/pull/4295

    通过环境变量 TRITON_CACHE_MANAGER 让 Triton 使用上述自定义缓存管理器，
    用于规避 Triton 编译时偶发的 FileNotFoundError（实验性特性）。
    """
    os.environ["TRITON_CACHE_MANAGER"] = (
        "liger_kernel.triton.monkey_patch:LigerTritonFileCacheManager"
    )
