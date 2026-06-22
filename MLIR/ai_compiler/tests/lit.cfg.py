# -*- Python -*-
# edge_ai_compiler_pro lit 主配置

import os
import lit.formats
from lit.llvm import llvm_config

config.name = "EDGE"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.edge_obj_root, "tests")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.md", "lit.cfg.py"]

config.edge_tools_dir = os.path.join(config.edge_obj_root, "bin")

# 把 LLVM 工具目录 (含 FileCheck/count/not, 来自 LLVM build 目录) 与本项目工具目录
# 都加入 PATH, 并注册工具替换.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.edge_tools_dir, append_path=True)

tool_dirs = [config.edge_tools_dir, config.llvm_tools_dir]
tools = [
    "edge-opt",
    "edge-introspect",
    "edge-memplan",
    "edge-run",
    "edge-quantize",
    "mlir-opt",
    "FileCheck",
    "count",
    "not",
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
