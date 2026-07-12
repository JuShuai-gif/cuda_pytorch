# ============================================================================
# __init__.py —— utils 包的统一导出入口
#
# 作用：
#   本文件是 mcunet/utils 软件包的初始化模块。Python 中，当某个目录包含 __init__.py
#   时，该目录会被视为一个包（Package）。本文件通过星号 (from ... import *) 将
#   utils 子包下所有子模块中 __all__ 列出的公共接口统一提升到包顶层，方便外部调用。
#
# 设计意图：
#   使用者只需 `from mcunet.utils import set_running_statistics, AverageMeter, ...`
#   即可访问所有工具函数，而不需要了解内部子模块的文件组织结构。这是一种常见的
#   Python 包设计模式，用于简化 API 表面并提供稳定的公共接口。
# ============================================================================

# 从 pytorch_utils 模块导入所有公开接口（如 rm_bn_from_net, get_net_device,
# count_parameters, count_net_flops, count_peak_activation_size 等）
from .pytorch_utils import *

# 从 my_modules 模块导入所有公开接口（如 MyModule, MyNetwork, init_models,
# set_bn_param, get_bn_param, replace_bn_with_gn, MyConv2d 等）
from .my_modules import *

# 从 common_tools 模块导入所有公开接口（如 sort_dict, get_same_padding,
# get_split_list, list_sum, list_mean, accuracy, AverageMeter 等）
from .common_tools import *

# 从 pytorch_modules 模块导入所有公开接口（如 make_divisible, build_activation,
# ShuffleLayer, MyGlobalAvgPool2d, Hswish, Hsigmoid, SEModule 等）
from .pytorch_modules import *

# 从 bn_utils 模块导入所有公开接口（如 set_running_statistics,
# adjust_bn_according_to_idx, copy_bn 等 BN 操作函数）
from .bn_utils import *

# 从 net_config 模块导入所有公开接口（如 get_network_config_with_activation_shape）
from .net_config import *
