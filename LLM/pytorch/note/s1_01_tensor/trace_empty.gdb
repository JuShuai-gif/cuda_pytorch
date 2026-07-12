# 追踪 torch.empty(2,3):Python -> C++ 绑定 -> 包回 Python 对象
# 用法:gdb -q -batch -x trace_empty.gdb --args python trace_empty.py
set pagination off
set breakpoint pending on

# [1] torch.empty 的 Python C 绑定入口
#     (import torch 不会命中,只有显式调用 torch.empty(...) 才触发)
break torch::autograd::THPVariable_empty
run
echo \n===== [1] THPVariable_empty (torch.empty 的 Python 绑定入口) =====\n
bt 6

# 改断 THPVariable_Wrap 的 move 重载,抓"把返回值包成 Python 对象"这一步
#（empty 返回临时 at::Tensor -> 走 move 重载)
delete 1
break THPVariable_Wrap(at::TensorBase&&)
continue
echo \n===== [2] THPVariable_Wrap (C++ at::Tensor 包成 Python 对象) =====\n
bt 12
continue
