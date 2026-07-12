# 追踪 t.view():Python 方法 -> C++ view -> 新建共享 Storage 的 TensorImpl
# 用法:gdb -q -batch -x trace_view.gdb --args python trace_view.py
set pagination off
set breakpoint pending on

# [1] Tensor.view 的 Python 方法绑定(import 不会命中)
break torch::autograd::THPVariable_view
run
echo \n===== [1] THPVariable_view (t.view 的 Python 方法绑定) =====\n
bt 6

# [2] C++ 实现;view_impl -> alias_with_sizes_and_strides
#     用 Storage(self.storage()) 复用原内存,只建新 TensorImpl(共享 Storage)
delete 1
break at::native::view
continue
echo \n===== [2] at::native::view (建共享 Storage 的新 TensorImpl) =====\n
bt 10
continue
