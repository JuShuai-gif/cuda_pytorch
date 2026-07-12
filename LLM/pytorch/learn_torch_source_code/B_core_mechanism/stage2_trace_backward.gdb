set pagination off
set breakpoint pending on
# entry point of the reverse-mode engine (called on the Python thread)
break torch::autograd::Engine::execute
# the generated backward node for x*x (runs on an engine worker thread)
break torch::autograd::generated::MulBackward0::apply

run
echo \n===== [1] Engine::execute  (backward entry, Python thread) =====\n
bt 12
continue
echo \n===== [2] MulBackward0::apply  (executing dy/dx on worker thread) =====\n
bt 14
continue
