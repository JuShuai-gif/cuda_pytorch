set pagination off
set breakpoint pending on
# break when the tensor is used in an op; inspect its TensorImpl
break at::TensorIteratorBase::build
run
# skip creation-time hits, stop on the sum()
echo \n===== inspect a live TensorImpl =====\n
# grab the first input tensor of the iterator config
frame 0
# print the TensorImpl of 'this' iterator is complex; instead inspect via a known tensor:
# Use the python-side tensor through its C++ TensorImpl pointer is hard here,
# so we demonstrate the struct layout via ptype instead.
ptype /o c10::TensorImpl
