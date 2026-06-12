#include <iostream>

extern "C" void print_plugin_contract() {
  std::cout << "TensorRT-LLM plugin contract demo\n";
  std::cout << "Inputs: Q/K/V or packed QKV, optional KV cache, sequence metadata\n";
  std::cout << "Output: fused attention result, optional updated KV cache\n";
}
