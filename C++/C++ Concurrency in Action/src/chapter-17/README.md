# Chapter 17 — 工程化实践

并发代码的测试、调试、构建和 CI/CD 最佳实践。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_sanitizer_demo.cpp` | Sanitizer 演示 | data race/TSan/ASan/UBSan、检测场景 |
| `02_stress_test.cpp` | 压力测试 | 不变式验证、长时间高负载、并发栈测试 |
| `03_concurrent_unit_test.cpp` | 单元测试框架 | 微型测试框架、并发断言、死锁超时 |
| `04_cmake_best_practices.cpp` | CMake 最佳实践 | 多构建配置、TSan 集成、LTO、测试超时 |

## 编译运行

```bash
# 普通编译
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)

./ch17_01_sanitizer_demo
./ch17_02_stress_test
./ch17_03_concurrent_unit_test
./ch17_04_cmake_best_practices

# 带 TSan 的构建
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" \
      -B build/tsan
cmake --build build/tsan -j$(nproc)
./build/tsan/ch17_01_sanitizer_demo
```

## 学习建议

1. TSan 应在每次提交前运行，捕获数据竞争
2. 压力测试的持续时间越长，发现时序 bug 的概率越高
3. 不变式验证是并发测试的核心方法论
4. CMake 多构建配置让 Debug/Release 切换更简单
5. CI 中必须设置 test timeout，防止死锁挂起流水线
