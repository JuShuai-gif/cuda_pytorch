# aos_vs_soa

AoS vs 并行数组（SoA）：对象大小与缓存局部性。

> 本代码是根据 PDF 中的概念重新编写的教学实现，不是原书源码的直接复制。

## 原理

PDF 第 120-127 页：遍历只访问部分字段时，**对象越小**遍历越快（空间局部性）。
书中 User 示例演进：

1. `BigUser`（128 字节，5 个 string 内联）：迭代慢；
2. `SmallUser`（40 字节，认证信息移出为指针）：约快 2 倍；
3. **并行数组**（只存 `level` 的 vector<short>、只存 `is_playing` 的
   vector<bool>）：更快，且 bool 版利用位数组。

书中实测（1M 用户）：11ms → 4ms → 0.7ms（level）/ 0.03ms（playing）。

## 构建与运行

```bash
cmake --build build --target ch04_aos_vs_soa_benchmark ch04_aos_vs_soa_tests
./build/chapter04_data_structures/ch04_aos_vs_soa_tests
./build/chapter04_data_structures/ch04_aos_vs_soa_benchmark
```

## 结果解释（GCC 13.3 Release，i9-14900HX，1M 用户）

本环境 `sizeof(BigUser)=168`、`sizeof(SmallUser)=48`（libstdc++ string 32 字节）。

num_users_at_level（找 level==5）：

| 表示 | mean | 相对 |
|---|---|---|
| `BigUser` AoS | ~5.5 ms | 1.0x |
| `SmallUser` AoS | ~4.2 ms | 1.3x |
| `vector<short>` SoA | ~0.25 ms | **~22x** |

num_playing_users：`vector<bool>` 位数组 ~0.77 ms（`BigUser` ~6.1ms，约 8 倍）。

## 重要警告（书中明确）

- 并行数组**破坏封装**、需手动保持索引同步、易错；
- 若算法要同时访问多个拆分字段，反而更慢；
- **"先采用良好的设计原则，遇到真实性能问题再考虑并行数组"**（PDF 127 页）。

## 结论

- 缩小对象、只遍历所需字段是普遍有效的优化方向；
- 并行数组是大刀阔斧的优化，收益大但维护成本高，谨慎使用。
