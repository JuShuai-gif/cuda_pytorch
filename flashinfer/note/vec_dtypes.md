# FlashInfer vec_dtypes.cuh 头文件函数用法总结

`vec_dtypes.cuh` 是 FlashInfer 库中的向量数据类型头文件，提供了 CUDA 设备端的高效向量操作、类型转换和内存访问原语。该文件定义了多种低精度浮点格式（FP8、FP4、BF16、FP16）和整数类型的向量化操作。

## 文件概览

- **文件名**: `vec_dtypes.cuh`
- **路径**: `/home/ghr/llm/flashinfer/include/flashinfer/vec_dtypes.cuh`
- **命名空间**: `flashinfer`
- **主要功能**:
  1. 原子内存操作（release/acquire语义）
  2. 类型转换模板类（`vec_cast`）
  3. 向量模板类（`vec_t`）支持多种数据类型和向量大小
  4. 快速反量化函数（FP8→FP16/BF16）
  5. 辅助函数（指数/尾数位数查询等）

## 1. 内存操作函数

### 1.1 原子存储操作

```cpp
__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr);
```

**功能**: 使用 release 语义原子存储 4 个 32 位整数到全局内存。

**参数**:
- `val`: 要存储的 `int4` 值
- `addr`: 目标内存地址

**使用场景**: 多线程同步，确保存储操作对所有后续的 acquire 加载可见。

### 1.2 原子加载操作

```cpp
__device__ __forceinline__ int4 ld_global_acquire(int4* addr);
```

**功能**: 使用 acquire 语义从全局内存原子加载 4 个 32 位整数。

**参数**:
- `addr`: 源内存地址

**返回值**: 加载的 `int4` 值

**使用场景**: 多线程同步，确保能看到所有之前的 release 存储操作。

### 1.3 Volatile 内存操作

```cpp
__device__ __forceinline__ void st_global_volatile(int4 const& val, int4* addr);
__device__ __forceinline__ int4 ld_global_volatile(int4* addr);
```

**功能**: volatile 内存操作，防止编译器优化，用于内存映射 I/O 或特殊硬件寄存器访问。

## 2. 类型转换模板类 `vec_cast`

`vec_cast` 模板类提供了不同类型之间的向量化转换，特别优化了低精度浮点格式的转换。

### 2.1 通用转换模板

```cpp
template <typename dst_t, typename src_t>
struct vec_cast {
    template <size_t vec_size>
    FLASHINFER_INLINE static void cast(dst_t* dst, const src_t* src);
};
```

**基本用法**: 将 `src_t` 类型的 `vec_size` 个元素转换为 `dst_t` 类型，结果存储在 `dst` 中。

### 2.2 特殊化的类型转换

文件为以下类型组合提供了优化的特化实现：

#### FP8 相关转换
- `vec_cast<__nv_fp8_e4m3, float>`: float → FP8 E4M3
- `vec_cast<__nv_fp8_e5m2, float>`: float → FP8 E5M2
- `vec_cast<__nv_fp8_e4m3, half>`: half → FP8 E4M3（支持硬件加速）
- `vec_cast<__nv_fp8_e5m2, half>`: half → FP8 E5M2（支持硬件加速）
- `vec_cast<half, __nv_fp8_e4m3>`: FP8 E4M3 → half
- `vec_cast<half, __nv_fp8_e5m2>`: FP8 E5M2 → half
- `vec_cast<nv_bfloat16, __nv_fp8_e4m3>`: FP8 E4M3 → bfloat16
- `vec_cast<nv_bfloat16, __nv_fp8_e5m2>`: FP8 E5M2 → bfloat16

#### FP16/BF16/float 相互转换
- `vec_cast<float, half>`: half → float
- `vec_cast<half, float>`: float → half
- `vec_cast<float, nv_bfloat16>`: bfloat16 → float
- `vec_cast<nv_bfloat16, float>`: float → bfloat16

### 2.3 转换示例

```cpp
// 将 float 向量转换为 FP8 E4M3
__nv_fp8_e4m3 fp8_data[vec_size];
float float_data[vec_size];
vec_cast<__nv_fp8_e4m3, float>::cast<vec_size>(fp8_data, float_data);

// 将 half 向量转换为 FP8 E5M2（使用硬件加速）
__nv_fp8_e5m2 fp8_data2[vec_size];
half half_data[vec_size];
vec_cast<__nv_fp8_e5m2, half>::cast<vec_size>(fp8_data2, half_data);
```

## 3. 向量模板类 `vec_t`

`vec_t` 是一个通用的向量模板类，支持多种数据类型和向量大小，提供了向量化加载、存储、填充和类型转换操作。

### 3.1 类定义

```cpp
template <typename float_t, size_t vec_size>
struct vec_t {
    // 元素访问
    FLASHINFER_INLINE float_t& operator[](size_t i);
    FLASHINFER_INLINE const float_t& operator[](size_t i) const;
    
    // 向量操作
    FLASHINFER_INLINE void fill(float_t val);
    FLASHINFER_INLINE void load(const float_t* ptr);
    FLASHINFER_INLINE void store(float_t* ptr) const;
    
    // 原子内存操作
    FLASHINFER_INLINE void load_global_acquire(float_t* addr);
    FLASHINFER_INLINE void store_global_release(float_t* addr) const;
    FLASHINFER_INLINE void load_global_volatile(float_t* addr);
    FLASHINFER_INLINE void store_global_volatile(float_t* addr) const;
    
    // 类型转换
    template <typename T>
    FLASHINFER_INLINE void cast_from(const vec_t<T, vec_size>& src);
    
    template <typename T>
    FLASHINFER_INLINE void cast_load(const T* ptr);
    
    template <typename T>
    FLASHINFER_INLINE void cast_store(T* ptr) const;
    
    // 内存拷贝
    FLASHINFER_INLINE static void memcpy(float_t* dst, const float_t* src);
    
    // 数据指针
    FLASHINFER_INLINE float_t* ptr();
};
```

### 3.2 支持的数据类型和向量大小

文件为以下数据类型提供了特化实现：

#### 3.2.1 FP8 类型
- `__nv_fp8_e4m3`（4位指数，3位尾数）
  - 支持向量大小: 1, 2, 4, 8, 16+
  - 16+ 向量使用 `int4` 数组存储
- `__nv_fp8_e5m2`（5位指数，2位尾数）
  - 支持向量大小: 1, 2, 4, 8, 16+

#### 3.2.2 FP4 类型（需要 CUDA >= 12.08）
- `__nv_fp4_e2m1`（2位指数，1位尾数）
  - 支持向量大小: 2, 4, 8, 16, 32+
  - 启用条件: `FLASHINFER_ENABLE_FP4_E2M1` 且 `CUDA_VERSION >= 12080`

#### 3.2.3 半精度浮点
- `half`（FP16）
  - 支持向量大小: 1, 2, 4, 8+
  - 8+ 向量使用 `int4` 数组存储
- `nv_bfloat16`（BF16）
  - 支持向量大小: 1, 2, 4, 8+

#### 3.2.4 其他类型
- `uint8_t`: 支持向量大小 1, 2, 4, 8, 16+
- `float`: 支持向量大小 1, 2, 4+

### 3.3 使用示例

```cpp
// 创建并初始化 half 向量
vec_t<half, 8> vec_half;
vec_half.fill(__float2half(1.0f));

// 从全局内存加载
half* device_ptr = ...;
vec_half.load(device_ptr);

// 存储到全局内存（带 release 语义）
vec_half.store_global_release(device_ptr);

// 类型转换：从 float 向量转换
vec_t<float, 8> vec_float;
vec_float.fill(2.0f);
vec_half.cast_from(vec_float);

// 类型转换加载：直接从 float 指针加载并转换为 half
float* float_ptr = ...;
vec_half.cast_load(float_ptr);

// 类型转换存储：将 half 向量转换为 float 并存储
vec_half.cast_store(float_ptr);
```

## 4. 辅助函数

### 4.1 浮点格式属性查询

```cpp
template <typename T>
constexpr FLASHINFER_INLINE int get_exponent_bits();

template <typename T>
constexpr FLASHINFER_INLINE int get_mantissa_bits();
```

**功能**: 查询浮点类型的指数位和尾数位数。

**支持的类型**:
- `__nv_fp8_e4m3`: 4位指数，3位尾数
- `__nv_fp8_e5m2`: 5位指数，2位尾数
- `half`: 5位指数，11位尾数
- `nv_bfloat16`: 8位指数，7位尾数

### 4.2 快速反量化函数

```cpp
template <typename fp8_dtype, typename fp16_dtype>
__device__ void fast_dequant_f8f16x4(uint32_t* input, uint2* output);
```

**功能**: 将 4 个 FP8 值（打包在 32 位中）快速反量化为 4 个 FP16/BF16 值。

**实现特点**:
- 针对 FP8 E5M2 → half 有硬件优化路径（使用 `__byte_perm`）
- 其他类型使用软件实现，参考 Marlin 反量化算法
- 支持 `half` 和 `nv_bfloat16` 输出

### 4.3 向量对类型映射

```cpp
template <typename T>
struct vec2_dtype;

template <typename T>
using vec2_dtype_t = typename vec2_dtype<T>::type;
```

**功能**: 获取类型的向量对类型（如 `half` → `half2`, `__nv_bfloat16` → `__nv_bfloat162`）。

**映射关系**:
- `half` → `half2`
- `__nv_bfloat16` → `__nv_bfloat162`
- `__nv_fp8_e4m3` → `__nv_fp8x2_e4m3`
- `__nv_fp8_e5m2` → `__nv_fp8x2_e5m2`
- 其他类型: 返回自身

### 4.4 向量对元素访问

```cpp
template <typename T, size_t VEC_SIZE>
FLASHINFER_INLINE vec2_dtype_t<T> get_vec2_element(vec_t<T, VEC_SIZE>& vec, int i);
```

**功能**: 从向量中获取第 i 个向量对元素（要求向量大小为 2 的倍数）。

## 5. 编译时配置

### 5.1 硬件 FP8 转换支持

```cpp
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED
#endif
```

**说明**: 在 SM90 及以上架构启用硬件 FP8 转换指令。

### 5.2 向后兼容性处理

```cpp
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 < 120200) && \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
// 为旧 CUDA 版本和旧架构提供 bfloat16 操作的替代实现
#endif
```

**说明**: 为 CUDA < 12.2 且架构 < SM80 的系统提供 `__hmul`, `__hmul2`, `__floats2bfloat162_rn` 等函数的替代实现。

## 6. 使用注意事项

1. **设备端使用**: 所有函数都标记为 `__device__`，只能在 CUDA 内核中使用。
2. **向量大小约束**: 每种数据类型有特定的向量大小要求（2的幂次，且满足最小对齐）。
3. **内存对齐**: 对于大向量（16+元素），假设数据按 16 字节对齐。
4. **硬件要求**: FP8 硬件转换需要 SM90+ 架构；FP4 支持需要 CUDA 12.08+。
5. **性能考虑**: 小向量（1, 2, 4 元素）使用寄存器存储；大向量使用共享内存或全局内存。

## 7. 典型使用场景

### 7.1 混合精度计算

```cpp
// 从全局内存加载 FP8 权重
vec_t<__nv_fp8_e4m3, 16> weight_vec;
weight_vec.load_global_acquire(weight_ptr);

// 转换为 half 进行计算
vec_t<half, 16> weight_half;
weight_half.cast_from(weight_vec);

// 执行计算...
```

### 7.2 内存高效存储

```cpp
// 计算得到 float 结果
vec_t<float, 8> result_float;
// ... 计算过程 ...

// 量化为 FP8 存储以节省内存
vec_t<__nv_fp8_e5m2, 8> result_fp8;
result_fp8.cast_from(result_float);
result_fp8.store_global_release(output_ptr);
```

### 7.3 线程间同步

```cpp
// 线程 0: 发布数据
vec_t<float, 4> data;
// ... 准备数据 ...
data.store_global_release(shared_addr);

// 线程 1: 获取数据（确保看到线程 0 的所有写入）
vec_t<float, 4> received_data;
received_data.load_global_acquire(shared_addr);
```

## 8. 性能优化建议

1. **使用合适的向量大小**: 根据 warp 大小（32 线程）和内存事务大小选择向量大小。
2. **利用硬件加速**: 在支持 SM90+ 的设备上，FP8 转换会自动使用硬件指令。
3. **批量操作**: 使用大向量减少内存事务次数。
4. **内存访问模式**: 确保向量加载/存储符合合并访问模式。
5. **寄存器压力**: 小向量更适合高寄存器压力的内核。

---

*文档生成时间: 2025-04-03*  
*基于文件: `/home/ghr/llm/flashinfer/include/flashinfer/vec_dtypes.cuh`*