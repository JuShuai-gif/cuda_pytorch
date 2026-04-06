#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

// FlashInfer 分页键值缓存系统头文件
// 本文件实现了分页键值缓存（Paged KV-Cache）的核心数据结构和CUDA内核
// 分页缓存将KV缓存划分为固定大小的页面，支持动态增长和高效内存管理
// 主要用于Transformer推理时的注意力机制，优化长序列处理的内存效率

#include <driver_types.h>

#include <vector>

#include "exception.h"
#include "fastdiv.cuh"
#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

// FlashInfer 命名空间，包含分页 KV-Cache 实现
// 该命名空间包含了所有分页键值缓存相关的数据结构和内核函数

/*!
 * \brief Paged key-value cache
 * \tparam layout The layout of last 3 dimensions in KV-Cache.
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
// 分页键值缓存核心数据结构
// 该结构体管理Transformer推理中的键值缓存，支持动态页面分配和高效内存访问
// 通过将KV缓存划分为固定大小的页面，可以灵活管理不同长度的序列，减少内存碎片
// 支持两种内存布局：HND（头-序列-维度）和NHD（序列-头-维度）
template <typename DType, typename IdType>
struct paged_kv_t {
  // 分页键值缓存数据结构
  // page_size: 每个页的大小（token 数量），决定了页面粒度
  // num_heads: 注意力头数，多头注意力机制中的头数
  // head_dim: 每个头的维度，每个注意力头的特征维度
  // batch_size: 批次大小，同时处理的请求数量
  // 页大小（快速除法对象），每个页面包含的token数量
  // 使用uint_fastdiv类型优化除法操作，提高页面索引计算性能
  // 页面是内存分配和管理的单元，固定大小便于内存管理和复用
  uint_fastdiv page_size;
  // 注意力头数，多头注意力机制中的头数
  // 每个头独立计算注意力权重，最后合并结果
  // 头数通常为模型架构的超参数，如32、64等
  uint32_t num_heads;
  // 每个注意力头的维度，每个头的特征维度大小
  // 决定了键值向量的长度，通常为64、128等
  // 总特征维度 = num_heads * head_dim
  uint32_t head_dim;
  // 批次大小，同时处理的请求（序列）数量
  // 批处理提高GPU利用率，但需要管理不同长度的序列
  // 分页缓存特别适合处理变长序列的批处理场景
  uint32_t batch_size;
  // 页间步长（跨页的步长），从一个页面到下一个页面的字节偏移量
  // 计算公式：stride_page = num_heads * page_size * head_dim
  // 用于在内存中导航到不同页面
  uint32_t stride_page;
  // 序列维度步长（跨页面内条目的步长），在同一页面内不同token间的偏移量
  // 根据布局不同：HND布局为head_dim，NHD布局为num_heads * head_dim
  // 用于访问同一页面内的不同序列位置
  uint32_t stride_n;
  // 头维度步长（跨头的步长），在同一页面内不同注意力头间的偏移量
  // 根据布局不同：HND布局为page_size * head_dim，NHD布局为head_dim
  // 用于访问同一页面内的不同注意力头
  uint32_t stride_h;

  // Internal layout:
  // [max_num_pages, num_heads, page_size, head_dim] if layout == HND
  // [max_num_pages, page_size, num_heads, head_dim] if layout == NHD
  // 键缓存数据指针，指向键（Key）缓存的内存起始位置
  // 存储所有页面中所有头的键向量，内存布局由stride参数定义
  // 通过get_k_ptr()等函数安全访问，支持保护性访问防止越界
  DType* k_data;
  // 值缓存数据指针，指向值（Value）缓存的内存起始位置
  // 存储所有页面中所有头的值向量，与键缓存有相同的布局
  // 通过get_v_ptr()等函数安全访问，与键缓存并行处理
  DType* v_data;
  // 页索引数组指针（映射逻辑页到物理页），实现逻辑页面到物理页面的映射
  // 逻辑页面是连续的页面索引，物理页面是实际内存位置
  // 这种映射支持页面复用和碎片整理，提高内存利用率
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  // 页索引指针数组（类似 CSR 格式的 indptr），第一个元素为 0，最后一个元素为 nnz_pages
  // 类似于稀疏矩阵的CSR格式，indptr[i]表示第i个请求的起始页面索引
  // indptr[i+1] - indptr[i] 表示第i个请求分配的页面数量
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  // 每个请求最后一页的有效token数量（偏移量）
  // 由于页面可能未完全填满，需要记录最后一页的实际使用长度
  // 序列总长度 = (页面数-1) * page_size + last_page_len
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  // 每个请求的起始位置（用于 RoPE 位置编码）
  // RoPE（旋转位置编码）需要绝对位置信息，该数组提供每个序列的起始位置
  // 用于计算相对位置编码，支持长上下文连续生成
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  // 构造一个空的分页键值缓存对象
  // 将所有指针初始化为nullptr，数值初始化为0
  // 用于创建未初始化的缓存对象，后续通过赋值或参数化构造函数初始化
  __host__ __device__ __forceinline__ paged_kv_t()
      : num_heads(0),
        page_size(),
        head_dim(0),
        batch_size(0),
        stride_page(0),
        stride_n(0),
        stride_h(0),
        k_data(nullptr),
        v_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache should be contiguous
   * \param v_data The start pointer of value cache, v_cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  // 构造分页键值缓存（连续内存布局）
  // 该构造函数适用于连续内存分配的KV缓存，计算内存步长基于布局
  // 参数说明：
  // - num_heads: 注意力头数，决定并行计算粒度
  // - page_size: 页面大小，影响内存碎片和利用率
  // - head_dim: 头维度，决定特征向量长度
  // - batch_size: 批次大小，支持批处理推理
  // - layout: 内存布局（HND或NHD），影响数据局部性和访问模式
  // - k_data/v_data: 键值缓存指针，需要连续内存分配
  // - indices: 页面索引数组，实现逻辑到物理页面映射
  // - indptr: 页面指针数组，类似CSR格式管理批次页面
  // - last_page_len: 最后一页有效长度，处理变长序列
  // - rope_pos_offset: RoPE位置偏移，支持位置编码
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* k_data,
                                      DType* v_data, IdType* indices, IdType* indptr,
                                      IdType* last_page_len, IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = num_heads * page_size * head_dim;
    this->k_data = k_data;
    this->v_data = v_data;
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache doesn't have to be contiguous
   * \param v_data The start pointer of value cache, v_cache doesn't have to be contiguous
   * \param kv_strides custom strides of each dimensions of k_data and v_data
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* k_data,
                                      DType* v_data, const int64_t* kv_strides, IdType* indices,
                                      IdType* indptr, IdType* last_page_len,
                                      IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = kv_strides[0];
    this->k_data = k_data;
    this->v_data = v_data;
    stride_n = layout == QKVLayout::kHND ? kv_strides[2] : kv_strides[1];
    stride_h = layout == QKVLayout::kHND ? kv_strides[1] : kv_strides[2];
  }

  // 获取指定批次的序列长度
  // 计算第batch_idx个请求的序列总长度（token数量）
  // 算法：总长度 = (分配的页面数-1) * 页面大小 + 最后一页的有效长度
  // 特殊情况：如果没有分配页面（indptr相等），则返回0
  // 该函数在主机和设备端均可调用，用于确定序列边界
  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  /*!
   * \brief Compute the offset of element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  // 计算元素在已分配缓冲区中的偏移量（线性地址）
  // 根据页面索引、头索引、条目索引和特征索引计算内存偏移
  // 计算公式：page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx
  // 这个偏移量用于直接访问KV缓存中的特定元素
  // 该函数在主机和设备端均可调用，是核心的内存访问函数
  __host__ __device__ __forceinline__ size_t get_elem_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx,
                                                             size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  /*!
   * \brief Compute the offset of element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  // 计算元素在页面内的偏移量（相对于页面起始位置）
  // 与get_elem_offset类似，但不包含页面偏移（page_idx * stride_page）
  // 适用于已知页面起始地址的情况，减少重复计算
  // 计算公式：head_idx * stride_h + entry_idx * stride_n + feat_idx
  // 用于页面内的局部访问，提高计算效率
  __host__ __device__ __forceinline__ size_t get_elem_offset_in_page(size_t head_idx,
                                                                     size_t entry_idx,
                                                                     size_t feat_idx) const {
    return head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  // 获取键指针（设备端函数，通过页迭代器、头索引、条目索引、特征索引）
  // 根据页迭代器（逻辑页面索引）获取实际的键数据指针
  // 使用__ldg()函数从全局内存安全加载页面索引，提高缓存效率
  // 该函数只在设备端调用，用于CUDA内核中的键数据访问
  // 参数page_iter是逻辑页面索引，通过indices数组映射到物理页面
  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return k_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  // 保护性获取键值偏移量（防止越界访问）
  // 与get_elem_offset类似，但添加边界检查，防止访问未分配的页面
  // 参数last_indptr表示有效的页面索引上限，page_iter必须小于此值
  // 如果page_iter >= last_indptr，返回0偏移（安全值）
  // 这种保护性访问在CUDA内核中很重要，可以避免非法内存访问
  __device__ __forceinline__ size_t protective_get_kv_offset(IdType page_iter, uint32_t head_idx,
                                                             uint32_t entry_idx, uint32_t feat_idx,
                                                             IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  // 保护性获取键指针（防止越界访问）
  // 基于protective_get_kv_offset计算保护性键指针
  // 如果page_iter超出有效范围，返回k_data + 0（即k_data本身）
  // 这种设计确保即使索引越界，也不会访问非法内存地址
  // 在并行处理变长序列时特别有用，不同线程可能访问不同数量的页面
  __device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return k_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }

  // 获取值指针（设备端函数，通过页迭代器、头索引、条目索引、特征索引）
  // 与get_k_ptr对称，用于访问值（Value）缓存数据
  // 使用相同的页面索引映射机制，确保键值对的一致性
  // 在注意力计算中，键和值通常需要配对访问，因此这两个函数一起使用
  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return v_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  // 保护性获取值指针（防止越界访问）
  // 与protective_get_k_ptr对称，提供值缓存的安全访问
  // 使用相同的边界检查逻辑，确保键值访问的一致性
  // 在保护性访问模式下，键和值的偏移计算使用相同的保护逻辑
  __device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return v_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }
};

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
   */
// 解码阶段追加新键值到分页键值缓存的 CUDA 内核
// 该内核用于Transformer解码（生成）阶段，将新生成的token的键值追加到缓存
// 每个CUDA块处理一个批次元素，块内线程处理不同的头和特征维度
// 使用向量化内存访问（vec_size）提高内存带宽利用率
// 内核假设每个批次只追加一个token（解码阶段特征）
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  // 线程索引：tx处理特征维度（向量化），ty处理注意力头
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  // 缓存参数
  uint32_t num_heads = paged_kv.num_heads;
  // 每个CUDA块处理一个批次元素
  uint32_t batch_idx = blockIdx.x;
  // 线程的y维度对应注意力头索引
  uint32_t head_idx = ty;

  // 计算当前批次的序列长度（已缓存的token数量）
  // 公式：(分配的页面数-1) * 页面大小 + 最后一页有效长度
  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];

  // 计算要追加的页面索引（逻辑页面索引）
  // 新token将追加到序列末尾对应的页面
  uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
  // 计算在页面内的条目索引（页面内位置）
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  // 获取键值缓存中的目标指针（要写入的位置）
  DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  
  // 向量化内存拷贝：从输入key拷贝到键缓存
  // 输入key的布局：[batch_size, num_heads, head_dim]
  vec_t<DType, vec_size>::memcpy(
      k_ptr, key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  // 向量化内存拷贝：从输入value拷贝到值缓存
  // 输入value的布局与key相同
  vec_t<DType, vec_size>::memcpy(
      v_ptr, value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param batch_indices The batch indices of elements to be appended
 * \param positions The positions of elements to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheKernel(paged_kv_t<DType, IdType> paged_kv,
                                         DType* __restrict__ append_key,
                                         DType* __restrict__ append_value,
                                         IdType* __restrict__ batch_indices,
                                         IdType* __restrict__ positions, uint32_t nnz,
                                         size_t append_k_stride_n, size_t append_k_stride_h,
                                         size_t append_v_stride_n, size_t append_v_stride_h) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t head_idx = ty;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(paged_kv.indptr[batch_indices[i]] * paged_kv.page_size + positions[i],
                              page_iter, entry_idx);
    DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        k_ptr, append_key + i * append_k_stride_n + head_idx * append_k_stride_h + tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        v_ptr, append_value + i * append_v_stride_n + head_idx * append_v_stride_h + tx * vec_size);
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
   * \return status Indicates whether CUDA calls are successful
   */
// 解码阶段追加键值缓存的主机端接口函数
// 该函数封装了CUDA内核启动逻辑，根据头维度分派不同的内核实例
// 自动计算向量化大小和线程块配置，优化内存访问模式
// 适用于Transformer解码阶段，批量追加新生成的token键值
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                     cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
   DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
     // 计算向量化大小：至少16字节或头维度的1/32，取较大值
     // 向量化提高内存吞吐量，但受内存对齐和硬件限制
     constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
     // 线程块x维度：处理特征维度，每个线程处理vec_size个元素
     uint32_t bdx = HEAD_DIM / vec_size;
     // 线程块y维度：处理注意力头，每个线程处理一个头
     uint32_t bdy = num_heads;
     // NOTE(Zihao): could be slow for small batch size, will optimize later
     // 网格配置：每个批次元素一个CUDA块
     dim3 nblks(batch_size);
     // 线程块配置：x维度处理特征，y维度处理头
     dim3 nthrs(bdx, bdy);
     // 获取对应头维度和向量大小的内核函数实例
     auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Append new keys/values to the paged key-value cache
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(paged_kv_t<DType, IdType> paged_kv, DType* append_key,
                               DType* append_value, IdType* batch_indices, IdType* positions,
                               uint32_t nnz, size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
                               cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t num_heads = paged_kv.num_heads;
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = AppendPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(int(nnz), num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    void* args[] = {(void*)&paged_kv,          (void*)&append_key,        (void*)&append_value,
                    (void*)&batch_indices,     (void*)&positions,         (void*)&nnz,
                    (void*)&append_k_stride_n, (void*)&append_k_stride_h, (void*)&append_v_stride_n,
                    (void*)&append_v_stride_h};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

// 分页键值缓存（MLA版本）数据结构
// MLA（Mixed Attention）混合注意力机制的特殊缓存结构
// 与标准paged_kv_t不同，MLA缓存存储压缩的键值对和键位置编码
// 用于优化长序列注意力计算，减少内存占用和计算复杂度
template <typename DType, typename IdType>
struct paged_kv_mla_t {
  // 分页键值缓存（MLA 版本，用于混合注意力）
  // 页大小（快速除法对象）
  uint_fastdiv page_size;
  // 压缩键值头维度
  uint32_t head_dim_ckv;
  // 键位置编码头维度
  uint32_t head_dim_kpe;
  // 批次大小
  uint32_t batch_size;
  // 压缩键值页间步长
  uint32_t stride_page_ckv;
  // 键位置编码页间步长
  uint32_t stride_page_kpe;
  // 压缩键值条目内步长
  uint32_t stride_n_ckv;
  // 键位置编码条目内步长
  uint32_t stride_n_kpe;

  // Internal layout:
  // [max_num_pages, page_size, head_dim]
  // 压缩键值缓存数据指针
  DType* ckv_data;
  // 键位置编码缓存数据指针
  DType* kpe_data;
  // 页索引数组指针
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  // 页索引指针数组
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  // 每个请求最后一页的偏移量
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  // 每个请求的起始位置（RoPE 偏移）
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  // 构造空的 MLA 分页键值缓存
  __host__ __device__ __forceinline__ paged_kv_mla_t()
      : head_dim_ckv(0),
        head_dim_kpe(0),
        batch_size(0),
        stride_page_ckv(0),
        stride_page_kpe(0),
        stride_n_ckv(0),
        stride_n_kpe(0),
        ckv_data(nullptr),
        kpe_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged mla kv cache
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_mla_t(uint32_t page_size, uint32_t head_dim_compressed_kv,
                                          uint32_t head_dim_kpe, uint32_t batch_size,
                                          DType* compressed_kv_data, DType* kpe_data,
                                          IdType* indices, IdType* indptr, IdType* last_page_len,
                                          IdType* rope_pos_offset = nullptr)
      : page_size(page_size),
        head_dim_ckv(head_dim_compressed_kv),
        head_dim_kpe(head_dim_kpe),
        batch_size(batch_size),
        ckv_data(compressed_kv_data),
        kpe_data(kpe_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page_ckv = page_size * head_dim_ckv;
    stride_n_ckv = head_dim_ckv;
    stride_page_kpe = page_size * head_dim_kpe;
    stride_n_kpe = head_dim_kpe;
  }

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param compressed_kv_strides custom strides of each dimensions of compressed-kv cache
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param kpe_strides custom strides of each dimensions of k-pe cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_mla_t(uint32_t page_size, uint32_t head_dim_compressed_kv,
                                          uint32_t head_dim_kpe, uint32_t batch_size,
                                          DType* compressed_kv_data,
                                          const int64_t* compressed_kv_strides, DType* kpe_data,
                                          const int64_t* kpe_strides, IdType* indices,
                                          IdType* indptr, IdType* last_page_len,
                                          IdType* rope_pos_offset = nullptr)
      : page_size(page_size),
        head_dim_ckv(head_dim_compressed_kv),
        head_dim_kpe(head_dim_kpe),
        batch_size(batch_size),
        ckv_data(compressed_kv_data),
        kpe_data(kpe_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page_ckv = compressed_kv_strides[0];
    stride_n_ckv = compressed_kv_strides[1];
    stride_page_kpe = kpe_strides[0];
    stride_n_kpe = kpe_strides[1];
  }

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_ckv(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_ckv + entry_idx * stride_n_ckv + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_ckv(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_ckv(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_kpe(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_kpe + entry_idx * stride_n_kpe + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_kpe(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_kpe(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __device__ __forceinline__ DType* get_ckv_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return ckv_data + get_elem_offset_ckv(__ldg(indices + page_idx), entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* get_kpe_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return kpe_data + get_elem_offset_kpe(__ldg(indices + page_idx), entry_idx, feat_idx);
  }
};

template <uint32_t head_dim_ckv, uint32_t head_dim_kpe, uint32_t vec_size, typename DType,
          typename IdType>
__global__ void AppendPagedKVMlaCacheKernel(paged_kv_mla_t<DType, IdType> paged_kv_mla,
                                            DType* __restrict__ append_ckv,
                                            DType* __restrict__ append_kpe,
                                            IdType* __restrict__ batch_indices,
                                            IdType* __restrict__ positions, uint32_t nnz,
                                            size_t append_ckv_stride_n,
                                            size_t append_kpe_stride_n) {
  uint32_t tx = threadIdx.x;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv_mla.page_size.divmod(
        paged_kv_mla.indptr[batch_indices[i]] * paged_kv_mla.page_size + positions[i], page_iter,
        entry_idx);
    DType* ckv_ptr = paged_kv_mla.get_ckv_ptr(page_iter, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(ckv_ptr, append_ckv + i * append_ckv_stride_n + tx * vec_size);

    if (tx * vec_size < head_dim_kpe) {
      DType* kpe_ptr = paged_kv_mla.get_kpe_ptr(page_iter, entry_idx, tx * vec_size);
      vec_t<DType, vec_size>::memcpy(kpe_ptr, append_kpe + i * append_kpe_stride_n + tx * vec_size);
    }
  }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVMlaCache(paged_kv_mla_t<DType, IdType> paged_kv, DType* append_ckv,
                                  DType* append_kpe, IdType* batch_indices, IdType* positions,
                                  uint32_t nnz, size_t append_ckv_stride_n,
                                  size_t append_kpe_stride_n, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  uint32_t head_dim_ckv = paged_kv.head_dim_ckv;
  uint32_t head_dim_kpe = paged_kv.head_dim_kpe;
  constexpr uint32_t HEAD_CKV_DIM = 512;
  constexpr uint32_t HEAD_KPE_DIM = 64;
  FLASHINFER_CHECK(head_dim_ckv == HEAD_CKV_DIM, "head_dim_ckv must be equal to 512");
  FLASHINFER_CHECK(head_dim_kpe == HEAD_KPE_DIM, "head_dim_kpe must be equal to 64");
  constexpr uint32_t vec_size = 2;

  uint32_t bdx = HEAD_CKV_DIM / vec_size;
  uint32_t num_threads = bdx;
  uint32_t smem_size = 0;
  auto kernel = AppendPagedKVMlaCacheKernel<HEAD_CKV_DIM, HEAD_KPE_DIM, vec_size, DType, IdType>;
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                     num_threads, smem_size));
  num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(int(nnz), num_sms));
  dim3 nblks(num_blocks_per_sm * num_sms);
  dim3 nthrs(bdx);
  void* args[] = {(void*)&paged_kv,
                  (void*)&append_ckv,
                  (void*)&append_kpe,
                  (void*)&batch_indices,
                  (void*)&positions,
                  (void*)&nnz,
                  (void*)&append_ckv_stride_n,
                  (void*)&append_kpe_stride_n};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_

// 文件结束：FlashInfer 分页键值缓存系统
// 本文件实现了高效的分页键值缓存管理，支持Transformer推理优化
// 主要特性：
// 1. 分页内存管理，减少内存碎片，支持动态序列增长
// 2. 两种内存布局（HND/NHD）支持，适应不同计算模式
// 3. 向量化内存访问，优化GPU内存带宽利用率
// 4. 保护性访问机制，防止越界内存访问
// 5. MLA（混合注意力）支持，优化长序列处理
// 6. 批处理变长序列，提高GPU利用率
