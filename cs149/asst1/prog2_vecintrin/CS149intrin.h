// 定义向量单元宽度（SIMD 通道数），此处设为 4 个通道
// Define vector unit width here
#define VECTOR_WIDTH 4

#ifndef CS149INTRIN_H_
#define CS149INTRIN_H_

#include <cstdlib>
#include <cmath>
#include "logger.h"

//*******************
//* 类型定义        *
//* Type Definition *
//*******************

// 全局日志记录器，用于追踪向量指令执行情况
extern Logger CS149Logger;

// 向量寄存器模板：包含 VECTOR_WIDTH 个元素
// 可以理解为一条 SIMD 指令同时操作 VECTOR_WIDTH 个数
template <typename T>
struct __cs149_vec {
  T value[VECTOR_WIDTH];
};

// 向量掩码，继承自 __cs149_vec<bool>
// 用于标记向量寄存器中哪些通道参与运算（true 表示活跃/参与）
// Declare a mask with __cs149_mask
struct __cs149_mask : __cs149_vec<bool> {};

// 声明浮点向量寄存器类型
// Declare a floating point vector register with __cs149_vec_float
#define __cs149_vec_float __cs149_vec<float>

// 声明整数向量寄存器类型
// Declare an integer vector register with __cs149_vec_int
#define __cs149_vec_int   __cs149_vec<int>

//***********************
//* 函数定义            *
//* Function Definition *
//***********************

// 返回一个掩码，前 N 个通道设为 1（活跃），其余通道设为 0（不活跃）
// 默认 N = VECTOR_WIDTH，即全部通道活跃
// Return a mask initialized to 1 in the first N lanes and 0 in the others
__cs149_mask _cs149_init_ones(int first = VECTOR_WIDTH);

// 返回 maska 的按位取反结果（活跃 <-> 不活跃）
// Return the inverse of maska
__cs149_mask _cs149_mask_not(__cs149_mask &maska);

// 返回 (maska | maskb)，即两个掩码的按位或
// Return (maska | maskb)
__cs149_mask _cs149_mask_or(__cs149_mask &maska, __cs149_mask &maskb);

// 返回 (maska & maskb)，即两个掩码的按位与
// Return (maska & maskb)
__cs149_mask _cs149_mask_and(__cs149_mask &maska, __cs149_mask &maskb);

// 统计 maska 中活跃通道的数量（值为 1 的通道个数）
// Count the number of 1s in maska
int _cs149_cntbits(__cs149_mask &maska);

// 条件赋值：若对应通道活跃，则将 vecResult 该通道设为 value；否则保留原值
// 不带 mask 的重载版本：将所有通道都设为 value（方便用户使用）
// Set register to value if vector lane is active
//  otherwise keep the old value
void _cs149_vset_float(__cs149_vec_float &vecResult, float value, __cs149_mask &mask);
void _cs149_vset_int(__cs149_vec_int &vecResult, int value, __cs149_mask &mask);
// For user's convenience, returns a vector register with all lanes initialized to value
__cs149_vec_float _cs149_vset_float(float value);
__cs149_vec_int _cs149_vset_int(int value);

// 条件移动：若对应通道活跃，则将 src 的值复制到 dest；否则 dest 保留原值
// Copy values from vector register src to vector register dest if vector lane active
// otherwise keep the old value
void _cs149_vmove_float(__cs149_vec_float &dest, __cs149_vec_float &src, __cs149_mask &mask);
void _cs149_vmove_int(__cs149_vec_int &dest, __cs149_vec_int &src, __cs149_mask &mask);

// 条件加载：若对应通道活跃，从内存数组 src 加载数据到向量寄存器 dest
// Load values from array src to vector register dest if vector lane active
//  otherwise keep the old value
void _cs149_vload_float(__cs149_vec_float &dest, float* src, __cs149_mask &mask);
void _cs149_vload_int(__cs149_vec_int &dest, int* src, __cs149_mask &mask);

// 条件存储：若对应通道活跃，将向量寄存器 src 的值写入内存数组 dest
// Store values from vector register src to array dest if vector lane active
//  otherwise keep the old value
void _cs149_vstore_float(float* dest, __cs149_vec_float &src, __cs149_mask &mask);
void _cs149_vstore_int(int* dest, __cs149_vec_int &src, __cs149_mask &mask);

// 条件加法：若对应通道活跃，计算 veca + vecb 存入 vecResult
// Return calculation of (veca + vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vadd_float(__cs149_vec_float &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vadd_int(__cs149_vec_int &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 条件减法：若对应通道活跃，计算 veca - vecb 存入 vecResult
// Return calculation of (veca - vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vsub_float(__cs149_vec_float &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vsub_int(__cs149_vec_int &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 条件乘法：若对应通道活跃，计算 veca * vecb 存入 vecResult
// Return calculation of (veca * vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vmult_float(__cs149_vec_float &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vmult_int(__cs149_vec_int &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 条件除法：若对应通道活跃，计算 veca / vecb 存入 vecResult
// Return calculation of (veca / vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vdiv_float(__cs149_vec_float &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vdiv_int(__cs149_vec_int &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);


// 条件绝对值：若对应通道活跃，计算 abs(veca) 存入 vecResult
// Return calculation of absolute value abs(veca) if vector lane active
//  otherwise keep the old value
void _cs149_vabs_float(__cs149_vec_float &vecResult, __cs149_vec_float &veca, __cs149_mask &mask);
void _cs149_vabs_int(__cs149_vec_int &vecResult, __cs149_vec_int &veca, __cs149_mask &mask);

// 条件大于比较：若对应通道活跃，比较 veca > vecb，结果存入掩码 vecResult
// Return a mask of (veca > vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vgt_float(__cs149_mask &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vgt_int(__cs149_mask &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 条件小于比较：若对应通道活跃，比较 veca < vecb，结果存入掩码 vecResult
// Return a mask of (veca < vecb) if vector lane active
//  otherwise keep the old value
void _cs149_vlt_float(__cs149_mask &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_vlt_int(__cs149_mask &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 条件等于比较：若对应通道活跃，比较 veca == vecb，结果存入掩码 vecResult
// Return a mask of (veca == vecb) if vector lane active
//  otherwise keep the old value
void _cs149_veq_float(__cs149_mask &vecResult, __cs149_vec_float &veca, __cs149_vec_float &vecb, __cs149_mask &mask);
void _cs149_veq_int(__cs149_mask &vecResult, __cs149_vec_int &veca, __cs149_vec_int &vecb, __cs149_mask &mask);

// 水平加法（相邻元素两两相加）：
//   例如 [0 1 2 3] -> [0+1, 0+1, 2+3, 2+3]
// Adds up adjacent pairs of elements, so
//  [0 1 2 3] -> [0+1 0+1 2+3 2+3]
void _cs149_hadd_float(__cs149_vec_float &vecResult, __cs149_vec_float &vec);

// 奇偶交错排列：将所有偶数索引元素移到数组前半部分，奇数索引元素移到后半部分
//   例如 [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
// Performs an even-odd interleaving where all even-indexed elements move to front half
//  of the array and odd-indexed to the back half, so
//  [0 1 2 3 4 5 6 7] -> [0 2 4 6 1 3 5 7]
void _cs149_interleave_float(__cs149_vec_float &vecResult, __cs149_vec_float &vec);

// 向日志中添加自定义调试信息
// Add a customized log to help debugging
void addUserLog(const char * logStr);

#endif
