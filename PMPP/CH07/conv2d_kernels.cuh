#pragma once
#ifndef CONV2D_KERNEL_H
#define CONV2D_KERNEL_H

#include <cuda_runtime.h>

#define FILTER_RADIUS 9
#define BLOCK_SIZE 32
#define TILE_SIZE 32

#define IN_TILE_SIZE 32
#define OUT_TILE_SIZE (IN_TILE_SIZE - 2 * FILTER_RADIUS)

// 使用宏定义，定义常量内存
#ifdef DEFINE_CONSTANT_MEMORY
__constant__ float constFilter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
#else
extern __constant__ float constFilter[(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1)];
#endif






#endif