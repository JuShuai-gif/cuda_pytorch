/*

  Note: This code was modified from example code
  originally provided by Intel.  To comply with Intel's open source
  licensing agreement, their copyright is retained below.

  -----------------------------------------------------------------

  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


//
// mandel --
//
// 对给定的复数 c = (c_re, c_im) 进行 Mandelbrot 迭代计算。
// 返回逃逸所需的迭代次数（若始终未逃逸则返回 count）。
//
// 迭代公式：z_{n+1} = z_n^2 + c，其中 z_0 = c
// 逃逸条件：|z_n|^2 > 4
//
static inline int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im; // z_0 = c
    int i;
    for (i = 0; i < count; ++i) {

        // 若 |z|^2 > 4，则 z 逃逸出 Mandelbrot 集，终止迭代
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        // 计算 z^2：实部 = re^2 - im^2，虚部 = 2 * re * im
        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        // z_{n+1} = z_n^2 + c
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i; // 返回实际迭代次数
}

//
// MandelbrotSerial --
//
// 计算一幅可视化 Mandelbrot 集的图像。
// 输出数组中每个元素的值表示该像素对应的复数逃逸出集合
// 所需的迭代次数（若迭代 maxIterations 次后仍未逃逸，则值为 maxIterations）。
//
// 参数说明：
// * x0, y0, x1, y1 — 复数平面中映射到图像视口的坐标范围
// * width, height — 输出图像的尺寸（宽度和高度）
// * startRow, totalRows — 指定计算图像的行范围（从 startRow 开始共 totalRows 行）
// * maxIterations — 最大迭代次数
// * output[] — 输出数组，长度为 width * height
//
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    // 计算每个像素在复数平面上对应的步长
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    // 计算需要处理的行范围
    int endRow = startRow + totalRows;

    // 遍历指定范围内的每一行
    for (int j = startRow; j < endRow; j++) {
        // 遍历该行的每一列
        for (int i = 0; i < width; ++i) {
            // 将像素坐标 (i, j) 映射到复数平面上的 (x, y)
            float x = x0 + i * dx;
            float y = y0 + j * dy;

            // 计算输出数组中的线性索引
            int index = (j * width + i);
            // 对复数 (x, y) 进行 Mandelbrot 迭代，结果存入输出数组
            output[index] = mandel(x, y, maxIterations);
        }
    }
}

