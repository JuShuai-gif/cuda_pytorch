import numpy as np

# 原始 tile(方便观察，用数字填充)
tile = np.arange(64).reshape(8,8)
# 复原之后的
swizzle_tile = np.arange(64).reshape(8,8)


print("=== 原始 tile (row-major) ===")
print(tile)

# 标准的 32 * 32 块
def swizzle_32_32(y,x):
    return y,x ^ y

"""
矩阵有 64 行,将矩阵按纵向分割为 两个 32 * 32的子矩阵,分别做 swizzle,计算时对 y 取模即可
"""
def swizzle_64_32(y,x):
    COLS = 32
    return y,x ^ (y%COLS)

"""
若列数为 64,可将矩阵按横向分为两个 32 * 32子矩阵,对每个子矩阵分别做 swizzle
"""

def swizzle_32_64(y,x):
    COLS = 32
    x_base = x // COLS * COLS # 当前子矩阵的起始列
    x_offset = x % COLS
    return y,x_base + (x_offset ^ y)

"""
向量化访问 half
half nums[32][64]
"""
def swizzle_half_32_64(y,x):
    x_offset = x % 2 # 组内偏移
    x = x //2        # 组号
    return y,(x^y) * 2 + x_offset

def test_swizzle_32_32():
    # 构造 swizzle 后的 tile
    swizzled = np.zeros_like(tile)

    # swizzled 写入
    for y in range(8):
        for x in range(8):
            y_new,x_new = swizzle_32_32(y,x)
            swizzled[y_new][x_new] = tile[y][x]

    print("\n=== swizzle 后 tile (x ^ y) ===")
    print(swizzled)

    # 复原之后的访问
    for y in range(8):
        for x in range(8):
            y_new,x_new = swizzle_32_32(y,x)
            swizzle_tile[y_new][x_new] = swizzled[y][x]
    print("复原之后的") 

    print(swizzle_tile)



if __name__ == "__main__":
    print("=== 原始 tile (row-major) ===")
    print(tile)
    
    test_swizzle_32_32()
    








