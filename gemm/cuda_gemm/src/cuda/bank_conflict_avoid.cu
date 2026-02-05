
// 计算行列索引对应的内存偏移量，假设矩阵是按行主序存储的
// ld 是矩阵的宽度（也就是每一行的元素数量）
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 从指针加载 float4 类型的数据
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <
    const int BLOCK_SIZE_M,  // 每个线程块计算的 C 矩阵的宽度
    const int BLOCK_SIZE_K,  // 每个线程块加载到共享内存的 A 矩阵的高度
    const int BLOCK_SIZE_N,  // 每个线程块计算的 C 矩阵的高度
    const int THREAD_SIZE_Y, // 每个线程计算 C 的小块的高度
    const int THREAD_SIZE_X,  // 每个线程计算 C 的小块的宽度
    const bool ENABLE_DOUBLE_BUFFER // 是否使用double buffering
    > 
__global__ void BankConflictAvoidMatMul( 
    float * __restrict__ A,// 输入矩阵 A M * K
    float * __restrict__ B,// 输入矩阵 B N * K
    float * __restrict__ C, // 输出矩阵 C  M * N
    const int K, // 矩阵 A 和 B 的内积维度
    const int N) // 矩阵 C 的列数
{
    // 块的索引：获取当前线程块的 x 和 y 坐标
    int bx = blockIdx.x;    // 当前线程块在 x 方向的索引
    int by = blockIdx.y;    // 当前线程块在 y 方向的索引

    // 线程索引：获取当前线程在块内的 x 和 y 坐标
    int tx = threadIdx.x;   // 当前线程在 x 方向的索引
    int ty = threadIdx.y;   // 当前线程在 y 方向的索引
    
    // size of thread block
    // size of thread block：线程块的大小（每个线程块的宽度和高度）
    // 横向每个线程块的大小（块内的线程数）
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    // 纵向每个线程块的大小（块内的线程数）
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;

    // 总共多少个线程块（每个线程块的线程数）
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    // 当前线程在整个线程块中的唯一 ID（跨越 x 和 y 方向）
    const int tid = ty * bszx + tx;

    // shared memory
    // __shared__ float As[BLOCK_SIZE_M * 2][BLOCK_SIZE_K];
    // __shared__ float Bs[BLOCK_SIZE_K * 2][BLOCK_SIZE_N];

    // 声明共享内存：为 A 和 B 矩阵分配共享内存，避免 bank 冲突
    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K + 1]; // avoid bank conflict
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    
    // 声明寄存器：用于存储部分计算结果和矩阵的切片
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    // 存储 A 矩阵的切片
    float frag_a[THREAD_SIZE_Y];
    // 存储 B 矩阵的切片
    float frag_b[THREAD_SIZE_X];
    
    // 计算每个线程加载一个 tile 行需要多少个线程（每个线程加载 4 个浮点数）
    // 每个线程加载 A 矩阵一行的数量
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    // 每个线程加载 B 矩阵一行的数量
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    // 计算当前线程负责加载的 A 和 B 的行列起始位置
    // A 矩阵的起始行
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    // B 矩阵的起始行
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    // A 矩阵的列位置
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
    // B 矩阵的列位置
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;
    
    // 计算当前线程跨多个行时的步长
    // A 矩阵的行步长
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    // B 矩阵的行步长
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;
    
    // 计算当前线程加载 A 矩阵的偏移量
    const int idx = A_TILE_ROW_START * BLOCK_SIZE_K + A_TILE_COL;
    // 计算当前线程加载 C 矩阵的偏移量
    const int idx_ = ty * THREAD_SIZE_Y * BLOCK_SIZE_K;
    
    // 处理矩阵乘法时，分块的大小为 BLOCK_SIZE_K，循环加载 A 和 B
    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {
        // load A from global memory to shared memory
        // 将 A 的一块加载到共享内存中
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            // 加载 A 矩阵的数据到共享内存中（避免银行冲突）
            // 计算 A 矩阵中的行列位置并加载到 As[共享内存]
            // 计算行
            int r = (idx + i * BLOCK_SIZE_K) / (BLOCK_SIZE_K + 1);
            // 计算列
            int c = (idx + i * BLOCK_SIZE_K) % (BLOCK_SIZE_K + 1);
            FETCH_FLOAT4(As[r][c]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // A矩阵的行
                    A_TILE_COL + tile_idx, // A矩阵的列
                    K )]);// A 矩阵的列数
        }

        // load B from global memory to shared memory
        // 将 B 的一块加载到共享内存中
        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            // 加载 B 矩阵的数据到共享内存
            FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // B 矩阵的行
                    B_TILE_COL + BLOCK_SIZE_N * bx, // B 矩阵的列
                    N )]);// B 矩阵的列数
        }
        // 同步线程，确保所有线程都完成数据加载
        __syncthreads();

        // 将共享内存加载到寄存器中
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            // load A from shared memory to register
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                // 加载 A 矩阵的共享内存数据到寄存器 frag_a
                int r = (idx_ + thread_y * BLOCK_SIZE_K + k) / (BLOCK_SIZE_K + 1);
                int c = (idx_ + thread_y * BLOCK_SIZE_K + k) % (BLOCK_SIZE_K + 1);
                frag_a[thread_y] = As[r][c];
            }

            // load B from shared memory to register
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                // 加载 B 矩阵的共享内存数据到寄存器 frag_b
                FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
            }
            
            // 计算 C 矩阵的部分结果并累加到 accum 中
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
                }
            }
            
        }
        __syncthreads();
    }

    // 将计算结果存回 C 矩阵
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            // 将计算的 C 的部分结果存回全局内存
            C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,// C 矩阵的行
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,// C 矩阵的列
                N)] = accum[thread_y][thread_x];
        }
    }
}