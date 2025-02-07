#include "gpu_kernel.cuh"

using namespace std;

static __global__ void ComplexMultiply_batch_kernel(Complex *a, Complex *b, Complex *block, int N, int count);
static __global__ void ComplexMultiplyDelay_batch_kernel(Complex *a, Complex *b, Complex *block, int *delay, int index, int delaycount, int N, int count);
static __global__ void ComplexMultiplyDelay_concurrent_batch_kernel(Complex *a, Complex *b, Complex *block, int *delay, int index, int delay_block_size, int N, int count, int numDMs);
static __global__ void DiscardSamples_batch_kernel(Complex *a, Complex *b, int M, int count);
static __global__ void DiscardSamples_concurrent_batch_kernel(Complex *a, Complex *b, int M, int count, int numDMs);
static __global__ void uint16ToComplex_kernel(Complex *dst, uint16_t *src, size_t size);
static __global__ void complexToUint16_kernel(uint16_t *dst, Complex *src, size_t size);
static __global__ void calculateIntensity_kernel(Complex *a, Complex *b, float *total_intensity, unsigned long size);

constexpr int BLOCK_SIZE = 256;

void ComplexMultiply_batch(Complex *a, Complex *b, int N, Complex *block, int count)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N * count + block_size - 1) / block_size;
    ComplexMultiply_batch_kernel<<<grid_size, block_size>>>(a, b, block, N, count);
    CUDA_CHECK(cudaGetLastError());
}

void ComplexMultiply_batch(Complex *a, Complex *b, int N, Complex *block, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N * count + block_size - 1) / block_size;
    ComplexMultiply_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, N, count);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ComplexMultiply_batch_kernel(Complex *a, Complex *b, Complex *block, int N, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * count)
        return;
    float2 input1, input2;
    input1.x = a[i].x / N;
    input1.y = a[i].y / N;
    input2.x = b[i % N].x;
    input2.y = b[i % N].y;
    block[i] = cuCmulf(input1, input2);
    // block[i].x = a[i].x / N * b[i % N].x - a[i].y / N * b[i % N].y;
    // block[i].y = a[i].x / N * b[i % N].y + a[i].y / N * b[i % N].x;
}

void ComplexMultiplyDelay_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N * count + block_size - 1) / block_size;
    ComplexMultiplyDelay_batch_kernel<<<grid_size, block_size>>>(a, b, block, delay, index, delay_block_size, N, count);
    CUDA_CHECK(cudaGetLastError());
}

void ComplexMultiplyDelay_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (N * count + block_size - 1) / block_size;
    ComplexMultiplyDelay_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, delay, index, delay_block_size, N, count);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void ComplexMultiplyDelay_batch_kernel(Complex *a, Complex *b, Complex *block, int *delay, int index, int delay_block_size, int N, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * count)
        return;
    int j = (index + i / N + delay[i % N]) % delay_block_size;
    Complex *c = block + j * N;
    float2 input1, input2;
    input1.x = a[i].x / N;
    input1.y = a[i].y / N;
    input2.x = b[i % N].x;
    input2.y = b[i % N].y;
    c[i % N] = cuCmulf(input1, input2);
    // c[i % N].x = a[i].x / N * b[i % N].x - a[i].y / N * b[i % N].y;
    // c[i % N].y = a[i].x / N * b[i % N].y + a[i].y / N * b[i % N].x;
    // c[i] = cuCmulf(a[i], b[i % N]);
}

void ComplexMultiplyDelay_concurrent_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count, int numDMs, cudaStream_t stream)
{
    // const int block_size = BLOCK_SIZE;
    // const int grid_size = (N * count * numDMs + block_size - 1) / block_size;
    // ComplexMultiplyDelay_concurrent_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, delay, index, delay_block_size, N, count, numDMs);
    // // // CUDA_CHECK(cudaGetLastError());

    // const int block_dim_x = BLOCK_SIZE;
    // // const int block_dim_x = 32;
    // int block_dim_y = numDMs;
    // dim3 block_size(block_dim_x, block_dim_y);
    // int grid_size;
    // if (N * count % block_dim_x != 0)
    //     grid_size = (N * count * numDMs + block_dim_x * block_dim_y - 1) / block_dim_x / block_dim_y;
    // else
    //     grid_size = N * count * numDMs / block_dim_x / block_dim_y;
    // ComplexMultiplyDelay_concurrent_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, delay, index, delay_block_size, N, count, numDMs);
    // // CUDA_CHECK(cudaGetLastError());

    // 使用二维网格，x维度为numDMs，y维度为count
    dim3 grid_size(numDMs, count);
    // 块大小设为256，根据GPU架构调整
    const int block_size = 256;
    ComplexMultiplyDelay_concurrent_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, block, delay, index, delay_block_size, N, count, numDMs);
}

static __global__ void ComplexMultiplyDelay_concurrent_batch_kernel(Complex *a, Complex *b, Complex *block, int *delay, int index, int delay_block_size, int N, int count, int numDMs)
{
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= N * count * numDMs)
    //     return;
    // int j = (index + (i % (N * count)) / N + delay[i / (N * count) * N + i % N]) % delay_block_size;
    // Complex *c = block + j / count * count * numDMs * N + i / (N * count) * count * N + (j % count) * N;
    // Complex input1, input2;
    // input1.x = a[i % (N * count)].x / N;
    // input1.y = a[i % (N * count)].y / N;
    // input2 = b[i / (N * count) * N + i % N];
    // c[i % N] = cuCmulf(input1, input2);

    // __shared__ Complex shared_input[BLOCK_SIZE];
    // int i = gridDim.x * blockDim.x * threadIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    // int shared_i = threadIdx.x;
    // int fft_i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= N * count * numDMs)
    //     return;
    // if (threadIdx.y == 0)
    // {
    //     shared_input[shared_i].x = a[fft_i].x / N;
    //     shared_input[shared_i].y = a[fft_i].y / N;
    // }
    // __syncthreads();
    // int j = (index + (i % (N * count)) / N + delay[i / (N * count) * N + i % N]) % delay_block_size;
    // Complex *c = block + j / count * count * numDMs * N + i / (N * count) * count * N + (j % count) * N;
    // Complex input2;
    // input2 = b[i / (N * count) * N + i % N];
    // c[i % N] = cuCmulf(shared_input[shared_i], input2);

    // 转换为float2以向量化内存访问
    float2 *a_float2 = reinterpret_cast<float2*>(a);
    float2 *b_float2 = reinterpret_cast<float2*>(b);
    float2 *block_float2 = reinterpret_cast<float2*>(block);
    
    const int dm_idx = blockIdx.x;
    const int count_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    // 预计算基址和常数
    const float inv_N = 1.0f / N;
    const int a_base = count_idx * N; // 当前count的a数组起始位置
    const int b_base = dm_idx * N;    // 当前DM的b数组起始位置

    // 每个线程处理多个n_idx，步长为blockDim.x
    for (int n_idx = tid; n_idx < N; n_idx += blockDim.x) {
        // 直接从全局内存读取delay值，合并访问
        int delay_val = delay[dm_idx * N + n_idx];
        
        // 计算延迟调整后的索引j
        int j = (index + count_idx + delay_val) % delay_block_size;
        int j_div_count = j / count;
        int j_mod_count = j % count;
        
        // 计算目标内存位置
        int block_offset = (j_div_count * numDMs + dm_idx) * (count * N) + j_mod_count * N + n_idx;
        
        // 向量化读取输入数据
        float2 a_val = a_float2[a_base + n_idx];
        float2 b_val = b_float2[b_base + n_idx];
        
        // 执行复数乘法（带归一化）
        Complex result;
        result.x = (a_val.x * b_val.x - a_val.y * b_val.y) * inv_N;
        result.y = (a_val.x * b_val.y + a_val.y * b_val.x) * inv_N;
        
        // 向量化写入结果
        block_float2[block_offset] = make_float2(result.x, result.y);
    }
}

void DiscardSamples_batch(Complex *a, Complex *b, int M, int count)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (M * count + block_size - 1) / block_size;
    DiscardSamples_batch_kernel<<<grid_size, block_size>>>(a, b, M, count);
    CUDA_CHECK(cudaGetLastError());
}

void DiscardSamples_batch(Complex *a, Complex *b, int M, int count, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (M * count + block_size - 1) / block_size;
    DiscardSamples_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, M, count);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void DiscardSamples_batch_kernel(Complex *a, Complex *b, int M, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * count)
        return;
    unsigned long index = i / M * 2 * M + i % M + M;
    b[i].x = a[index].x / (2 * M);
    b[i].y = a[index].y / (2 * M);
    // b[i].x = a[i / M * 2 * M + i % M + M].x / (2 * M);
    // b[i].y = a[i / M * 2 * M + i % M + M].y / (2 * M);
}

void DiscardSamples_concurrent_batch(Complex *a, Complex *b, int M, int count, int numDMs, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (M * count * numDMs + block_size - 1) / block_size;
    DiscardSamples_concurrent_batch_kernel<<<grid_size, block_size, 0, stream>>>(a, b, M, count, numDMs);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void DiscardSamples_concurrent_batch_kernel(Complex *a, Complex *b, int M, int count, int numDMs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * count * numDMs)
        return;
    // unsigned long index = i / M * 2 * M + i % M + M;
    // b[i].x = a[index].x / (2 * M);
    // b[i].y = a[index].y / (2 * M);
    const int group = i / M;
    const int offset = i % M;
    const int index = group * 2 * M + offset + M;

    float2 *a_f2 = reinterpret_cast<float2 *>(a);
    float2 *b_f2 = reinterpret_cast<float2 *>(b);

    const float scale = 1.0f / (2 * M);

    float2 val = a_f2[index];
    val.x *= scale;
    val.y *= scale;

    b_f2[i] = val;
}

void uint16ToComplex(Complex *dst, uint16_t *src, size_t size)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    uint16ToComplex_kernel<<<grid_size, block_size>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

void uint16ToComplex(Complex *dst, uint16_t *src, size_t size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    uint16ToComplex_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void uint16ToComplex_kernel(Complex *dst, uint16_t *src, size_t size)
{
    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if (i >= size)
    //     return;
    // float x, y;
    // x = src[2 * i];
    // y = src[2 * i + 1];
    // dst[i].x = (x == 0) ? 0 : static_cast<float>(x) - 32768.0f;
    // dst[i].y = (y == 0) ? 0 : static_cast<float>(y) - 32768.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;

    uint32_t val = reinterpret_cast<uint32_t *>(src)[i];
    uint16_t y = val & 0xFFFF;
    uint16_t x = (val >> 16) & 0xFFFF;

    float fx = static_cast<float>(x) - 32768.0f * (x != 0);
    float fy = static_cast<float>(y) - 32768.0f * (y != 0);
    dst[i].x = fx;
    dst[i].y = fy;
}

void complexToUint16(uint16_t *dst, Complex *src, size_t size)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    complexToUint16_kernel<<<grid_size, block_size>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

void complexToUint16(uint16_t *dst, Complex *src, size_t size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    complexToUint16_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void complexToUint16_kernel(uint16_t *dst, Complex *src, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    float2 val = reinterpret_cast<float2 *>(src)[i];
    uint16_t y = static_cast<uint16_t>(val.y + 32768.0f * (val.y != 0));
    uint16_t x = static_cast<uint16_t>(val.x + 32768.0f * (val.x != 0));
    uint32_t val32 = (x << 16) | y;
    reinterpret_cast<uint32_t *>(dst)[i] = val32;
    // dst[2 * i] = (src[i].x == 0) ? 0 : static_cast<uint16_t>(src[i].x + 32768.0f);
    // dst[2 * i + 1] = (src[i].y == 0) ? 0 : static_cast<uint16_t>(src[i].y + 32768.0f);
}

void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    calculateIntensity_kernel<<<grid_size, block_size>>>(a, b, total_intensity, size);
    CUDA_CHECK(cudaGetLastError());
}

void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size, cudaStream_t stream)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    calculateIntensity_kernel<<<grid_size, block_size, 0, stream>>>(a, b, total_intensity, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void calculateIntensity_kernel(Complex *a, Complex *b, float *total_intensity, unsigned long size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    total_intensity[i] = sqrtf(a[i].x * a[i].x + a[i].y * a[i].y + b[i].x * b[i].x + b[i].y * b[i].y);
}

__global__ void initializeBoolArray_kernel(bool *array, int size, bool value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        array[idx] = value;
    }
}

void initializeBoolArray(bool *array, int size, bool value)
{
    const int block_size = BLOCK_SIZE;
    const int grid_size = (size + block_size - 1) / block_size;
    initializeBoolArray_kernel<<<grid_size, block_size>>>(array, size, value);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void waitForCPU_kernel(bool *flag)
{
    while (*flag)
        ;
    *flag = true;
}

void waitForCPU(bool *flag, cudaStream_t stream)
{
    waitForCPU_kernel<<<1, 1, 0, stream>>>(flag);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void unblockCPU_kernel(bool *flag)
{
    *flag = false;
}

void unblockCPU(bool *flag, cudaStream_t stream)
{
    unblockCPU_kernel<<<1, 1, 0, stream>>>(flag);
    CUDA_CHECK(cudaGetLastError());
}