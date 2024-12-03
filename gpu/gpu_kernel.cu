#include "gpu_kernel.cuh"

using namespace std;

static __global__ void ComplexMultiply_batch_kernel(Complex *a, Complex *b, Complex *block, int N, int count);
static __global__ void ComplexMultiplyDelay_batch_kernel(Complex *a, Complex *b, Complex *block, int *delay, int index, int delaycount, int N, int count);
static __global__ void DiscardSamples_batch_kernel(Complex *a, Complex *b, int M, int count);
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size)
        return;
    float x, y;
    x = src[2 * i];
    y = src[2 * i + 1];
    dst[i].x = (x == 0) ? 0 : static_cast<float>(x) - 32768.0f;
    dst[i].y = (y == 0) ? 0 : static_cast<float>(y) - 32768.0f;
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
    dst[2 * i] = (src[i].x == 0) ? 0 : static_cast<uint16_t>(src[i].x + 32768.0f);
    dst[2 * i + 1] = (src[i].y == 0) ? 0 : static_cast<uint16_t>(src[i].y + 32768.0f);
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