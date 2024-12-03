#include "msosm_gpu_batch.h"

MSOSM_GPU_BATCH::MSOSM_GPU_BATCH(float bw, float dm, float f0) : Prepare_MSOSM(bw, dm, f0) {}

void MSOSM_GPU_BATCH::get_device_info()
{
    GPU_GetDevInfo();
}

void MSOSM_GPU_BATCH::initialize_uint16(unsigned long fftpoint, int count)
{
    if (verbose)
    {
        get_device_info();
    }
    calculate_min_order();
    if (fftpoint != 0)
    {
        M = fftpoint / 2;
        if (M < M_min)
        {
            cout << "Warning: The filter order is too small for the given FFT point. The minimum FFT point " << 2 * M_min << " is used." << endl;
            M = M_min;
        }
    }
    fftpoint = 2 * M;
    this->fftpoint = fftpoint;
    this->count = count;
    segmentation();
    generate_dedisp_params();

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_d, (count + 1) * M * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(input_buffer_d, 0, (count + 1) * M * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_int16_d, count * M * sizeof(uint16_pair)));

    CUDA_CHECK(cudaMalloc((void **)&dedisp_params_d, fftpoint * sizeof(Complex)));
    CUDA_CHECK(cudaMemcpy(dedisp_params_d, dedisp_params, fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));

    // Create cuFTT plan
    int n[1] = {fftpoint};             // 1D FFT Size
    int inembed[] = {(count + 1) * M}; // Input Size
    int onembed[] = {fftpoint};        // Output Size
    int istride = 1;                   // Input Stride
    int ostride = 1;                   // Output Stride
    int idist = M;                     // Input distance between consecutive FFT batches
    int odist = fftpoint;              // Output distance between consecutive FFT batches
    CUFFT_CHECK(cufftPlanMany(&p_f, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, count));
    CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, count));

    CUDA_CHECK(cudaMalloc((void **)&fft_block_d, count * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&ifft_block_d, count * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&delay_points_d, fftpoint * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(delay_points_d, delay_points, fftpoint * sizeof(int), cudaMemcpyHostToDevice));

    delay_block_size = delaycount + count - 1;
    if (delay_block_size % count != 0)
    {
        delay_block_size += count - delay_block_size % count;
    }
    if (verbose)
    {
        cout << "delay block size: " << delay_block_size << endl;
    }
    CUDA_CHECK(cudaMalloc((void **)&delay_block_d, delay_block_size * fftpoint * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(delay_block_d, 0, delay_block_size * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_d, count * M * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_int16_d, count * M * sizeof(uint16_pair)));
}

void MSOSM_GPU_BATCH::filter_block_uint16(uint16_pair *input)
{
    // 拷贝新的count段输入数据到输入缓冲区
    CUDA_CHECK(cudaMemcpy(input_buffer_int16_d, input, count * M * sizeof(uint16_pair), cudaMemcpyHostToDevice));
    // 拷贝重合部分到输入缓冲区
    CUDA_CHECK(cudaMemcpy(input_buffer_d, input_buffer_d + count * M, M * sizeof(Complex), cudaMemcpyDeviceToDevice));
    // 转换为复数float
    uint16ToComplex(input_buffer_d + M, (uint16_t *)input_buffer_int16_d, count * M);
    // count个重合点数为M的FFT同时计算
    CUFFT_CHECK(cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD));
    // 将FFT结果拷贝到delay block对应位置
    Complex *current_block = delay_block_d + current_index * fftpoint;
    ComplexMultiplyDelay_batch(fft_block_d, dedisp_params_d, fftpoint, delay_block_d, delay_points_d, delay_block_size,current_index, count);
    // 增加当前位置的偏移量，每次增加count个位置
    increment();
    // 同时计算count个IFFT，这里IFFT必须是整块的，要求delay block长度是count的整数倍
    CUFFT_CHECK(cufftExecC2C(p_b, current_block, ifft_block_d, CUFFT_INVERSE));
    // 每个IFFT结果都需要丢弃前M个采样点，这里使用核函数来并行处理
    DiscardSamples_batch(ifft_block_d, output_buffer_d, M, count);
}

void MSOSM_GPU_BATCH::get_output(Complex *output)
{
    CUDA_CHECK(cudaMemcpy(output, output_buffer_d, count * M * sizeof(Complex), cudaMemcpyDeviceToHost));
}

void MSOSM_GPU_BATCH::increment()
{
    current_index += count;
    if (current_index >= delay_block_size)
    {
        current_index -= delay_block_size;
    }
}

MSOSM_GPU_BATCH::~MSOSM_GPU_BATCH()
{
    cufftDestroy(p_f);
    cufftDestroy(p_b);
}

int MSOSM_GPU_BATCH::next_power_of_2(int n)
{
    if (n <= 0)
    {
        return 1;
    }
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}