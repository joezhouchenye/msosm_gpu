#include "msosm_gpu_concurrent.h"

MSOSM_GPU_DM_concurrent::MSOSM_GPU_DM_concurrent(float *bw, float *dm, float *f0, int numDMs) : Prepare_MSOSM_DM_concurrent(bw, dm, f0, numDMs)
{
    CUDA_CHECK(cudaEventCreateWithFlags(&dm_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&fft_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&output_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreate(&dm_stream));
    CUDA_CHECK(cudaStreamCreate(&fft_stream));
    CUDA_CHECK(cudaStreamCreate(&output_stream));
    delay_block_size = new int[numDMs];

    wait_for_cpu = false;
}

void MSOSM_GPU_DM_concurrent::get_device_info()
{
    GPU_GetDevInfo();
}

void MSOSM_GPU_DM_concurrent::initialize_uint16(unsigned long fftpoint, int count, unsigned long input_size)
{
    if (verbose)
    {
        get_device_info();
    }
    calculate_min_order();
    if (fftpoint != 0)
    {
        M_common = fftpoint / 2;
        if (M_common < M_common_min)
        {
            cout << "Warning: The filter order is too small for the given FFT point. The minimum FFT point " << 2 * M_common_min << " is used." << endl;
            M_common = M_common_min;
        }
    }
    fftpoint = 2 * M_common;
    this->fftpoint = fftpoint;
    if (input_size != 0)
        count = input_size / M_common;
    this->count = count;
    if (verbose)
    {
        cout << "FFTPoint: " << fftpoint << endl;
        cout << "Count: " << count << endl;
        cout << "M_common: " << M_common << endl;
    }
    // Create cuFTT plan
    int n[1] = {fftpoint};                    // 1D FFT Size
    int inembed[] = {(count + 1) * M_common}; // Input Size
    int onembed[] = {fftpoint};               // Output Size
    int istride = 1;                          // Input Stride
    int ostride = 1;                          // Output Stride
    int idist = M_common;                     // Input distance between consecutive FFT batches
    int odist = fftpoint;                     // Output distance between consecutive FFT batches
    CUFFT_CHECK(cufftPlanMany(&p_f, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, count));
    cufftSetStream(p_f, fft_stream);

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_d, (count + 1) * M_common * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(input_buffer_d, 0, (count + 1) * M_common * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_uint16_d, count * M_common * sizeof(uint16_pair)));

    CUDA_CHECK(cudaMalloc((void **)&fft_block_d, count * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&dedisp_params_d, numDMs * fftpoint * sizeof(Complex)));
    CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, count * numDMs));
    // CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, count));
    cufftSetStream(p_b, dm_stream);
    CUDA_CHECK(cudaMalloc((void **)&ifft_block_d, numDMs * count * fftpoint * sizeof(Complex)));
    CUDA_CHECK(cudaMalloc((void **)&delay_points_d, numDMs * fftpoint * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&output_buffer_d, (numDMs * count * M_common + 1) * sizeof(Complex)));
    cudaMemset(output_buffer_d + numDMs * count * M_common, 0, sizeof(Complex));

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_uint16_d, (numDMs * count * M_common + 1) * sizeof(uint16_pair)));
    cudaMemset(output_buffer_uint16_d + numDMs * count * M_common, 0, sizeof(uint16_pair));

    max_delay_block_size = 0;
    for (int i = 0; i < numDMs; i++)
    {
        segmentation(i);
        generate_dedisp_params(i);

        CUDA_CHECK(cudaMemcpy(dedisp_params_d + i * fftpoint, dedisp_params[i], fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(delay_points_d + i * fftpoint, delay_points[i], fftpoint * sizeof(int), cudaMemcpyHostToDevice));

        delay_block_size[i] = delaycount[i] + count - 1;
        if (delay_block_size[i] % count != 0)
        {
            delay_block_size[i] += count - delay_block_size[i] % count;
        }
        if (delay_block_size[i] > max_delay_block_size)
        {
            max_delay_block_size = delay_block_size[i];
        }
        if (verbose)
        {
            cout << "delay block size " << i << ": " << delay_block_size[i] << endl;
        }
    }
    CUDA_CHECK(cudaMalloc((void **)&delay_block_d, numDMs * max_delay_block_size * fftpoint * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(delay_block_d, 0, numDMs * max_delay_block_size * fftpoint * sizeof(Complex)));
    if (verbose)
        line();
}

void MSOSM_GPU_DM_concurrent::filter_block_uint16(uint16_pair *input)
{
    cudaStreamWaitEvent(fft_stream, dm_event, 0);
    // 拷贝新的count段输入数据到输入缓冲区
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_uint16_d, input, count * M_common * sizeof(uint16_pair), cudaMemcpyHostToDevice, fft_stream));
    // 拷贝重合部分到输入缓冲区
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_d, input_buffer_d + count * M_common, M_common * sizeof(Complex), cudaMemcpyDeviceToDevice, fft_stream));
    // 转换为复数float
    uint16ToComplex(input_buffer_d + M_common, (uint16_t *)(input_buffer_uint16_d), count * M_common, fft_stream);
    // count个重合点数为M的FFT同时计算
    CUFFT_CHECK(cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD));

    cudaEventRecord(fft_event, fft_stream);

    cudaStreamWaitEvent(dm_stream, fft_event, 0);
    // 将FFT结果拷贝到delay block对应位置
    current_block_d = delay_block_d + current_index * fftpoint * numDMs;
    ComplexMultiplyDelay_concurrent_batch(fft_block_d, dedisp_params_d, fftpoint, delay_block_d, delay_points_d, max_delay_block_size, current_index, count, numDMs, dm_stream);
    cudaEventRecord(dm_event, dm_stream);

    // 同时计算count个IFFT，这里IFFT必须是整块的，要求delay block长度是count的整数倍
    CUFFT_CHECK(cufftExecC2C(p_b, current_block_d, ifft_block_d, CUFFT_INVERSE));
    // 增加当前位置的偏移量，每次增加count个位置
    current_index += count;
    if (current_index >= max_delay_block_size)
    {
        current_index -= max_delay_block_size;
    }

    // 每个IFFT结果都需要丢弃前M个采样点，这里使用核函数来并行处理
    cudaStreamWaitEvent(dm_stream, output_event, 0);
    DiscardSamples_concurrent_batch(ifft_block_d, output_buffer_d, M_common, count, numDMs, dm_stream);
}

void MSOSM_GPU_DM_concurrent::get_output(Complex *output)
{
    cudaEventRecord(ready_event, dm_stream);
    while (wait_for_cpu)
        ;
    wait_for_cpu = true;
    cudaStreamWaitEvent(output_stream, ready_event, 0);
    CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_d, (count * M_common * numDMs + 1) * sizeof(Complex), cudaMemcpyDeviceToHost, output_stream));
    cudaEventRecord(output_event, output_stream);
}

void MSOSM_GPU_DM_concurrent::get_output(uint16_pair *output)
{
    complexToUint16((uint16_t *)output_buffer_uint16_d, output_buffer_d, count * M_common * numDMs, dm_stream);
    cudaEventRecord(ready_event, dm_stream);
    while (wait_for_cpu)
        ;
    wait_for_cpu = true;
    cudaStreamWaitEvent(output_stream, ready_event, 0);
    CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_uint16_d, (count * M_common * numDMs + 1) * sizeof(uint16_pair), cudaMemcpyDeviceToHost, output_stream));
    cudaEventRecord(output_event, output_stream);
}

void MSOSM_GPU_DM_concurrent::synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

void MSOSM_GPU_DM_concurrent::reset_device()
{
    CUDA_CHECK(cudaDeviceReset());
}

MSOSM_GPU_DM_concurrent::~MSOSM_GPU_DM_concurrent()
{
    CUDA_CHECK(cudaDeviceReset());
}

int MSOSM_GPU_DM_concurrent::next_power_of_2(int n)
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