#include "msosm_gpu_batch_stream.h"

MSOSM_GPU_DM_BATCH_stream::MSOSM_GPU_DM_BATCH_stream(float *bw, float *dm, float *f0, int numStreams) : Prepare_MSOSM_DM_stream(bw, dm, f0, numStreams)
{
    current_index = new int[numStreams];
    memset(current_index, 0, numStreams * sizeof(int));
    stream_event = new cudaEvent_t[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaEventCreate(&(stream_event[i])));
    }
    CUDA_CHECK(cudaEventCreate(&fft_event));
    stream = new cudaStream_t[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&(stream[i])));
    }
    CUDA_CHECK(cudaStreamCreate(&fft_stream));
    delay_block_d = new Complex *[numStreams];
    ifft_block_d = new Complex *[numStreams];
    output_buffer_int16_d = new uint16_pair *[numStreams];
    output_buffer_d = new Complex *[numStreams];
    dedisp_params_d = new Complex *[numStreams];
    delay_points_d = new int *[numStreams];
    delay_block_size = new int[numStreams];
    p_b = new cufftHandle[numStreams];
    current_block = new Complex *[numStreams];

    wait_for_cpu = new bool[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        wait_for_cpu[i] = false;
    }
}

void MSOSM_GPU_DM_BATCH_stream::get_device_info()
{
    GPU_GetDevInfo();
}

void MSOSM_GPU_DM_BATCH_stream::initialize_uint16(unsigned long fftpoint, int count)
{
    if (verbose)
    {
        get_device_info();
    }
    calculate_min_order();
    if (fftpoint != 0)
    {
        M_stream = fftpoint / 2;
        if (M_stream < M_stream_min)
        {
            cout << "Warning: The filter order is too small for the given FFT point. The minimum FFT point " << 2 * M_stream_min << " is used." << endl;
            M_stream = M_stream_min;
        }
    }
    fftpoint = 2 * M_stream;
    this->fftpoint = fftpoint;
    this->count = count;
    // Create cuFTT plan
    int n[1] = {fftpoint};                    // 1D FFT Size
    int inembed[] = {(count + 1) * M_stream}; // Input Size
    int onembed[] = {fftpoint};               // Output Size
    int istride = 1;                          // Input Stride
    int ostride = 1;                          // Output Stride
    int idist = M_stream;                     // Input distance between consecutive FFT batches
    int odist = fftpoint;                     // Output distance between consecutive FFT batches
    CUFFT_CHECK(cufftPlanMany(&p_f, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, count));
    cufftSetStream(p_f, fft_stream);

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_d, (count + 1) * M_stream * sizeof(Complex)));
    CUDA_CHECK(cudaMemset(input_buffer_d, 0, (count + 1) * M_stream * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&input_buffer_int16_d, count * M_stream * sizeof(uint16_pair)));

    CUDA_CHECK(cudaMalloc((void **)&fft_block_d, count * fftpoint * sizeof(Complex)));

    for (int i = 0; i < numStreams; i++)
    {
        segmentation(i);
        generate_dedisp_params(i);

        CUDA_CHECK(cudaMalloc((void **)&(dedisp_params_d[i]), fftpoint * sizeof(Complex)));
        CUDA_CHECK(cudaMemcpy(dedisp_params_d[i], dedisp_params[i], fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));

        CUFFT_CHECK(cufftPlan1d(&(p_b[i]), fftpoint, CUFFT_C2C, count));
        cufftSetStream(p_b[i], stream[i]);

        CUDA_CHECK(cudaMalloc((void **)&(ifft_block_d[i]), count * fftpoint * sizeof(Complex)));

        CUDA_CHECK(cudaMalloc((void **)&(delay_points_d[i]), fftpoint * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(delay_points_d[i], delay_points[i], fftpoint * sizeof(int), cudaMemcpyHostToDevice));

        delay_block_size[i] = delaycount[i] + count - 1;
        if (delay_block_size[i] % count != 0)
        {
            delay_block_size[i] += count - delay_block_size[i] % count;
        }
        if (verbose)
        {
            cout << "delay block size " << i << ": " << delay_block_size[i] << endl;
        }
        CUDA_CHECK(cudaMalloc((void **)&(delay_block_d[i]), delay_block_size[i] * fftpoint * sizeof(Complex)));
        CUDA_CHECK(cudaMemset(delay_block_d[i], 0, delay_block_size[i] * fftpoint * sizeof(Complex)));

        CUDA_CHECK(cudaMalloc((void **)&(output_buffer_d[i]), (count * M_stream + 1) * sizeof(Complex)));
        cudaMemset(output_buffer_d[i] + count * M_stream, 0, sizeof(Complex));

        CUDA_CHECK(cudaMalloc((void **)&(output_buffer_int16_d[i]), count * M_stream * sizeof(uint16_pair)));
    }
    if (verbose)
        line();
}

void MSOSM_GPU_DM_BATCH_stream::filter_block_uint16(uint16_pair *input)
{
    for (int i = 0; i < numStreams; i++)
    {
        cudaStreamWaitEvent(fft_stream, stream_event[i], 0);
    }
    // 拷贝新的count段输入数据到输入缓冲区
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_int16_d, input, count * M_stream * sizeof(uint16_pair), cudaMemcpyHostToDevice, fft_stream));
    // 拷贝重合部分到输入缓冲区
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_d, input_buffer_d + count * M_stream, M_stream * sizeof(Complex), cudaMemcpyDeviceToDevice, fft_stream));
    // 转换为复数float
    uint16ToComplex(input_buffer_d + M_stream, (uint16_t *)(input_buffer_int16_d), count * M_stream, fft_stream);
    // count个重合点数为M的FFT同时计算
    CUFFT_CHECK(cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD));

    cudaEventRecord(fft_event, fft_stream);

    for (int i = 0; i < numStreams; i++)
    {
        cudaStreamWaitEvent(stream[i], fft_event, 0);
        // 将FFT结果拷贝到delay block对应位置
        current_block[i] = delay_block_d[i] + current_index[i] * fftpoint;
        ComplexMultiplyDelay_batch(fft_block_d, dedisp_params_d[i], fftpoint, delay_block_d[i], delay_points_d[i], delay_block_size[i], current_index[i], count, stream[i]);
        cudaEventRecord(stream_event[i], stream[i]);
    }
    for (int i = 0; i < numStreams; i++)
    {
        // 同时计算count个IFFT，这里IFFT必须是整块的，要求delay block长度是count的整数倍
        CUFFT_CHECK(cufftExecC2C(p_b[i], current_block[i], ifft_block_d[i], CUFFT_INVERSE));
        // 增加当前位置的偏移量，每次增加count个位置
        current_index[i] += count;
        if (current_index[i] >= delay_block_size[i])
        {
            current_index[i] -= delay_block_size[i];
        }
    }
    for (int i = 0; i < numStreams; i++)
    {
        // 每个IFFT结果都需要丢弃前M个采样点，这里使用核函数来并行处理
        DiscardSamples_batch(ifft_block_d[i], output_buffer_d[i], M_stream, count, stream[i]);
    }
}

void MSOSM_GPU_DM_BATCH_stream::get_output(Complex **output)
{
    for (int i = 0; i < numStreams; i++)
    {
        while (wait_for_cpu[i])
            ;
        wait_for_cpu[i] = true;
        CUDA_CHECK(cudaMemcpyAsync(output[i], output_buffer_d[i], (count * M_stream + 1) * sizeof(Complex), cudaMemcpyDeviceToHost, stream[i]));
    }
}

void MSOSM_GPU_DM_BATCH_stream::synchronize()
{
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamSynchronize(stream[i]));
    }
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(fft_stream));
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaEventDestroy(stream_event[i]));
    }
    CUDA_CHECK(cudaEventDestroy(fft_event));
    CUDA_CHECK(cudaFree(input_buffer_d));
    CUDA_CHECK(cudaFree(input_buffer_int16_d));
    CUDA_CHECK(cudaFree(fft_block_d));
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaFree(ifft_block_d[i]));
    }
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaFree(delay_block_d[i]));
    }
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaFree(output_buffer_d[i]));
    }
}

MSOSM_GPU_DM_BATCH_stream::~MSOSM_GPU_DM_BATCH_stream()
{
    cufftDestroy(p_f);
    for (int i = 0; i < numStreams; i++)
    {
        cufftDestroy(p_b[i]);
    }
}

int MSOSM_GPU_DM_BATCH_stream::next_power_of_2(int n)
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