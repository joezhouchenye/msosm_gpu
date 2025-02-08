#include "osm_gpu_concurrent.h"

OSM_GPU_DM_concurrent::OSM_GPU_DM_concurrent(float *bw, float *dm, float *f0, int numDMs) : Prepare_OSM_DM_concurrent(bw, dm, f0, numDMs)
{
    CUDA_CHECK(cudaEventCreateWithFlags(&dm_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&fft_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&output_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreate(&dm_stream));
    CUDA_CHECK(cudaStreamCreate(&fft_stream));
    CUDA_CHECK(cudaStreamCreate(&output_stream));
    wait_for_cpu = false;
}

void OSM_GPU_DM_concurrent::get_device_info()
{
    GPU_GetDevInfo();
}

void OSM_GPU_DM_concurrent::initialize_uint16(unsigned long fftpoint, int count)
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
    this->count = count;

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

    CUFFT_CHECK(cufftPlan1d(&p_b, fftpoint, CUFFT_C2C, count * numDMs));
    cufftSetStream(p_b, dm_stream);

    CUDA_CHECK(cudaMalloc((void **)&filtered_block_d, numDMs * count * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&ifft_block_d, numDMs * count * fftpoint * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_d, (numDMs * count * M_common + 1) * sizeof(Complex)));

    CUDA_CHECK(cudaMalloc((void **)&output_buffer_uint16_d, (numDMs * count * M_common + 1) * sizeof(uint16_pair)));

    CUDA_CHECK(cudaMalloc((void **)&dedisp_params_d, numDMs * fftpoint * sizeof(Complex)));

    for (int i = 0; i < numDMs; i++)
    {
        generate_dedisp_params(i);
        CUDA_CHECK(cudaMemcpy(dedisp_params_d + i * fftpoint, dedisp_params[i], fftpoint * sizeof(Complex), cudaMemcpyHostToDevice));
    }
}

void OSM_GPU_DM_concurrent::filter_block_uint16(uint16_pair *input)
{
    cudaStreamWaitEvent(fft_stream, dm_event, 0);
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_uint16_d, input, count * M_common * sizeof(uint16_pair), cudaMemcpyHostToDevice, fft_stream));
    CUDA_CHECK(cudaMemcpyAsync(input_buffer_d, input_buffer_d + count * M_common, M_common * sizeof(Complex), cudaMemcpyDeviceToDevice, fft_stream));
    uint16ToComplex(input_buffer_d + M_common, (uint16_t *)input_buffer_uint16_d, count * M_common, fft_stream);
    CUFFT_CHECK(cufftExecC2C(p_f, input_buffer_d, fft_block_d, CUFFT_FORWARD));
    cudaEventRecord(fft_event, fft_stream);

    cudaStreamWaitEvent(dm_stream, fft_event, 0);
    ComplexMultiply_concurrent_batch(fft_block_d, dedisp_params_d, fftpoint, filtered_block_d, count, numDMs, dm_stream);
    cudaEventRecord(dm_event, dm_stream);

    CUFFT_CHECK(cufftExecC2C(p_b, filtered_block_d, ifft_block_d, CUFFT_INVERSE));

    cudaStreamWaitEvent(dm_stream, output_event, 0);
    DiscardSamples_concurrent_batch(ifft_block_d, output_buffer_d, M_common, count, numDMs, dm_stream);
}

void OSM_GPU_DM_concurrent::get_output(Complex *output)
{
    cudaEventRecord(ready_event, dm_stream);
    while (wait_for_cpu)
        ;
    wait_for_cpu = true;
    cudaStreamWaitEvent(output_stream, ready_event, 0);
    CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_d, (count * M_common * numDMs + 1) * sizeof(Complex), cudaMemcpyDeviceToHost, output_stream));
    cudaEventRecord(output_event, output_stream);
}

void OSM_GPU_DM_concurrent::get_output(uint16_pair *output)
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

void OSM_GPU_DM_concurrent::synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

OSM_GPU_DM_concurrent::~OSM_GPU_DM_concurrent()
{
    CUDA_CHECK(cudaDeviceReset());
}
