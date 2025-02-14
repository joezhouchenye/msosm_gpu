#pragma once

#include "prepare_concurrent.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

class OSM_GPU_DM_concurrent : public Prepare_OSM_DM_concurrent
{
public:
    OSM_GPU_DM_concurrent(float *bw, float *dm, float *f0, int numDMs);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint = 0, int count = 1, bool compute_only=false);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex *output);
    void get_output(uint16_pair *output);
    void synchronize();
    void reset_device();
    ~OSM_GPU_DM_concurrent();

public:
    int count;
    bool wait_for_cpu;

private:
    bool compute_only;
    Complex *output_buffer_d;
    unsigned long fftpoint;
    Complex *input_buffer_d;
    uint16_pair *input_buffer_uint16_d;
    Complex *fft_block_d;
    Complex *filtered_block_d;
    Complex *ifft_block_d;
    uint16_pair *output_buffer_uint16_d;
    Complex *dedisp_params_d;
    cufftHandle p_f, p_b;
    cudaStream_t fft_stream;
    cudaStream_t dm_stream;
    cudaStream_t output_stream;
    cudaEvent_t dm_event;
    cudaEvent_t fft_event;
    cudaEvent_t ready_event;
    cudaEvent_t output_event;
};