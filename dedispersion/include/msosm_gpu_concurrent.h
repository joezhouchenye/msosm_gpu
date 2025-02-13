#pragma once

#include "prepare_concurrent.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

typedef struct callBackData {
  uint16_pair **input;
  Complex **output;
  uint16_pair **input_host;
  Complex **output_host;
} callBackData_t;

constexpr int buffer_size = 32;
class MSOSM_GPU_DM_concurrent : public Prepare_MSOSM_DM_concurrent
{
public:
    MSOSM_GPU_DM_concurrent(float *bw, float *dm, float *f0, int numDMs);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint, int count = 1, unsigned long input_size=0);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex *output);
    void get_output(uint16_pair *output);
    void synchronize();
    void reset_device();
    ~MSOSM_GPU_DM_concurrent();

private:
    int next_power_of_2(int n);

private:
    int current_index = 0;

public:
    int count;
    bool wait_for_cpu;

private:
    Complex *output_buffer_d;
    Complex *current_block_d;
    unsigned long fftpoint;
    Complex *delay_block_d;
    Complex *input_buffer_d;
    uint16_pair *input_buffer_uint16_d;
    Complex *fft_block_d;
    Complex *ifft_block_d;
    uint16_pair *output_buffer_uint16_d;
    Complex *dedisp_params_d;
    int *delay_points_d;
    cufftHandle p_f;
    cufftHandle p_b;
    int *delay_block_size;
    int max_delay_block_size;
    cudaStream_t fft_stream;
    cudaStream_t dm_stream;
    cudaStream_t output_stream;
    cudaEvent_t dm_event;
    cudaEvent_t fft_event;
    cudaEvent_t ready_event;
    cudaEvent_t output_event;
};