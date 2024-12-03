#pragma once

#include "prepare_stream.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

typedef struct callBackData {
  uint16_pair **input;
  Complex **output;
  uint16_pair **input_host;
  Complex **output_host;
} callBackData_t;

class MSOSM_GPU_BATCH_stream : public Prepare_MSOSM_stream
{
public:
    MSOSM_GPU_BATCH_stream(float *bw, float *dm, float *f0, int numStreams);
    void get_device_info();
    void initialize_uint16(unsigned long *fftpoint, int count = 1);
    void filter_block_uint16(uint16_pair **input);
    void get_output(Complex **output);
    void synchronize();
    ~MSOSM_GPU_BATCH_stream();

private:
    int next_power_of_2(int n);

private:
    int *current_index = 0;

public:
    int count;
    Complex **output_buffer_d;

private:
    Complex **current_block;
    unsigned long *fftpoint;
    Complex **delay_block_d;
    Complex **input_buffer_d;
    uint16_pair **input_buffer_int16_d;
    Complex **fft_block_d;
    Complex **ifft_block_d;
    uint16_pair **output_buffer_int16_d;
    Complex **dedisp_params_d;
    int **delay_points_d;
    cufftHandle *p_f, *p_b;
    int *delay_block_size;
    cudaStream_t *stream;
};

constexpr int buffer_size = 32;
class MSOSM_GPU_DM_BATCH_stream : public Prepare_MSOSM_DM_stream
{
public:
    MSOSM_GPU_DM_BATCH_stream(float *bw, float *dm, float *f0, int numStreams);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint, int count = 1);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex **output);
    void synchronize();
    ~MSOSM_GPU_DM_BATCH_stream();

private:
    int next_power_of_2(int n);

private:
    int *current_index = 0;

public:
    int count;
    bool *wait_for_cpu;

private:
    Complex **output_buffer_d;
    Complex **current_block;
    unsigned long fftpoint;
    Complex **delay_block_d;
    Complex *input_buffer_d;
    uint16_pair *input_buffer_int16_d;
    Complex *fft_block_d;
    Complex **ifft_block_d;
    uint16_pair **output_buffer_int16_d;
    Complex **dedisp_params_d;
    int **delay_points_d;
    cufftHandle p_f;
    cufftHandle *p_b;
    int *delay_block_size;
    cudaStream_t fft_stream;
    cudaStream_t *stream;
    cudaEvent_t *stream_event;
    cudaEvent_t fft_event;
};