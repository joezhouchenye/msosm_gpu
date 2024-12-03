#pragma once

#include "prepare.h"
#include "gpu_kernel.cuh"
#include "psrdada.h"

class MSOSM_GPU_BATCH : public Prepare_MSOSM
{
public:
    MSOSM_GPU_BATCH(float bw = 0, float dm = 0, float f0 = 0);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint = 0, int count = 1);
    void filter_block_uint16(uint16_pair *input);
    void get_output(Complex *output);
    ~MSOSM_GPU_BATCH();

private:
    int next_power_of_2(int n);

private:
    bool reverse_flag = false;
    int current_index = 0;

public:
    int count;
    Complex *output_buffer_d;

private:
    void increment();
    unsigned long fftpoint;
    Complex *delay_block_d;
    Complex *input_buffer_d;
    uint16_pair *input_buffer_int16_d;
    Complex *fft_block_d;
    Complex *ifft_block_d;
    uint16_pair *output_buffer_int16_d;
    Complex *dedisp_params_d;
    int *delay_points_d;
    cufftHandle p_f, p_b;
    int delay_block_size;
};

class MSOSM_GPU_DM_BATCH_loop : public Prepare_MSOSM_DM_loop
{
public:
    MSOSM_GPU_DM_BATCH_loop(float *bw, float *dm, float *f0, int numOutput);
    void get_device_info();
    void initialize_uint16(unsigned long fftpoint, int count = 1);
    void input_process(uint16_pair *input);
    void filter_block_uint16(Complex **output, int i);
    void synchronize();
    ~MSOSM_GPU_DM_BATCH_loop();

private:
    int next_power_of_2(int n);

private:
    int *current_index = 0;

public:
    int count;
    Complex **output_buffer_d;

private:
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
};