#pragma once

#include "gpu_common.cuh"

void ComplexMultiply_batch(Complex *a, Complex *b, int N, Complex *block, int count);
void ComplexMultiplyDelay_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count);
void DiscardSamples_batch(Complex *a, Complex *b, int M, int count);
void uint16ToComplex(Complex *dst, uint16_t *src, size_t size);
void complexToUint16(uint16_t *dst, Complex *src, size_t size);
void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size);

// Stream version
void ComplexMultiply_batch(Complex *a, Complex *b, int N, Complex *block, int count, cudaStream_t stream);
void ComplexMultiplyDelay_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count, cudaStream_t stream);
void DiscardSamples_batch(Complex *a, Complex *b, int M, int count, cudaStream_t stream);
void uint16ToComplex(Complex *dst, uint16_t *src, size_t size, cudaStream_t stream);
void complexToUint16(uint16_t *dst, Complex *src, size_t size, cudaStream_t stream);
void calculateIntensity(Complex *a, Complex *b, float *total_intensity, unsigned long size, cudaStream_t stream);

// Concurrent version
void ComplexMultiply_concurrent_batch(Complex *a, Complex *b, int N, Complex *block, int count, int numDMs, cudaStream_t stream);
void ComplexMultiplyDelay_concurrent_batch(Complex *a, Complex *b, int N, Complex *block, int *delay, int delay_block_size, int index, int count, int numDMs, cudaStream_t stream);
void DiscardSamples_concurrent_batch(Complex *a, Complex *b, int M, int count, int numDMs, cudaStream_t stream);

void initializeBoolArray(bool* array, int size, bool value);
void waitForCPU(bool* flag, cudaStream_t stream);
void unblockCPU(bool* flag, cudaStream_t stream);