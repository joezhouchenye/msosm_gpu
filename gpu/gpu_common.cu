#include "gpu_common.cuh"

using namespace std;

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err == cudaSuccess)
        return;
    cerr << "Cuda error: " << cudaGetErrorString(err) << endl;
    if (msg)
        cerr << msg << endl;
    exit(1);
}

// Device information
void GPU_GetDevInfo()
{
    if (verbose)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        cout << "Number of GPU devices: " << deviceCount << endl;
        line();
        cudaDeviceProp prop;
        for (int i = 0; i < deviceCount; i++)
        {
            cout << "Device " << i << " information:" << endl;
            cudaGetDeviceProperties(&prop, i);
            cout << "Device " << prop.name << endl;
            cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
            cout << "Clock rate: " << prop.clockRate << " KHz" << endl;
            cout << "Amount of global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << endl;
            cout << "Amount of constant memory: " << prop.totalConstMem / (1024.0) << " KB" << endl;
            cout << "Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
            cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
            cout << "Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << endl;
            cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
            cout << "Max Block Size: ";
            cout << prop.maxThreadsDim[0] << " ";
            cout << prop.maxThreadsDim[1] << " ";
            cout << prop.maxThreadsDim[2] << endl;
            cout << "Max Grid Size: ";
            cout << prop.maxGridSize[0] << " ";
            cout << prop.maxGridSize[1] << " ";
            cout << prop.maxGridSize[2] << endl;
            cout << "Number of SMs: " << prop.multiProcessorCount << endl;
            cout << "Maximum number of blocks per SM: " << prop.maxBlocksPerMultiProcessor << endl;
            cout << "Maximum amount of shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << endl;
            cout << "Maximum amount of shared memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << endl;
            cout << "Maximum number of registers per block: " << prop.regsPerBlock / 1024 << " K" << endl;
            cout << "Maximum number of registers per SM: " << prop.regsPerMultiprocessor / 1024 << " K" << endl;
            cout << "Maximum number of threads per block: " << prop.maxThreadsPerBlock << endl;
            cout << "Maximum number of threads per SM: " << prop.maxThreadsPerMultiProcessor << endl;
            cout << "Maximum concurrent streams: " << prop.asyncEngineCount << endl;
            line();
        }
    }
}