#include "globals.h"
#include "msosm_gpu_batch_stream.h"
#include "simulated_complex_signal.h"

int main(int argc, char *argv[])
{
    verbose = false;
    int numStreams = 1;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"channel", required_argument, nullptr, 'c'},
        {nullptr, 0, nullptr, 0}};

    for (;;)
    {
        switch (getopt_long(argc, argv, "", long_options, nullptr))
        {
        case 'v':
            verbose = true;
            continue;
        case 'c':
            numStreams = stoi(optarg);
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }
    // Pulsar signal parameters
    float bw = 16e6;
    float dm = 750;
    float f0 = 1e9;
    const int inputSize = 50;
    unsigned long block_size = 8388608;
    unsigned long fftpoint = 0;
    float period = (float)block_size / bw;
    // Batch Size
    int count = 32;
    cout << "Batch Size: " << count << endl;

    // Generate simulated complex signal
    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal(inputSize, false);
    unsigned long signal_size = simulated_signal.signal_size;

    float *bw_stream, *dm_stream, *f0_stream;
    bw_stream = new float[numStreams];
    dm_stream = new float[numStreams];
    f0_stream = new float[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        bw_stream[i] = bw;
        dm_stream[i] = dm;
        f0_stream[i] = f0;
    }
    unsigned long *fftpoint_stream = new unsigned long[numStreams];
    memset(fftpoint_stream, 0, numStreams * sizeof(unsigned long));

    // Initialize MSOSM_GPU_BATCH_stream object
    MSOSM_GPU_BATCH_stream msosm(bw_stream, dm_stream, f0_stream, numStreams);
    msosm.initialize_uint16(fftpoint_stream, count);

    cout << "Filter length: " << msosm.M[0] << endl;
    cout << "Delay count: " << msosm.delaycount[0] << endl;

    unsigned long M = msosm.M[0];

    // Initialize input CPU memory space
    uint16_pair *input[numStreams];
    input[0] = simulated_signal.signal_u16;
    for (int i = 1; i < numStreams; i++)
    {
        cudaError_t error;
        error = cudaMallocHost((void **)&(input[i]), signal_size * sizeof(uint16_pair));
        if (error != cudaSuccess)
        {
            cout << "Host Memory Allocation Failed" << endl;
            exit(1);
        }
        memcpy(input[i], input[0], signal_size * sizeof(uint16_pair));
    }

    // plot_abs(input[0], simulated_signal.Np);

    // Initialize output CPU memory space
    Complex *output[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        cudaError_t error;
        error = cudaMallocHost((void **)&(output[i]), signal_size * sizeof(Complex));
        if (error != cudaSuccess)
        {
            cout << "Host Memory Allocation Failed" << endl;
            exit(1);
        }
    }

    uint16_pair **input_stream = new uint16_pair *[numStreams];
    Complex **output_stream = new Complex *[numStreams];
    unsigned long process_len = count * M;
    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < signal_size / process_len; i++)
    {
        for (int j = 0; j < numStreams; j++)
        {
            input_stream[j] = input[j] + i * process_len;
            output_stream[j] = output[j] + i * process_len;
        }
        msosm.filter_block_uint16(input_stream);
        msosm.get_output(output_stream);
    }

    msosm.synchronize();

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Time taken (stream num: " << numStreams << "): " << duration.count() / 1000000.0 << " ms" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    // plot_abs(output[0], block_size);
    // // if (numStreams > 2)
    // //     plot_abs(output[1], block_size);
    // // if (numStreams > 1)
    // plot_abs(output[numStreams - 1], block_size);

    // // for (int i = 0; i < numStreams; i++)
    // // {
    // //     plot_abs(output[i], block_size);
    // // }
    return 0;
}