#include "globals.h"
#include "msosm_gpu_batch.h"
#include "simulated_complex_signal.h"

int main(int argc, char *argv[])
{
    verbose = false;
    int numOutput = 1;
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
            numOutput = stoi(optarg);
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
    int count = 64;
    cout << "Batch Size: " << count << endl;

    // Generate simulated complex signal
    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal(inputSize, false);
    unsigned long signal_size = simulated_signal.signal_size;

    float *bw_stream, *dm_stream, *f0_stream;
    bw_stream = new float[numOutput];
    dm_stream = new float[numOutput];
    f0_stream = new float[numOutput];
    for (int i = 0; i < numOutput; i++)
    {
        bw_stream[i] = bw;
        dm_stream[i] = dm - i * 1.0;
        f0_stream[i] = f0;
    }

    // Initialize MSOSM_GPU_BATCH_stream object
    MSOSM_GPU_DM_BATCH_loop msosm(bw_stream, dm_stream, f0_stream, numOutput);
    // msosm.override_order(8388608);
    msosm.initialize_uint16(fftpoint, count);

    cout << "Filter length: " << msosm.M_loop << endl;
    cout << "Delay count: " << msosm.delaycount[0] << endl;

    unsigned long M = msosm.M_loop;
    unsigned long process_len = count * M;

    // Initialize input CPU memory space
    uint16_pair *input;
    input = simulated_signal.signal_u16;

    // plot_abs(input, simulated_signal.Np);

    // Initialize output CPU memory space
    Complex *output[numOutput];
    for (int i = 0; i < numOutput; i++)
    {
        output[i] = (Complex *)malloc(process_len * sizeof(Complex));
        // output[i] = (Complex *)malloc(signal_size * sizeof(Complex));
        if (output[i] == NULL)
        {
            cout << "Memory Allocation Failed" << endl;
            exit(1);
        }
    }
    for (int i = 0; i < numOutput; i++)
    {
        cudaError_t error;
        error = cudaHostRegister(output[i], process_len * sizeof(Complex), cudaHostRegisterDefault);
        // error = cudaHostRegister(output[i], signal_size * sizeof(Complex), cudaHostRegisterDefault);
        if (error != cudaSuccess)
        {
            cout << "Host Memory Registration Failed" << endl;
            exit(1);
        }
    }
    // for (int i = 0; i < numOutput; i++)
    // {
    //     cudaError_t error;
    //     error = cudaMallocHost((void **)&(output[i]), signal_size * sizeof(Complex));
    //     if (error != cudaSuccess)
    //     {
    //         cout << "Host Memory Allocation Failed" << endl;
    //         exit(1);
    //     }
    // }

    uint16_pair *input_stream;
    Complex **output_stream = new Complex *[numOutput];
    for (int i = 0; i < numOutput; i++)
    {
        output_stream[i] = output[i];
    }

    cout << "Start Processing" << endl;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < signal_size / process_len; i++)
    {
        input_stream = input + i * process_len;
        // for (int j = 0; j < numOutput; j++)
        // {
        //     output_stream[j] = output[j] + i * process_len;
        // }
        msosm.input_process(input_stream);
        for (int j = 0; j < numOutput; j++)
        {
            msosm.filter_block_uint16(output_stream, j);
        }
    }

    msosm.synchronize();

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Time taken (stream num: " << numOutput << "): " << duration.count() / 1000000.0 << " ms" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    // plot_abs(output[0], block_size);
    // // if (numOutput > 2)
    // //     plot_abs(output[1], block_size);
    // // if (numOutput > 1)
    // plot_abs(output[numOutput - 1], block_size);

    // for (int i = 0; i < numOutput; i++)
    // {
    //     plot_abs(output[i], block_size);
    // }
    return 0;
}