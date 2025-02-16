#include "globals.h"
#include "msosm_gpu_batch_stream.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

void output_thread(Complex *dst, Complex *src, unsigned long process_len, int num_process, MSOSM_GPU_DM_BATCH_stream *msosm, int stream_id)
{
    int count = 0;
    bool flag = false;
    src[process_len].x = 1;
    while (count < num_process)
    {
        while (src[process_len].x == 1)
            ;
        nvtxRangePush("Output Copy");
        src[process_len].x = 1;
        // memcpy(dst, src, process_len * sizeof(Complex));
        memcpy(dst + count * process_len, src, process_len * sizeof(Complex));
        count++;
        msosm->wait_for_cpu[stream_id] = false;
        nvtxRangePop();
    }
}

void output_thread_file(uint16_pair *src, unsigned long process_len, int num_process, MSOSM_GPU_DM_BATCH_stream *msosm, int stream_id)
{
    src[process_len].first = 1;
    // Get the current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert to tm structure for local time
    std::tm now_tm = *std::localtime(&now_time_t);

    // Use a stringstream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

    ofstream outfile;
    // Open file in binary write mode
    outfile.open(oss.str() + "_" + to_string(stream_id) + ".bin", ios::binary | ios::out);
    int count = 0;
    bool flag = false;
    while (count < num_process)
    {
        while (src[process_len].first == 1)
            ;
        nvtxRangePush("Output Copy");
        src[process_len].first = 1;
        outfile.write((char *)src, process_len * sizeof(uint16_pair));
        count++;
        msosm->wait_for_cpu[stream_id] = false;
        nvtxRangePop();
    }
}

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
    float bw = 128e6;
    float dm = 100;
    float f0 = 1e9;
    const int inputSize = 16;
    unsigned long block_size = 8388608*2;
    unsigned long fftpoint = 0;
    // fftpoint = 33554432;
    // fftpoint = 65536;
    float period = (float)block_size / bw;
    // Batch Size
    int count = 32;
    cout << "Batch Size: " << count << endl;
    cout << "Number of Streams: " << numStreams << endl;

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
        dm_stream[i] = dm - i * 0.1;
        f0_stream[i] = f0;
    }

    // Initialize MSOSM_GPU_BATCH_stream object
    MSOSM_GPU_DM_BATCH_stream msosm(bw_stream, dm_stream, f0_stream, numStreams);
    msosm.initialize_uint16(fftpoint, count);

    cout << "Filter length: " << msosm.M_stream << endl;
    cout << "Delay count: " << msosm.delaycount[0] << endl;

    unsigned long M = msosm.M_stream;
    unsigned long process_len = count * M;

    // Initialize input CPU memory space
    uint16_pair *input;
    input = simulated_signal.signal_u16;

    // Initialize output CPU memory space
    // Complex **output_check = new Complex *[numStreams];
    // if (signal_size * sizeof(Complex) * numStreams / 1024 / 1024 / 1024 > 16)
    // {
    //     cout << "Memory Size Exceeds 16GB" << endl;
    //     exit(1);
    // }
    // for (int i = 0; i < numStreams; i++)
    // {
    //     output_check[i] = new Complex[signal_size];
    //     // cudaMallocHost((void **)&(output_check[i]), process_len * sizeof(Complex));
    // }

    // Complex *output[numStreams];
    // for (int i = 0; i < numStreams; i++)
    // {
    //     output[i] = (Complex *)malloc((process_len + 1) * sizeof(Complex));
    //     // output[i] = (Complex *)malloc(signal_size * sizeof(Complex));
    //     if (output[i] == NULL)
    //     {
    //         cout << "Memory Allocation Failed" << endl;
    //         exit(1);
    //     }
    // }
    // for (int i = 0; i < numStreams; i++)
    // {
    //     cudaError_t error;
    //     error = cudaHostRegister(output[i], (process_len + 1) * sizeof(Complex), cudaHostRegisterDefault);
    //     // error = cudaHostRegister(output[i], signal_size * sizeof(Complex), cudaHostRegisterDefault);
    //     if (error != cudaSuccess)
    //     {
    //         cout << "Host Memory Registration Failed" << endl;
    //         exit(1);
    //     }
    // }

    uint16_pair *output[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        output[i] = (uint16_pair *)malloc((process_len + 1) * sizeof(uint16_pair));
        if (output[i] == NULL)
        {
            cout << "Memory Allocation Failed" << endl;
            exit(1);
        }
    }
    for (int i = 0; i < numStreams; i++)
    {
        cudaError_t error;
        error = cudaHostRegister(output[i], (process_len + 1) * sizeof(uint16_pair), cudaHostRegisterDefault);
        if (error != cudaSuccess)
        {
            cout << "Host Memory Registration Failed" << endl;
            exit(1);
        }
    }

    uint16_pair *input_stream;
    uint16_pair **output_stream = new uint16_pair *[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        output_stream[i] = output[i];
    }

    cout << "Start Processing" << endl;
    cout << "Signal Size: " << signal_size << endl;
    cout << "Process Size: " << process_len << endl;

    vector<thread> threads;

    for (int i = 0; i < numStreams; i++)
    {
        // threads.emplace_back(output_thread, output_check[i], output_stream[i], process_len, signal_size / process_len, &msosm, i);
        threads.emplace_back(output_thread_file, output_stream[i], process_len, signal_size / process_len, &msosm, i);
    }

    cout << "Output Bytes: " << signal_size * sizeof(Complex) * numStreams << endl;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < signal_size / process_len; i++)
    {
        input_stream = input + i * process_len;
        msosm.filter_block_uint16(input_stream);
        msosm.get_output(output_stream);
    }

    for (auto &t : threads)
    {
        t.join();
    }
    msosm.synchronize();

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Time taken (stream num: " << numStreams << "): " << duration.count() / 1000000.0 << " ms" << endl;
    cout << "Tansfer speed requirement: " << signal_size * sizeof(uint16_pair) * numStreams / (duration.count() / 1000000000.0) / 1024.0 / 1024.0 / 1024.0 * 8.0 << " Gbps" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    // plot_abs(output_check[0], block_size);
    // plot_abs(output[0], block_size);
    // // if (numStreams > 2)
    // //     plot_abs(output[1], block_size);
    // // if (numStreams > 1)
    // plot_abs(output[numStreams - 1], block_size);

    // for (int i = 0; i < numStreams; i++)
    // {
    //     plot_abs(output_check[i], block_size);
    // }
    // show();
    return 0;
}