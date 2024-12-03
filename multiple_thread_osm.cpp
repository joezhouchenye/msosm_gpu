#include "globals.h"
#include "osm_gpu_batch.h"
#include "simulated_complex_signal.h"

void osm_thread(OSM_GPU_BATCH osm, uint16_pair *input, unsigned long signal_size, Complex *output)
{
    unsigned long process_len = osm.count * osm.M;
    for (int i = 0; i < signal_size / process_len; i++)
    {
        osm.filter_block_uint16(input + i * process_len);
        osm.get_output(output + i * process_len);
    }
}

int main(int argc, char *argv[])
{
    verbose = false;
    int numThreads = 1;
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
            numThreads = stoi(optarg);
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
    float period = (float)block_size / bw;
    // Batch Size
    int count = 1;

    // Generate simulated complex signal
    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal(inputSize, false);
    unsigned long signal_size = simulated_signal.signal_size;

    vector<OSM_GPU_BATCH> osm;
    vector<thread> threads;

    // Initialize all OSM_GPU_BATCH objects
    for (int i = 0; i < numThreads; i++)
    {
        osm.emplace_back(OSM_GPU_BATCH(bw, dm, f0));
        // Use minimum FFT point size
        osm[i].initialize_uint16(0, count, true);
    }

    cout << "Filter length: " << osm[0].M << endl;

    // Initialize input CPU memory space
    uint16_pair *input[numThreads];
    input[0] = simulated_signal.signal_u16;
    for (int i = 1; i < numThreads; i++)
    {
        cudaMallocHost((void **)&input[i], signal_size * sizeof(uint16_pair));
        memcpy(input[i], input[0], signal_size * sizeof(uint16_pair));
    }
    // plot_abs(input[0], simulated_signal.Np);

    // Initialize output CPU memory space
    Complex *output[numThreads];
    for (int i = 0; i < numThreads; i++)
    {
        cudaMallocHost((void **)&(output[i]), signal_size * sizeof(Complex));
    }

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    // Start the threads
    for (int i = 0; i < numThreads; i++)
    {
        threads.emplace_back(osm_thread, osm[i], input[i], signal_size, output[i]);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Time taken (thread num: " << numThreads << "): " << duration.count()/1000000.0 << " ms" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    plot_abs(output[0], block_size);
    if (numThreads > 1)
        plot_abs(output[numThreads - 1], block_size);

    return 0;
}