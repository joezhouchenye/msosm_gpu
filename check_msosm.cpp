#include "globals.h"
#include "msosm_gpu_concurrent.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    verbose = false;
    int count = 32;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    // Compare proces length with OSM
    unsigned long osm_process_len = 268435456;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0}};

    for (;;)
    {
        switch (getopt_long(argc, argv, "", long_options, nullptr))
        {
        case 'v':
            verbose = true;
            continue;
        case 'b':
            count = stoi(optarg);
            continue;
        case 'w':
            bw = stof(optarg);
            continue;
        case 'd':
            dm = stof(optarg);
            continue;
        case 'f':
            f0 = stof(optarg);
            continue;
        case 'n':
            fftpoint = stoul(optarg);
            continue;
        case 'p':
            osm_process_len = stoul(optarg);
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    cout << "Bandwidth: " << bw / 1e6 << " MHz" << endl;
    cout << "Dispersion measure: " << dm << " pc cm^-3" << endl;
    cout << "Start frequency: " << f0 / 1e6 << " MHz" << endl;

    MSOSM_GPU_DM_concurrent *msosm;
    SimulatedComplexSignal *simulated_signal; // Generate simulated complex signal
    // Use different parameters here to reduce generation time,
    // since we only need to test the speed
    unsigned long signal_size = 0;
    uint16_pair *input;

    msosm = new MSOSM_GPU_DM_concurrent(&bw, &dm, &f0, 1);
    msosm->initialize_uint16(fftpoint, count, 0, true);
    unsigned long M = msosm->M_common;
    unsigned long process_len = count * M;

    cout << "Nd: " << msosm->Nd[0] << endl;
    unsigned long max_process_len;
    max_process_len = count * M;

    cout << "Max Process Length: " << max_process_len << endl;
    cout << "Compared with OSM Process Length: " << osm_process_len << endl;

    int inputSize;
    float period = 0.002048;
    unsigned long block_size = static_cast<unsigned long>(period * bw);

    inputSize = max_process_len / block_size;
    if (inputSize == 0)
        inputSize = 1;
    if (inputSize > osm_process_len / block_size)
    {
        cout << "The compared OSM process length is too short" << endl;
    }
    else
    {
        inputSize = osm_process_len / block_size;
        if (inputSize == 0)
            inputSize = 1;
    }
    simulated_signal = new SimulatedComplexSignal(bw, dm, f0, period, "uint16");
    simulated_signal->generate_pulsar_signal(inputSize, false, 0, false);
    signal_size = simulated_signal->signal_size;
    cout << "Signal Size: " << signal_size << endl;
    input = simulated_signal->signal_u16;

    cudaError_t error;
    error = cudaHostRegister(input, signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed for input" << endl;
        exit(1);
    }

    int process_count = signal_size / process_len;
    cout << "Process Count: " << process_count << endl;

    // uint16_pair *result;
    // result = (uint16_pair *)malloc(signal_size * sizeof(uint16_pair));

    // uint16_pair *output;
    // output = (uint16_pair *)malloc((process_len + 1) * sizeof(uint16_pair));
    // if (output == NULL)
    // {
    //     cout << "Memory Allocation Failed" << endl;
    //     exit(1);
    // }
    // error = cudaHostRegister(output, (process_len + 1) * sizeof(uint16_pair), cudaHostRegisterDefault);
    // if (error != cudaSuccess)
    // {
    //     cout << "Host Memory Registration Failed" << endl;
    //     exit(1);
    // }

    uint16_pair *output;
    output = (uint16_pair *)malloc((signal_size + process_count) * sizeof(uint16_pair));
    if (output == NULL)
    {
        cout << "Memory Allocation Failed" << endl;
        exit(1);
    }
    error = cudaHostRegister(output, (signal_size + process_count) * sizeof(uint16_pair), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed" << endl;
        exit(1);
    }

    uint16_pair *current_input;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int k = 0; k < process_count; k++)
    {
        current_input = input + k * process_len;
        msosm->filter_block_uint16(current_input);
        msosm->get_output(output + k * (process_len + 1));
        msosm->synchronize();
        // cudaMemcpy(result + k * process_len, output, process_len * sizeof(uint16_pair), cudaMemcpyHostToHost);
    }

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    double time = duration.count() / 1000000.0;

    // plot_abs(result + (inputSize-1)*block_size, block_size);
    plot_abs(output + (inputSize-1)*block_size, block_size);
    show();

    cudaHostUnregister(input);
    msosm->reset_device();

    cout << "Time taken (ms):" << time << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}