#include "globals.h"
#include "msosm_gpu_concurrent.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int count = 4;
    int startcount = -1;
    int endcount = 0;
    // Pulsar signal parameters
    float bw = 16e6;
    float dm = 75;
    float f0 = 1e9;
    // Compare proces length with OSM
    unsigned long osm_process_len = 33554432;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"channel", required_argument, nullptr, 'c'},
        {"start", required_argument, nullptr, 's'},
        {"end", required_argument, nullptr, 'e'},
        {"stop", required_argument, nullptr, 'e'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
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
        case 'c':
            numDMs = stoi(optarg);
            continue;
        case 's':
            startcount = stoi(optarg);
            continue;
        case 'e':
            endcount = stoi(optarg);
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

    // Batch Size
    if (startcount == -1)
    {
        // Use given count value
        startcount = count;
        endcount = count;
        cout << "Default batch size " << count << " is used" << endl;
    }
    else if ((startcount > endcount) || (startcount < 0))
    {
        cout << "Invalid count range" << endl;
        exit(1);
    }
    else
    {
        cout << "Batch Size Range:" << endl;
        for (int i = startcount; i <= endcount; i++)
        {
            if (i == startcount)
            {
                cout << "[" << i << ", ";
            }
            else if (i == endcount)
            {
                cout << i << "]" << endl;
            }
            else
            {
                cout << i << ", ";
            }
        }
    }

    unsigned long fftpoint = 0;

    float *bw_i, *DM_i, *f0_i;
    bw_i = new float[numDMs];
    DM_i = new float[numDMs];
    f0_i = new float[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        bw_i[i] = bw;
        DM_i[i] = dm;
        f0_i[i] = f0;
    }

    MSOSM_GPU_DM_concurrent *msosm[endcount - startcount + 1];
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 0;
    uint16_pair *input;

    for (int i = startcount; i <= endcount; i++)
    {
        int index = i - startcount;
        msosm[index] = new MSOSM_GPU_DM_concurrent(bw_i, DM_i, f0_i, numDMs);
        count = static_cast<unsigned long>(pow(2, i));
        msosm[index]->initialize_uint16(fftpoint, count);
        unsigned long M = msosm[index]->M_common;
        unsigned long process_len = count * M;
        if (i == startcount)
        {
            unsigned long max_process_len = static_cast<unsigned long>(pow(2, endcount) * M);
            cout << "Max Process Length: " << max_process_len << endl;
            // Generate simulated complex signal
            // Use different parameters here to reduce generation time,
            // since we only need to test the speed
            int inputSize;
            unsigned long block_size = 8388608;
            float period = (float)block_size / 16e6;
            inputSize = osm_process_len / block_size;
            simulated_signal = new SimulatedComplexSignal(16e6, 75, f0, period, "uint16");
            simulated_signal->generate_pulsar_signal(inputSize, false, 0, false);
            signal_size = simulated_signal->signal_size;
            cout << "Signal Size: " << signal_size << endl;
            input = simulated_signal->signal_u16;
        }
        cudaError_t error;
        error = cudaHostRegister(input, signal_size * sizeof(uint16_pair), cudaHostRegisterDefault);
        if (error != cudaSuccess)
        {
            cout << "Host Memory Registration Failed for input" << endl;
            exit(1);
        }

        Complex *output;
        output = (Complex *)malloc((numDMs * process_len + 1) * sizeof(Complex));
        if (output == NULL)
        {
            cout << "Memory Allocation Failed" << endl;
            exit(1);
        }
        error = cudaHostRegister(output, (numDMs * process_len + 1) * sizeof(Complex), cudaHostRegisterDefault);
        if (error != cudaSuccess)
        {
            cout << "Host Memory Registration Failed" << endl;
            exit(1);
        }

        uint16_pair *current_input;

        // Start the timer
        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < signal_size / process_len; i++)
        {
            current_input = input + i * process_len;
            msosm[index]->filter_block_uint16(current_input);
        }
        msosm[index]->synchronize();

        // Stop the timer
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        if (i == startcount)
        {
            cout << "Time taken (ms):" << endl;
            cout << "[";
        }
        if (i == endcount)
            cout << duration.count() / 1000000.0 << "]" << endl;
        else
            cout << duration.count() / 1000000.0 << ", ";
        cudaHostUnregister(input);
        msosm[index]->reset_device();
    }

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}