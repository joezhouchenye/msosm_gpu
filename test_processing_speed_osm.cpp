#include "globals.h"
#include "osm_gpu_concurrent.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int count = 4;
    int startcount = -1;
    int endcount = -1;
    int repeat = 10;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    unsigned long compare_process_len = 268435456;
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
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
        {"repeat", required_argument, nullptr, 'r'},
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
        case 'n':
            fftpoint = stoul(optarg);
            continue;
        case 'p':
            compare_process_len = stoul(optarg);
            continue;
        case 'r':
            repeat = stoi(optarg);
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

    OSM_GPU_DM_concurrent *osm[endcount - startcount + 1];
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 0;
    uint16_pair *input;

    double sum[endcount - startcount + 1] = {0};
    double sum_of_squares[endcount - startcount + 1] = {0};

    for (int i = startcount; i <= endcount; i++)
    {
        int index = i - startcount;
        osm[index] = new OSM_GPU_DM_concurrent(bw_i, DM_i, f0_i, numDMs);
        if (startcount != -1)
            count = static_cast<unsigned long>(pow(2, i));
        osm[index]->initialize_uint16(fftpoint, count, true);
        unsigned long M = osm[index]->M_common;
        unsigned long process_len = count * M;
        if (i == startcount)
        {
            cout << "Nd: " << osm[index]->Nd[0] << endl;
            unsigned long max_process_len;
            if (startcount == -1)
                max_process_len = count * M;
            else
                max_process_len = static_cast<unsigned long>(pow(2, endcount) * M);
            cout << "Max Process Length: " << max_process_len << endl;
            // Generate simulated complex signal
            // Use different parameters here to reduce generation time,
            // since we only need to test the speed
            int inputSize;
            unsigned long block_size = 8388608;
            float period = (float)block_size / 16e6;
            // Assume max_process_len and block_size are powers of 2
            inputSize = max_process_len / block_size;
            if (inputSize == 0)
                inputSize = 1;
            if (inputSize > compare_process_len / block_size)
            {
                cout << "The compared OSM process length is too short" << endl;
            }
            else
            {
                inputSize = compare_process_len / block_size;
                if (inputSize == 0)
                    inputSize = 1;
            }
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

        uint16_pair *output;
        output = (uint16_pair *)malloc((numDMs * process_len + 1) * sizeof(uint16_pair));
        if (output == NULL)
        {
            cout << "Memory Allocation Failed" << endl;
            exit(1);
        }
        error = cudaHostRegister(output, (numDMs * process_len + 1) * sizeof(uint16_pair), cudaHostRegisterDefault);
        if (error != cudaSuccess)
        {
            cout << "Host Memory Registration Failed" << endl;
            exit(1);
        }

        for (int j = 0; j < repeat; j++)
        {

            uint16_pair *current_input;

            // Start the timer
            auto start = chrono::high_resolution_clock::now();

            for (int k = 0; k < signal_size / process_len; k++)
            {
                current_input = input + k * process_len;
                osm[index]->filter_block_uint16(current_input);
                osm[index]->get_output(output);
            }
            osm[index]->synchronize();

            // Stop the timer
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
            double time = duration.count() / 1000000.0;
            sum[index] += time;
            sum_of_squares[index] += time * time;
        }
        cudaHostUnregister(input);
        osm[index]->reset_device();
        double mean = sum[index] / repeat;
        double variance = sum_of_squares[index] / repeat - mean * mean;
        double standard_deviation = sqrt(variance);
        if (i == startcount)
        {
            cout << "Time taken (ms) with " << repeat << " runs:" << endl;
            cout << "[";
        }
        if (i == endcount)
            cout << "[" << mean << ", " << standard_deviation << "]" << "]" << endl;
        else
            cout << "[" << mean << ", " << standard_deviation << "]" << ", ";
    }

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}