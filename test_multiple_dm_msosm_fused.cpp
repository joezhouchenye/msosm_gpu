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
    int startDM = -1;
    int endDM = -1;
    int repeat = 10;
    // Pulsar signal parameters
    float bw = 128e6;
    float dm = 75;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
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
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
        {"repeat", required_argument, nullptr, 'r'},
        {"trial", required_argument, nullptr, 't'},
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
            startDM = stoi(optarg);
            continue;
        case 'e':
            endDM = stoi(optarg);
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
        case 'r':
            repeat = stoi(optarg);
            continue;
        case 't':
            numDMs = stoi(optarg);
            continue;
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    cout << "Bandwidth: " << bw / 1e6 << " MHz" << endl;
    cout << "Max Dispersion measure: " << dm << " pc cm^-3" << endl;
    cout << "Start frequency: " << f0 / 1e6 << " MHz" << endl;

    // numDMs
    if (startDM == -1)
    {
        // Use given numDM value
        cout << "Default DM trials " << numDMs << " is used" << endl;
    }
    else if ((startDM > endDM) || (startDM < 0))
    {
        cout << "Invalid DM trials range" << endl;
        exit(1);
    }
    else
    {
        cout << "DM Trial Range:" << endl;
        for (int i = startDM; i <= endDM; i++)
        {
            if (i == startDM)
            {
                cout << "[" << i << ", ";
            }
            else if (i == endDM)
            {
                cout << i << "]" << endl;
            }
            else
            {
                cout << i << ", ";
            }
        }
    }

    MSOSM_GPU_DM_concurrent *msosm[endDM - startDM + 1];
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 0;
    uint16_pair *input;

    double sum[endDM - startDM + 1] = {0};
    double sum_of_squares[endDM - startDM + 1] = {0};

    for (int i = startDM; i <= endDM; i++)
    {
        if (startDM != -1)
            numDMs = static_cast<int>(pow(2, i));
        float *bw_i, *DM_i, *f0_i;
        bw_i = new float[numDMs];
        DM_i = new float[numDMs];
        f0_i = new float[numDMs];
        for (int j = 0; j < numDMs; j++)
        {
            bw_i[j] = bw;
            DM_i[j] = dm;
            f0_i[j] = f0;
        }
        int index = i - startDM;
        msosm[index] = new MSOSM_GPU_DM_concurrent(bw_i, DM_i, f0_i, numDMs);
        msosm[index]->initialize_uint16(fftpoint, count);
        unsigned long M = msosm[index]->M_common;
        unsigned long process_len = count * M;
        if (i == startDM)
        {
            cout << "Nd: " << msosm[index]->Nd[0] << endl;
            unsigned long max_process_len;
            max_process_len = count * M;
            cout << "Max Process Length: " << max_process_len << endl;
            cout << "Compared with OSM Process Length: " << osm_process_len << endl;
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

        for (int j = 0; j < repeat; j++)
        {
            uint16_pair *current_input;

            // Start the timer
            auto start = chrono::high_resolution_clock::now();

            for (int k = 0; k < signal_size / process_len; k++)
            {
                current_input = input + k * process_len;
                msosm[index]->filter_block_uint16(current_input);
            }
            msosm[index]->synchronize();

            // Stop the timer
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
            double time = duration.count() / 1000000.0;
            sum[index] += time;
            sum_of_squares[index] += time * time;
        }
        cudaHostUnregister(input);
        msosm[index]->reset_device();
        double mean = sum[index] / repeat;
        double variance = (sum_of_squares[index] / repeat) - (mean * mean);
        double standard_deviation = sqrt(variance);
        if (i == startDM)
        {
            cout << "Time taken (ms) with " << repeat << " runs:" << endl;
            cout << "[";
        }
        if (i == endDM)
            cout << "[" << mean << ", " << standard_deviation << "]" << "]" << endl;
        else
            cout << "[" << mean << ", " << standard_deviation << "]" << ", ";
    }

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}