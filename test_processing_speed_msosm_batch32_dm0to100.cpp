#include "globals.h"
#include "msosm_gpu_concurrent.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>
#include <sstream> // 用于字符串流
#include <string>  // 用于字符串操作

// 解析字符串并返回整数数组
std::vector<unsigned long> parseArray(const std::string &input)
{
    std::vector<unsigned long> numbers;
    std::string processedInput = input;

    // 去掉方括号
    processedInput.erase(0, 1);                      // 移除开头的 '['
    processedInput.erase(processedInput.size() - 1); // 移除结尾的 ']'

    // 分割字符串，根据逗号进行分割
    size_t pos = 0;
    while ((pos = processedInput.find(',')) != std::string::npos)
    {
        std::string token = processedInput.substr(0, pos);
        numbers.push_back(std::stoul(token)); // 将字符串转换为整数并添加到数组
        processedInput.erase(0, pos + 1);     // 移除已处理的部分
    }

    // 处理最后一个数字
    if (!processedInput.empty())
    {
        numbers.push_back(std::stoul(processedInput)); // 转换最后一个数字
    }

    return numbers; // 返回解析后的整数数组
}

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int count = 32;
    int startdm = -1;
    int enddm = -1;
    int dmcount = 10;
    float dmstep = 0;
    int repeat = 10;
    string input_size_string = "0";
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
        {"channel", required_argument, nullptr, 'c'},
        {"start", required_argument, nullptr, 's'},
        {"end", required_argument, nullptr, 'e'},
        {"stop", required_argument, nullptr, 'e'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"dmcount", required_argument, nullptr, 'u'},
        {"f0", required_argument, nullptr, 'f'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"compare", required_argument, nullptr, 'p'},
        {"repeat", required_argument, nullptr, 'r'},
        {"size", required_argument, nullptr, 'i'},
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
            startdm = stoi(optarg);
            continue;
        case 'e':
            enddm = stoi(optarg);
            continue;
        case 'w':
            bw = stof(optarg);
            continue;
        case 'd':
            dm = stof(optarg);
            continue;
        case 'u':
            dmcount = stoi(optarg);
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
        case 'i':
            input_size_string = optarg;
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

    // DM values
    if (startdm == -1)
    {
        // Use given DM value
        cout << "Default DM value " << dm << " is used" << endl;
        dmcount = 1;
    }
    else if ((startdm > enddm) || (startdm < 0))
    {
        cout << "Invalid DM range" << endl;
        exit(1);
    }
    else
    {
        dmstep = (enddm - startdm) / (dmcount - 1);
        cout << "DM step: " << dmstep << " pc cm^-3" << endl;
        cout << "DM Range:" << endl;
        for (int i = 1; i <= dmcount; i++)
        {
            if (i == 1)
            {
                cout << "[" << startdm << ", ";
            }
            else if (i == dmcount)
            {
                cout << enddm << "]" << endl;
            }
            else
            {
                cout << startdm + (i - 1) * dmstep << ", ";
            }
        }
    }

    vector<unsigned long> input_size;
    int *count_input;
    unsigned long *fft_point_i;
    if (input_size_string != "0")
    {
        input_size = parseArray(input_size_string);
        // Get the length of input_size
        int input_size_len = input_size.size();

        for (int i = 0; i < dmcount; i++)
        {
            if (i == 0)
                cout << "Input size: " << "[";
            if (i == dmcount - 1)
                cout << input_size[i] << "]" << endl;
            else
                cout << input_size[i] << ", ";
        }
        if (input_size_len != dmcount)
        {
            cout << "Wrong input size" << endl;
            exit(1);
        }
        count_input = new int[dmcount];
        fft_point_i = new unsigned long[dmcount];
    }

    MSOSM_GPU_DM_concurrent *msosm[dmcount];
    SimulatedComplexSignal *simulated_signal;
    unsigned long signal_size = 0;
    uint16_pair *input;

    double sum[dmcount] = {0};
    double sum_of_squares[dmcount] = {0};

    for (int i = 1; i <= dmcount; i++)
    {
        int index = i - 1;
        float *bw_i, *DM_i, *f0_i;
        bw_i = new float[numDMs];
        DM_i = new float[numDMs];
        f0_i = new float[numDMs];
        for (int j = 0; j < numDMs; j++)
        {
            bw_i[j] = bw;
            if (startdm != -1)
                dm = startdm + (i - 1) * dmstep;
            DM_i[j] = dm;
            f0_i[j] = f0;
        }
        msosm[index] = new MSOSM_GPU_DM_concurrent(bw_i, DM_i, f0_i, numDMs);
        if (input_size_string != "0")
        {
            msosm[index]->initialize_uint16(fftpoint, count, input_size[index], true);
            count = msosm[index]->count;
            fft_point_i[index] = msosm[index]->M_common * 2;
            count_input[index] = count;
        }
        else
            msosm[index]->initialize_uint16(fftpoint, count, 0, true);
        unsigned long M = msosm[index]->M_common;
        unsigned long process_len = count * M;
        if (i == 1)
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
        if (process_len > signal_size)
        {
            cout << "Warning: Process length " << process_len << " is larger than signal size" << endl;
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
                msosm[index]->filter_block_uint16(current_input);
                msosm[index]->get_output(output);
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
        if (i == 1)
        {
            cout << "Time taken (ms) with " << repeat << " runs:" << endl;
            cout << "[";
        }
        if (i == dmcount)
            cout << "[" << mean << ", " << standard_deviation << "]" << "]" << endl;
        else
            cout << "[" << mean << ", " << standard_deviation << "]" << ", ";
    }

    if (input_size_string != "0")
    {
        cout << "Batch Size:" << endl;
        for (int i = 0; i < dmcount; i++)
        {
            if (i == 0)
                cout << "[";
            if (i == dmcount - 1)
                cout << count_input[i] << "]" << endl;
            else
                cout << count_input[i] << ", ";
        }

        cout << "FFT length:" << endl;
        for (int i = 0; i < dmcount; i++)
        {
            if (i == 0)
                cout << "[";
            if (i == dmcount - 1)
                cout << fft_point_i[i] << "]" << endl;
            else
                cout << fft_point_i[i] << ", ";
        }
    }
    else
    {
        cout << "Batch Size: " << count << endl;
        cout << "FFT length: " << 2 * msosm[0]->M_common << endl;
    }

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    return 0;
}