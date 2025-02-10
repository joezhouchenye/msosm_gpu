#include "globals.h"
#include "msosm_gpu_concurrent.h"
#include "simulated_complex_signal.h"
#include <fstream>
#include <iomanip>

void output_thread(Complex *dst, Complex *src, unsigned long process_len, int num_process, int numDMs, MSOSM_GPU_DM_concurrent *msosm)
{
    int count = 0;
    bool flag = false;
    src[process_len * numDMs].x = 1;
    while (count < num_process)
    {
        while (src[process_len * numDMs].x == 1)
            ;
        nvtxRangePush("Output Copy");
        src[process_len * numDMs].x = 1;
        // memcpy(dst, src, process_len * numDMs * sizeof(Complex));
        memcpy(dst + count * process_len * numDMs, src, process_len * numDMs * sizeof(Complex));
        count++;
        msosm->wait_for_cpu = false;
        nvtxRangePop();
    }
}

// void output_thread_file(Complex *src, unsigned long process_len, int num_process, int numDMs, MSOSM_GPU_DM_concurrent *msosm)
// {
//     src[process_len * numDMs].x = 1;
//     // Get the current time
//     auto now = std::chrono::system_clock::now();
//     std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

//     // Convert to tm structure for local time
//     std::tm now_tm = *std::localtime(&now_time_t);

//     // Use a stringstream to format the date and time
//     std::ostringstream oss;
//     oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

//     // FILE *fp[numDMs];
//     // // Open file in binary write mode
//     // for (int i = 0; i < numDMs; i++)
//     // {
//     //     string filename = "/media/joe/MyFile/" + oss.str() + "_" + to_string(i) + ".bin";
//     //     fp[i] = fopen(filename.c_str(), "wb");
//     // }
//     ofstream outfile[numDMs];
//     // Open file in binary write mode
//     for (int i = 0; i < numDMs; i++)
//     {
//         // outfile[i].open("/media/joe/MyFile/" + oss.str() + "_" + to_string(i) + ".bin", ios::binary | ios::out);
//         outfile[i].open(oss.str() + "_" + to_string(i) + ".bin", ios::binary | ios::out);
//     }
//     // ofstream outfile;
//     // // Open file in binary write mode
//     // outfile.open("test.bin", ios::binary | ios::out);
//     int count = 0;
//     bool flag = false;
//     while (count < num_process)
//     {
//         while (src[process_len * numDMs].x == 1)
//             ;
//         nvtxRangePush("Output Copy");
//         src[process_len * numDMs].x = 1;
//         // for (int i = 0; i < numDMs; i++)
//         // {
//         //     fwrite(src + i * process_len, sizeof(Complex), process_len, fp[i]);
//         // }
//         for (int i = 0; i < numDMs; i++)
//         {
//             outfile[i].write((char *)src + i * process_len * sizeof(Complex), process_len * sizeof(Complex));
//         }
//         // cout << "test" << endl;
//         // outfile.write((char *)src, numDMs * process_len * sizeof(Complex));
//         // outfile.flush();
//         // memcpy(dst, src, process_len * numDMs * sizeof(Complex));
//         // memcpy(dst + count * process_len * numDMs, src, process_len * numDMs * sizeof(Complex));
//         count++;
//         msosm->wait_for_cpu = false;
//         nvtxRangePop();
//     }
// }

void output_thread_file(uint16_pair *src, unsigned long process_len, int num_process, int numDMs, MSOSM_GPU_DM_concurrent *msosm)
{
    src[process_len * numDMs].first = 1;
    // Get the current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    // Convert to tm structure for local time
    std::tm now_tm = *std::localtime(&now_time_t);

    // Use a stringstream to format the date and time
    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S");

    ofstream outfile[numDMs];
    // Open file in binary write mode
    for (int i = 0; i < numDMs; i++)
    {
        // outfile[i].open("/media/joe/MyFile/" + oss.str() + "_" + to_string(i) + ".bin", ios::binary | ios::out);
        outfile[i].open(oss.str() + "_" + to_string(i) + ".bin", ios::binary | ios::out);
    }
    int count = 0;
    bool flag = false;
    while (count < num_process)
    {
        while (src[process_len * numDMs].first == 1)
            ;
        nvtxRangePush("Output Copy");
        src[process_len * numDMs].first = 1;
        // for (int i = 0; i < numDMs; i++)
        // {
        //     outfile[i].write((char *)src + i * process_len * sizeof(uint16_pair), process_len * sizeof(uint16_pair));
        // }
        count++;
        msosm->wait_for_cpu = false;
        nvtxRangePop();
    }
}

int main(int argc, char *argv[])
{
    verbose = false;
    int numDMs = 1;
    int count = 8;
    float bw = 128e6;
    float dm = 615.566;
    float f0 = 1e9;
    unsigned long fftpoint = 0;
    const struct option long_options[] = {
        {"verbose", no_argument, nullptr, 'v'},
        {"batch", required_argument, nullptr, 'b'},
        {"channel", required_argument, nullptr, 'c'},
        {"fftpoint", required_argument, nullptr, 'n'},
        {"bw", required_argument, nullptr, 'w'},
        {"dm", required_argument, nullptr, 'd'},
        {"f0", required_argument, nullptr, 'f'},
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
        case 'n':
            fftpoint = stoul(optarg);
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
        default:
            continue;
        case -1:
            break;
        }
        break;
    }

    // Pulsar signal parameters
    const int inputSize = 16;
    unsigned long block_size = 8388608 * 2;
    float period = (float)block_size / bw;
    // Batch Size
    cout << "Batch Size: " << count << endl;
    cout << "Number of Streams: " << numDMs << endl;

    // Generate simulated complex signal
    SimulatedComplexSignal simulated_signal(bw, dm, f0, period, "uint16");
    simulated_signal.generate_pulsar_signal(inputSize, false);
    unsigned long signal_size = simulated_signal.signal_size;

    float *bw_i, *DM_i, *f0_i;
    bw_i = new float[numDMs];
    DM_i = new float[numDMs];
    f0_i = new float[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        bw_i[i] = bw;
        DM_i[i] = dm - i * 0.1;
        DM_i[i] = dm;
        f0_i[i] = f0;
    }

    // Initialize MSOSM_GPU_concurrent object
    MSOSM_GPU_DM_concurrent msosm(bw_i, DM_i, f0_i, numDMs);
    msosm.initialize_uint16(fftpoint, count);

    cout << "Filter length: " << msosm.M_common << endl;
    cout << "Delay count: " << msosm.delaycount[0] << endl;

    unsigned long M = msosm.M_common;
    unsigned long process_len = count * M;

    cout << "Signal Size: " << signal_size << endl;
    cout << "Process Size: " << process_len << endl;

    // Initialize input CPU memory space
    uint16_pair *input;
    input = simulated_signal.signal_u16;

    // Initialize output CPU memory space
    // Complex *output_check;
    // if (process_len * sizeof(Complex) * numDMs / 1024 / 1024 / 1024 > 16)
    // {
    //     cout << "Memory Size Exceeds 16GB" << endl;
    //     exit(1);
    // }
    // cudaMallocHost((void **)&output_check, numDMs * process_len * sizeof(Complex));

    Complex *output_check;
    if (signal_size * sizeof(Complex) * numDMs / 1024 / 1024 / 1024 > 16)
    {
        cout << "Memory Size Exceeds 16GB" << endl;
        exit(1);
    }
    cudaMallocHost((void **)&output_check, numDMs * signal_size * sizeof(Complex));

    Complex *output;
    output = (Complex *)malloc((numDMs * process_len + 1) * sizeof(Complex));
    if (output == NULL)
    {
        cout << "Memory Allocation Failed" << endl;
        exit(1);
    }
    cudaError_t error;
    error = cudaHostRegister(output, (numDMs * process_len + 1) * sizeof(Complex), cudaHostRegisterDefault);
    if (error != cudaSuccess)
    {
        cout << "Host Memory Registration Failed" << endl;
        exit(1);
    }

    // uint16_pair *output;
    // output = (uint16_pair *)malloc((numDMs * process_len + 1) * sizeof(Complex));
    // if (output == NULL)
    // {
    //     cout << "Memory Allocation Failed" << endl;
    //     exit(1);
    // }
    // cudaError_t error;
    // error = cudaHostRegister(output, (numDMs * process_len + 1) * sizeof(uint16_pair), cudaHostRegisterDefault);
    // if (error != cudaSuccess)
    // {
    //     cout << "Host Memory Registration Failed" << endl;
    //     exit(1);
    // }

    uint16_pair *current_input;

    cout << "Start Processing" << endl;

    vector<thread> threads;

    threads.emplace_back(output_thread, output_check, output, process_len, signal_size / process_len, numDMs, &msosm);
    // threads.emplace_back(output_thread_file, output, process_len, signal_size / process_len, numDMs, &msosm);
    sleep(1);

    cout << "Output Bytes: " << signal_size * sizeof(Complex) * numDMs << endl;

    // Start the timer
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < signal_size / process_len; i++)
    {
        current_input = input + i * process_len;
        msosm.filter_block_uint16(current_input);
        msosm.get_output(output);
    }

    for (auto &t : threads)
    {
        t.join();
    }
    msosm.synchronize();

    // Stop the timer
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    cout << "Time taken (DM trial num: " << numDMs << " " << "Batch Size: " << count << "): " << duration.count() / 1000000.0 << " ms" << endl;
    cout << "Tansfer speed requirement: " << signal_size * sizeof(Complex) * numDMs / (duration.count() / 1000000000.0) / 1024.0 / 1024.0 / 1024.0 * 8.0 << " Gbps" << endl;

    auto timedata = 1.0 / bw * signal_size * 1000;
    cout << "Real-time data time: " << timedata << " ms" << endl;

    // plot_abs_concurrent_all(output_check, process_len, block_size, numDMs);
    // // // plot_abs_concurrent(output_check+49*block_size*numDMs, process_len, block_size, numDMs, 1);
    // show();
    
    // For single DM test
    float *data = new float[block_size];
    for (int i = 0; i < block_size; i++)
    {
        data[i] = sqrt(pow(output_check[i].x, 2) + pow(output_check[i].y, 2));
    }

    ofstream outfile("data.bin", ios::binary | ios::out);
    outfile.write((char *)data, block_size * sizeof(float));
    outfile.close();
    return 0;
}