#include "globals.h"

bool verbose = false;

void banner()
{
    cout << "    ╔╦╗╔═╗╔═╗╔═╗╔╦╗" << endl;
    cout << "    ║║║╚═╗║ ║╚═╗║║║" << endl;
    cout << "    ╩ ╩╚═╝╚═╝╚═╝╩ ╩" << endl;
}

void line()
{
    cout << "------------------------------------------------------------" << endl;
}

void plot_abs(uint16_pair *signal, unsigned long size)
{
    vector<float> v_signal(size);
    float first, second;
    for (unsigned long i = 0; i < size; i++)
    {
        first = (signal[i].first == 0) ? 0.0f : static_cast<float>(signal[i].first) - 32768.0f;
        second = (signal[i].second == 0) ? 0.0f : static_cast<float>(signal[i].second) - 32768.0f;
        v_signal.at(i) = sqrt(pow(first, 2) + pow(second, 2));
    }
    plt::plot(v_signal);
}

void plot_abs(Complex *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        v_signal.at(i) = sqrt(pow(signal[i].x, 2) + pow(signal[i].y, 2));
    }
    plt::plot(v_signal);
}

void show()
{
    plt::show();
    plt::detail::_interpreter::kill();
}

void plot_pol1(uint16_pair *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        uint16_t tmp = signal[i].first;
        v_signal.at(i) = (tmp == 0) ? 0.0f : static_cast<float>(tmp) - 32768.0f;
    }
    plt::plot(v_signal);
}

void plot_pol2(uint16_pair *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        uint16_t tmp = signal[i].second;
        v_signal.at(i) = (tmp == 0) ? 0.0f : static_cast<float>(tmp) - 32768.0f;
    }
    plt::plot(v_signal);
}

template <typename T>
void plot(T *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        v_signal.at(i) = static_cast<float>(signal[i]);
    }
    plt::plot(v_signal);
}

template void plot<float>(float *, unsigned long);
template void plot<unsigned long>(unsigned long *, unsigned long);

void plot_abs_concurrent_all(Complex *signal, int block_size, unsigned long size, int numDMs)
{
    vector<vector<float>> v_signal(numDMs, vector<float>(size));
    int cycles = size / block_size;
    Complex tmp;
    for (int i = 0; i < cycles; i++)
    {
        for (int j = 0; j < numDMs; j++)
        {
            for (int k = 0; k < block_size; k++)
            {
                tmp = signal[i * (block_size * numDMs) + j * block_size + k];
                v_signal.at(j).at(i * block_size + k) = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2));
            }
        }
    }
    if (cycles == 0)
    {
        for (int i = 0; i < numDMs; i++)
        {
            for (int j = 0; j < size; j++)
            {
                tmp = signal[i * block_size + j];
                v_signal.at(i).at(j) = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2));
            }
        }
    }
    for (int i = 0; i < numDMs; i++)
    {
        plt::plot(v_signal.at(i));
    }
}

void plot_abs_concurrent(Complex *signal, unsigned long block_size, unsigned long size, int numDMs, int index)
{
    vector<float> v_signal(size);
    int cycles = size / block_size;
    Complex tmp;
    for (int i = 0; i < cycles; i++)
    {
        for (int k = 0; k < block_size; k++)
        {
            tmp = signal[i * (block_size * numDMs) + index * block_size + k];
            v_signal.at(i * block_size + k) = sqrt(pow(tmp.x, 2) + pow(tmp.y, 2));
        }
    }
    plt::plot(v_signal);
}
