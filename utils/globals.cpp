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

template void plot<float>(float*, unsigned long);
template void plot<unsigned long>(unsigned long*, unsigned long);