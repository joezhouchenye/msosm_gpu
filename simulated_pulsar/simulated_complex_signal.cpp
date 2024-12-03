#include "simulated_complex_signal.h"

/**
 * @brief SimulatedComplexSignal constructor
 * @param bw Bandwidth
 * @param dm Dispersion measure
 * @param f0 Start frequency
 * @param period Pulsar period
 *
 * This constructor initializes the parameters of the simulated complex signal.
 */
SimulatedComplexSignal::SimulatedComplexSignal(float bw, float dm, float f0, float period, string mode)
{
    this->data_type = mode;

    this->bw = bw;
    this->dm = dm;
    this->f0 = f0;
    this->period = period;

    this->fs = bw;
    this->kdm = 4.15 * 1e15;
    this->w0 = 2 * pi * f0;
    this->Np = static_cast<unsigned long>(period * fs);
    this->Nd = static_cast<unsigned long>(floor(kdm * dm * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));

    if (verbose)
    {
        cout << "Test signal parameters:" << endl;
        cout << "Bandwidth: " << bw << "Hz" << endl;
        cout << "Dispersion measure: " << dm << "pc cm^-3" << endl;
        cout << "Start frequency: " << f0 << "Hz" << endl;
        cout << "Pulsar period: " << period << "s" << endl;
        cout << "Period samples: " << Np << endl;
        cout << "Dispersion samples: " << Nd << endl;
        line();
    }
}

void SimulatedComplexSignal::generate_pulsar_signal(unsigned long repeat, bool add_noise, float SNR)
{
    if (verbose)
        cout << "Generating pulsar signal" << endl;

    if (add_noise && verbose)
    {
        cout << "Noise will be added..." << endl;
        cout << "SNR: " << SNR << " dB" << endl;
        line();
    }

    signal_size = repeat * Np;
    cudaMallocHost((void **)&this->signal, sizeof(fftwf_complex) * signal_size);
    if (data_type == "uint16")
    {
        cudaMallocHost((void **)&this->signal_u16, sizeof(uint16_pair) * signal_size);
    }

    range = static_cast<unsigned long>(ceil((float)Nd / (float)Np));
    if (verbose)
    {
        cout << "Dispersion spread periods: " << range << endl;
    }
    unsigned long order = static_cast<unsigned long>(pow(2, ceil(log2(range * Np))));
    unsigned long fftpoint = 2 * order;
    if (verbose)
    {
        cout << "Using FFT points = " << fftpoint;
        cout << " (period points = " << Np << ")" << endl;
    }

    fftwf_complex *signal, *signal_fd;
    signal = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * Np);
    signal_fd = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    // Dispersion filter frequency response
    // float w[fftpoint];
    float *w;
    w = (float *)malloc(sizeof(float) * order);
    float step = 2 * pi * fs / order;
    fftwf_complex *H1, *H;
    H1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * order);
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    for (unsigned long i = 0; i < order / 2; i++)
    {
        w[i] = -pi * fs + step * i;
        w[i] = w[i] + pi * fs;
        H1[order / 2 + i][0] = cos(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
        H1[order / 2 + i][1] = sin(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
    }
    for (unsigned long i = order / 2; i < order; i++)
    {
        w[i] = -pi * fs + step * i;
        w[i] = w[i] + pi * fs;
        H1[i - order / 2][0] = cos(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
        H1[i - order / 2][1] = sin(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
    }
    fftwf_plan p;
    p = fftwf_plan_dft_1d(order, H1, H1, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    for (unsigned long i = 0; i < order; i++)
    {
        H1[i][0] = H1[i][0] / order;
        H1[i][1] = H1[i][1] / order;
    }

    memset(H, 0, sizeof(fftwf_complex) * fftpoint);
    memcpy(H + order, H1, sizeof(fftwf_complex) * order);
    fftwf_free(H1);
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    // Add dispersion
    fftwf_plan p_f, p_b;
    p_f = fftwf_plan_dft_1d(fftpoint, signal_fd, signal_fd, FFTW_FORWARD, FFTW_ESTIMATE);
    p_b = fftwf_plan_dft_1d(fftpoint, signal_fd, signal_fd, FFTW_BACKWARD, FFTW_ESTIMATE);

    memset(this->signal, 0, sizeof(fftwf_complex) * Np);

    /// Generate original signal
    memset(signal, 0, sizeof(fftwf_complex) * Np);
    unsigned long pulse_position = Np / 2;
    signal[pulse_position][0] = 1;
    signal[pulse_position + 2][0] = 1;

    for (unsigned long i = 0; i < range; i++)
    {
        memset(signal_fd, 0, sizeof(fftwf_complex) * fftpoint);
        memcpy(signal_fd + i * Np, signal, sizeof(fftwf_complex) * Np);
        fftwf_execute(p_f);
        float real, imag;
        for (unsigned long j = 0; j < fftpoint; j++)
        {
            real = signal_fd[j][0] * H[j][0] - signal_fd[j][1] * H[j][1];
            imag = signal_fd[j][0] * H[j][1] + signal_fd[j][1] * H[j][0];
            signal_fd[j][0] = real;
            signal_fd[j][1] = imag;
        }
        fftwf_execute(p_b);
        for (unsigned long j = 0; j < fftpoint; j++)
        {
            signal_fd[j][0] = signal_fd[j][0] / fftpoint;
            signal_fd[j][1] = signal_fd[j][1] / fftpoint;
        }
        for (unsigned long j = 0; j < Np; j++)
        {
            this->signal[j][0] = this->signal[j][0] + signal_fd[j][0];
            this->signal[j][1] = this->signal[j][1] + signal_fd[j][1];
        }
    }

    fftwf_destroy_plan(p_f);
    fftwf_destroy_plan(p_b);

    fftwf_free(signal);
    fftwf_free(signal_fd);
    fftwf_free(H);
    fftwf_free(w);

    signal_power = 0;
    for (unsigned long i = 0; i < Np; i++)
    {
        signal_power = signal_power + this->signal[i][0] * this->signal[i][0] + this->signal[i][1] * this->signal[i][1];
    }
    signal_power = 10 * log10(signal_power / Np);
    if (verbose)
    {
        cout << "Signal measured power: " << signal_power << " dB" << endl;
        line();
    }

    fftwf_complex *noise = NULL;
    AWGN g(SNR - signal_power, signal_size);
    if (add_noise)
    {
        noise = g.generateNoiseSamples();
    }

    if (verbose)
    {
        cout << "Repeat signal..." << endl;
        cout << "Repeat: " << repeat << endl;
        cout << "Signal Size: " << signal_size << endl;
    }

    if (!add_noise && verbose)
    {
        cout << "No noise added!" << endl;
    }

    if (verbose)
        line();

    for (unsigned long i = 0; i < repeat; i++)
    {
        if (i != 0)
        {
            memcpy(this->signal + i * Np, this->signal, sizeof(fftwf_complex) * Np);
        }
        if (!add_noise && data_type == "uint16")
        {
            for (unsigned long j = 0; j < Np; j++)
            {
                this->signal_u16[i * Np + j].first = this->signal[i * Np + j][0] == 0 ? 0 : static_cast<uint16_t>(this->signal[i * Np + j][0] * 32767.0f + 32768.0f);
                this->signal_u16[i * Np + j].second = this->signal[i * Np + j][1] == 0 ? 0 : static_cast<uint16_t>(this->signal[i * Np + j][1] * 32767.0f + 32768.0f);
            }
        }
        if (add_noise)
        {
            for (unsigned long j = 0; j < Np; j++)
            {
                this->signal[i * Np + j][0] = this->signal[i * Np + j][0] + noise[i * Np + j][0];
                this->signal[i * Np + j][1] = this->signal[i * Np + j][1] + noise[i * Np + j][1];
                if (data_type == "uint16")
                {
                    this->signal_u16[i * Np + j].first = (this->signal[i * Np + j][0] == 0) ? 0 : static_cast<uint16_t>(this->signal[i * Np + j][0] * 32767.0f + 32768.0f);
                    this->signal_u16[i * Np + j].second = (this->signal[i * Np + j][1] == 0) ? 0 : static_cast<uint16_t>(this->signal[i * Np + j][1] * 32767.0f + 32768.0f);
                }
            }
        }
    }

    if (add_noise)
    {
        g.deallocate();
    }
}

void SimulatedComplexSignal::plot_abs(const fftwf_complex *signal, unsigned long size)
{
    vector<float> v_signal(size);
    for (unsigned long i = 0; i < size; i++)
    {
        v_signal.at(i) = sqrt(pow(signal[i][0], 2) + pow(signal[i][1], 2));
    }
    plt::plot(v_signal);
    plt::show();
    cout << "Max: " << *max_element(v_signal.begin(), v_signal.end()) << endl;
}

void SimulatedComplexSignal::plot_abs(const uint16_pair *signal, unsigned long size)
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
    plt::show();
    cout << "Max: " << *max_element(v_signal.begin(), v_signal.end()) << endl;
}

SimulatedComplexSignal::~SimulatedComplexSignal()
{
    // Fix segmentation fault caused by matplotlibcpp
    // fix_matplotlibcpp();
}

void SimulatedComplexSignal::fix_matplotlibcpp()
{
    plt::detail::_interpreter::kill();
}