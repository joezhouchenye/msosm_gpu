#include "prepare.h"

Prepare_MSOSM::Prepare_MSOSM(float bw, float dm, float f0) : bw(bw), dm(dm), f0(f0)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = 2 * pi * f0;
    Nd = static_cast<unsigned long>(floor(kdm * dm * (1 / pow(f0, 2) - 1 / pow(f0 + bw, 2)) / (1 / fs)));
    if (verbose)
    {
        cout << "Sampling frequency: " << fs << endl;
        cout << "Start frequency: " << f0 << endl;
        cout << "Dispersion smearing points: " << Nd << endl;
    }
}

void Prepare_MSOSM::calculate_min_order()
{
    // Multi-segment Overlap-Save method
    // Ensure each segment at least has 8 points.
    double order;
    order = 2 * sqrt(Nd);
    M = static_cast<unsigned long>(pow(2, ceil(log2(order))));
    M_min = M;
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
        cout << "Minimum filter order: " << M_min << endl;
    }
}

void Prepare_MSOSM::override_order(unsigned long order)
{
    M = order;
}

void Prepare_MSOSM::generate_dedisp_params()
{
    // Dispersion filter frequency response
    float *w;
    w = (float *)fftwf_malloc(sizeof(float) * M);
    float step = 2 * pi * fs / M;
    unsigned long fftpoint = 2 * M;
    fftwf_complex *H;
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    fftwf_complex *h;
    h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M);

    for (unsigned long i = 0; i < M / 2; i++)
    {
        w[i] = -pi * fs + step * i;
        w[i] = w[i] + pi * fs;
        h[M / 2 + i][0] = cos(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
        h[M / 2 + i][1] = -sin(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
    }
    for (unsigned long i = M / 2; i < M; i++)
    {
        w[i] = -pi * fs + step * i;
        w[i] = w[i] + pi * fs;
        h[i - M / 2][0] = cos(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
        h[i - M / 2][1] = -sin(4 * pi * pi * kdm * dm * w[i] * w[i] / (w[i] + w0) / w0 / w0);
    }

    fftwf_plan p;
    p = fftwf_plan_dft_1d(M, h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    for (unsigned long i = 0; i < M; i++)
    {
        H[i][0] = h[i][0] / M;
        H[i][1] = h[i][1] / M;
    }

    memcpy(H, h, sizeof(fftwf_complex) * M);
    fftwf_free(h);
    fftwf_free(w);

    memset(H + M, 0, sizeof(fftwf_complex) * (fftpoint - M));
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params = H;
}

void Prepare_MSOSM::segmentation()
{
    delaycount = static_cast<int>(ceil(Nd / static_cast<float>(M)));
    // delaycount = 1;
    if (verbose)
    {
        cout << "Delaycount: " << delaycount << endl;
        line();
    }
    float *w;
    w = (float *)malloc(sizeof(float) * (delaycount + 1));
    w[0] = -1;
    w[delaycount] = 1;
    int pos_index = 1;
    for (int i = 1; i < delaycount; i++)
    {
        w[i] = (sqrt(1 / (1 / pow(w0, 2) - i * M / fs / (4 * pi * pi * kdm * dm))) - w0 - pi * fs) / pi / fs;
        if (w[i] < 0)
            pos_index = i + 1;
    }
    negcount = pos_index;
    poscount = delaycount - pos_index + 1;
    segneg = (unsigned long *)malloc(sizeof(unsigned long) * (negcount + 1));
    segpos = (unsigned long *)malloc(sizeof(unsigned long) * (poscount + 1));
    delayneg = (int *)malloc(sizeof(int) * negcount);
    delaypos = (int *)malloc(sizeof(int) * poscount);
    for (unsigned long i = 0; i < negcount; i++)
    {
        segneg[i] = static_cast<unsigned long>(floor((w[i] + 2) * M));
    }
    segneg[negcount] = 2 * M;
    segpos[0] = 0;
    for (unsigned long i = 0; i < poscount; i++)
    {
        segpos[i + 1] = static_cast<unsigned long>(floor(w[i + negcount] * M));
    }
    for (int i = 0; i < negcount; i++)
    {
        delayneg[i] = i;
    }
    for (int i = 0; i < poscount; i++)
    {
        delaypos[i] = delayneg[negcount - 1] + i;
    }
    free(w);

    // Delay of each FFT point
    delay_points = (int *)malloc(sizeof(int) * 2 * M);
    for (unsigned long j = 0; j < negcount; j++)
    {
        for (unsigned long k = segneg[j]; k < segneg[j + 1]; k++)
        {
            delay_points[k] = delayneg[j];
        }
    }
    for (unsigned long j = 0; j < poscount; j++)
    {
        for (unsigned long k = segpos[j]; k < segpos[j + 1]; k++)
        {
            delay_points[k] = delaypos[j];
        }
    }
    free(delayneg);
    free(delaypos);
    free(segneg);
    free(segpos);
}