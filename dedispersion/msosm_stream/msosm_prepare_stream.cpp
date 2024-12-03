#include "prepare_stream.h"

Prepare_MSOSM_stream::Prepare_MSOSM_stream(float *bw, float *dm, float *f0, int numStreams) : bw(bw), dm(dm), f0(f0), numStreams(numStreams)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = new float[numStreams];
    Nd = new unsigned long[numStreams];
    for (int i = 0; i < numStreams; i++)
    {
        w0[i] = 2 * pi * f0[i];
        Nd[i] = static_cast<unsigned long>(floor(kdm * dm[i] * (1 / pow(f0[i], 2) - 1 / pow(f0[i] + bw[i], 2)) / (1 / fs[i])));
    }
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
        for (int i = 0; i < numStreams; i++)
        {
            cout << "Sampling frequency " << i << ": " << fs[i] << endl;
            cout << "Start frequency " << i << ": " << f0[i] << endl;
            cout << "Dispersion smearing points " << i << ": " << Nd[i] << endl;
        }
    }
    M = new unsigned long[numStreams];
    M_min = new unsigned long[numStreams];
    delaycount = new int[numStreams];
    dedisp_params = new fftwf_complex *[numStreams];
    delay_points = new int *[numStreams];
}

void Prepare_MSOSM_stream::calculate_min_order()
{
    // Multi-segment Overlap-Save method
    // Ensure each segment at least has 8 points.
    for (int i = 0; i < numStreams; i++)
    {
        double order;
        order = 2 * sqrt(Nd[i]);
        M[i] = static_cast<unsigned long>(pow(2, ceil(log2(order))));
        M_min[i] = M[i];
        if (verbose)
        {
            cout << "Minimum filter order " << i << ": " << M_min[i] << endl;
        }
    }
}

void Prepare_MSOSM_stream::override_order(unsigned long order, int i)
{
    M[i] = order;
}

void Prepare_MSOSM_stream::generate_dedisp_params(int i)
{
    // Dispersion filter frequency response
    float *w;
    w = (float *)fftwf_malloc(sizeof(float) * M[i]);
    float step = 2 * pi * fs[i] / M[i];
    unsigned long fftpoint = 2 * M[i];
    fftwf_complex *H;
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    fftwf_complex *h;
    h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M[i]);

    for (unsigned long j = 0; j < M[i] / 2; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[M[i] / 2 + j][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[M[i] / 2 + j][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }
    for (unsigned long j = M[i] / 2; j < M[i]; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[j - M[i] / 2][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[j - M[i] / 2][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }

    fftwf_plan p;
    p = fftwf_plan_dft_1d(M[i], h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    for (unsigned long j = 0; j < M[i]; j++)
    {
        H[j][0] = h[j][0] / M[i];
        H[j][1] = h[j][1] / M[i];
    }

    memcpy(H, h, sizeof(fftwf_complex) * M[i]);
    fftwf_free(h);
    fftwf_free(w);

    memset(H + M[i], 0, sizeof(fftwf_complex) * (fftpoint - M[i]));
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params[i] = H;
}

void Prepare_MSOSM_stream::segmentation(int i)
{
    delaycount[i] = static_cast<int>(ceil(Nd[i] / static_cast<float>(M[i])));
    if (verbose)
    {
        cout << "Delaycount " << i << ": " << delaycount[i] << endl;
    }
    float *w;
    w = (float *)malloc(sizeof(float) * (delaycount[i] + 1));
    w[0] = -1;
    w[delaycount[i]] = 1;
    int pos_index = 1;
    for (int j = 1; j < delaycount[i]; j++)
    {
        w[j] = (sqrt(1 / (1 / pow(w0[i], 2) - j * M[i] / fs[i] / (4 * pi * pi * kdm * dm[i]))) - w0[i] - pi * fs[i]) / pi / fs[i];
        if (w[j] < 0)
            pos_index = j + 1;
    }
    negcount = pos_index;
    poscount = delaycount[i] - pos_index + 1;
    segneg = (unsigned long *)malloc(sizeof(unsigned long) * (negcount + 1));
    segpos = (unsigned long *)malloc(sizeof(unsigned long) * (poscount + 1));
    delayneg = (int *)malloc(sizeof(int) * negcount);
    delaypos = (int *)malloc(sizeof(int) * poscount);
    for (unsigned long j = 0; j < negcount; j++)
    {
        segneg[j] = static_cast<unsigned long>(floor((w[j] + 2) * M[i]));
    }
    segneg[negcount] = 2 * M[i];
    segpos[0] = 0;
    for (unsigned long j = 0; j < poscount; j++)
    {
        segpos[j + 1] = static_cast<unsigned long>(floor(w[j + negcount] * M[i]));
    }
    for (int j = 0; j < negcount; j++)
    {
        delayneg[j] = j;
    }
    for (int j = 0; j < poscount; j++)
    {
        delaypos[j] = delayneg[negcount - 1] + j;
    }
    free(w);

    // Delay of each FFT point
    delay_points[i] = (int *)malloc(sizeof(int) * 2 * M[i]);
    for (unsigned long j = 0; j < negcount; j++)
    {
        for (unsigned long k = segneg[j]; k < segneg[j + 1]; k++)
        {
            delay_points[i][k] = delayneg[j];
        }
    }
    for (unsigned long j = 0; j < poscount; j++)
    {
        for (unsigned long k = segpos[j]; k < segpos[j + 1]; k++)
        {
            delay_points[i][k] = delaypos[j];
        }
    }
    free(delayneg);
    free(delaypos);
    free(segneg);
    free(segpos);
}