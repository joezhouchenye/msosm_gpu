#include "prepare.h"

Prepare_MSOSM_DM_loop::Prepare_MSOSM_DM_loop(float *bw, float *dm, float *f0, int numOutput) : bw(bw), dm(dm), f0(f0), numOutput(numOutput)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = new float[numOutput];
    Nd = new unsigned long[numOutput];
    for (int i = 0; i < numOutput; i++)
    {
        w0[i] = 2 * pi * f0[i];
        Nd[i] = static_cast<unsigned long>(floor(kdm * dm[i] * (1 / pow(f0[i], 2) - 1 / pow(f0[i] + bw[i], 2)) / (1 / fs[i])));
    }
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
        for (int i = 0; i < numOutput; i++)
        {
            cout << "Sampling frequency " << i << ": " << fs[i] << endl;
            cout << "Start frequency " << i << ": " << f0[i] << endl;
            cout << "Dispersion smearing points " << i << ": " << Nd[i] << endl;
        }
    }
    M = new unsigned long[numOutput];
    M_min = new unsigned long[numOutput];
    delaycount = new int[numOutput];
    dedisp_params = new fftwf_complex *[numOutput];
    delay_points = new int *[numOutput];
}

void Prepare_MSOSM_DM_loop::calculate_min_order()
{
    // Multi-segment Overlap-Save method
    // Ensure each segment at least has 8 points.
    for (int i = 0; i < numOutput; i++)
    {
        double order;
        order = 2 * sqrt(Nd[i]);
        M[i] = static_cast<unsigned long>(pow(2, ceil(log2(order))));
        M_min[i] = M[i];
        if (verbose)
        {
            cout << "Minimum filter order " << i << ": " << M_min[i] << endl;
        }
        if (M[i] > M_loop)
            M_loop = M[i];
    }
    M_loop_min = M_loop;
}

void Prepare_MSOSM_DM_loop::override_order(unsigned long order)
{
    M_loop = order;
}

void Prepare_MSOSM_DM_loop::generate_dedisp_params(int i)
{
    // Dispersion filter frequency response
    float *w;
    w = (float *)fftwf_malloc(sizeof(float) * M_loop);
    float step = 2 * pi * fs[i] / M_loop;
    unsigned long fftpoint = 2 * M_loop;
    fftwf_complex *H;
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    fftwf_complex *h;
    h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M_loop);

    for (unsigned long j = 0; j < M_loop / 2; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[M_loop / 2 + j][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[M_loop / 2 + j][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }
    for (unsigned long j = M_loop / 2; j < M_loop; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[j - M_loop / 2][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[j - M_loop / 2][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }

    fftwf_plan p;
    p = fftwf_plan_dft_1d(M_loop, h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    for (unsigned long j = 0; j < M_loop; j++)
    {
        H[j][0] = h[j][0] / M_loop;
        H[j][1] = h[j][1] / M_loop;
    }

    memcpy(H, h, sizeof(fftwf_complex) *M_loop);
    fftwf_free(h);
    fftwf_free(w);

    memset(H + M_loop, 0, sizeof(fftwf_complex) * (fftpoint - M_loop));
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params[i] = H;
}

void Prepare_MSOSM_DM_loop::segmentation(int i)
{
    delaycount[i] = static_cast<int>(ceil(Nd[i] / static_cast<float>(M_loop)));
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
        w[j] = (sqrt(1 / (1 / pow(w0[i], 2) - j * M_loop / fs[i] / (4 * pi * pi * kdm * dm[i]))) - w0[i] - pi * fs[i]) / pi / fs[i];
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
        segneg[j] = static_cast<unsigned long>(floor((w[j] + 2) * M_loop));
    }
    segneg[negcount] = 2 * M_loop;
    segpos[0] = 0;
    for (unsigned long j = 0; j < poscount; j++)
    {
        segpos[j + 1] = static_cast<unsigned long>(floor(w[j + negcount] * M_loop));
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
    delay_points[i] = (int *)malloc(sizeof(int) * 2 * M_loop);
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