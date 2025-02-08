#include "prepare_concurrent.h"

Prepare_OSM_DM_concurrent::Prepare_OSM_DM_concurrent(float *bw, float *dm, float *f0, int numDMs) : bw(bw), dm(dm), f0(f0), numDMs(numDMs)
{
    fs = bw;
    kdm = 4.15 * 1e15;
    w0 = new float[numDMs];
    Nd = new unsigned long[numDMs];
    for (int i = 0; i < numDMs; i++)
    {
        w0[i] = 2 * pi * f0[i];
        Nd[i] = static_cast<unsigned long>(floor(kdm * dm[i] * (1 / pow(f0[i], 2) - 1 / pow(f0[i] + bw[i], 2)) / (1 / fs[i])));
    }
    if (verbose)
    {
        cout << "Prepare parameters..." << endl;
        for (int i = 0; i < numDMs; i++)
        {
            cout << "Sampling frequency " << i << ": " << fs[i] << endl;
            cout << "Start frequency " << i << ": " << f0[i] << endl;
            cout << "Dispersion smearing points " << i << ": " << Nd[i] << endl;
        }
    }
    M = new unsigned long[numDMs];
    M_min = new unsigned long[numDMs];
    dedisp_params = new fftwf_complex *[numDMs];
}

void Prepare_OSM_DM_concurrent::calculate_min_order()
{
    // Overlap-Save method
    // Ensure FFT length is 2 * log2(Nd).
    for (int i = 0; i < numDMs; i++)
    {
        int order;
        order = next_power_of_2(Nd[i]);
        M[i] = 2 * order;
        M_min[i] = M[i];
        if (verbose)
        {
            cout << "Minimum filter order " << i << ": " << M_min[i] << endl;
        }
        if (M[i] > M_common)
            M_common = M[i];
    }
    M_common_min = M_common;
}

void Prepare_OSM_DM_concurrent::override_order(unsigned long order)
{
    M_common = order;
}

void Prepare_OSM_DM_concurrent::generate_dedisp_params(int i)
{
    // Dispersion filter frequency response
    float *w;
    w = (float *)fftwf_malloc(sizeof(float) * M_common);
    float step = 2 * pi * fs[i] / M_common;
    unsigned long fftpoint = 2 * M_common;
    fftwf_complex *H;
    H = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * fftpoint);

    fftwf_complex *h;
    h = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * M_common);

    for (unsigned long j = 0; j < M_common / 2; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[M_common / 2 + j][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[M_common / 2 + j][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }
    for (unsigned long j = M_common / 2; j < M_common; j++)
    {
        w[j] = -pi * fs[i] + step * j;
        w[j] = w[j] + pi * fs[i];
        h[j - M_common / 2][0] = cos(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
        h[j - M_common / 2][1] = -sin(4 * pi * pi * kdm * dm[i] * w[j] * w[j] / (w[j] + w0[i]) / w0[i] / w0[i]);
    }

    fftwf_plan p;
    p = fftwf_plan_dft_1d(M_common, h, h, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    for (unsigned long j = 0; j < M_common; j++)
    {
        H[j][0] = h[j][0] / M_common;
        H[j][1] = h[j][1] / M_common;
    }

    memcpy(H, h, sizeof(fftwf_complex) *M_common);
    fftwf_free(h);
    fftwf_free(w);

    memset(H + M_common, 0, sizeof(fftwf_complex) * (fftpoint - M_common));
    p = fftwf_plan_dft_1d(fftpoint, H, H, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    dedisp_params[i] = H;
}

int Prepare_OSM_DM_concurrent::next_power_of_2(int n)
{
    if (n <= 0)
    {
        return 1;
    }
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}
