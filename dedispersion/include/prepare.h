#pragma once

#include "globals.h"

using namespace std;

class Prepare_MSOSM
{
public:
    Prepare_MSOSM(float bw = 0, float dm = 0, float f0 = 0);
    void calculate_min_order();
    void override_order(unsigned long order);
    void generate_dedisp_params();
    void segmentation();

public:
    // Filter order for the Multi-segment Overlap-Save method
    unsigned long M;
    // Minimum filter order for the Multi-segment Overlap-Save method
    unsigned long M_min;
    // Segment indexes for the negative and positive frequency components
    unsigned long *segneg, *segpos;
    // Segment delays for the negative and positive frequency components
    int *delayneg, *delaypos;
    // Number of segments for the negative and positive frequency components
    int negcount, poscount;
    // Number of delays needed
    int delaycount;
    // Dedispersion parameters
    fftwf_complex *dedisp_params;
    // Delay of each FFT point
    int *delay_points;
    // Dispersion smearing points
    unsigned long Nd;

private:
    // Bandwidth
    float bw;
    // Sampling frequency
    float fs;
    // Dispersion constant
    float kdm;
    // Dispersion measure
    float dm;
    // Start frequency of the band
    float f0;
    // Start angular frequency
    float w0;
};

class Prepare_MSOSM_DM_loop
{
public:
    Prepare_MSOSM_DM_loop(float *bw, float *dm, float *f0, int numOutput);
    void calculate_min_order();
    void override_order(unsigned long order);
    void generate_dedisp_params(int i);
    void segmentation(int i);

public:
    // Filter order for the Multi-segment Overlap-Save method
    unsigned long *M;
    // Minimum filter order for the Multi-segment Overlap-Save method
    unsigned long *M_min;
    // Common filter order for all streams
    unsigned long M_loop = 0;
    // Common minimum filter order for all streams
    unsigned long M_loop_min = 0;
    // Segment indexes for the negative and positive frequency components
    unsigned long *segneg, *segpos;
    // Segment delays for the negative and positive frequency components
    int *delayneg, *delaypos;
    // Number of segments for the negative and positive frequency components
    int negcount, poscount;
    // Number of delays needed
    int *delaycount;
    // Dedispersion parameters
    fftwf_complex **dedisp_params;
    // Delay of each FFT point
    int **delay_points;
    // Dispersion smearing points
    unsigned long *Nd;
    // Number of streams
    int numOutput;

private:
    // Bandwidth
    float *bw;
    // Sampling frequency
    float *fs;
    // Dispersion constant
    float kdm;
    // Dispersion measure
    float *dm;
    // Start frequency of the band
    float *f0;
    // Start angular frequency
    float *w0;
};

class Prepare_OSM
{
public:
    Prepare_OSM(float bw = 0, float dm = 0, float f0 = 0);
    void calculate_min_order();
    void override_order(unsigned long order);
    void generate_dedisp_params();

private:
    int next_power_of_2(int n);

public:
    // Filter order for the Overlap-Save method
    unsigned long M;
    // Minimum filter order for the Overlap-Save method
    unsigned long M_min;
    // Segment indexes for the negative and positive frequency components
    fftwf_complex *dedisp_params;
    // Dispersion smearing points
    unsigned long Nd;

private:
    // Bandwidth
    float bw;
    // Sampling frequency
    float fs;
    // Dispersion constant
    float kdm;
    // Dispersion measure
    float dm;
    // Start frequency of the band
    float f0;
    // Start angular frequency
    float w0;
};