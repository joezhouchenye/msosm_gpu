# Multi-Segment Overlap-Save Method for Pulsar Coherent Dedispersion

## Usage:

```
msosm CLI:
--verbose Verbose mode
--file    Input PSRDADA file

If simulated data is used:
--bw      Signal bandwith in Hz (default: 16e6)
--dm      Dispersion measure in pc cm^-3 (default: 750)
--f0      Band start frequency in Hz (default: 1e9)
--period  Signal period in s (default: 0.024)
--snr     Add gaussian noise to achieve an SNR in dB
--show    Plot the results
```

## Custom Class Usage:

- GPU version: class MSOSM_GPU in `msosm_gpu.h`
- CPU version: class MSOSM_CPU in `msosm_cpu.h`