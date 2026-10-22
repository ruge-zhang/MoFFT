# Linux SME1 benchmark

The accepted CPU measurement compares MoFFT with a fixed FFTW 3.3.11 build: all 54 precision/length conditions, 30 batches per implementation and 3,240 retained timing samples. It measures a complex-to-complex (C2C) fast Fourier transform (FFT), not an application pipeline.

## CPU and build configuration

| Component | Observed or fixed configuration |
|---|---|
| CPU | AArch64; MIDR `0x00000000480fd220` |
| Logical CPUs | 608 exposed by Linux; experiment uses one thread |
| Cache at the measured CPU | 32 KiB L1 data, 32 KiB L1 instruction, 768 KiB unified L2; 64-byte lines |
| Vector lengths | Scalable Vector Extension (SVE) and SME: 64 bytes / 512 bits; SME1 path |
| Frequency | `userspace` governor; approximately 1.55 GHz in before/after observations, not a continuous trace |
| OS | Linux `5.10.0-285.6.v2403.ky10p.aarch64` |
| Toolchain | BiSheng Enterprise 5.2.0 / Clang 19.1.7; CMake 3.28.2; Python 3.12.7 |
| Placement | One pinned CPU, one thread, whole-machine exclusive; 1 CPU / 4096 MB requested |
| FFTW | Static 3.3.11; FP32 with NEON, FP64 with NEON disabled; no OpenMP or MPI |
| Transform | FP32/FP64, forward, out-of-place C2C; real input `sin(0.01*i)`, imaginary `cos(0.013*i)` |
| Qualification | CPU calibration and complete plan search accepted; final library passed 54 numerical conditions and 216 state transitions |

The logical-CPU count is not a measured physical-core count. Cache entries are CPU observations, not proof of timed working-set residency. Results describe this fixed FFTW build, not every FFTW instruction path or tuning option.

## Timing and statistics

The timer surrounds execute calls; MoFFT execution-status checks remain inside. Input preparation, plan creation, repetition calibration, warmup and destruction are outside. FFTW uses `FFTW_MEASURE`. Warmup requires at least 1,000 executions with a 0.25-second target. The two implementations alternate first position; all 30 batches per group are retained.

Throughput is `5*N*log2(N)/median_seconds/1e9` GFLOP/s. Error bars invert the original 95% per-library median bootstrap intervals from 2,000 MT19937_64 resamples. They are not paired speedup intervals or familywise intervals across lengths.

The coefficient of variation (CV) is population standard deviation divided by mean. All 108 groups passed the unchanged 5% threshold; maximum CV was 2.345843%. Whole-system load above 2.0 is warning-only. CPU identity, affinity, exclusivity, vector length, numerical correctness and sample-quality checks remain required. Linux does not apply Apple's QoS/thermal interfaces.

Four panels show FP64 powers/non-powers and FP32 powers/non-powers. Lengths are equally spaced categories. Each precision displays 26 predefined lengths, producing 104 points across both libraries. Additional accepted `N=144` samples remain in the full data but outside the predefined display, independently of results.

## Replot the accepted results

- [Original accepted PNG](../assets/overall-lingsheng-linux.png), copied byte-for-byte.
- [Compact raw samples and frozen statistics](../benchmarks/results/lingsheng-linux-reference.json), with all 108 groups and 3,240 original-precision samples.
- [Plot data](../benchmarks/results/lingsheng-linux-reference-figure.csv) and [SHA256 manifest](../benchmarks/results/lingsheng-linux-reference.manifest.json).
- [Standalone replotter](../benchmarks/plot_linux_overall.py), with no FFT execution or cluster connection.

With Python 3.10 or newer, from the MoFFT checkout root:

```sh
python3 benchmarks/plot_linux_overall.py --check-only
```

With Matplotlib installed (validated with 3.9.2), in the same environment and checkout root:

```sh
python3 benchmarks/plot_linux_overall.py --output build/figures/linux-reference
```

Output is PNG/SVG/PDF, the 104-point CSV and plotting provenance. Existing directories are refused. The script checks digests and coverage, then uses frozen medians/intervals without new statistics. Font/backend differences can change image bytes; compare the data for numerical identity.

## Repeat from source

The complete [workflow instructions](linux-sme1-workflow.md) cover a new workspace: fixed FFTW source build, source-built probes/candidates, CPU calibration, kernel wisdom, plan search, final-library checks, timing and login analysis. The unmeasured bootstrap and original Apple commands do not replace this sequence.

The historical measured implementation was based on adaptation commit `006dd05` plus the SME1-specific memory cost keys and registry syntax correction now included here. New source/toolchain/CPU combinations require their own calibration and matching wisdom. Never relabel old digests to bypass validation. Packaging/replotting does not create a new hardware measurement.

| Build setting | Value |
|---|---|
| Generator / type | Unix Makefiles / Release |
| Port and tools | `MOFFT_SME1_PORT=ON`, `MOFFT_BUILD_TESTS=OFF`, `MOFFT_BUILD_TOOLS=OFF` |
| Tuning | Explicit accepted `MOFFT_PROFILE`, `MOFFT_KERNEL_WISDOM`, `MOFFT_WISDOM` and actual `MOFFT_WISDOM_MACHINE` |
| Capture | C++17, `-O3`, `--rtlib=compiler-rt -lgcc_s -lm`; static MoFFT and both FFTW precisions |
| FFTW common | `--disable-shared --enable-static --disable-fortran --disable-openmp --disable-mpi`; Clang, `CFLAGS='-O3 -fPIC'` |
| FFTW precision | FP32 `--enable-float --enable-neon`; FP64 `--disable-neon` |

Published reference data omit cluster-machine names, scheduler identifiers, accounts and internal paths. Private execution records remain separate. SME2 hardware numerical/performance validation requires an SME2 device.
