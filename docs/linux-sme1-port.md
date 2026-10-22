# Linux AArch64 SME1 port

This port adds an explicit Linux path for the first Arm Scalable Matrix Extension (SME). The original SME2 path remains the default; an SME2 machine is not automatically switched to SME1.

## Platform changes

| Component | Default SME2 path | Opt-in SME1 path |
|---|---|---|
| Selection | Original build arguments | `MOFFT_SME1_PORT=ON`; generator `--sme1` |
| Platform | Original Apple arm64 restrictions | Linux AArch64, 512-bit streaming vector length |
| Operations | Original grouped operations and matrix-FMA candidates | Exclude SME2 grouped operations; retain SME outer products, existing SVE twiddle operations and single-row ZA extraction |
| Candidates | Original selection space | Ported width-2 candidates, no temporary ZA tile rotation |
| Public interface | Original FFT API and runtime | Same API and runtime |
| Streaming entry | Original Apple assembly | Compiler-managed Linux C entry with private ZA |
| Initial profile | Original Apple bootstrap | Separate Linux bootstrap, explicitly `measured=false` |

These candidate restrictions describe this implementation, not a universal hardware limit. FP32/FP64 and forward/backward generation remain available. Generation alone does not establish target correctness for every size, layout or caller state.

The Linux entry uses `__arm_new("za")` and `__arm_locally_streaming` from [Arm ACLE](https://arm-software.github.io/acle/main/acle.html). The compiler implements the [AAPCS64](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst) private-ZA/lazy-save protocol. A handwritten SMSTART/SMSTOP wrapper does not replace it.

## Basic library build

Use Linux AArch64, an installed Clang toolchain with SME ACLE and `<arm_sme.h>`, CMake 3.24 or newer, and Python 3.10 or newer. Before target execution, verify OS support, required SME floating-point capabilities and an actual streaming vector length of 64 bytes. Compiler flags do not establish hardware support. This port does not change system vector-length settings.

In a Linux AArch64 shell, from the MoFFT checkout root, replace the compiler paths:

```sh
cmake -S . -B build/linux-sme1 -DCMAKE_BUILD_TYPE=Release \
  -DMOFFT_SME1_PORT=ON -DMOFFT_BUILD_TOOLS=OFF \
  -DCMAKE_C_COMPILER=/absolute/path/to/clang \
  -DCMAKE_CXX_COMPILER=/absolute/path/to/clang++ \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -lgcc_s"
```

In the same shell and checkout root:

```sh
cmake --build build/linux-sme1 --target mofft mofft_correctness
```

The compiler-rt settings provide Linux Clang's SME ABI helpers and must match the installed runtime. They do not change Apple linking. Execute the correctness program only after verifying target instruction and ABI support.

## Calibration and complete workflow

The Linux bootstrap contains model assumptions for initial generation, not a measured CPU profile. Recalibrate the actual CPU/system/vector/toolchain combination; a profile for one CPU is not a general calibration for every machine.

The complete [Linux workflow](linux-sme1-workflow.md) is in `tools/linux_sme1`: source-built probes, candidate validation, kernel wisdom, full plan search, final correctness/lifecycle checks, raw capture, login acceptance and plotting. Original Apple tools remain separate and are disabled in an SME1 build. Do not run original SME2 microbenchmarks on an SME1 CPU.

For SME1 only, two-vector loads/stores use distinct `sve_load2` and `sve_store2` cost keys. Default SME2 keys remain unchanged. The SME1 CMake path applies a syntax-only post-generation correction to locality-dispatch statement terminators; the default SME2 path does not invoke it.

## SME2 compatibility

Default arguments, Apple entry, runtime, API, Apple profiles and build options remain on their original path. Run the compiler, SME1 and workflow regression tests, and compare default SME2 sources/selections under identical profiles and parameters. Compiler digests follow upstream identity rules: regenerate matching wisdom after a source change; do not forge digests or disable validation.

Generated-source and compiled-object equality under one Linux compiler are distinct from SME2 hardware correctness and performance. Hardware regression requires the same SME2 device, compiler, profiles and timing protocol, with `MOFFT_SME1_PORT=OFF`. Run original correctness/candidate checks before the original benchmark. Do not silently substitute a bootstrap while claiming the same tuning.

The original Apple `-mno-implicit-sme` setting remains. Linux SME2 object comparisons use identical flags supported by the installed Linux compiler; they do not claim Apple Clang object equality.

Git attributes keep compiler Python files in canonical LF form on every checkout. Compiler identity hashes raw bytes; changing only Windows CRLF endings would otherwise invalidate measured wisdom. This preserves the normal digest check rather than relabelling old wisdom.
