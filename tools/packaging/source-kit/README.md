# MoFFT 0.4.0 Preview 1 — Source Distribution

MoFFT is a matrix-oriented FFT library for Apple arm64 systems with SME2. This
release is a **source distribution** containing the library runtime, four
pre-generated SME2 kernel banks, and the included plan cases. It is a macOS
arm64 preview, not a portable build or a drop-in FFTW ABI replacement.

The source archive is the canonical download. Building it does not require
Python, a kernel generator, calibration data, or a tuning run. It requires
CMake 3.24 or newer, Apple Clang (or compatible Clang with the required SME2
support), and the Apple arm64 SDK. The generated kernels require an SME2-capable
Apple processor at runtime.

## Build and install

Download the [MoFFT @MOFFT_RELEASE_VERSION@ source archive](https://github.com/ruge-zhang/MoFFT/releases/download/v@MOFFT_RELEASE_VERSION@/mofft-@MOFFT_RELEASE_VERSION@.tar.gz)
and its adjacent `.sha256` file from GitHub Releases, then verify and unpack it:

```sh
shasum -a 256 -c mofft-@MOFFT_RELEASE_VERSION@.tar.gz.sha256
tar -xzf mofft-@MOFFT_RELEASE_VERSION@.tar.gz
cd mofft-@MOFFT_RELEASE_VERSION@
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$HOME/.local"
```

The build produces a static library and a relocatable CMake package. A consumer
can use it with:

```cmake
find_package(MoFFT 0.4 CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE MoFFT::mofft)
```

For example, build the included C demo after installing to the default prefix:

```sh
cmake -S examples/consumer -B consumer-build \
  -DCMAKE_PREFIX_PATH="$HOME/.local"
cmake --build consumer-build --parallel
./consumer-build/mofft-demo 256 fp32 forward pretuned
```

The library API supports FP32 and FP64 complex transforms. `PRETUNED` uses an
included plan case when available and otherwise uses the built-in analytical
planner. `MEASURE` searches among the kernels already included in the source
distribution; it does not generate or compile new kernels at runtime. Plan
recipes are scoped to the configured build identity and CPU family.

This preview includes M5-oriented source artifacts only. Performance and
hardware behavior can vary with the compiler and operating-system SDK used to
build it; the package makes no cross-platform or FFTW-performance guarantee.
See the repository documentation and release notes for the exact preview
scope. The library is distributed under the Apache License 2.0; see `LICENSE`.
