# Downloading and installing MoFFT

## MoFFT 0.4.0 Preview 1

This preview is a C library for Apple M5/macOS arm64, not an FFTW ABI
replacement. It supports one-dimensional FP32/FP64 complex transforms in both
directions and placements. It has passed functional checks; this release does
not claim a new, formally tuned performance result.

- [Library source package](https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1.tar.gz)
  ([SHA-256](https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1.tar.gz.sha256)).
- [Optional prebuilt macOS arm64 SDK](https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1-macos-arm64.tar.gz)
  ([SHA-256](https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1-macos-arm64.tar.gz.sha256)).
- [Release notes](https://github.com/ruge-zhang/MoFFT/releases/tag/v0.4.0-preview.1)
  and [development repository](https://github.com/ruge-zhang/MoFFT).

The source package contains the complete library sources and pre-generated
kernel C files, rather than the Python-based development toolchain. Consumers
do not run Python, calibration, or kernel generation to build this package.

## Build and install

Requirements: Apple M5, macOS 26.4 or newer, Apple Clang 21 with SME/SME2 ACLE
support, and CMake 3.24 or newer. FP64 additionally requires SME F64F64.
Only this M5/macOS configuration is validated for this preview.

```sh
curl -LO https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1.tar.gz
curl -LO https://github.com/ruge-zhang/MoFFT/releases/download/v0.4.0-preview.1/mofft-0.4.0-preview.1.tar.gz.sha256
shasum -a 256 -c mofft-0.4.0-preview.1.tar.gz.sha256
tar -xzf mofft-0.4.0-preview.1.tar.gz

cmake -S mofft-0.4.0-preview.1 -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
cmake --install build
```

The install prefix contains `include/mofft.h`, `lib/libmofft.a`, and a relocatable
CMake package. To skip compiling the library, unpack the prebuilt SDK and use
its package directory as the prefix instead.

## Link into an application

```cmake
find_package(MoFFT 0.4 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE MoFFT::mofft)
```

Pass `-DCMAKE_PREFIX_PATH="$PWD/install"` when configuring the application.
For a plain C application:

```c
#include <mofft.h>

int main(void) {
    mofft_complex_f64 input[256] = {{1.0, 0.0}};
    mofft_complex_f64 output[256];
    mofft_plan_f64 *plan = mofft_plan_dft_1d_f64(
        256, input, output, MOFFT_FORWARD, MOFFT_FLAG_PRETUNED);
    if (!plan) return 1;
    mofft_status status = mofft_execute_plan_f64(plan);
    mofft_plan_destroy_f64(plan);
    return status == MOFFT_SUCCESS ? 0 : 1;
}
```

Save it as `example.c`, then compile and run:

```sh
clang -mmacosx-version-min=26.4 example.c -Iinstall/include \
  -Linstall/lib -lmofft -lm -o example
./example
```

Application files using only the public header do not need SME compiler flags.
Use `f32` types/functions for FP32. The same input/output pointer selects
in-place operation. Use `MOFFT_BACKWARD` for the inverse direction. Neither
direction normalizes: a forward/backward roundtrip returns `N*x`.

## Planning policies and saved plans

- `MOFFT_FLAG_PRETUNED` uses embedded preset plans where available, otherwise
  the analytical planner. This preview includes exact presets for lengths
  169, 256 and 1024 in both precisions. Other supported lengths still use
  precompiled kernels and do not trigger generation or measurement.
- `MOFFT_FLAG_MEASURE` times a bounded set of compiled kernel banks,
  decompositions and layouts at plan creation. Create once and reuse the plan.
  This is not JIT compilation or a guarantee of a global optimum. Use a quiet,
  thermally stable machine for meaningful tuning.
- `MOFFT_FLAG_WISDOM_ONLY` requires an exact embedded case rather than allowing
  model fallback.

Explicit recipe export/import lets applications save a selected plan and load
it on a later run of the same library build and CPU family. Recipes contain no
pointers or generated code; there is no implicit filesystem cache. See
[planning policies and saved recipes](BINARY_DISTRIBUTION.md) and the
[public C header](../include/mofft.h) for the status-returning API and diagnostics.

## License and scope

MoFFT uses [Apache-2.0](../LICENSE). The interface is inspired by FFTW's lifecycle
but does not implement FFTW's ABI, flags, wisdom format, real transforms,
multidimensional transforms or its full API surface.
