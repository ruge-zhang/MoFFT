# Binary distribution and planning policies

MoFFT's native library can be distributed separately from its Python kernel
compiler. A consumer links the supplied library and public C header; it does not
run calibration, Python, kernel generation, or CMake code generation.

The current prebuilt delivery targets **macOS arm64 on M5-class hardware with
SME/SME2**. FP64 additionally requires SME F64F64. This is not a universal Arm
binary or an iOS Simulator binary. Missing hardware features are reported before
entering generated code. The iOS packaging path is retained for future porting
work, but is not part of this validated macOS delivery; linking an iOS archive
does not establish on-device behavior.

## Two consumer paths

| Policy | Planning work | Generated code | Timing at execution |
| --- | --- | --- | --- |
| PRETUNED (default) | Use a matching compiled plan recipe; use the model for an uncovered length unless strict wisdom is requested | Already in the archive | Only execute the chosen plan |
| MEASURE | Time a bounded set of linked kernel banks, decompositions and layout choices on private scratch buffers | Already in the archive | Only execute the chosen plan |

`MEASURE` is a search over the candidates available in that particular binary,
not a proof of a global optimum. It takes extra time at **plan creation**, not on
every execution. Create once and reuse the plan. Its timing is only meaningful
in a quiet, thermally stable window; short checks under normal desktop activity
are functional tests, not formal performance results. Measurement does not
overwrite caller input buffers. Measured plan choices remain process-local
unless the caller explicitly exports a recipe; the library does not silently
write a cache into a user's filesystem.

## M5 quick start

The macOS arm64 package includes a precompiled demo. After extracting it, enter
the package directory and run it without any compiler, Python or generator:

```sh
./bin/mofft-demo 256 fp32 pretuned
./bin/mofft-demo 4096 fp64 measure --save-plan fp64-4096.plan
./bin/mofft-demo 4096 fp64 --load-plan fp64-4096.plan
```

The demo checks an impulse transform; it is a functional example, not the
formal benchmark. The same plain C example is included as source if you want
to rebuild it or start an application:

```sh
cmake -S examples/consumer -B build -DCMAKE_PREFIX_PATH="$PWD"
cmake --build build

# Use a plan bundled in the library when one matches, or the model otherwise.
./build/mofft-demo 4096 fp64 pretuned

# Search this binary's linked candidates on the M5 and save the selected plan.
./build/mofft-demo 4096 fp64 measure --save-plan fp64-4096.plan

# Reuse it on a later run without repeating the search.
./build/mofft-demo 4096 fp64 --load-plan fp64-4096.plan
```

The example accepts `fp32` or `fp64`, and `forward` or `inverse`. PRETUNED is
the default; MEASURE runs only when explicitly requested. A saved recipe is a
small versioned text record containing the selected decomposition and layout.
It is tied to the library build and CPU family; import fails if either differs.
The example checks that the requested size, precision, direction and placement
match before executing. A recipe contains no buffer addresses or private
machine paths. MoFFT never reads or writes a plan cache file implicitly.

Recipes are not FFTW wisdom and do not contain generated code. The selected
kernels are already compiled into the static library. Recipes cannot be reused
with another binary build or CPU family.

For application-managed storage, call `mofft_plan_export_f32/f64` with a NULL
buffer and zero capacity to obtain the byte count, allocate that many bytes,
then export and preserve all bytes including the final NUL. Pass the same byte
count to `mofft_plan_import_f32/f64` when loading. Check the returned status and
the shape in `mofft_plan_diagnostics` before executing. Import restores a fresh
plan, not its former buffer addresses: use `mofft_execute_f32/f64` with your
current arrays, rather than the bound-array `mofft_execute_plan_f32/f64` helper.

A binary built with `MOFFT_BUILD_KERNEL_PORTFOLIO=ON` includes the default bank
and three alternative expression-family banks. Legal VG2/VG4, temporary-tile,
and pipeline forms are selected by the producer's model for each family. This
allows on-device code selection without executing a compiler on iOS. A binary
without that option has only bank 0: its MEASURE policy can still search FFT
decompositions and layouts, but cannot compare different kernel banks.

## FFTW-inspired C interface

The argument order and plan/execute/destroy lifecycle follow FFTW's familiar
shape. MoFFT does **not** claim compatibility with `fftw3.h`, FFTW's ABI, flags,
complex typedefs, or its real/multidimensional APIs. Existing MoFFT explicit-buffer
APIs remain supported.

```c
#include <mofft.h>

int main(void) {
    mofft_complex_f32 input[1024] = {{0}}, output[1024];
    input[0].real = 1.0f;
    /* 0: use the packaged preset/model; MOFFT_FLAG_MEASURE: retune locally. */
    unsigned flags = 0;
    mofft_plan_f32 *plan = mofft_plan_dft_1d_f32(
        1024, input, output, MOFFT_FORWARD, flags);
    if (!plan) return 1;
    mofft_status status = mofft_execute_plan_f32(plan);
    mofft_plan_destroy_f32(plan);
    return status == MOFFT_SUCCESS ? 0 : 1;
}
```

The FP64 equivalents use `mofft_complex_f64`, `mofft_plan_dft_1d_f64`,
`mofft_execute_plan_f64`, and `mofft_plan_destroy_f64`. Passing the same pointer
for input and output selects in-place operation. Forward uses sign -1 and
backward sign +1; neither direction normalizes, so inverse(forward(x)) = N*x.

For error details and selection diagnostics, use the status-returning API:

```c
mofft_plan_options options;
mofft_plan_options_init(&options);
options.mode = MOFFT_PLANNER_MEASURE;
options.candidate_limit = 32;
options.timing_samples = 7;
mofft_plan_diagnostics selected;
mofft_plan_f64 *plan = NULL;
mofft_status status = mofft_plan_create_with_options_f64(
    &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &selected);
/* On success, inspect selected.radices, kernel_bank, candidates_measured,
   used_wisdom, used_model_fallback, and planning_seconds. */
```

Use `MOFFT_PLAN_STRICT_WISDOM` in `options.flags`, or
`MOFFT_FLAG_WISDOM_ONLY` in a bound-array constructor, when an uncovered case
must fail instead of using the model. "Generated code is bundled" and "this
exact transform length has a measured bundled plan" are separate properties;
diagnostics must not conflate them.

## Use a macOS package

Unpack the arm64 archive to any directory. No installation into a system
directory is required. With CMake:

```cmake
find_package(MoFFT CONFIG REQUIRED)
target_link_libraries(my_fft_program PRIVATE MoFFT::mofft)
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/unpacked/MoFFT`.
The package does not require the original producer's source/build directory.
For a direct C build, use its `include/` and `lib/` directories and link
`-lmofft -lm`. Consumers do not need SME compiler flags; those instructions are
already in the archive.

## Future iOS packaging path (not validated by the current delivery)

Add `MoFFT.xcframework` to the app's Xcode target and include `mofft.h`.
It contains a static, device-arm64 slice; do not request dynamic embedding.
Normal app signing and provisioning are still required. The library contains
no signing identity, device identifier, or developer account. Distributing the
library does not make an unsigned iPad application directly installable.

The repository's `ios-app` package-check harness can link an installed package
through `MoFFT::mofft` without including headers or generated code from the
source tree. Any device execution, including PRETUNED and MEASURE coverage in
both precisions and directions, remains a separate validation step. Device and
signing parameters are supplied locally, never committed.

## Produce a frozen binary from existing cases

The producer selects the calibrated profile, kernel wisdom, and plan wisdom:

```sh
cmake -S . -B build/binary -DCMAKE_BUILD_TYPE=Release \
  -DMOFFT_BUILD_KERNEL_PORTFOLIO=ON \
  -DMOFFT_PROFILE=/path/to/profile.json \
  -DMOFFT_KERNEL_WISDOM=/path/to/kernel-wisdom.json \
  -DMOFFT_WISDOM=/path/to/plan-wisdom.json
cmake --build build/binary --parallel 2
ctest --test-dir build/binary --output-on-failure
cmake --install build/binary --prefix build/package
python3 tools/packaging/package.py verify-macos --prefix build/package
python3 tools/packaging/package.py package-macos --prefix build/package \
  --output build/MoFFT-macos-arm64.tar.gz
```

When the package is expected to contain an exact PRETUNED case, pass
`--require-wisdom-size N` to both `verify-macos` and `package-macos`. The
verification then requires strict FP32 and FP64 wisdom hits at that size, so a
model-only bootstrap package cannot be mistaken for a wisdom-bearing package.

For future iOS packaging, use the iOS toolchain and a profile/wisdom set matched
to the actual target, specify `MOFFT_WISDOM_MACHINE` for that target, and run the
package script's `verify-ios` and `package-ios` subcommands. An archive that
merely links for iOS has not passed an on-device test.

The current consumer example and package workflow target M5 macOS. An iOS
archive can be cross-linked and packaged, but device execution is a separate
validation step; host linking alone does not establish iPad runtime behavior.

Existing wisdom compatibility checks remain enforced. Do not edit machine,
profile, or compiler hashes merely to make old results load. Historical
measurement files must be available and compatible to reproduce the old tuned
binary; a bootstrap-model build must never be advertised as that binary.

## Remeasure and regenerate on a native producer

```sh
# Short, explicitly nonformal check of the complete producer workflow:
python3 tools/retune_native.py --workdir build/tune-smoke --smoke

# Kernel measurement, plan search, freezing, validation, and packaging:
python3 tools/retune_native.py --workdir build/tune --exclusive-confirmed

# Reuse a matched set of existing producer inputs without remeasurement:
python3 tools/retune_native.py --workdir build/reuse \
  --profile /path/to/profile.json \
  --kernel-wisdom /path/to/kernel-wisdom.json \
  --plan-wisdom /path/to/plan-wisdom.json
```

The smoke workflow measures only a limited kernel subset and a few FFT sizes;
it is not release tuning. The regular workflow measures legal kernel candidates
and searches the requested size list, then compiles its results into the final
library. This producer workflow is separate from the lightweight MEASURE API
inside a downloaded binary. Neither workflow publishes a GitHub release.

## Release checks

- Test both precisions, directions, and placements against reference outputs.
- Test PRETUNED hits, uncovered-case fallback/strict failure, and MEASURE.
- Relocate/extract the actual archive and build a consumer from that artifact.
- Before offering an iOS package, run its actual archive on each intended device;
  host linking is not enough.
- Scan binaries and metadata for local paths, personal identifiers, raw logs,
  credentials, signing details, and unintended source/data files.
- Record the target capabilities and whether preset selections were measured
  or model-based; do not promise the same performance across different CPUs.
- Publish only the validated archives, public usage documentation, license,
  checksums, and accurate release notes. Keep raw experiment files local.
