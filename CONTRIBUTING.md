# Contributing to MoFFT

Thank you for improving MoFFT. Contributions should preserve numerical
correctness, deterministic generation, and reproducible performance evidence.

## Development setup

MoFFT's generated kernels require an Apple Arm machine with SME/SME2 support.
The Python compiler tests are platform-independent:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -p 'test_compiler.py'
```

On supported hardware, also run:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --build build --target mofft_validate_candidates
cmake --build build --target mofft_check_codegen
```

## Pull requests

- Keep changes focused and explain their effect on the compiler model, emitted
  instructions, numerical behavior, or experiment protocol.
- Add a regression test for changes that affect rewrites, scheduling, code
  generation, planning, or runtime layout.
- Do not commit generated kernels or build directories.
- Mark all non-exclusive performance data as nonformal. Include raw samples,
  machine/profile hashes, load average, and dispersion when reporting speedups.
- Do not claim unpublished Apple port numbers. Resource relationships inferred
  from throughput must remain labeled as inferences.

By contributing, you agree that your contribution is licensed under
Apache-2.0.
