#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
out="$root/build/full-dry-run"
fftw="$root/build/deps/fftw-3.3.11/install"
mkdir -p "$out"

if [ ! -f "$fftw/lib/libfftw3.a" ] || [ ! -f "$fftw/lib/libfftw3f.a" ]; then
  "$root/tools/build_fftw_3_3_11.sh" >/dev/null
fi
cmake -S "$root" -B "$out/bootstrap-build" -DCMAKE_BUILD_TYPE=Release \
  -DMOFFT_BUILD_TESTS=ON -DMOFFT_BUILD_TOOLS=ON
cmake --build "$out/bootstrap-build" -j"$(sysctl -n hw.ncpu)"

"$out/bootstrap-build/mofft_microbench" --iterations 50000 --samples 5 \
  > "$out/m5-microbench-dry-run.json"
PYTHONPATH="$root/compiler" python3 -m mofft_compiler.calibrate \
  "$out/m5-microbench-dry-run.json" --output "$out/apple-m5-dry-run-profile.json"

cmake -S "$root" -B "$out/measured-build" -DCMAKE_BUILD_TYPE=Release \
  -DMOFFT_WITH_FFTW=ON -DMOFFT_PROFILE="$out/apple-m5-dry-run-profile.json" \
  -DCMAKE_PREFIX_PATH="$fftw"
cmake --build "$out/measured-build" -j"$(sysctl -n hw.ncpu)"
ctest --test-dir "$out/measured-build" --output-on-failure
"$out/measured-build/mofft_benchmark" --full-dry-run --batches 3 \
  --output "$out/m5-full-dry-run.json"

plot_python=python3
if ! "$plot_python" -c 'import matplotlib' 2>/dev/null; then
  if [ ! -x "$out/plot-venv/bin/python" ]; then
    python3 -m venv "$out/plot-venv"
    "$out/plot-venv/bin/python" -m pip install -q 'matplotlib>=3.8'
  fi
  plot_python="$out/plot-venv/bin/python"
fi
"$plot_python" "$root/benchmarks/plot_results.py" \
  --results "$out/m5-full-dry-run.json" --label "Apple M5" \
  --output "$out/m5-full-dry-run.png"
echo "$out"
