from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import tempfile

from .emitter import (RADICES, _compiler_digest, emit_candidate_source,
                      profile_digest)
from .patterns import Candidate, enumerate_candidates
from .profile import MachineProfile


def _name(candidate: Candidate, serial: int, context: str) -> str:
    return (f"candidate_{serial}_r{candidate.radix}_{candidate.precision}_"
            f"{candidate.direction}_{candidate.stage}_"
            f"{candidate.pattern.value}_vg{candidate.matrix_vg_width}_"
            f"rot{int(candidate.rotate_temp_tiles)}_"
            f"pipe{candidate.batch_pipeline_depth}_{context}")


def _wrapper(candidate: Candidate, name: str, context: str) -> str:
    ctype = "mofft_complex_f32" if candidate.precision == "fp32" else "mofft_complex_f64"
    if candidate.stage == "first":
        call = f"{name}(input, batch, output, batch);"
    elif context == "normal":
        call = (f"{name}(input, batch, 0, twiddles, batch, "
                "1, output, batch);")
    elif context == "broadcast":
        # The harness transports the requested locality repeat in
        # twiddle_stride.  Broadcast input remains radix-major, while the
        # compact twiddle table has one element per repeat-wide group.
        call = (f"size_t repeat = twiddle_stride; "
                f"{name}(input, batch, 0, twiddles, "
                "batch / repeat, repeat, output, batch);")
    elif context == "direct_broadcast":
        # The direct layout stores each repeat-wide radix tile contiguously.
        # Its row stride and input repeat are therefore both repeat; twiddles
        # retain the same compact group-major layout as broadcast.
        call = (f"size_t repeat = row_stride; "
                f"{name}(input, repeat, repeat, twiddles, "
                "batch / repeat, repeat, output, batch);")
    else:
        raise ValueError(f"unsupported validation context {context}")
    return f"""
void {name}_wrapper(const {ctype} *input, size_t row_stride,
                    const {ctype} *twiddles, size_t twiddle_stride,
                    {ctype} *output, size_t batch) __arm_streaming __arm_inout("za") {{
  {call}
}}
"""


def _table(candidates: list[tuple[Candidate, str, str]], precision: str) -> str:
    rows = []
    for candidate, name, context in candidates:
        if candidate.precision != precision:
            continue
        direction = -1 if candidate.direction == "forward" else 1
        stage = 0 if candidate.stage == "first" else 1
        label = (f"r{candidate.radix}/{candidate.precision}/"
                 f"{candidate.direction}/{candidate.stage}/"
                 f"{candidate.pattern.value}/vg{candidate.matrix_vg_width}/"
                 f"rot{int(candidate.rotate_temp_tiles)}/"
                 f"pipe{candidate.batch_pipeline_depth}/"
                 f"{context}")
        context_id = {"normal": 0, "broadcast": 1,
                      "direct_broadcast": 2, "transposed": 3}[context]
        rows.append(f'  {{"{label}", {candidate.radix}, {direction}, {stage}, '
                    f'{context_id}, (void *){name}_wrapper}}')
    return ",\n".join(rows)


def _harness(candidates: list[tuple[Candidate, str, str]]) -> str:
    return f"""
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mofft.h"

#define PI 3.141592653589793238462643383279502884
extern void mofft_candidate_entry(void *, const void *, size_t, const void *,
                                  size_t, void *, size_t);
struct test_case {{ const char *name; int radix, direction, other, context; void *body; }};
static const struct test_case f32_cases[] = {{
{_table(candidates, "fp32")}
}};
static const struct test_case f64_cases[] = {{
{_table(candidates, "fp64")}
}};

__attribute__((optnone)) static int test_f32(const struct test_case *tc) {{
  const size_t batch = 19, count = (size_t)tc->radix * batch;
  mofft_complex_f32 *in = calloc(count, sizeof(*in));
  mofft_complex_f32 *tw = calloc(count, sizeof(*tw));
  mofft_complex_f32 *out = calloc(count, sizeof(*out));
  if (!in || !tw || !out) return 1;
  for (int j=0;j<tc->radix;++j) for (size_t b=0;b<batch;++b) {{
    size_t q=(size_t)j*batch+b;
    in[q].real=(float)(sin(q*.173)+.01*(q%11));
    in[q].imag=(float)(cos(q*.097)-.02*(q%7));
    double a=tc->direction*2.0*PI*j*(double)b/(tc->radix*batch+3.0);
    tw[q].real=(float)cos(a); tw[q].imag=(float)sin(a);
  }}
  mofft_candidate_entry(tc->body,in,batch,tw,batch,out,batch);
  double error=0;
  for(size_t b=0;b<batch;++b) for(int k=0;k<tc->radix;++k) {{
    double rr=0,ii=0;
    for(int j=0;j<tc->radix;++j) {{
      size_t q=(size_t)j*batch+b; double xr=in[q].real,xi=in[q].imag;
      if(tc->other) {{ size_t tq=tc->context ? (size_t)j : q;
                      double nr=xr*tw[tq].real-xi*tw[tq].imag;
                      xi=xr*tw[tq].imag+xi*tw[tq].real; xr=nr; }}
      double a=tc->direction*2.0*PI*j*k/tc->radix, cr=cos(a),ci=sin(a);
      rr+=xr*cr-xi*ci; ii+=xr*ci+xi*cr;
    }}
    size_t q=tc->context==3 ? b*(size_t)tc->radix+k : (size_t)k*batch+b;
    double e=hypot(out[q].real-rr,out[q].imag-ii); if(e>error) error=e;
  }}
  free(in); free(tw); free(out);
  double tolerance=64.0*FLT_EPSILON*tc->radix;
  if(error>tolerance) {{ fprintf(stderr,"%s error %.9g tolerance %.9g\\n",tc->name,error,tolerance); return 1; }}
  return 0;
}}

__attribute__((optnone)) static int test_f64(const struct test_case *tc) {{
  const size_t batch = 11, count = (size_t)tc->radix * batch;
  mofft_complex_f64 *in = calloc(count, sizeof(*in));
  mofft_complex_f64 *tw = calloc(count, sizeof(*tw));
  mofft_complex_f64 *out = calloc(count, sizeof(*out));
  if (!in || !tw || !out) return 1;
  for (int j=0;j<tc->radix;++j) for (size_t b=0;b<batch;++b) {{
    size_t q=(size_t)j*batch+b;
    in[q].real=sin(q*.173)+.01*(q%11); in[q].imag=cos(q*.097)-.02*(q%7);
    double a=tc->direction*2.0*PI*j*(double)b/(tc->radix*batch+3.0);
    tw[q].real=cos(a); tw[q].imag=sin(a);
  }}
  mofft_candidate_entry(tc->body,in,batch,tw,batch,out,batch);
  double error=0;
  for(size_t b=0;b<batch;++b) for(int k=0;k<tc->radix;++k) {{
    double rr=0,ii=0;
    for(int j=0;j<tc->radix;++j) {{
      size_t q=(size_t)j*batch+b; double xr=in[q].real,xi=in[q].imag;
      if(tc->other) {{ size_t tq=tc->context ? (size_t)j : q;
                      double nr=xr*tw[tq].real-xi*tw[tq].imag;
                      xi=xr*tw[tq].imag+xi*tw[tq].real; xr=nr; }}
      double a=tc->direction*2.0*PI*j*k/tc->radix, cr=cos(a),ci=sin(a);
      rr+=xr*cr-xi*ci; ii+=xr*ci+xi*cr;
    }}
    size_t q=tc->context==3 ? b*(size_t)tc->radix+k : (size_t)k*batch+b;
    double e=hypot(out[q].real-rr,out[q].imag-ii); if(e>error) error=e;
  }}
  free(in); free(tw); free(out);
  double tolerance=192.0*DBL_EPSILON*tc->radix;
  if(error>tolerance) {{ fprintf(stderr,"%s error %.17g tolerance %.17g\\n",tc->name,error,tolerance); return 1; }}
  return 0;
}}

static double now_seconds(void) {{
  struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}}
static int compare_double(const void *a, const void *b) {{
  const double x=*(const double *)a,y=*(const double *)b;
  return (x>y)-(x<y);
}}

__attribute__((optnone)) static void benchmark_f32(const struct test_case *tc, size_t batch, size_t repeat) {{
  const size_t count=(size_t)tc->radix*batch;
  enum {{ rounds=9, repetitions=100 }};
  mofft_complex_f32 *in=calloc(count,sizeof(*in)),*tw=calloc(count,sizeof(*tw));
  mofft_complex_f32 *out=calloc(count,sizeof(*out)); double samples[rounds];
  for(size_t q=0;q<count;++q) {{ in[q].real=(float)sin(q*.017); in[q].imag=(float)cos(q*.013); tw[q].real=1; }}
  for(int q=0;q<10;++q)mofft_candidate_entry(tc->body,in,repeat,tw,repeat,out,batch);
  for(int r=0;r<rounds;++r) {{ double a=now_seconds(); for(int q=0;q<repetitions;++q)mofft_candidate_entry(tc->body,in,repeat,tw,repeat,out,batch); samples[r]=1e9*(now_seconds()-a)/repetitions; }}
  qsort(samples,rounds,sizeof(double),compare_double);
  printf("benchmark,%s,%zu,%.3f\\n",tc->name,repeat,samples[rounds/2]);
  free(in);free(tw);free(out);
}}

__attribute__((optnone)) static void benchmark_f64(const struct test_case *tc, size_t batch, size_t repeat) {{
  const size_t count=(size_t)tc->radix*batch;
  enum {{ rounds=9, repetitions=100 }};
  mofft_complex_f64 *in=calloc(count,sizeof(*in)),*tw=calloc(count,sizeof(*tw));
  mofft_complex_f64 *out=calloc(count,sizeof(*out)); double samples[rounds];
  for(size_t q=0;q<count;++q) {{ in[q].real=sin(q*.017); in[q].imag=cos(q*.013); tw[q].real=1; }}
  for(int q=0;q<10;++q)mofft_candidate_entry(tc->body,in,repeat,tw,repeat,out,batch);
  for(int r=0;r<rounds;++r) {{ double a=now_seconds(); for(int q=0;q<repetitions;++q)mofft_candidate_entry(tc->body,in,repeat,tw,repeat,out,batch); samples[r]=1e9*(now_seconds()-a)/repetitions; }}
  qsort(samples,rounds,sizeof(double),compare_double);
  printf("benchmark,%s,%zu,%.3f\\n",tc->name,repeat,samples[rounds/2]);
  free(in);free(tw);free(out);
}}

__attribute__((optnone)) int main(int argc, char **argv) {{
  int failures=0;
  for(size_t i=0;i<sizeof(f32_cases)/sizeof(f32_cases[0]);++i) failures+=test_f32(&f32_cases[i]);
  for(size_t i=0;i<sizeof(f64_cases)/sizeof(f64_cases[0]);++i) failures+=test_f64(&f64_cases[i]);
  printf("validated %zu generated candidates; failures=%d\\n",
         sizeof(f32_cases)/sizeof(f32_cases[0])+sizeof(f64_cases)/sizeof(f64_cases[0]), failures);
  if(!failures && argc>=2 && argc<=4 && !strcmp(argv[1],"--benchmark")) {{
    const size_t benchmark_batch = argc >= 3 ?
        (size_t)strtoull(argv[2],NULL,10) : 256;
    const size_t benchmark_repeat = argc >= 4 ?
        (size_t)strtoull(argv[3],NULL,10) :
        (benchmark_batch < 32 ? benchmark_batch : 32);
    if(benchmark_batch==0 || benchmark_repeat==0 ||
       benchmark_repeat>benchmark_batch || benchmark_batch%benchmark_repeat)
      return 2;
    for(int pass=0;pass<3;++pass) {{
      size_t n=sizeof(f32_cases)/sizeof(f32_cases[0]);
      for(size_t q=0;q<n;++q) {{ size_t i=pass==1?n-1-q:q; benchmark_f32(&f32_cases[i],benchmark_batch,benchmark_repeat); }}
      n=sizeof(f64_cases)/sizeof(f64_cases[0]);
      for(size_t q=0;q<n;++q) {{ size_t i=pass==1?n-1-q:q; benchmark_f64(&f64_cases[i],benchmark_batch,benchmark_repeat); }}
    }}
  }}
  return failures ? 1 : 0;
}}
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compile and numerically validate every legal kernel candidate")
    parser.add_argument("--cc", default="xcrun clang")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--radices", default=",".join(map(str, RADICES)))
    parser.add_argument("--precisions", default="fp32,fp64")
    parser.add_argument("--directions", default="forward,backward")
    parser.add_argument("--stages", default="first,other")
    parser.add_argument("--contexts",
                        default="normal,transposed,broadcast,direct_broadcast")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--temp-tile-rotation", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--benchmark-batches", default="256")
    parser.add_argument(
        "--benchmark-repeats", default="32",
        help="comma-separated locality-repeat values")
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--profile", type=Path,
                        help="machine profile recorded in benchmark wisdom")
    parser.add_argument("--keep", type=Path)
    parser.add_argument("--emit-source", type=Path,
                        help="write the candidate corpus without compiling it")
    args = parser.parse_args(argv)
    radices = tuple(int(value) for value in args.radices.split(",") if value)
    precisions = tuple(value for value in args.precisions.split(",") if value)
    directions = tuple(value for value in args.directions.split(",") if value)
    stages = tuple(value for value in args.stages.split(",") if value)
    benchmark_batches = tuple(int(value) for value in
                              args.benchmark_batches.split(",") if value)
    if not benchmark_batches or any(value <= 0 for value in benchmark_batches):
        parser.error("--benchmark-batches must contain positive integers")
    benchmark_repeats = tuple(int(value) for value in
                              args.benchmark_repeats.split(",") if value)
    if not benchmark_repeats or any(value <= 0 for value in benchmark_repeats):
        parser.error("--benchmark-repeats must contain positive integers")
    contexts = tuple(value for value in args.contexts.split(",") if value)
    invalid_contexts = set(contexts) - {
        "normal", "transposed", "broadcast", "direct_broadcast"}
    if invalid_contexts:
        parser.error(f"unknown contexts: {sorted(invalid_contexts)}")
    if args.benchmark_output is not None and args.profile is None:
        parser.error("--benchmark-output requires --profile")
    candidates: list[tuple[Candidate, str, str]] = []
    serial = 0
    for precision in precisions:
        for direction in directions:
            for stage in stages:
                for radix in radices:
                    for candidate in enumerate_candidates(radix, precision,
                                                          stage, direction,
                                                          args.temp_tile_rotation):
                        valid_stage_contexts = (
                            {"normal", "broadcast", "direct_broadcast"}
                            if stage == "other" else {"normal", "transposed"})
                        applicable = tuple(x for x in contexts
                                           if x in valid_stage_contexts)
                        for context in applicable:
                            if (candidate.batch_pipeline_depth == 2 and
                                    context != "direct_broadcast"):
                                continue
                            candidates.append((candidate, _name(
                                candidate, serial, context), context))
                            serial += 1
    temporary = None
    if args.emit_source:
        work = args.emit_source.resolve().parent
        work.mkdir(parents=True, exist_ok=True)
    elif args.keep:
        work = args.keep.resolve(); work.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="mofft-candidates-")
        work = Path(temporary.name)
    source = (args.emit_source.resolve() if args.emit_source else
              work / "candidate_corpus.c")
    executable = work / "candidate_corpus"
    chunks = [emit_candidate_source(candidate, name, context) +
              _wrapper(candidate, name, context)
              for candidate, name, context in candidates]
    source.write_text("\n".join(chunks) + _harness(candidates))
    if args.emit_source:
        print(source)
        return
    command = args.cc.split() + [
        "-O3", "-march=armv9.2-a+sme+sme2+sme-f64f64",
        "-mno-implicit-sme", f"-I{args.root / 'include'}", str(source),
        str(args.root / "tests/candidate_entry.S"), "-lm", "-o", str(executable)]
    subprocess.run(command, check=True)
    completed_outputs: list[tuple[int, int, str]] = []
    if args.benchmark:
        for benchmark_batch in benchmark_batches:
            for benchmark_repeat in benchmark_repeats:
                if (benchmark_repeat > benchmark_batch or
                        benchmark_batch % benchmark_repeat):
                    continue
                completed = subprocess.run(
                    [str(executable), "--benchmark", str(benchmark_batch),
                     str(benchmark_repeat)], check=True, text=True,
                    capture_output=args.benchmark_output is not None)
                completed_outputs.append(
                    (benchmark_batch, benchmark_repeat,
                     completed.stdout or ""))
        if not completed_outputs:
            parser.error("no repeat divides any requested benchmark batch")
    else:
        subprocess.run([str(executable)], check=True, text=True)
    if args.benchmark_output is not None:
        if not args.benchmark:
            parser.error("--benchmark-output requires --benchmark")
        measurements: dict[tuple, list[float]] = {}
        pattern = re.compile(
            r"^benchmark,r(\d+)/(fp(?:32|64))/(forward|backward)/"
            r"(first|other)/([^/]+)/vg([24])/"
            r"rot([01])/pipe([12])/"
            r"(normal|transposed|broadcast|direct_broadcast),"
            r"(\d+),([0-9.]+)$")
        for benchmark_batch, benchmark_repeat, output_text in completed_outputs:
            print(output_text, end="")
            for line in output_text.splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                (radix, precision, direction, stage, expression, width,
                 rotate, pipeline_depth, context, locality_repeat, ns) = (
                    match.groups())
                if int(locality_repeat) != benchmark_repeat:
                    raise RuntimeError("benchmark reported the wrong repeat")
                locality_repeat = (benchmark_repeat if context in
                                   ("broadcast", "direct_broadcast") else 0)
                key = (benchmark_batch, locality_repeat, int(radix),
                       precision, direction, stage, context, expression,
                       int(width), bool(int(rotate)), int(pipeline_depth))
                measurements.setdefault(key, []).append(float(ns))
        entries = []
        for key in sorted(measurements):
            (benchmark_batch, locality_repeat, radix, precision, direction,
             stage, context, expression, width, rotate, pipeline_depth) = key
            samples = measurements[key]
            if len(samples) < 3 or len(samples) % 3:
                raise RuntimeError(
                    f"expected complete three-pass measurements for {key}")
            entry = {
                "radix": radix, "precision": precision,
                "direction": direction, "stage": stage,
                "batch": benchmark_batch,
                "context": context, "pattern": expression,
                "matrix_vg_width": width,
                "rotate_temp_tiles": rotate,
                "batch_pipeline_depth": pipeline_depth,
                "median_nanoseconds": statistics.median(samples),
                "pass_nanoseconds": samples,
            }
            if locality_repeat:
                entry["locality_repeat"] = locality_repeat
            entries.append(entry)
        payload = {
            "schema_version": 4,
            "compiler_sha256": _compiler_digest(),
            "profile_sha256": profile_digest(MachineProfile.load(args.profile)),
            "formal_measurement": False,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "batches": list(benchmark_batches),
            "locality_repeats": list(benchmark_repeats),
            "passes": 3, "pass_order": "forward,reverse,forward",
            "rounds_per_pass": 9, "repetitions_per_round": 100,
            "entries": entries,
        }
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
