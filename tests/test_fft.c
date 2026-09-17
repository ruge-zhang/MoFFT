#include "mofft.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.141592653589793238462643383279502884

static int check_f64(size_t n, mofft_direction direction, int in_place) {
  mofft_complex_f64 *input = calloc(n, sizeof(*input));
  mofft_complex_f64 *output = calloc(n, sizeof(*output));
  mofft_complex_f64 *reference = calloc(n, sizeof(*reference));
  if (!input || !output || !reference) return 1;
  for (size_t i = 0; i < n; ++i) {
    input[i].real = sin((double)i * 0.17) + (double)(i % 7) * 0.03;
    input[i].imag = cos((double)i * 0.11) - (double)(i % 5) * 0.02;
  }
  for (size_t k = 0; k < n; ++k) {
    for (size_t j = 0; j < n; ++j) {
      double angle = (double)direction * 2.0 * PI * (double)j * (double)k / (double)n;
      reference[k].real += input[j].real * cos(angle) - input[j].imag * sin(angle);
      reference[k].imag += input[j].real * sin(angle) + input[j].imag * cos(angle);
    }
  }
  mofft_plan_f64 *plan = NULL;
  mofft_status status = mofft_plan_create_f64(&plan, n, direction,
      in_place ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "plan f64 n=%zu: %s\n", n, mofft_status_string(status)); return 1;
  }
  if (in_place) {
    memcpy(output, input, n * sizeof(*input));
    status = mofft_execute_f64(plan, output, output);
  } else {
    status = mofft_execute_f64(plan, input, output);
  }
  double max_error = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double e = hypot(output[i].real - reference[i].real,
                     output[i].imag - reference[i].imag);
    if (e > max_error) max_error = e;
  }
  double tolerance = 96.0 * DBL_EPSILON * (double)n;
  if (status != MOFFT_SUCCESS || max_error > tolerance) {
    fprintf(stderr, "f64 n=%zu dir=%d place=%d error=%g tolerance=%g\n",
            n, direction, in_place, max_error, tolerance);
    status = MOFFT_INTERNAL_ERROR;
  }
  mofft_plan_destroy_f64(plan); free(input); free(output); free(reference);
  return status != MOFFT_SUCCESS;
}

static int check_f32(size_t n, mofft_direction direction, int in_place) {
  mofft_complex_f32 *input = calloc(n, sizeof(*input));
  mofft_complex_f32 *output = calloc(n, sizeof(*output));
  mofft_complex_f32 *reference = calloc(n, sizeof(*reference));
  if (!input || !output || !reference) return 1;
  for (size_t i = 0; i < n; ++i) {
    input[i].real = (float)(sin((double)i * 0.17) + (double)(i % 7) * 0.03);
    input[i].imag = (float)(cos((double)i * 0.11) - (double)(i % 5) * 0.02);
  }
  for (size_t k = 0; k < n; ++k) {
    double rr = 0.0, ii = 0.0;
    for (size_t j = 0; j < n; ++j) {
      double angle = (double)direction * 2.0 * PI * (double)j * (double)k / (double)n;
      rr += input[j].real * cos(angle) - input[j].imag * sin(angle);
      ii += input[j].real * sin(angle) + input[j].imag * cos(angle);
    }
    reference[k].real = (float)rr; reference[k].imag = (float)ii;
  }
  mofft_plan_f32 *plan = NULL;
  mofft_status status = mofft_plan_create_f32(&plan, n, direction,
      in_place ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "plan f32 n=%zu: %s\n", n, mofft_status_string(status)); return 1;
  }
  if (in_place) {
    memcpy(output, input, n * sizeof(*input));
    status = mofft_execute_f32(plan, output, output);
  } else {
    status = mofft_execute_f32(plan, input, output);
  }
  double max_error = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double e = hypot((double)output[i].real - reference[i].real,
                     (double)output[i].imag - reference[i].imag);
    if (e > max_error) max_error = e;
  }
  double tolerance = 32.0 * FLT_EPSILON * (double)n;
  if (status != MOFFT_SUCCESS || max_error > tolerance) {
    fprintf(stderr, "f32 n=%zu dir=%d place=%d error=%g tolerance=%g\n",
            n, direction, in_place, max_error, tolerance);
    status = MOFFT_INTERNAL_ERROR;
  }
  mofft_plan_destroy_f32(plan); free(input); free(output); free(reference);
  return status != MOFFT_SUCCESS;
}

static int check_roundtrip(size_t n) {
  int failed = 0;
  mofft_complex_f32 *x32 = calloc(n, sizeof(*x32));
  mofft_complex_f32 *y32 = calloc(n, sizeof(*y32));
  mofft_complex_f32 *z32 = calloc(n, sizeof(*z32));
  mofft_complex_f64 *x64 = calloc(n, sizeof(*x64));
  mofft_complex_f64 *y64 = calloc(n, sizeof(*y64));
  mofft_complex_f64 *z64 = calloc(n, sizeof(*z64));
  if (!x32 || !y32 || !z32 || !x64 || !y64 || !z64) return 1;
  for (size_t i = 0; i < n; ++i) {
    double real = sin((double)i * .017) + (double)(i % 17) * .003;
    double imag = cos((double)i * .013) - (double)(i % 13) * .002;
    x32[i].real = (float)real; x32[i].imag = (float)imag;
    x64[i].real = real; x64[i].imag = imag;
  }
  mofft_plan_f32 *f32 = NULL, *b32 = NULL;
  mofft_plan_f64 *f64 = NULL, *b64 = NULL;
  if (mofft_plan_create_f32(&f32,n,MOFFT_FORWARD,MOFFT_OUT_OF_PLACE) ||
      mofft_plan_create_f32(&b32,n,MOFFT_BACKWARD,MOFFT_OUT_OF_PLACE) ||
      mofft_plan_create_f64(&f64,n,MOFFT_FORWARD,MOFFT_OUT_OF_PLACE) ||
      mofft_plan_create_f64(&b64,n,MOFFT_BACKWARD,MOFFT_OUT_OF_PLACE) ||
      mofft_execute_f32(f32,x32,y32) || mofft_execute_f32(b32,y32,z32) ||
      mofft_execute_f64(f64,x64,y64) || mofft_execute_f64(b64,y64,z64)) {
    failed = 1;
  } else {
    double e32 = 0.0, e64 = 0.0;
    for (size_t i = 0; i < n; ++i) {
      e32 = fmax(e32, hypot((double)z32[i].real / n - x32[i].real,
                            (double)z32[i].imag / n - x32[i].imag));
      e64 = fmax(e64, hypot(z64[i].real / n - x64[i].real,
                            z64[i].imag / n - x64[i].imag));
    }
    double stages = fmax(1.0, log2((double)n));
    double t32 = 128.0 * FLT_EPSILON * stages;
    double t64 = 512.0 * DBL_EPSILON * stages;
    if (e32 > t32 || e64 > t64) {
      fprintf(stderr,"roundtrip n=%zu f32=%g/%g f64=%g/%g\n",
              n,e32,t32,e64,t64); failed = 1;
    }
  }
  mofft_plan_destroy_f32(f32); mofft_plan_destroy_f32(b32);
  mofft_plan_destroy_f64(f64); mofft_plan_destroy_f64(b64);
  free(x32); free(y32); free(z32); free(x64); free(y64); free(z64);
  return failed;
}

static int check_specific_plans(void) {
  const size_t n = 256;
  const int factors[] = {32, 8};
  mofft_complex_f32 *x32 = calloc(n, sizeof(*x32));
  mofft_complex_f32 *a32 = calloc(n, sizeof(*a32));
  mofft_complex_f32 *b32 = calloc(n, sizeof(*b32));
  mofft_complex_f64 *x64 = calloc(n, sizeof(*x64));
  mofft_complex_f64 *a64 = calloc(n, sizeof(*a64));
  mofft_complex_f64 *b64 = calloc(n, sizeof(*b64));
  if (!x32 || !a32 || !b32 || !x64 || !a64 || !b64) return 1;
  for (size_t i = 0; i < n; ++i) {
    x32[i].real = (float)sin(.03 * i);
    x32[i].imag = (float)cos(.02 * i);
    x64[i].real = x32[i].real;
    x64[i].imag = x32[i].imag;
  }
  mofft_plan_f32 *d32 = NULL, *s32 = NULL;
  mofft_plan_f64 *d64 = NULL, *s64 = NULL;
  int failed =
      mofft_plan_create_f32(&d32, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE) ||
      mofft_plan_create_with_radices_f32(&s32, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 2) ||
      mofft_plan_create_f64(&d64, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE) ||
      mofft_plan_create_with_radices_f64(&s64, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 2);
  if (!failed) {
    failed = mofft_execute_f32(d32, x32, a32) ||
             mofft_execute_f32(s32, x32, b32) ||
             mofft_execute_f64(d64, x64, a64) ||
             mofft_execute_f64(s64, x64, b64);
  }
  double e32 = 0.0, e64 = 0.0;
  for (size_t i = 0; i < n && !failed; ++i) {
    e32 = fmax(e32, hypot((double)a32[i].real - b32[i].real,
                          (double)a32[i].imag - b32[i].imag));
    e64 = fmax(e64, hypot(a64[i].real - b64[i].real,
                          a64[i].imag - b64[i].imag));
  }
  if (e32 > 64.0 * FLT_EPSILON * n || e64 > 256.0 * DBL_EPSILON * n)
    failed = 1;
  const int invalid[] = {17, 16};
  mofft_plan_f32 *bad = NULL;
  if (mofft_plan_create_with_radices_f32(&bad, n, MOFFT_FORWARD,
      MOFFT_OUT_OF_PLACE, invalid, 2) != MOFFT_INVALID_ARGUMENT) failed = 1;
  mofft_plan_destroy_f32(d32); mofft_plan_destroy_f32(s32);
  mofft_plan_destroy_f64(d64); mofft_plan_destroy_f64(s64);
  free(x32); free(a32); free(b32); free(x64); free(a64); free(b64);
  return failed;
}

static int check_transposed_first_kernels(void) {
  const int radices[] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64};
  int failed = 0;
  for (size_t item = 0; item < sizeof(radices) / sizeof(radices[0]); ++item) {
    const int factors[] = {2, radices[item]};
    const size_t n = (size_t)2 * (size_t)radices[item];
    mofft_complex_f32 *x32 = calloc(n, sizeof(*x32));
    mofft_complex_f32 *y32 = calloc(n, sizeof(*y32));
    mofft_complex_f64 *x64 = calloc(n, sizeof(*x64));
    mofft_complex_f64 *y64 = calloc(n, sizeof(*y64));
    mofft_complex_f64 *reference = calloc(n, sizeof(*reference));
    if (!x32 || !y32 || !x64 || !y64 || !reference) return 1;
    for (size_t i = 0; i < n; ++i) {
      x64[i].real = sin(.13 * i) + .01 * (i % 3);
      x64[i].imag = cos(.07 * i) - .02 * (i % 5);
      x32[i].real = (float)x64[i].real;
      x32[i].imag = (float)x64[i].imag;
    }
    for (size_t k = 0; k < n; ++k)
      for (size_t j = 0; j < n; ++j) {
        double angle = -2.0 * PI * (double)j * (double)k / (double)n;
        reference[k].real += x64[j].real * cos(angle) -
                             x64[j].imag * sin(angle);
        reference[k].imag += x64[j].real * sin(angle) +
                             x64[j].imag * cos(angle);
      }
    mofft_plan_f32 *p32 = NULL;
    mofft_plan_f64 *p64 = NULL;
    if (mofft_plan_create_with_radices_f32(&p32, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 2) ||
        mofft_plan_create_with_radices_f64(&p64, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 2) ||
        mofft_execute_f32(p32, x32, y32) ||
        mofft_execute_f64(p64, x64, y64)) {
      failed = 1;
    } else {
      double e32 = 0.0, e64 = 0.0;
      for (size_t i = 0; i < n; ++i) {
        e32 = fmax(e32, hypot((double)y32[i].real - reference[i].real,
                              (double)y32[i].imag - reference[i].imag));
        e64 = fmax(e64, hypot(y64[i].real - reference[i].real,
                              y64[i].imag - reference[i].imag));
      }
      if (e32 > 64.0 * FLT_EPSILON * n ||
          e64 > 256.0 * DBL_EPSILON * n) {
        fprintf(stderr, "transposed first radix=%d f32=%g f64=%g\n",
                radices[item], e32, e64);
        failed = 1;
      }
    }
    mofft_plan_destroy_f32(p32); mofft_plan_destroy_f64(p64);
    free(x32); free(y32); free(x64); free(y64); free(reference);
  }
  return failed;
}

static int check_transpose_strategies(void) {
  const size_t n = 4096;
  const int factors[] = {16, 16, 16};
  mofft_complex_f32 *x32 = calloc(n, sizeof(*x32));
  mofft_complex_f32 *linear32 = calloc(n, sizeof(*linear32));
  mofft_complex_f32 *blocked32 = calloc(n, sizeof(*blocked32));
  mofft_complex_f64 *x64 = calloc(n, sizeof(*x64));
  mofft_complex_f64 *linear64 = calloc(n, sizeof(*linear64));
  mofft_complex_f64 *blocked64 = calloc(n, sizeof(*blocked64));
  if (!x32 || !linear32 || !blocked32 || !x64 || !linear64 || !blocked64)
    return 1;
  for (size_t i = 0; i < n; ++i) {
    x64[i].real = sin(.017 * i); x64[i].imag = cos(.013 * i);
    x32[i].real = (float)x64[i].real; x32[i].imag = (float)x64[i].imag;
  }
  mofft_plan_f32 *a32 = NULL, *b32 = NULL;
  mofft_plan_f64 *a64 = NULL, *b64 = NULL;
  int failed =
      mofft_plan_create_with_radices_f32(&a32, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 3) ||
      mofft_plan_create_with_radices_f32(&b32, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 3) ||
      mofft_plan_create_with_radices_f64(&a64, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 3) ||
      mofft_plan_create_with_radices_f64(&b64, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, factors, 3) ||
      mofft_plan_set_direct_input_f32(a32, 0) ||
      mofft_plan_set_direct_input_f32(b32, 0) ||
      mofft_plan_set_direct_input_f64(a64, 0) ||
      mofft_plan_set_direct_input_f64(b64, 0) ||
      mofft_plan_set_blocked_transpose_mask_f32(b32, UINT64_C(1)) ||
      mofft_plan_set_blocked_transpose_mask_f64(b64, UINT64_C(2));
  if (!failed)
    failed = mofft_execute_f32(a32, x32, linear32) ||
             mofft_execute_f32(b32, x32, blocked32) ||
             mofft_execute_f64(a64, x64, linear64) ||
             mofft_execute_f64(b64, x64, blocked64);
  double e32 = 0.0, e64 = 0.0;
  for (size_t i = 0; i < n && !failed; ++i) {
    e32 = fmax(e32, hypot((double)linear32[i].real - blocked32[i].real,
                          (double)linear32[i].imag - blocked32[i].imag));
    e64 = fmax(e64, hypot(linear64[i].real - blocked64[i].real,
                          linear64[i].imag - blocked64[i].imag));
  }
  if (e32 != 0.0 || e64 != 0.0) {
    fprintf(stderr, "transpose strategy mismatch f32=%g f64=%g\n", e32, e64);
    failed = 1;
  }
  mofft_plan_destroy_f32(a32); mofft_plan_destroy_f32(b32);
  mofft_plan_destroy_f64(a64); mofft_plan_destroy_f64(b64);
  free(x32); free(linear32); free(blocked32);
  free(x64); free(linear64); free(blocked64);
  return failed;
}

static int check_stage_layouts(void) {
  const size_t n = 4096;
  static const struct {
    int factors[6];
    size_t count;
  } cases[] = {
      {{8, 16, 32}, 3},
      {{8, 8, 64}, 3},
      {{8, 8, 8, 8}, 4},
      {{4, 4, 16, 16}, 4},
      {{4, 4, 4, 4, 4, 4}, 6},
  };
  int failed = 0;
  mofft_complex_f32 *input32 = calloc(n, sizeof(*input32));
  mofft_complex_f32 *transpose32 = calloc(n, sizeof(*transpose32));
  mofft_complex_f32 *section32 = calloc(n, sizeof(*section32));
  mofft_complex_f64 *input64 = calloc(n, sizeof(*input64));
  mofft_complex_f64 *transpose64 = calloc(n, sizeof(*transpose64));
  mofft_complex_f64 *section64 = calloc(n, sizeof(*section64));
  if (!input32 || !transpose32 || !section32 || !input64 || !transpose64 ||
      !section64) {
    free(input32); free(transpose32); free(section32);
    free(input64); free(transpose64); free(section64);
    return 1;
  }
  for (size_t i = 0; i < n; ++i) {
    input64[i].real = sin(.017 * i);
    input64[i].imag = cos(.013 * i);
    input32[i].real = (float)input64[i].real;
    input32[i].imag = (float)input64[i].imag;
  }
  for (size_t test_case = 0;
       test_case < sizeof(cases) / sizeof(cases[0]); ++test_case) {
    for (int direction = -1; direction <= 1; direction += 2) {
      for (int in_place = 0; in_place <= 1; ++in_place) {
        const mofft_placement placement = in_place ? MOFFT_IN_PLACE
                                                    : MOFFT_OUT_OF_PLACE;
        mofft_plan_f32 *a32 = NULL, *b32 = NULL;
        mofft_plan_f64 *a64 = NULL, *b64 = NULL;
        memcpy(transpose32, input32, n * sizeof(*input32));
        memcpy(section32, input32, n * sizeof(*input32));
        memcpy(transpose64, input64, n * sizeof(*input64));
        memcpy(section64, input64, n * sizeof(*input64));
        failed |= mofft_plan_create_with_radices_f32(
            &a32, n, (mofft_direction)direction, placement,
            cases[test_case].factors, cases[test_case].count) ||
            mofft_plan_create_with_radices_f32(
                &b32, n, (mofft_direction)direction, placement,
                cases[test_case].factors, cases[test_case].count) ||
            mofft_plan_create_with_radices_f64(
                &a64, n, (mofft_direction)direction, placement,
                cases[test_case].factors, cases[test_case].count) ||
            mofft_plan_create_with_radices_f64(
                &b64, n, (mofft_direction)direction, placement,
                cases[test_case].factors, cases[test_case].count) ||
            mofft_plan_set_stage_layout_f32(
                b32, MOFFT_STAGE_LAYOUT_SECTION) ||
            mofft_plan_set_stage_layout_f64(
                b64, MOFFT_STAGE_LAYOUT_SECTION);
        if (!failed) {
          const mofft_complex_f32 *source_a32 =
              in_place ? transpose32 : input32;
          const mofft_complex_f32 *source_b32 =
              in_place ? section32 : input32;
          const mofft_complex_f64 *source_a64 =
              in_place ? transpose64 : input64;
          const mofft_complex_f64 *source_b64 =
              in_place ? section64 : input64;
          failed |= mofft_execute_f32(a32, source_a32, transpose32) ||
                    mofft_execute_f32(b32, source_b32, section32) ||
                    mofft_execute_f64(a64, source_a64, transpose64) ||
                    mofft_execute_f64(b64, source_b64, section64);
        }
        double error32 = 0.0, error64 = 0.0;
        for (size_t i = 0; i < n && !failed; ++i) {
          error32 = fmax(error32,
              hypot((double)transpose32[i].real - section32[i].real,
                    (double)transpose32[i].imag - section32[i].imag));
          error64 = fmax(error64,
              hypot(transpose64[i].real - section64[i].real,
                    transpose64[i].imag - section64[i].imag));
        }
        if (error32 > 128.0 * FLT_EPSILON * n ||
            error64 > 512.0 * DBL_EPSILON * n) {
          fprintf(stderr,
                  "stage layout mismatch case=%zu dir=%d place=%d f32=%g "
                  "f64=%g\n",
                  test_case, direction, in_place, error32, error64);
          failed = 1;
        }
        mofft_plan_destroy_f32(a32); mofft_plan_destroy_f32(b32);
        mofft_plan_destroy_f64(a64); mofft_plan_destroy_f64(b64);
      }
    }
  }
  free(input32); free(transpose32); free(section32);
  free(input64); free(transpose64); free(section64);
  return failed;
}

static int check_broadcast_other_kernels(void) {
  const int radices[] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64};
  int failed = 0;
  for (size_t item = 0; item < sizeof(radices) / sizeof(radices[0]); ++item) {
    const int factors[] = {13, radices[item], 2};
    const size_t n = (size_t)26 * (size_t)radices[item];
    mofft_complex_f32 *x32 = calloc(n, sizeof(*x32));
    mofft_complex_f32 *y32 = calloc(n, sizeof(*y32));
    mofft_complex_f32 *z32 = calloc(n, sizeof(*z32));
    mofft_complex_f64 *x64 = calloc(n, sizeof(*x64));
    mofft_complex_f64 *y64 = calloc(n, sizeof(*y64));
    mofft_complex_f64 *z64 = calloc(n, sizeof(*z64));
    if (!x32 || !y32 || !z32 || !x64 || !y64 || !z64) return 1;
    for (size_t i = 0; i < n; ++i) {
      x64[i].real = sin(.017 * i); x64[i].imag = cos(.013 * i);
      x32[i].real = (float)x64[i].real; x32[i].imag = (float)x64[i].imag;
    }
    mofft_plan_f32 *f32 = NULL, *b32 = NULL, *reference32 = NULL;
    mofft_plan_f64 *f64 = NULL, *b64 = NULL, *reference64 = NULL;
    failed |=
        mofft_plan_create_with_radices_f32(&f32, n, MOFFT_FORWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_create_with_radices_f32(&b32, n, MOFFT_BACKWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_create_with_radices_f64(&f64, n, MOFFT_FORWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_create_with_radices_f64(&b64, n, MOFFT_BACKWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_create_with_radices_f32(&reference32, n, MOFFT_FORWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_create_with_radices_f64(&reference64, n, MOFFT_FORWARD,
            MOFFT_OUT_OF_PLACE, factors, 3) ||
        mofft_plan_set_direct_input_f32(reference32, 0) ||
        mofft_plan_set_direct_input_f64(reference64, 0);
    if (!failed)
      failed |= mofft_execute_f32(f32, x32, y32) ||
                mofft_execute_f32(reference32, x32, z32) ||
                mofft_execute_f64(f64, x64, y64) ||
                mofft_execute_f64(reference64, x64, z64);
    double path_error32 = 0.0, path_error64 = 0.0;
    for (size_t i = 0; i < n && !failed; ++i) {
      path_error32 = fmax(path_error32,
          hypot((double)y32[i].real - z32[i].real,
                (double)y32[i].imag - z32[i].imag));
      path_error64 = fmax(path_error64,
          hypot(y64[i].real - z64[i].real, y64[i].imag - z64[i].imag));
    }
    if (path_error32 > 128.0 * FLT_EPSILON * radices[item] *
                           log2((double)n) ||
        path_error64 > 192.0 * DBL_EPSILON * radices[item] *
                           log2((double)n)) {
      fprintf(stderr,
              "direct-input mismatch radix=%d errors=(%.9g, %.17g)\n",
              radices[item], path_error32, path_error64);
      failed = 1;
    }
    if (!failed)
      failed |= mofft_execute_f32(b32, y32, z32) ||
                mofft_execute_f64(b64, y64, z64);
    double e32 = 0.0, e64 = 0.0;
    for (size_t i = 0; i < n && !failed; ++i) {
      e32 = fmax(e32, hypot((double)z32[i].real / n - x32[i].real,
                            (double)z32[i].imag / n - x32[i].imag));
      e64 = fmax(e64, hypot(z64[i].real / n - x64[i].real,
                            z64[i].imag / n - x64[i].imag));
    }
    if (e32 > 128.0 * FLT_EPSILON * log2((double)n) ||
        e64 > 512.0 * DBL_EPSILON * log2((double)n)) {
      fprintf(stderr, "broadcast other radix=%d f32=%g f64=%g\n",
              radices[item], e32, e64);
      failed = 1;
    }
    mofft_plan_destroy_f32(f32); mofft_plan_destroy_f32(b32);
    mofft_plan_destroy_f32(reference32);
    mofft_plan_destroy_f64(f64); mofft_plan_destroy_f64(b64);
    mofft_plan_destroy_f64(reference64);
    free(x32); free(y32); free(z32); free(x64); free(y64); free(z64);
  }
  return failed;
}

int main(void) {
  const size_t sizes[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 32, 45, 64, 144, 225, 256};
  int failures = 0;
  const double small_stage = mofft_estimated_stage_cost(
      8, 64, 0, 4096, 512);
  const double large_stage = mofft_estimated_stage_cost(
      8, 64, 0, 262144, 32768);
  if (!isfinite(small_stage) || !isfinite(large_stage) ||
      large_stage <= small_stage ||
      mofft_estimated_stage_cost(8, 16, 0, 4096, 512) != DBL_MAX)
    failures++;
  {
    const int plan_radices[] = {8, 8, 8, 8};
    const double materialized = mofft_estimated_plan_cost(
        4096, 64, plan_radices, 4, 0, 0);
    const double blocked = mofft_estimated_plan_cost(
        4096, 64, plan_radices, 4, 7, 0);
    const double direct = mofft_estimated_plan_cost(
        4096, 64, plan_radices, 4, 0, 1);
    const double section = mofft_estimated_plan_cost_with_layout(
        4096, 64, plan_radices, 4, 0, 0, MOFFT_STAGE_LAYOUT_SECTION);
    mofft_plan_cost_breakdown section_detail, linear_detail, blocked_detail;
    const int section_status =
        mofft_estimated_plan_cost_breakdown_with_layout(
            4096, 64, plan_radices, 4, 0, 0,
            MOFFT_STAGE_LAYOUT_SECTION, &section_detail);
    const int linear_status =
        mofft_estimated_plan_cost_breakdown_with_layout(
            4096, 64, plan_radices, 4, 0, 0,
            MOFFT_STAGE_LAYOUT_TRANSPOSE, &linear_detail);
    const int blocked_status =
        mofft_estimated_plan_cost_breakdown_with_layout(
            4096, 64, plan_radices, 4, 7, 0,
            MOFFT_STAGE_LAYOUT_TRANSPOSE, &blocked_detail);
    /* A measured layout profile may legitimately rank linear and blocked
     * transpose differently for this shape. Test accounting invariants here,
     * not a bootstrap-profile-specific ordering. */
    if (!isfinite(materialized) || !isfinite(blocked) || !isfinite(direct) ||
        !isfinite(section) ||
        section_status != MOFFT_SUCCESS || linear_status != MOFFT_SUCCESS ||
        blocked_status != MOFFT_SUCCESS ||
        section_detail.logical_read_bytes != 336896 ||
        section_detail.logical_write_bytes != 262144 ||
        section_detail.transferred_read_bytes != 336896 ||
        section_detail.transferred_write_bytes != 262144 ||
        section_detail.peak_working_set_bytes != 196608 ||
        section_detail.layout_cost != 0.0 ||
        linear_detail.logical_read_bytes != 533504 ||
        linear_detail.logical_write_bytes != 458752 ||
        linear_detail.transferred_read_bytes != 992256 ||
        linear_detail.transferred_write_bytes != 458752 ||
        blocked_detail.transferred_read_bytes != 533504 ||
        !(linear_detail.layout_cost > 0.0) ||
        fabs(section_detail.total_cost - section) > 1e-12 ||
        fabs(linear_detail.total_cost - materialized) > 1e-12 ||
        mofft_estimated_plan_cost(4096, 64, plan_radices, 3, 0, 1) !=
            DBL_MAX ||
        mofft_estimated_plan_cost_breakdown_with_layout(
            4096, 64, plan_radices, 4, 0, 0,
            MOFFT_STAGE_LAYOUT_SECTION, NULL) != MOFFT_INVALID_ARGUMENT ||
        mofft_estimated_plan_cost_with_layout(
            4096, 64, plan_radices, 4, 0, 0,
            (mofft_stage_layout)99) != DBL_MAX) {
      fprintf(stderr,
              "plan-model mismatch costs=(linear %.9g, blocked %.9g, "
              "direct %.9g, section %.9g) layout=(%.9g, %.9g)\n",
              materialized, blocked, direct, section,
              linear_detail.layout_cost, blocked_detail.layout_cost);
      failures++;
    }
  }
  failures += check_specific_plans();
  failures += check_transposed_first_kernels();
  failures += check_transpose_strategies();
  failures += check_stage_layouts();
  failures += check_broadcast_other_kernels();
  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    for (int direction = -1; direction <= 1; direction += 2) {
      for (int in_place = 0; in_place <= 1; ++in_place) {
        failures += check_f32(sizes[i], (mofft_direction)direction, in_place);
        failures += check_f64(sizes[i], (mofft_direction)direction, in_place);
      }
    }
  }
  mofft_plan_f32 *bad = NULL;
  if (mofft_plan_create_f32(&bad, 17, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE)
      != MOFFT_UNSUPPORTED_SIZE) failures++;
  const size_t extended[] = {256,512,1024,2048,4096,8192,16384,32768,
    65536,131072,262144,144,169,196,225,1728,2197,2744,3375,20736,28561,
    38416,50625,248832,371293,537824,759375};
  for (size_t i=0;i<sizeof(extended)/sizeof(extended[0]);++i) {
    mofft_plan_f32 *p32=NULL; mofft_plan_f64 *p64=NULL;
    if (mofft_plan_create_f32(&p32,extended[i],MOFFT_FORWARD,
                              MOFFT_OUT_OF_PLACE)!=MOFFT_SUCCESS) failures++;
    if (mofft_plan_create_f64(&p64,extended[i],MOFFT_FORWARD,
                              MOFFT_OUT_OF_PLACE)!=MOFFT_SUCCESS) failures++;
    mofft_plan_destroy_f32(p32); mofft_plan_destroy_f64(p64);
  }
  const size_t roundtrip[] = {169,196,1024,1728,2197,38416,248832};
  for (size_t i=0;i<sizeof(roundtrip)/sizeof(roundtrip[0]);++i)
    failures += check_roundtrip(roundtrip[i]);
  if (failures) fprintf(stderr, "%d correctness checks failed\n", failures);
  return failures ? 1 : 0;
}
