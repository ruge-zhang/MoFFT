#include "mofft.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_N 30
#define TEST_PI 3.141592653589793238462643383279502884

static void fill_input_f32(mofft_complex_f32 *input, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    input[i].real = (float)(sin((double)i * 0.31) + (double)(i % 5) * 0.07);
    input[i].imag = (float)(cos((double)i * 0.23) - (double)(i % 7) * 0.03);
  }
}

static void fill_input_f64(mofft_complex_f64 *input, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    input[i].real = sin((double)i * 0.31) + (double)(i % 5) * 0.07;
    input[i].imag = cos((double)i * 0.23) - (double)(i % 7) * 0.03;
  }
}

static void dft_f32(const mofft_complex_f32 *input, mofft_complex_f32 *output,
                    size_t n, mofft_direction direction) {
  for (size_t k = 0; k < n; ++k) {
    double real = 0.0, imag = 0.0;
    for (size_t j = 0; j < n; ++j) {
      const double angle = (double)direction * 2.0 * TEST_PI *
                           (double)j * (double)k / (double)n;
      real += input[j].real * cos(angle) - input[j].imag * sin(angle);
      imag += input[j].real * sin(angle) + input[j].imag * cos(angle);
    }
    output[k].real = (float)real;
    output[k].imag = (float)imag;
  }
}

static void dft_f64(const mofft_complex_f64 *input, mofft_complex_f64 *output,
                    size_t n, mofft_direction direction) {
  for (size_t k = 0; k < n; ++k) {
    double real = 0.0, imag = 0.0;
    for (size_t j = 0; j < n; ++j) {
      const double angle = (double)direction * 2.0 * TEST_PI *
                           (double)j * (double)k / (double)n;
      real += input[j].real * cos(angle) - input[j].imag * sin(angle);
      imag += input[j].real * sin(angle) + input[j].imag * cos(angle);
    }
    output[k].real = real;
    output[k].imag = imag;
  }
}

static int compare_f32(const mofft_complex_f32 *actual,
                       const mofft_complex_f32 *expected, size_t n) {
  double max_error = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double error = hypot((double)actual[i].real - expected[i].real,
                               (double)actual[i].imag - expected[i].imag);
    if (error > max_error) max_error = error;
  }
  if (max_error > 2e-4 * (double)n) {
    fprintf(stderr, "fp32 max error %g\n", max_error);
    return 1;
  }
  return 0;
}

static int compare_f64(const mofft_complex_f64 *actual,
                       const mofft_complex_f64 *expected, size_t n) {
  double max_error = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double error = hypot(actual[i].real - expected[i].real,
                               actual[i].imag - expected[i].imag);
    if (error > max_error) max_error = error;
  }
  if (max_error > 2e-11 * (double)n) {
    fprintf(stderr, "fp64 max error %g\n", max_error);
    return 1;
  }
  return 0;
}

static int test_measured_planner(void) {
  mofft_complex_f32 input32[TEST_N], output32[TEST_N], reference32[TEST_N];
  mofft_complex_f64 input64[TEST_N], output64[TEST_N], reference64[TEST_N];
  fill_input_f32(input32, TEST_N);
  fill_input_f64(input64, TEST_N);
  const mofft_direction directions[] = {MOFFT_FORWARD, MOFFT_BACKWARD};
  const mofft_placement placements[] = {MOFFT_OUT_OF_PLACE, MOFFT_IN_PLACE};
  mofft_plan_options options;
  mofft_plan_options_init(&options);
  options.mode = MOFFT_PLANNER_MEASURE;
  options.candidate_limit = 4;
  options.timing_samples = 3;

  for (size_t d = 0; d < sizeof(directions) / sizeof(directions[0]); ++d) {
    for (size_t p = 0; p < sizeof(placements) / sizeof(placements[0]); ++p) {
      mofft_plan_diagnostics diagnostics32, diagnostics64;
      mofft_plan_f32 *plan32 = NULL;
      mofft_plan_f64 *plan64 = NULL;
      if (mofft_plan_create_with_options_f32(
              &plan32, TEST_N, directions[d], placements[p], &options,
              &diagnostics32) != MOFFT_SUCCESS ||
          mofft_plan_create_with_options_f64(
              &plan64, TEST_N, directions[d], placements[p], &options,
              &diagnostics64) != MOFFT_SUCCESS) {
        fprintf(stderr, "measured planner failed\n");
        mofft_plan_destroy_f32(plan32);
        mofft_plan_destroy_f64(plan64);
        return 1;
      }
      if (!diagnostics32.measured || !diagnostics64.measured ||
          diagnostics32.candidates_measured == 0 ||
          diagnostics32.candidates_measured > options.candidate_limit ||
          diagnostics64.candidates_measured == 0 ||
          diagnostics64.candidates_measured > options.candidate_limit ||
          diagnostics32.kernel_bank >= diagnostics32.kernel_bank_count ||
          diagnostics64.kernel_bank >= diagnostics64.kernel_bank_count) {
        fprintf(stderr, "measured planner returned invalid diagnostics\n");
        mofft_plan_destroy_f32(plan32);
        mofft_plan_destroy_f64(plan64);
        return 1;
      }
      dft_f32(input32, reference32, TEST_N, directions[d]);
      dft_f64(input64, reference64, TEST_N, directions[d]);
      const mofft_complex_f32 *run_input32 = input32;
      mofft_complex_f32 *run_output32 = output32;
      const mofft_complex_f64 *run_input64 = input64;
      mofft_complex_f64 *run_output64 = output64;
      if (placements[p] == MOFFT_IN_PLACE) {
        memcpy(output32, input32, sizeof(input32));
        memcpy(output64, input64, sizeof(input64));
        run_input32 = output32;
        run_output32 = output32;
        run_input64 = output64;
        run_output64 = output64;
      }
      const int failed =
          mofft_execute_f32(plan32, run_input32, run_output32) != MOFFT_SUCCESS ||
          mofft_execute_f64(plan64, run_input64, run_output64) != MOFFT_SUCCESS ||
          compare_f32(run_output32, reference32, TEST_N) ||
          compare_f64(run_output64, reference64, TEST_N);
      mofft_plan_destroy_f32(plan32);
      mofft_plan_destroy_f64(plan64);
      if (failed) return 1;
    }
  }
  return 0;
}

static int test_large_measured_plan(void) {
  const size_t n = 262144;
  mofft_complex_f64 *input =
      (mofft_complex_f64 *)malloc(n * sizeof(*input));
  mofft_complex_f64 *original =
      (mofft_complex_f64 *)malloc(n * sizeof(*original));
  mofft_complex_f64 *output =
      (mofft_complex_f64 *)malloc(n * sizeof(*output));
  if (!input || !original || !output) {
    free(input); free(original); free(output);
    fprintf(stderr, "large measured-plan test allocation failed\n");
    return 1;
  }
  for (size_t i = 0; i < n; ++i) {
    input[i].real = (double)((int)(i % 101) - 50) / 101.0;
    input[i].imag = (double)((int)(i % 67) - 33) / 67.0;
  }
  memcpy(original, input, n * sizeof(*input));

  mofft_plan_options options;
  mofft_plan_options_init(&options);
  options.mode = MOFFT_PLANNER_MEASURE;
  options.candidate_limit = 4;
  options.timing_samples = 3;
  mofft_plan_diagnostics diagnostics;
  mofft_plan_f64 *plan = NULL;
  const mofft_status status = mofft_plan_create_with_options_f64(
      &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  if (status != MOFFT_SUCCESS || !plan || !diagnostics.measured ||
      diagnostics.candidates_measured == 0 ||
      diagnostics.candidates_measured > options.candidate_limit ||
      diagnostics.kernel_bank_count == 0 ||
      diagnostics.kernel_bank >= diagnostics.kernel_bank_count ||
      !(diagnostics.selected_median_seconds > 0.0) ||
      memcmp(input, original, n * sizeof(*input)) != 0) {
    fprintf(stderr, "large FP64 measured-plan creation failed: %s\n",
            mofft_status_string(status));
    mofft_plan_destroy_f64(plan);
    free(input); free(original); free(output);
    return 1;
  }
  const mofft_status execute_status = mofft_execute_f64(plan, input, output);
  int non_finite = 0;
  for (size_t i = 0; i < n; ++i)
    if (!isfinite(output[i].real) || !isfinite(output[i].imag)) {
      non_finite = 1;
      break;
    }
  mofft_plan_destroy_f64(plan);
  free(input); free(original); free(output);
  if (execute_status != MOFFT_SUCCESS || non_finite) {
    fprintf(stderr, "large FP64 measured-plan execution failed: %s\n",
            mofft_status_string(execute_status));
    return 1;
  }
  return 0;
}

static int test_pretuned_and_bound_api(void) {
  mofft_plan_options options;
  mofft_plan_diagnostics diagnostics;
  mofft_plan_options_init(&options);
  mofft_plan_f32 *fallback = NULL;
  if (mofft_plan_create_with_options_f32(
          &fallback, TEST_N, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options,
          &diagnostics) != MOFFT_SUCCESS ||
      diagnostics.measured || diagnostics.radix_count == 0 ||
      diagnostics.kernel_bank_count == 0 ||
      diagnostics.kernel_bank != 0) {
    mofft_plan_destroy_f32(fallback);
    fprintf(stderr, "PRETUNED planner diagnostics failed\n");
    return 1;
  }
  mofft_plan_destroy_f32(fallback);

  mofft_plan_f32 *strict = NULL;
  options.flags = MOFFT_PLAN_STRICT_WISDOM;
  const mofft_status strict_status = mofft_plan_create_with_options_f32(
      &strict, 17, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, NULL);
  mofft_plan_destroy_f32(strict);
  if (strict_status != MOFFT_WISDOM_NOT_FOUND) {
    fprintf(stderr, "strict wisdom expected miss, got %s\n",
            mofft_status_string(strict_status));
    return 1;
  }

  mofft_complex_f32 input[TEST_N], output[TEST_N], reference[TEST_N];
  fill_input_f32(input, TEST_N);
  dft_f32(input, reference, TEST_N, MOFFT_FORWARD);
  mofft_plan_f32 *bound = mofft_plan_dft_1d_f32(
      TEST_N, input, output, MOFFT_FORWARD, 0);
  if (!bound || mofft_execute_plan_f32(bound) != MOFFT_SUCCESS ||
      compare_f32(output, reference, TEST_N)) {
    mofft_plan_destroy_f32(bound);
    fprintf(stderr, "bound-array PRETUNED API failed\n");
    return 1;
  }
  mofft_plan_destroy_f32(bound);
  return 0;
}

static int test_each_compiled_bank(void) {
  mofft_complex_f32 input[TEST_N], output[TEST_N], reference[TEST_N];
  mofft_complex_f64 input64[TEST_N], output64[TEST_N], reference64[TEST_N];
  const int radices[] = {3, 10};
  /* The diagnostics API reports the actual compiled bank count without
   * exposing generated headers to library consumers. */
  mofft_plan_options options;
  mofft_plan_diagnostics diagnostics;
  mofft_plan_options_init(&options);
  mofft_plan_f32 *probe = NULL;
  if (mofft_plan_create_with_options_f32(&probe, TEST_N, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, &options, &diagnostics) != MOFFT_SUCCESS)
    return 1;
  mofft_plan_destroy_f32(probe);
  fill_input_f32(input, TEST_N);
  fill_input_f64(input64, TEST_N);
  const unsigned int banks = diagnostics.kernel_bank_count;
  const mofft_direction directions[] = {MOFFT_FORWARD, MOFFT_BACKWARD};
  for (unsigned int bank = 0; bank < banks; ++bank) {
    for (size_t d = 0; d < 2; ++d) {
      mofft_plan_f32 *plan32 = NULL;
      mofft_plan_f64 *plan64 = NULL;
      if (mofft_plan_create_with_radices_f32(&plan32, TEST_N, directions[d],
              MOFFT_OUT_OF_PLACE, radices, 2) != MOFFT_SUCCESS ||
          mofft_plan_create_with_radices_f64(&plan64, TEST_N, directions[d],
              MOFFT_OUT_OF_PLACE, radices, 2) != MOFFT_SUCCESS ||
          mofft_plan_set_kernel_bank_f32(plan32, bank) != MOFFT_SUCCESS ||
          mofft_plan_set_kernel_bank_f64(plan64, bank) != MOFFT_SUCCESS) {
        mofft_plan_destroy_f32(plan32);
        mofft_plan_destroy_f64(plan64);
        return 1;
      }
      dft_f32(input, reference, TEST_N, directions[d]);
      dft_f64(input64, reference64, TEST_N, directions[d]);
      const int failed =
          mofft_execute_f32(plan32, input, output) != MOFFT_SUCCESS ||
          mofft_execute_f64(plan64, input64, output64) != MOFFT_SUCCESS ||
          compare_f32(output, reference, TEST_N) ||
          compare_f64(output64, reference64, TEST_N);
      mofft_plan_destroy_f32(plan32);
      mofft_plan_destroy_f64(plan64);
      if (failed) {
        fprintf(stderr, "kernel bank %u correctness failed\n", bank);
        return 1;
      }
    }
  }
  return 0;
}

static int test_single_stage_in_place(void) {
  enum { n = 64 };
  const int radix[] = {n};
  mofft_complex_f32 input32[n], output32[n], reference32[n];
  mofft_complex_f64 input64[n], output64[n], reference64[n];
  mofft_plan_options options;
  mofft_plan_diagnostics diagnostics;
  mofft_plan_options_init(&options);
  mofft_plan_f32 *probe = NULL;
  if (mofft_plan_create_with_options_f32(&probe, n, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, &options, &diagnostics) != MOFFT_SUCCESS) {
    fprintf(stderr, "unable to query compiled kernel banks\n");
    return 1;
  }
  mofft_plan_destroy_f32(probe);
  const mofft_direction directions[] = {MOFFT_FORWARD, MOFFT_BACKWARD};
  for (unsigned int bank = 0; bank < diagnostics.kernel_bank_count; ++bank) {
    for (size_t d = 0; d < sizeof(directions) / sizeof(directions[0]); ++d) {
      fill_input_f32(input32, n);
      fill_input_f64(input64, n);
      dft_f32(input32, reference32, n, directions[d]);
      dft_f64(input64, reference64, n, directions[d]);
      memcpy(output32, input32, sizeof(input32));
      memcpy(output64, input64, sizeof(input64));
      mofft_plan_f32 *plan32 = NULL;
      mofft_plan_f64 *plan64 = NULL;
      if (mofft_plan_create_with_radices_f32(&plan32, n, directions[d],
              MOFFT_IN_PLACE, radix, 1) != MOFFT_SUCCESS ||
          mofft_plan_create_with_radices_f64(&plan64, n, directions[d],
              MOFFT_IN_PLACE, radix, 1) != MOFFT_SUCCESS ||
          mofft_plan_set_kernel_bank_f32(plan32, bank) != MOFFT_SUCCESS ||
          mofft_plan_set_kernel_bank_f64(plan64, bank) != MOFFT_SUCCESS) {
        mofft_plan_destroy_f32(plan32);
        mofft_plan_destroy_f64(plan64);
        return 1;
      }
      const int failed =
          mofft_execute_f32(plan32, output32, output32) != MOFFT_SUCCESS ||
          mofft_execute_f64(plan64, output64, output64) != MOFFT_SUCCESS ||
          compare_f32(output32, reference32, n) ||
          compare_f64(output64, reference64, n);
      mofft_plan_destroy_f32(plan32);
      mofft_plan_destroy_f64(plan64);
      if (failed) {
        fprintf(stderr, "radix-64 in-place correctness failed (bank %u)\n",
                bank);
        return 1;
      }
    }
  }
  return 0;
}

int main(void) {
  if (test_measured_planner() || test_pretuned_and_bound_api() ||
      test_each_compiled_bank() || test_single_stage_in_place() ||
      test_large_measured_plan())
    return 1;
  puts("planner API and compiled kernel-bank correctness passed");
  return 0;
}
