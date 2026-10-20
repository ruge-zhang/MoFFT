#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mofft.h>

#ifndef MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE
#define MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE 0
#endif

static int check_f32(mofft_planner_mode mode) {
  const size_t n = 8;
  mofft_complex_f32 input[8] = {{1.0f, 0.0f}};
  mofft_complex_f32 output[8] = {{0.0f, 0.0f}};
  mofft_plan_options options;
  mofft_plan_options_init(&options);
  options.mode = mode;
  /* Keep this package smoke check short; deployments can choose a larger
   * candidate and sample budget when they explicitly request MEASURE. */
  options.candidate_limit = 2;
  options.timing_samples = 1;
  mofft_plan_diagnostics diagnostics = {0};
  mofft_plan_f32 *plan = NULL;
  mofft_status status = mofft_plan_create_with_options_f32(
      &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 plan creation failed: %s\n",
            mofft_status_string(status));
    return 0;
  }
  status = mofft_execute_f32(plan, input, output);
  mofft_plan_destroy_f32(plan);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 execution failed: %s\n",
            mofft_status_string(status));
    return 0;
  }
  if (mode == MOFFT_PLANNER_MEASURE &&
      (!diagnostics.measured || diagnostics.candidates_measured == 0 ||
       diagnostics.kernel_bank_count == 0)) {
    fprintf(stderr, "FP32 MEASURE diagnostics were incomplete\n");
    return 0;
  }
  if (mode == MOFFT_PLANNER_PRETUNED &&
      !(diagnostics.used_wisdom || diagnostics.used_model_fallback)) {
    fprintf(stderr, "FP32 PRETUNED diagnostics were incomplete\n");
    return 0;
  }
  for (size_t i = 0; i < n; ++i) {
    if (fabsf(output[i].real - 1.0f) > 1e-5f ||
        fabsf(output[i].imag) > 1e-5f) {
      fprintf(stderr, "FP32 impulse check failed at bin %zu\n", i);
      return 0;
    }
  }
  return 1;
}

static int check_f64(mofft_planner_mode mode) {
  const size_t n = 8;
  mofft_complex_f64 input[8] = {{1.0, 0.0}};
  mofft_complex_f64 output[8] = {{0.0, 0.0}};
  mofft_plan_options options;
  mofft_plan_options_init(&options);
  options.mode = mode;
  options.candidate_limit = 2;
  options.timing_samples = 1;
  mofft_plan_diagnostics diagnostics = {0};
  mofft_plan_f64 *plan = NULL;
  mofft_status status = mofft_plan_create_with_options_f64(
      &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP64 plan creation failed: %s\n",
            mofft_status_string(status));
    return 0;
  }
  status = mofft_execute_f64(plan, input, output);
  mofft_plan_destroy_f64(plan);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP64 execution failed: %s\n",
            mofft_status_string(status));
    return 0;
  }
  if (mode == MOFFT_PLANNER_MEASURE &&
      (!diagnostics.measured || diagnostics.candidates_measured == 0 ||
       diagnostics.kernel_bank_count == 0)) {
    fprintf(stderr, "FP64 MEASURE diagnostics were incomplete\n");
    return 0;
  }
  if (mode == MOFFT_PLANNER_PRETUNED &&
      !(diagnostics.used_wisdom || diagnostics.used_model_fallback)) {
    fprintf(stderr, "FP64 PRETUNED diagnostics were incomplete\n");
    return 0;
  }
  for (size_t i = 0; i < n; ++i) {
    if (fabs(output[i].real - 1.0) > 1e-12 ||
        fabs(output[i].imag) > 1e-12) {
      fprintf(stderr, "FP64 impulse check failed at bin %zu\n", i);
      return 0;
    }
  }
  return 1;
}

static int check_fftw_style_helpers(void) {
  const size_t n = 8;
  mofft_complex_f32 input32[8] = {{1.0f, 0.0f}};
  mofft_complex_f32 output32[8] = {{0.0f, 0.0f}};
  mofft_plan_f32 *plan32 = mofft_plan_dft_1d_f32(
      n, input32, output32, MOFFT_FORWARD, 0);
  if (!plan32 || mofft_execute_plan_f32(plan32) != MOFFT_SUCCESS) {
    if (plan32) mofft_plan_destroy_f32(plan32);
    fprintf(stderr, "FFTW-style PRETUNED helper failed\n");
    return 0;
  }
  mofft_plan_destroy_f32(plan32);

  mofft_complex_f64 input64[8] = {{1.0, 0.0}};
  mofft_complex_f64 output64[8] = {{0.0, 0.0}};
  mofft_plan_f64 *plan64 = mofft_plan_dft_1d_f64(
      n, input64, output64, MOFFT_FORWARD, MOFFT_FLAG_MEASURE);
  if (!plan64 || mofft_execute_plan_f64(plan64) != MOFFT_SUCCESS) {
    if (plan64) mofft_plan_destroy_f64(plan64);
    fprintf(stderr, "FFTW-style MEASURE helper failed\n");
    return 0;
  }
  mofft_plan_destroy_f64(plan64);
  return fabsf(output32[0].real - 1.0f) <= 1e-5f &&
         fabs(output64[0].real - 1.0) <= 1e-12;
}

int main(void) {
#if MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE > 0
  {
    const size_t n = MOFFT_CONSUMER_REQUIRE_WISDOM_SIZE;
    mofft_complex_f32 *input32 = calloc(n, sizeof(*input32));
    mofft_complex_f32 *output32 = calloc(n, sizeof(*output32));
    mofft_complex_f64 *input64 = calloc(n, sizeof(*input64));
    mofft_complex_f64 *output64 = calloc(n, sizeof(*output64));
    mofft_plan_options options;
    mofft_plan_diagnostics diag32 = {0}, diag64 = {0};
    mofft_plan_f32 *plan32 = NULL;
    mofft_plan_f64 *plan64 = NULL;
    mofft_status status32, status64;
    if (!input32 || !output32 || !input64 || !output64) {
      fprintf(stderr, "wisdom-hit test allocation failed\n");
      free(input32); free(output32); free(input64); free(output64);
      return 1;
    }
    input32[0].real = 1.0f;
    input64[0].real = 1.0;
    mofft_plan_options_init(&options);
    options.flags = MOFFT_PLAN_STRICT_WISDOM;
    status32 = mofft_plan_create_with_options_f32(
        &plan32, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diag32);
    status64 = mofft_plan_create_with_options_f64(
        &plan64, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diag64);
    if (status32 != MOFFT_SUCCESS || status64 != MOFFT_SUCCESS ||
        !diag32.used_wisdom || !diag64.used_wisdom ||
        diag32.used_model_fallback || diag64.used_model_fallback ||
        mofft_execute_f32(plan32, input32, output32) != MOFFT_SUCCESS ||
        mofft_execute_f64(plan64, input64, output64) != MOFFT_SUCCESS) {
      fprintf(stderr, "strict FP32/FP64 wisdom-hit check failed at N=%zu\n", n);
      mofft_plan_destroy_f32(plan32); mofft_plan_destroy_f64(plan64);
      free(input32); free(output32); free(input64); free(output64);
      return 1;
    }
    for (size_t i = 0; i < n; ++i) {
      if (fabsf(output32[i].real - 1.0f) > 1e-5f ||
          fabsf(output32[i].imag) > 1e-5f ||
          fabs(output64[i].real - 1.0) > 1e-12 ||
          fabs(output64[i].imag) > 1e-12) {
        fprintf(stderr, "strict wisdom impulse check failed at bin %zu\n", i);
        mofft_plan_destroy_f32(plan32); mofft_plan_destroy_f64(plan64);
        free(input32); free(output32); free(input64); free(output64);
        return 1;
      }
    }
    mofft_plan_destroy_f32(plan32); mofft_plan_destroy_f64(plan64);
    free(input32); free(output32); free(input64); free(output64);
    printf("Strict PRETUNED FP32/FP64 wisdom hit passed at N=%zu.\n", n);
  }
#endif
  if (!check_f32(MOFFT_PLANNER_PRETUNED) ||
      !check_f64(MOFFT_PLANNER_PRETUNED) ||
      !check_f32(MOFFT_PLANNER_MEASURE) ||
      !check_f64(MOFFT_PLANNER_MEASURE) ||
      !check_fftw_style_helpers())
    return 1;
  printf("MoFFT %s installed-package consumer passed (FP32/FP64, "
         "PRETUNED/MEASURE).\n", mofft_version());
  return 0;
}

