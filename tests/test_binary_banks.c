#include "mofft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Independent spectral oracle: three known Fourier modes, not a roundtrip
 * through another generated kernel. Tests execution only, never performance. */
#define DEFINE_CHECK(SUFFIX, SCALAR, TOLERANCE) \
static int check_##SUFFIX(size_t n, unsigned bank, int sign, int inplace, \
                           const int *radices, size_t depth) { \
  mofft_complex_##SUFFIX *input = calloc(n, sizeof(*input)); \
  mofft_complex_##SUFFIX *output = calloc(n, sizeof(*output)); \
  mofft_plan_##SUFFIX *plan = NULL; \
  if (!input || !output) { free(input); free(output); return 1; } \
  const double pi = acos(-1.0); \
  for (size_t j = 0; j < n; ++j) { \
    const double a = 2*pi*3*j/n, b = 2*pi*5*j/n; \
    input[j].real = (SCALAR)(0.25 + cos(a) + 0.5*cos(b)); \
    input[j].imag = (SCALAR)(sin(a) - 0.5*sin(b)); \
  } \
  mofft_status status = radices ? mofft_plan_create_with_radices_##SUFFIX( \
      &plan, n, sign, inplace ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE, radices, depth) : \
      mofft_plan_create_##SUFFIX(&plan, n, sign, \
      inplace ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE); \
  if (status == MOFFT_SUCCESS) status = mofft_plan_set_kernel_bank_##SUFFIX(plan, bank); \
  if (status == MOFFT_SUCCESS) status = mofft_plan_set_stage_layout_##SUFFIX(plan, \
      bank & 1 ? MOFFT_STAGE_LAYOUT_SECTION : MOFFT_STAGE_LAYOUT_TRANSPOSE); \
  if (inplace) memcpy(output, input, n*sizeof(*input)); \
  if (status == MOFFT_SUCCESS) status = mofft_execute_##SUFFIX( \
      plan, inplace ? output : input, output); \
  double error = 0; \
  if (status == MOFFT_SUCCESS) for (size_t k = 0; k < n; ++k) { \
    double expected = k == 0 ? 0.25 : 0; \
    if (k == (sign < 0 ? 3 % n : (n - 3 % n) % n)) expected += 1; \
    if (k == (sign < 0 ? (n - 5 % n) % n : 5 % n)) expected += 0.5; \
    double e = hypot((double)output[k].real/n - expected, (double)output[k].imag/n); \
    if (!isfinite(e)) { status = MOFFT_INTERNAL_ERROR; break; } \
    if (e > error) error = e; \
  } \
  mofft_plan_destroy_##SUFFIX(plan); free(input); free(output); \
  if (status != MOFFT_SUCCESS || error > TOLERANCE) { \
    fprintf(stderr, #SUFFIX " N=%zu bank=%u sign=%d inplace=%d error=%g status=%s\n", \
        n, bank, sign, inplace, error, mofft_status_string(status)); return 1; \
  } \
  return 0; \
}

DEFINE_CHECK(f32, float, 5e-5)
DEFINE_CHECK(f64, double, 2e-11)

int main(void) {
  mofft_plan_f32 *probe = NULL;
  mofft_plan_options options;
  mofft_plan_options_init(&options);
  mofft_plan_diagnostics info;
  if (mofft_plan_create_with_options_f32(&probe, 16, MOFFT_FORWARD,
      MOFFT_OUT_OF_PLACE, &options, &info) != MOFFT_SUCCESS) return 1;
  mofft_plan_destroy_f32(probe);
  const int radices[] = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64};
  const size_t sizes[] = {121,169,196,225,256,1024,1728,2197,2744,3375,
      20736,28561,38416,50625,65536,248832,262144,371293,537824,759375};
  size_t checks = 0;
  for (unsigned bank = 0; bank < info.kernel_bank_count; ++bank) {
    for (size_t r = 0; r < sizeof(radices)/sizeof(radices[0]); ++r) {
      const int factors[] = {radices[r], 2};
      for (int sign = -1; sign <= 1; sign += 2) for (int inplace=0; inplace<2; ++inplace) {
        if (check_f32(radices[r], bank, sign, inplace, factors, 1) ||
            check_f64(radices[r], bank, sign, inplace, factors, 1) ||
            check_f32(2*radices[r], bank, sign, inplace, factors, 2) ||
            check_f64(2*radices[r], bank, sign, inplace, factors, 2)) return 1;
        checks += 4;
      }
    }
    for (size_t s=0; s<sizeof(sizes)/sizeof(sizes[0]); ++s)
      for (int sign=-1; sign<=1; sign+=2) for (int inplace=0; inplace<2; ++inplace) {
        if (check_f32(sizes[s],bank,sign,inplace,NULL,0) ||
            check_f64(sizes[s],bank,sign,inplace,NULL,0)) return 1;
        checks += 2;
      }
  }
  printf("All %u compiled banks passed %zu spectral-oracle checks (FP32/FP64).\n",
         info.kernel_bank_count, checks);
  return 0;
}
