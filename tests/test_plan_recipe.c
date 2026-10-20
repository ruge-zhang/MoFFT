#include "mofft.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TEST_N 96
#define RECIPE_PREFIX "MOFFT_PLAN_V1 "

static int valid_build_id(void) {
  const char *id = mofft_build_id();
  if (!id || strlen(id) != 64) return 0;
  for (size_t i = 0; i < 64; ++i)
    if (!((id[i] >= '0' && id[i] <= '9') ||
          (id[i] >= 'a' && id[i] <= 'f')))
      return 0;
  return 1;
}

static int expect_bad_f32_recipe(const char *recipe, size_t size,
                                mofft_status expected) {
  mofft_plan_f32 *plan = (mofft_plan_f32 *)(uintptr_t)1;
  const mofft_status status =
      mofft_plan_import_f32(&plan, recipe, size, NULL);
  if (status != expected || plan != NULL) {
    fprintf(stderr, "FP32 recipe rejection: expected %s, got %s\n",
            mofft_status_string(expected), mofft_status_string(status));
    if (plan && plan != (mofft_plan_f32 *)(uintptr_t)1)
      mofft_plan_destroy_f32(plan);
    return 1;
  }
  return 0;
}

static int expect_bad_f64_recipe(const char *recipe, size_t size,
                                mofft_status expected) {
  mofft_plan_f64 *plan = (mofft_plan_f64 *)(uintptr_t)1;
  const mofft_status status =
      mofft_plan_import_f64(&plan, recipe, size, NULL);
  if (status != expected || plan != NULL) {
    fprintf(stderr, "FP64 recipe rejection: expected %s, got %s\n",
            mofft_status_string(expected), mofft_status_string(status));
    if (plan && plan != (mofft_plan_f64 *)(uintptr_t)1)
      mofft_plan_destroy_f64(plan);
    return 1;
  }
  return 0;
}

#define DEFINE_RECIPE_CASE(SUFFIX, SCALAR, EPSILON)                         \
static double max_error_##SUFFIX(                                           \
    const mofft_complex_##SUFFIX *actual,                                   \
    const mofft_complex_##SUFFIX *expected, size_t n) {                     \
  double error = 0.0;                                                       \
  for (size_t i = 0; i < n; ++i) {                                         \
    const double e = hypot((double)actual[i].real - expected[i].real,       \
                           (double)actual[i].imag - expected[i].imag);      \
    const double scale = 1.0 + hypot((double)expected[i].real,              \
                                    (double)expected[i].imag);              \
    if (!isfinite(e) || !isfinite(scale)) return INFINITY;                  \
    if (e / scale > error) error = e / scale;                              \
  }                                                                         \
  return error;                                                             \
}                                                                           \
static void direct_dft_##SUFFIX(const mofft_complex_##SUFFIX *input,        \
    mofft_complex_##SUFFIX *output, size_t n, mofft_direction direction) {  \
  const double tau = 2.0 * acos(-1.0);                                     \
  for (size_t k = 0; k < n; ++k) {                                        \
    double real = 0.0, imag = 0.0;                                         \
    for (size_t j = 0; j < n; ++j) {                                      \
      const double angle = (double)direction * tau * (double)j *           \
                           (double)k / (double)n;                          \
      const double c = cos(angle), s = sin(angle);                         \
      real += (double)input[j].real * c - (double)input[j].imag * s;       \
      imag += (double)input[j].real * s + (double)input[j].imag * c;       \
    }                                                                       \
    output[k].real = (SCALAR)real;                                         \
    output[k].imag = (SCALAR)imag;                                         \
  }                                                                         \
}                                                                           \
static int check_recipe_case_##SUFFIX(                                     \
    size_t n, const int *radices, size_t depth,                            \
    mofft_direction direction, mofft_placement placement,                  \
    unsigned int bank) {                                                    \
  mofft_complex_##SUFFIX input[MAX_TEST_N];                                \
  mofft_complex_##SUFFIX original_buffer[MAX_TEST_N];                       \
  mofft_complex_##SUFFIX imported_buffer[MAX_TEST_N];                       \
  mofft_complex_##SUFFIX original_output[MAX_TEST_N];                       \
  mofft_complex_##SUFFIX imported_output[MAX_TEST_N];                       \
  mofft_complex_##SUFFIX reference[MAX_TEST_N];                             \
  if (n > MAX_TEST_N) return 1;                                             \
  for (size_t i = 0; i < n; ++i) {                                         \
    input[i].real = (SCALAR)(0.2 + sin((double)(i * 7 + 3)) * 0.7);         \
    input[i].imag = (SCALAR)(cos((double)(i * 5 + 1)) * 0.4 - 0.1);        \
  }                                                                         \
  direct_dft_##SUFFIX(input, reference, n, direction);                     \
  mofft_plan_##SUFFIX *original = NULL, *imported = NULL;                  \
  mofft_plan_diagnostics info;                                              \
  memset(&info, 0, sizeof(info));                                           \
  mofft_status status = mofft_plan_create_with_radices_##SUFFIX(           \
      &original, n, direction, placement, radices, depth);                 \
  if (status == MOFFT_SUCCESS)                                             \
    status = mofft_plan_set_kernel_bank_##SUFFIX(original, bank);          \
  const uint64_t mask = depth > 1 ? (uint64_t)(bank % 4U) : 0;             \
  if (status == MOFFT_SUCCESS)                                             \
    status = mofft_plan_set_blocked_transpose_mask_##SUFFIX(original, mask); \
  if (status == MOFFT_SUCCESS)                                             \
    status = mofft_plan_set_direct_input_##SUFFIX(original, bank & 1U);     \
  const mofft_stage_layout layout = (bank & 2U)                            \
      ? MOFFT_STAGE_LAYOUT_SECTION : MOFFT_STAGE_LAYOUT_TRANSPOSE;          \
  if (status == MOFFT_SUCCESS)                                             \
    status = mofft_plan_set_stage_layout_##SUFFIX(original, layout);       \
  size_t required = 0;                                                     \
  if (status == MOFFT_SUCCESS)                                             \
    status = mofft_plan_export_##SUFFIX(original, NULL, 0, &required);     \
  if (status != MOFFT_SUCCESS || required < 2) {                            \
    fprintf(stderr, #SUFFIX " recipe size query failed: %s\n",           \
            mofft_status_string(status));                                  \
    mofft_plan_destroy_##SUFFIX(original);                                  \
    return 1;                                                               \
  }                                                                         \
  char *recipe = (char *)malloc(required);                                 \
  if (!recipe) { mofft_plan_destroy_##SUFFIX(original); return 1; }         \
  char small[1] = {'x'};                                                    \
  size_t short_required = 0;                                               \
  const mofft_status short_status = mofft_plan_export_##SUFFIX(            \
      original, small, sizeof(small), &short_required);                    \
  if (short_status != MOFFT_BUFFER_TOO_SMALL || short_required != required || \
      small[0] != 'x') {                                                   \
    fprintf(stderr, #SUFFIX " undersized recipe buffer contract failed\n"); \
    free(recipe); mofft_plan_destroy_##SUFFIX(original); return 1;         \
  }                                                                         \
  status = mofft_plan_export_##SUFFIX(original, recipe, required,           \
                                      &short_required);                     \
  if (status != MOFFT_SUCCESS || short_required != required ||             \
      recipe[required - 1] != '\0' || recipe[required - 2] != '\n' ||      \
      !strstr(recipe, mofft_build_id())) {                                 \
    fprintf(stderr, #SUFFIX " recipe export failed: %s\n",                \
            mofft_status_string(status));                                  \
    free(recipe); mofft_plan_destroy_##SUFFIX(original); return 1;         \
  }                                                                         \
  status = mofft_plan_import_##SUFFIX(&imported, recipe, required, &info);  \
  if (status != MOFFT_SUCCESS || !imported || !info.used_wisdom ||         \
      info.measured || info.candidates_measured != 0 ||                    \
      info.planning_seconds != 0.0 || info.selected_median_seconds != 0.0 || \
      info.length != n || info.precision_bits != sizeof(SCALAR) * 8 ||     \
      info.direction != direction || info.placement != placement ||        \
      info.radix_count != depth || info.kernel_bank != bank ||             \
      info.blocked_transpose_mask != mask ||                               \
      info.direct_input != (int)(bank & 1U) || info.stage_layout != layout || \
      memcmp(info.radices, radices, depth * sizeof(radices[0])) != 0) {     \
    fprintf(stderr, #SUFFIX " recipe import/diagnostics failed: %s\n",    \
            mofft_status_string(status));                                  \
    free(recipe); mofft_plan_destroy_##SUFFIX(original);                   \
    mofft_plan_destroy_##SUFFIX(imported); return 1;                       \
  }                                                                         \
  if (placement == MOFFT_IN_PLACE) {                                       \
    memcpy(original_buffer, input, n * sizeof(input[0]));                  \
    memcpy(imported_buffer, input, n * sizeof(input[0]));                  \
    status = mofft_execute_##SUFFIX(original, original_buffer,              \
                                    original_buffer);                      \
    if (status == MOFFT_SUCCESS)                                           \
      status = mofft_execute_##SUFFIX(imported, imported_buffer,            \
                                      imported_buffer);                    \
    memcpy(original_output, original_buffer, n * sizeof(input[0]));        \
    memcpy(imported_output, imported_buffer, n * sizeof(input[0]));        \
  } else {                                                                  \
    status = mofft_execute_##SUFFIX(original, input, original_output);     \
    if (status == MOFFT_SUCCESS)                                           \
      status = mofft_execute_##SUFFIX(imported, input, imported_output);   \
  }                                                                         \
  const double original_error = max_error_##SUFFIX(                        \
      original_output, reference, n);                                      \
  const double imported_error = max_error_##SUFFIX(                        \
      imported_output, reference, n);                                      \
  const double replay_error = max_error_##SUFFIX(                          \
      imported_output, original_output, n);                                \
  const int failed = status != MOFFT_SUCCESS ||                            \
      original_error > (EPSILON) || imported_error > (EPSILON) ||          \
      replay_error > (EPSILON);                                            \
  if (failed)                                                               \
    fprintf(stderr, #SUFFIX " recipe execution N=%zu bank=%u dir=%d place=%d " \
            "status=%s oracle=%g replay=%g\n", n, bank, direction,       \
            placement, mofft_status_string(status), imported_error,        \
            replay_error);                                                 \
  free(recipe);                                                            \
  mofft_plan_destroy_##SUFFIX(original);                                   \
  mofft_plan_destroy_##SUFFIX(imported);                                   \
  return failed;                                                           \
}

DEFINE_RECIPE_CASE(f32, float, 2.0e-5)
DEFINE_RECIPE_CASE(f64, double, 2.0e-10)

static int test_rejections(void) {
  const int radices[] = {4, 4, 4};
  mofft_plan_f32 *plan = NULL;
  if (mofft_plan_create_with_radices_f32(&plan, 64, MOFFT_FORWARD,
          MOFFT_OUT_OF_PLACE, radices, 3) != MOFFT_SUCCESS)
    return 1;
  size_t size = 0;
  if (mofft_plan_export_f32(plan, NULL, 0, &size) != MOFFT_SUCCESS) {
    mofft_plan_destroy_f32(plan);
    return 1;
  }
  char *recipe = (char *)malloc(size + 2);
  if (!recipe) { mofft_plan_destroy_f32(plan); return 1; }
  if (mofft_plan_export_f32(plan, recipe, size, &size) != MOFFT_SUCCESS) {
    free(recipe); mofft_plan_destroy_f32(plan); return 1;
  }
  mofft_plan_destroy_f32(plan);

  int failed = 0;
  failed |= expect_bad_f32_recipe(recipe, size - 1, MOFFT_INVALID_ARGUMENT);
  failed |= expect_bad_f32_recipe(recipe, size - 2, MOFFT_INVALID_ARGUMENT);

  char saved = recipe[0];
  recipe[0] = 'X';
  failed |= expect_bad_f32_recipe(recipe, size, MOFFT_INCOMPATIBLE_RECIPE);
  recipe[0] = saved;

  const size_t build_id_offset = strlen(RECIPE_PREFIX);
  saved = recipe[build_id_offset];
  recipe[build_id_offset] = saved == '0' ? '1' : '0';
  failed |= expect_bad_f32_recipe(recipe, size, MOFFT_INCOMPATIBLE_RECIPE);
  recipe[build_id_offset] = saved;

  const size_t cpu_family_offset = build_id_offset + 65;
  saved = recipe[cpu_family_offset];
  recipe[cpu_family_offset] = saved == '0' ? '1' : '0';
  failed |= expect_bad_f32_recipe(recipe, size, MOFFT_INCOMPATIBLE_RECIPE);
  recipe[cpu_family_offset] = saved;

  failed |= expect_bad_f64_recipe(recipe, size, MOFFT_INCOMPATIBLE_RECIPE);

  memcpy(recipe + size, "x\0", 2);
  failed |= expect_bad_f32_recipe(recipe, size + 2, MOFFT_INVALID_ARGUMENT);
  free(recipe);
  return failed;
}

int main(void) {
  if (!valid_build_id()) {
    fprintf(stderr, "MoFFT binary build ID is not 64 lowercase hex digits\n");
    return 1;
  }

  mofft_plan_f32 *probe = NULL;
  mofft_plan_options options;
  mofft_plan_diagnostics diagnostics;
  mofft_plan_options_init(&options);
  const mofft_status probe_status = mofft_plan_create_with_options_f32(
      &probe, 64, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  if (probe_status != MOFFT_SUCCESS || !probe ||
      diagnostics.kernel_bank_count == 0 || diagnostics.length != 64 ||
      diagnostics.precision_bits != 32 ||
      diagnostics.direction != MOFFT_FORWARD ||
      diagnostics.placement != MOFFT_OUT_OF_PLACE) {
    fprintf(stderr, "unable to query compiled kernel banks: %s\n",
            mofft_status_string(probe_status));
    mofft_plan_destroy_f32(probe);
    return 1;
  }
  mofft_plan_destroy_f32(probe);

  const size_t lengths[] = {64, 96};
  const int radices[][3] = {{4, 4, 4}, {3, 4, 8}};
  const mofft_direction directions[] = {MOFFT_FORWARD, MOFFT_BACKWARD};
  size_t cases = 0;
  for (unsigned int bank = 0; bank < diagnostics.kernel_bank_count; ++bank) {
    for (size_t n = 0; n < sizeof(lengths) / sizeof(lengths[0]); ++n) {
      for (size_t d = 0; d < sizeof(directions) / sizeof(directions[0]); ++d) {
        for (int p = 0; p < 2; ++p) {
          const mofft_placement placement = p ? MOFFT_IN_PLACE
                                              : MOFFT_OUT_OF_PLACE;
          if (check_recipe_case_f32(lengths[n], radices[n], 3,
                                    directions[d], placement, bank) ||
              check_recipe_case_f64(lengths[n], radices[n], 3,
                                    directions[d], placement, bank))
            return 1;
          cases += 2;
        }
      }
    }
    /* Exercise the special single-stage radix-64 in-place path too. */
    const int single_radix[] = {64};
    for (size_t d = 0; d < sizeof(directions) / sizeof(directions[0]); ++d) {
      if (check_recipe_case_f32(64, single_radix, 1, directions[d],
                                MOFFT_IN_PLACE, bank) ||
          check_recipe_case_f64(64, single_radix, 1, directions[d],
                                MOFFT_IN_PLACE, bank))
        return 1;
      cases += 2;
    }
  }

  if (test_rejections()) return 1;
  printf("Plan recipe replay passed %zu numeric cases across %u compiled banks.\n",
         cases, diagnostics.kernel_bank_count);
  return 0;
}
