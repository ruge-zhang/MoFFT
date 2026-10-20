#include "mofft.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int roundtrip_f32(unsigned bank) {
  const size_t n = 256;
  mofft_complex_f32 *data = calloc(n, sizeof(*data));
  mofft_complex_f32 *spectrum = calloc(n, sizeof(*spectrum));
  mofft_complex_f32 *roundtrip = calloc(n, sizeof(*roundtrip));
  mofft_plan_f32 *forward = NULL, *backward = NULL, *imported = NULL;
  mofft_status status = MOFFT_SUCCESS;
  char *recipe = NULL;
  size_t recipe_size = 0;
  int failed = 0;

  if (!data || !spectrum || !roundtrip) {
    failed = 1;
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    data[i].real = (float)(0.25 + sin((double)i * 0.13));
    data[i].imag = (float)cos((double)i * 0.07);
  }
  status = mofft_plan_create_f32(&forward, n, MOFFT_FORWARD,
                                 MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_kernel_bank_f32(forward, bank);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f32(forward, data, spectrum);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_export_f32(forward, NULL, 0, &recipe_size);
  if (status == MOFFT_SUCCESS && recipe_size > 0) {
    recipe = malloc(recipe_size);
    if (!recipe) {
      failed = 1;
      goto done;
    }
    status = mofft_plan_export_f32(forward, recipe, recipe_size,
                                   &recipe_size);
  }
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_import_f32(&imported, recipe, recipe_size, NULL);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f32(imported, data, roundtrip);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 bank %u recipe check: %s\n", bank,
            mofft_status_string(status));
    failed = 1;
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    if (fabsf(roundtrip[i].real - spectrum[i].real) > 2.0e-4f ||
        fabsf(roundtrip[i].imag - spectrum[i].imag) > 2.0e-4f) {
      fprintf(stderr, "FP32 bank %u imported recipe mismatch at %zu\n", bank, i);
      failed = 1;
      break;
    }
  }
  if (!failed)
    status = mofft_plan_create_f32(&backward, n, MOFFT_BACKWARD,
                                   MOFFT_OUT_OF_PLACE);
  if (!failed && status == MOFFT_SUCCESS)
    status = mofft_plan_set_kernel_bank_f32(backward, bank);
  if (!failed && status == MOFFT_SUCCESS)
    status = mofft_execute_f32(backward, spectrum, roundtrip);
  if (!failed && status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 bank %u inverse: %s\n", bank,
            mofft_status_string(status));
    failed = 1;
  }
  if (!failed) {
    for (size_t i = 0; i < n; ++i) {
      if (fabsf(roundtrip[i].real / (float)n - data[i].real) > 2.0e-5f ||
          fabsf(roundtrip[i].imag / (float)n - data[i].imag) > 2.0e-5f) {
        fprintf(stderr, "FP32 bank %u roundtrip mismatch at %zu\n", bank, i);
        failed = 1;
        break;
      }
    }
  }

done:
  free(recipe);
  mofft_plan_destroy_f32(forward);
  mofft_plan_destroy_f32(backward);
  mofft_plan_destroy_f32(imported);
  free(data);
  free(spectrum);
  free(roundtrip);
  return failed;
}

static int roundtrip_f64(unsigned bank) {
  const size_t n = 256;
  mofft_complex_f64 *data = calloc(n, sizeof(*data));
  mofft_complex_f64 *spectrum = calloc(n, sizeof(*spectrum));
  mofft_complex_f64 *roundtrip = calloc(n, sizeof(*roundtrip));
  mofft_plan_f64 *forward = NULL, *backward = NULL;
  mofft_status status = MOFFT_SUCCESS;
  int failed = 0;

  if (!data || !spectrum || !roundtrip) {
    failed = 1;
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    data[i].real = 0.25 + sin((double)i * 0.13);
    data[i].imag = cos((double)i * 0.07);
  }
  status = mofft_plan_create_f64(&forward, n, MOFFT_FORWARD,
                                 MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_kernel_bank_f64(forward, bank);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_create_f64(&backward, n, MOFFT_BACKWARD,
                                   MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_kernel_bank_f64(backward, bank);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f64(forward, data, spectrum);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f64(backward, spectrum, roundtrip);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP64 bank %u roundtrip: %s\n", bank,
            mofft_status_string(status));
    failed = 1;
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    if (fabs(roundtrip[i].real / (double)n - data[i].real) > 2.0e-11 ||
        fabs(roundtrip[i].imag / (double)n - data[i].imag) > 2.0e-11) {
      fprintf(stderr, "FP64 bank %u roundtrip mismatch at %zu\n", bank, i);
      failed = 1;
      break;
    }
  }

done:
  mofft_plan_destroy_f64(forward);
  mofft_plan_destroy_f64(backward);
  free(data);
  free(spectrum);
  free(roundtrip);
  return failed;
}

int main(void) {
  mofft_plan_f32 *probe = NULL;
  mofft_plan_diagnostics diagnostics = {0};
  mofft_plan_options options;
  mofft_plan_options_init(&options);
  if (mofft_plan_create_with_options_f32(&probe, 256, MOFFT_FORWARD,
      MOFFT_OUT_OF_PLACE, &options, &diagnostics) != MOFFT_SUCCESS) {
    fprintf(stderr, "could not create the planning probe\n");
    return 1;
  }
  mofft_plan_destroy_f32(probe);
  if (diagnostics.kernel_bank_count != 4) {
    fprintf(stderr, "expected four compiled kernel banks, found %u\n",
            diagnostics.kernel_bank_count);
    return 1;
  }
  for (unsigned bank = 0; bank < diagnostics.kernel_bank_count; ++bank) {
    if (roundtrip_f32(bank) || roundtrip_f64(bank)) return 1;
  }
  printf("FP32/FP64 roundtrip and recipe checks passed for %u kernel banks.\n",
         diagnostics.kernel_bank_count);
  return 0;
}
