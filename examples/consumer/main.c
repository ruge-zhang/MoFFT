#include <mofft.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { PRECISION_F32, PRECISION_F64 } precision_t;

static void usage(const char *program) {
  fprintf(stderr,
          "Usage: %s LENGTH fp32|fp64 [forward|inverse] [pretuned|measure]\n"
          "          [--save-plan FILE | --load-plan FILE]\n"
          "PRETUNED uses the packaged plan when available, then the model.\n"
          "MEASURE searches plans using kernels already in the library.\n"
          "A saved plan recipe can be loaded on later runs of the matching build/CPU.\n",
          program);
}

static int parse_length(const char *text, size_t *length) {
  char *end = NULL;
  unsigned long long value;
  if (!text || !*text || *text == '-') return 0;
  value = strtoull(text, &end, 10);
  if (*end != '\0' || value < 2 || value > SIZE_MAX) return 0;
  *length = (size_t)value;
  return 1;
}

static char *read_recipe(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  long length;
  char *text;
  if (!file) {
    perror("could not open plan recipe");
    return NULL;
  }
  if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
      fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    fprintf(stderr, "invalid or empty plan recipe\n");
    return NULL;
  }
  text = (char *)malloc((size_t)length);
  if (!text) {
    fclose(file);
    fprintf(stderr, "could not allocate memory for plan recipe\n");
    return NULL;
  }
  if (fread(text, 1, (size_t)length, file) != (size_t)length ||
      text[length - 1] != '\0') {
    fclose(file);
    free(text);
    fprintf(stderr, "plan recipe is truncated or lacks its final NUL byte\n");
    return NULL;
  }
  fclose(file);
  *size = (size_t)length;
  return text;
}

static int write_recipe(const char *path, const void *plan, precision_t precision) {
  size_t required = 0;
  char *text;
  FILE *file;
  mofft_status status;
  if (precision == PRECISION_F32)
    status = mofft_plan_export_f32((const mofft_plan_f32 *)plan, NULL, 0,
                                   &required);
  else
    status = mofft_plan_export_f64((const mofft_plan_f64 *)plan, NULL, 0,
                                   &required);
  if (status != MOFFT_SUCCESS || required == 0) {
    fprintf(stderr, "could not size plan recipe: %s\n", mofft_status_string(status));
    return 0;
  }
  text = (char *)malloc(required);
  if (!text) {
    fprintf(stderr, "could not allocate plan recipe\n");
    return 0;
  }
  if (precision == PRECISION_F32)
    status = mofft_plan_export_f32((const mofft_plan_f32 *)plan, text,
                                   required, &required);
  else
    status = mofft_plan_export_f64((const mofft_plan_f64 *)plan, text,
                                   required, &required);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "could not export plan recipe: %s\n", mofft_status_string(status));
    free(text);
    return 0;
  }
  file = fopen(path, "wb");
  if (!file) {
    perror("could not create plan recipe");
    free(text);
    return 0;
  }
  int ok = fwrite(text, 1, required, file) == required;
  if (fclose(file) != 0) ok = 0;
  if (!ok) fprintf(stderr, "could not write complete plan recipe\n");
  free(text);
  return ok;
}

static void print_diagnostics(const mofft_plan_diagnostics *diagnostics) {
  printf("plan: bank %u of %u, radices [", diagnostics->kernel_bank,
         diagnostics->kernel_bank_count);
  for (size_t i = 0; i < diagnostics->radix_count; ++i)
    printf("%s%d", i ? ", " : "", diagnostics->radices[i]);
  printf("], measured=%s, wisdom=%s, model-fallback=%s\n",
         diagnostics->measured ? "yes" : "no",
         diagnostics->used_wisdom ? "yes" : "no",
         diagnostics->used_model_fallback ? "yes" : "no");
  if (diagnostics->measured)
    printf("planning %.6f s, measured %zu candidates\n",
           diagnostics->planning_seconds, diagnostics->candidates_measured);
}

static int run_f32(size_t n, mofft_direction direction, mofft_planner_mode mode,
                   const char *save_path, const char *load_path) {
  mofft_complex_f32 *input = (mofft_complex_f32 *)calloc(n, sizeof(*input));
  mofft_complex_f32 *output = (mofft_complex_f32 *)calloc(n, sizeof(*output));
  mofft_plan_f32 *plan = NULL;
  mofft_plan_diagnostics diagnostics = {0};
  mofft_status status;
  int ok = 0;
  if (!input || !output) {
    fprintf(stderr, "input/output allocation failed\n");
    goto done;
  }
  input[0].real = 1.0f;
  if (load_path) {
    size_t recipe_size = 0;
    char *recipe = read_recipe(load_path, &recipe_size);
    if (!recipe) goto done;
    status = mofft_plan_import_f32(&plan, recipe, recipe_size, &diagnostics);
    free(recipe);
  } else {
    mofft_plan_options options;
    mofft_plan_options_init(&options);
    options.mode = mode;
    status = mofft_plan_create_with_options_f32(
        &plan, n, direction, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  }
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 plan creation/import failed: %s\n",
            mofft_status_string(status));
    goto done;
  }
  if (!load_path && mode == MOFFT_PLANNER_MEASURE &&
      (!diagnostics.measured || diagnostics.candidates_measured == 0)) {
    fprintf(stderr, "FP32 MEASURE did not report any measured candidates\n");
    goto done;
  }
  if (diagnostics.length != n || diagnostics.precision_bits != 32 ||
      diagnostics.direction != direction ||
      diagnostics.placement != MOFFT_OUT_OF_PLACE) {
    fprintf(stderr, "FP32 recipe does not match the requested transform\n");
    goto done;
  }
  print_diagnostics(&diagnostics);
  status = mofft_execute_f32(plan, input, output);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP32 execution failed: %s\n", mofft_status_string(status));
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(output[i].real) || !isfinite(output[i].imag) ||
        fabsf(output[i].real - 1.0f) > 1e-3f * fmaxf(1.0f, log2f((float)n)) ||
        fabsf(output[i].imag) > 1e-3f * fmaxf(1.0f, log2f((float)n))) {
      fprintf(stderr, "FP32 impulse check failed at bin %zu\n", i);
      goto done;
    }
  }
  if (save_path && !write_recipe(save_path, plan, PRECISION_F32)) goto done;
  printf("FP32 %s impulse FFT passed at N=%zu.\n",
         direction == MOFFT_FORWARD ? "forward" : "inverse", n);
  ok = 1;
done:
  mofft_plan_destroy_f32(plan);
  free(input);
  free(output);
  return ok;
}

static int run_f64(size_t n, mofft_direction direction, mofft_planner_mode mode,
                   const char *save_path, const char *load_path) {
  mofft_complex_f64 *input = (mofft_complex_f64 *)calloc(n, sizeof(*input));
  mofft_complex_f64 *output = (mofft_complex_f64 *)calloc(n, sizeof(*output));
  mofft_plan_f64 *plan = NULL;
  mofft_plan_diagnostics diagnostics = {0};
  mofft_status status;
  int ok = 0;
  if (!input || !output) {
    fprintf(stderr, "input/output allocation failed\n");
    goto done;
  }
  input[0].real = 1.0;
  if (load_path) {
    size_t recipe_size = 0;
    char *recipe = read_recipe(load_path, &recipe_size);
    if (!recipe) goto done;
    status = mofft_plan_import_f64(&plan, recipe, recipe_size, &diagnostics);
    free(recipe);
  } else {
    mofft_plan_options options;
    mofft_plan_options_init(&options);
    options.mode = mode;
    status = mofft_plan_create_with_options_f64(
        &plan, n, direction, MOFFT_OUT_OF_PLACE, &options, &diagnostics);
  }
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP64 plan creation/import failed: %s\n",
            mofft_status_string(status));
    goto done;
  }
  if (!load_path && mode == MOFFT_PLANNER_MEASURE &&
      (!diagnostics.measured || diagnostics.candidates_measured == 0)) {
    fprintf(stderr, "FP64 MEASURE did not report any measured candidates\n");
    goto done;
  }
  if (diagnostics.length != n || diagnostics.precision_bits != 64 ||
      diagnostics.direction != direction ||
      diagnostics.placement != MOFFT_OUT_OF_PLACE) {
    fprintf(stderr, "FP64 recipe does not match the requested transform\n");
    goto done;
  }
  print_diagnostics(&diagnostics);
  status = mofft_execute_f64(plan, input, output);
  if (status != MOFFT_SUCCESS) {
    fprintf(stderr, "FP64 execution failed: %s\n", mofft_status_string(status));
    goto done;
  }
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(output[i].real) || !isfinite(output[i].imag) ||
        fabs(output[i].real - 1.0) > 1e-10 * fmax(1.0, log2((double)n)) ||
        fabs(output[i].imag) > 1e-10 * fmax(1.0, log2((double)n))) {
      fprintf(stderr, "FP64 impulse check failed at bin %zu\n", i);
      goto done;
    }
  }
  if (save_path && !write_recipe(save_path, plan, PRECISION_F64)) goto done;
  printf("FP64 %s impulse FFT passed at N=%zu.\n",
         direction == MOFFT_FORWARD ? "forward" : "inverse", n);
  ok = 1;
done:
  mofft_plan_destroy_f64(plan);
  free(input);
  free(output);
  return ok;
}

int main(int argc, char **argv) {
  size_t n;
  precision_t precision;
  mofft_direction direction = MOFFT_FORWARD;
  mofft_planner_mode mode = MOFFT_PLANNER_PRETUNED;
  const char *save_path = NULL, *load_path = NULL;
  if (argc < 3 || !parse_length(argv[1], &n)) {
    usage(argv[0]);
    return 2;
  }
  if (strcmp(argv[2], "fp32") == 0) precision = PRECISION_F32;
  else if (strcmp(argv[2], "fp64") == 0) precision = PRECISION_F64;
  else {
    usage(argv[0]);
    return 2;
  }
  for (int i = 3; i < argc; ++i) {
    if (strcmp(argv[i], "forward") == 0) direction = MOFFT_FORWARD;
    else if (strcmp(argv[i], "inverse") == 0) direction = MOFFT_BACKWARD;
    else if (strcmp(argv[i], "pretuned") == 0) mode = MOFFT_PLANNER_PRETUNED;
    else if (strcmp(argv[i], "measure") == 0) mode = MOFFT_PLANNER_MEASURE;
    else if (strcmp(argv[i], "--save-plan") == 0 && i + 1 < argc && !save_path)
      save_path = argv[++i];
    else if (strcmp(argv[i], "--load-plan") == 0 && i + 1 < argc && !load_path)
      load_path = argv[++i];
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if ((save_path && load_path) ||
      (precision == PRECISION_F32 && n > SIZE_MAX / sizeof(mofft_complex_f32)) ||
      (precision == PRECISION_F64 && n > SIZE_MAX / sizeof(mofft_complex_f64))) {
    usage(argv[0]);
    return 2;
  }
  printf("MoFFT %s, build %s\n", mofft_version(), mofft_build_id());
  if (load_path && mode != MOFFT_PLANNER_PRETUNED)
    fprintf(stderr, "note: --load-plan reuses the recipe; planner mode is ignored\n");
  if (precision == PRECISION_F32)
    return run_f32(n, direction, mode, save_path, load_path) ? 0 : 1;
  return run_f64(n, direction, mode, save_path, load_path) ? 0 : 1;
}
