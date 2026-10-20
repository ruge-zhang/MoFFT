#ifndef MOFFT_H
#define MOFFT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float real, imag; } mofft_complex_f32;
typedef struct { double real, imag; } mofft_complex_f64;

typedef enum {
  MOFFT_SUCCESS = 0,
  MOFFT_INVALID_ARGUMENT = 1,
  MOFFT_UNSUPPORTED_SIZE = 2,
  MOFFT_ALLOCATION_FAILURE = 3,
  MOFFT_UNSUPPORTED_HARDWARE = 4,
  MOFFT_INTERNAL_ERROR = 5,
  MOFFT_WISDOM_NOT_FOUND = 6,
  MOFFT_INCOMPATIBLE_RECIPE = 7,
  MOFFT_BUFFER_TOO_SMALL = 8
} mofft_status;

typedef enum { MOFFT_FORWARD = -1, MOFFT_BACKWARD = 1 } mofft_direction;
typedef enum { MOFFT_OUT_OF_PLACE = 0, MOFFT_IN_PLACE = 1 } mofft_placement;
typedef enum {
  MOFFT_TRANSPOSE_LINEAR = 0,
  MOFFT_TRANSPOSE_BLOCKED = 1
} mofft_transpose_strategy;
typedef enum {
  MOFFT_STAGE_LAYOUT_TRANSPOSE = 0,
  MOFFT_STAGE_LAYOUT_SECTION = 1
} mofft_stage_layout;

/* Plan creation policy. PRETUNED uses compiled wisdom when present and the
 * fast analytical planner otherwise. MEASURE times a bounded set of complete
 * plans using kernels already linked into this binary; it never generates or
 * compiles kernels at runtime. */
typedef enum {
  MOFFT_PLANNER_PRETUNED = 0,
  MOFFT_PLANNER_MEASURE = 1
} mofft_planner_mode;

enum {
  /* For PRETUNED mode, fail instead of falling back to the analytical model
   * when this exact length/precision has no compiled wisdom entry. */
  MOFFT_PLAN_STRICT_WISDOM = 1u << 0
};

/* Zero candidate_limit/timing_samples selects the library defaults. The
 * implementation bounds these to at most 64 candidates and 31 samples. */
typedef struct {
  mofft_planner_mode mode;
  uint32_t flags;
  size_t candidate_limit;
  uint32_t timing_samples;
} mofft_plan_options;

#define MOFFT_MAX_PLAN_DEPTH 64
typedef struct {
  int measured;
  int used_wisdom;
  int used_model_fallback;
  size_t candidates_measured;
  size_t radix_count;
  int radices[MOFFT_MAX_PLAN_DEPTH];
  unsigned int kernel_bank;
  unsigned int kernel_bank_count;
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
  double planning_seconds;
  double selected_median_seconds;
  /* Shape fields make an imported recipe directly usable to allocate buffers. */
  size_t length;
  int precision_bits;
  mofft_direction direction;
  mofft_placement placement;
} mofft_plan_diagnostics;

/* FFTW-inspired flags for the bound-array convenience constructors below.
 * These are MoFFT API flags, not binary-compatible FFTW flags. */
enum {
  /* Explicit spelling for the default PRETUNED policy; equivalent to 0. */
  MOFFT_FLAG_PRETUNED = 0,
  MOFFT_FLAG_MEASURE = 1u << 0,
  MOFFT_FLAG_WISDOM_ONLY = 1u << 1
};

typedef struct {
  double kernel_cost;
  double memory_cost;
  double layout_cost;
  double total_cost;
  size_t logical_read_bytes;
  size_t logical_write_bytes;
  size_t transferred_read_bytes;
  size_t transferred_write_bytes;
  size_t peak_working_set_bytes;
} mofft_plan_cost_breakdown;

typedef struct mofft_plan_f32 mofft_plan_f32;
typedef struct mofft_plan_f64 mofft_plan_f64;

mofft_status mofft_plan_create_f32(mofft_plan_f32 **plan, size_t length,
                                   mofft_direction direction,
                                   mofft_placement placement);
mofft_status mofft_plan_create_f64(mofft_plan_f64 **plan, size_t length,
                                   mofft_direction direction,
                                   mofft_placement placement);

/* Initialize to the fast PRETUNED/default policy. A NULL options pointer to
 * mofft_plan_create_with_options_* has the same meaning. */
void mofft_plan_options_init(mofft_plan_options *options);
mofft_status mofft_plan_create_with_options_f32(
    mofft_plan_f32 **plan, size_t length, mofft_direction direction,
    mofft_placement placement, const mofft_plan_options *options,
    mofft_plan_diagnostics *diagnostics);
mofft_status mofft_plan_create_with_options_f64(
    mofft_plan_f64 **plan, size_t length, mofft_direction direction,
    mofft_placement placement, const mofft_plan_options *options,
    mofft_plan_diagnostics *diagnostics);

/* Bound-array FFTW-inspired lifecycle. The constructor returns NULL on
 * failure; flags accept 0 (PRETUNED/default), MOFFT_FLAG_MEASURE, or
 * MOFFT_FLAG_WISDOM_ONLY. The plan retains the supplied buffer addresses. */
mofft_plan_f32 *mofft_plan_dft_1d_f32(
    size_t length, const mofft_complex_f32 *input,
    mofft_complex_f32 *output, mofft_direction direction, uint32_t flags);
mofft_plan_f64 *mofft_plan_dft_1d_f64(
    size_t length, const mofft_complex_f64 *input,
    mofft_complex_f64 *output, mofft_direction direction, uint32_t flags);
mofft_status mofft_execute_plan_f32(const mofft_plan_f32 *plan);
mofft_status mofft_execute_plan_f64(const mofft_plan_f64 *plan);

/* Expert planner interface used by mofft-plan-search.  Radices are ordered
 * from the outermost stage to the first (innermost) stage. */
mofft_status mofft_plan_create_with_radices_f32(
    mofft_plan_f32 **plan, size_t length, mofft_direction direction,
    mofft_placement placement, const int *radices, size_t radix_count);
mofft_status mofft_plan_create_with_radices_f64(
    mofft_plan_f64 **plan, size_t length, mofft_direction direction,
    mofft_placement placement, const int *radices, size_t radix_count);
/* Plan-search control: set after creation and before concurrent execution. */
mofft_status mofft_plan_set_transpose_strategy_f32(
    mofft_plan_f32 *plan, mofft_transpose_strategy strategy);
mofft_status mofft_plan_set_transpose_strategy_f64(
    mofft_plan_f64 *plan, mofft_transpose_strategy strategy);
/* Bit level selects cache-blocking for that outer-to-inner stage index. */
mofft_status mofft_plan_set_blocked_transpose_mask_f32(
    mofft_plan_f32 *plan, uint64_t blocked_levels);
mofft_status mofft_plan_set_blocked_transpose_mask_f64(
    mofft_plan_f64 *plan, uint64_t blocked_levels);
mofft_status mofft_plan_set_direct_input_f32(mofft_plan_f32 *plan,
                                              int enabled);
mofft_status mofft_plan_set_direct_input_f64(mofft_plan_f64 *plan,
                                              int enabled);
mofft_status mofft_plan_set_stage_layout_f32(mofft_plan_f32 *plan,
                                              mofft_stage_layout layout);
mofft_status mofft_plan_set_stage_layout_f64(mofft_plan_f64 *plan,
                                              mofft_stage_layout layout);
mofft_status mofft_plan_set_kernel_bank_f32(mofft_plan_f32 *plan,
                                             unsigned int bank);
mofft_status mofft_plan_set_kernel_bank_f64(mofft_plan_f64 *plan,
                                             unsigned int bank);
mofft_status mofft_plan_get_kernel_bank_f32(const mofft_plan_f32 *plan,
                                             unsigned int *bank);
mofft_status mofft_plan_get_kernel_bank_f64(const mofft_plan_f64 *plan,
                                             unsigned int *bank);

/* The build ID identifies the compiled code/model/wisdom set without exposing
 * producer source paths or tuning inputs. Recipes are explicit text records
 * scoped to this build and CPU family; they contain no pointers or timings. */
const char *mofft_build_id(void);
/* Query the export size with buffer=NULL, capacity=0. On success, *required
 * includes the final NUL byte. An undersized buffer is left unchanged and
 * returns MOFFT_BUFFER_TOO_SMALL with the required size. */
mofft_status mofft_plan_export_f32(const mofft_plan_f32 *plan, char *buffer,
                                   size_t capacity, size_t *required);
mofft_status mofft_plan_export_f64(const mofft_plan_f64 *plan, char *buffer,
                                   size_t capacity, size_t *required);
/* recipe_size includes the final NUL byte. Import allocates a fresh plan but
 * never measures or generates code. A build/CPU-family/precision mismatch
 * returns MOFFT_INCOMPATIBLE_RECIPE. On failure, *plan is NULL. On success,
 * diagnostics (when non-NULL) reports the shape and selected recipe. Buffer
 * addresses are not restored: execute with mofft_execute_f32/f64. */
mofft_status mofft_plan_import_f32(mofft_plan_f32 **plan, const char *recipe,
                                   size_t recipe_size,
                                   mofft_plan_diagnostics *diagnostics);
mofft_status mofft_plan_import_f64(mofft_plan_f64 **plan, const char *recipe,
                                   size_t recipe_size,
                                   mofft_plan_diagnostics *diagnostics);

mofft_status mofft_execute_f32(const mofft_plan_f32 *plan,
                               const mofft_complex_f32 *input,
                               mofft_complex_f32 *output);
mofft_status mofft_execute_f64(const mofft_plan_f64 *plan,
                               const mofft_complex_f64 *input,
                               mofft_complex_f64 *output);

void mofft_plan_destroy_f32(mofft_plan_f32 *plan);
void mofft_plan_destroy_f64(mofft_plan_f64 *plan);
double mofft_estimated_radix_cost(int radix, int precision_bits,
                                  int first_stage);
double mofft_estimated_stage_cost(int radix, int precision_bits,
                                  int first_stage, size_t transform_length,
                                  size_t stage_length);
/* Estimate a complete expert plan, including layout conversion and the cache
 * line utilization of direct strided input. Radices use the same outermost to
 * innermost order as mofft_plan_create_with_radices_*(). */
double mofft_estimated_plan_cost(size_t length, int precision_bits,
                                 const int *radices, size_t radix_count,
                                 uint64_t blocked_transpose_mask,
                                 int direct_input);
double mofft_estimated_plan_cost_with_layout(
    size_t length, int precision_bits, const int *radices, size_t radix_count,
    uint64_t blocked_transpose_mask, int direct_input,
    mofft_stage_layout layout);
/* Return the same complete-plan estimate with auditable compute, memory,
 * layout, and traffic components. Transferred bytes include the analytical
 * cache-line overfetch bound; calibrated layout efficiency affects cost. */
mofft_status mofft_estimated_plan_cost_breakdown_with_layout(
    size_t length, int precision_bits, const int *radices, size_t radix_count,
    uint64_t blocked_transpose_mask, int direct_input,
    mofft_stage_layout layout, mofft_plan_cost_breakdown *breakdown);
int mofft_wisdom_is_formal(void);
const char *mofft_wisdom_source(void);
const char *mofft_status_string(mofft_status status);
const char *mofft_version(void);

#ifdef __cplusplus
}
#endif
#endif
