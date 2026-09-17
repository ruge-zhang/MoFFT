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
  MOFFT_INTERNAL_ERROR = 5
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
