#include "mofft.h"
#include "mofft_generated_kernels.h"
#include "mofft_generated_wisdom.h"
#include <arm_sve.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MOFFT_MAX_DEPTH 64
#define MOFFT_PI 3.141592653589793238462643383279502884

static const int k_radices[] = {64, 32, 16, 15, 14, 13, 12, 11, 10,
                                9, 8, 7, 6, 5, 4, 3, 2};

static mofft_status set_specific_factors(size_t length, const int *radices,
                                         size_t radix_count, int *factors,
                                         size_t *lengths, int *depth,
                                         size_t *workspace_count);

static size_t saturating_multiply_size(size_t a, size_t b) {
  return a != 0 && b > SIZE_MAX / a ? SIZE_MAX : a * b;
}

static size_t saturating_add_size(size_t a, size_t b) {
  return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

struct mofft_plan_f32 {
  size_t length;
  int direction;
  mofft_placement placement;
  int depth;
  int factors[MOFFT_MAX_DEPTH];
  size_t level_lengths[MOFFT_MAX_DEPTH];
  size_t offsets[MOFFT_MAX_DEPTH];
  mofft_complex_f32 *scratch;
  mofft_complex_f32 *twiddles;
  size_t workspace_count;
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
};

struct mofft_plan_f64 {
  size_t length;
  int direction;
  mofft_placement placement;
  int depth;
  int factors[MOFFT_MAX_DEPTH];
  size_t level_lengths[MOFFT_MAX_DEPTH];
  size_t offsets[MOFFT_MAX_DEPTH];
  mofft_complex_f64 *scratch;
  mofft_complex_f64 *twiddles;
  size_t workspace_count;
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
};

static void *aligned_allocate(size_t alignment, size_t bytes) {
  void *result = NULL;
  if (bytes == 0) return NULL;
  return posix_memalign(&result, alignment, bytes) == 0 ? result : NULL;
}

typedef struct {
  double kernel_cost;
  double memory_cost;
  double total_cost;
  size_t logical_read_bytes;
  size_t logical_write_bytes;
  size_t working_set_bytes;
} mofft_stage_cost;

static int estimate_stage_cost(int radix, int precision_bits, int first_stage,
                               size_t transform_length, size_t stage_length,
                               size_t extra_read_bytes,
                               mofft_stage_cost *result) {
  if (radix < 2 || (precision_bits != 32 && precision_bits != 64) ||
      transform_length == 0 || stage_length == 0 ||
      stage_length > transform_length ||
      transform_length % (size_t)radix != 0 || !result)
    return 0;
  const size_t lanes = precision_bits == 32 ? 16 : 8;
  const size_t independent = transform_length / (size_t)radix;
  const double batches = (double)((independent + lanes - 1) / lanes);
  result->kernel_cost = mofft_generated_radix_cost_stage(
      radix, precision_bits, first_stage != 0) * batches;
  const size_t complex_bytes = precision_bits == 32 ? 8 : 16;
  const size_t data_bytes = saturating_multiply_size(
      transform_length, complex_bytes);
  /* Every stage traverses all N inputs and outputs. Later stages additionally
   * consume a compact twiddle table whose live extent is stage_length. Using
   * stage_length for the data arrays incorrectly classified late stages of a
   * large FFT as L1/L2-resident. */
  const size_t twiddle_bytes = first_stage
      ? 0 : saturating_multiply_size(stage_length, complex_bytes);
  result->logical_read_bytes = saturating_add_size(data_bytes, twiddle_bytes);
  result->logical_write_bytes = data_bytes;
  result->working_set_bytes = saturating_add_size(
      saturating_multiply_size(data_bytes, 2), twiddle_bytes);
  const size_t transferred_reads = saturating_add_size(
      result->logical_read_bytes, extra_read_bytes);
  result->memory_cost = mofft_generated_memory_cost(
      result->working_set_bytes, transferred_reads,
      result->logical_write_bytes);
  result->total_cost = result->kernel_cost > result->memory_cost
      ? result->kernel_cost : result->memory_cost;
  return 1;
}

double mofft_estimated_stage_cost(int radix, int precision_bits,
                                  int first_stage, size_t transform_length,
                                  size_t stage_length) {
  mofft_stage_cost cost;
  return estimate_stage_cost(radix, precision_bits, first_stage,
                             transform_length, stage_length, 0, &cost)
             ? cost.total_cost : DBL_MAX;
}

static size_t saturating_divide_by_utilization(size_t bytes,
                                                size_t useful_line_bytes,
                                                size_t cache_line_bytes) {
  if (useful_line_bytes >= cache_line_bytes) return bytes;
  if (useful_line_bytes == 0 || bytes > SIZE_MAX / cache_line_bytes)
    return SIZE_MAX;
  const size_t product = bytes * cache_line_bytes;
  if (product > SIZE_MAX - (useful_line_bytes - 1)) return SIZE_MAX;
  return (product + useful_line_bytes - 1) / useful_line_bytes;
}

mofft_status mofft_estimated_plan_cost_breakdown_with_layout(
    size_t length, int precision_bits, const int *radices, size_t radix_count,
    uint64_t blocked_transpose_mask, int direct_input,
    mofft_stage_layout layout, mofft_plan_cost_breakdown *breakdown) {
  if (!radices || radix_count == 0 || radix_count > MOFFT_MAX_DEPTH ||
      length < 2 || (precision_bits != 32 && precision_bits != 64) ||
      (direct_input != 0 && direct_input != 1) ||
      (layout != MOFFT_STAGE_LAYOUT_TRANSPOSE &&
       layout != MOFFT_STAGE_LAYOUT_SECTION) ||
      blocked_transpose_mask >> (radix_count - 1) || !breakdown)
    return MOFFT_INVALID_ARGUMENT;
  memset(breakdown, 0, sizeof(*breakdown));
  const size_t lanes = precision_bits == 32 ? 16 : 8;
  const size_t complex_bytes = precision_bits == 32 ? 8 : 16;
  const size_t cache_line_bytes = mofft_generated_cache_line_bytes();
  const size_t data_bytes = saturating_multiply_size(length, complex_bytes);
  size_t n = length;
  for (size_t level = 0; level < radix_count; ++level) {
    const int radix = radices[level];
    int supported = 0;
    for (size_t i = 0; i < sizeof(k_radices) / sizeof(k_radices[0]); ++i)
      supported |= k_radices[i] == radix;
    if (!supported || n % (size_t)radix != 0)
      return MOFFT_INVALID_ARGUMENT;
    const size_t remainder = n / (size_t)radix;
    size_t extra_read_bytes = 0;
    const int fused_layout =
#ifdef MOFFT_FUSE_FIRST_TRANSPOSE
        radix_count == 2 && level == 0;
#else
        0;
#endif
    if (remainder != 1 && layout == MOFFT_STAGE_LAYOUT_SECTION) {
      /* A section remains contiguous through every stage.  The kernel reads
       * radix rows separated by the number of sections, but each row consumes
       * `remainder` adjacent complex values.  Charge cache-line overfetch for
       * short rows; unlike the transpose layout there is no materialization
       * pass, so its read/write bandwidth is already in the stage cost. */
      const size_t useful_line_bytes = saturating_multiply_size(
          remainder, complex_bytes);
      const size_t transferred = saturating_divide_by_utilization(
          data_bytes, useful_line_bytes, cache_line_bytes);
      extra_read_bytes = transferred > data_bytes
                             ? transferred - data_bytes : 0;
    } else if (remainder != 1 && !fused_layout) {
      const size_t parent_batch = length / n;
      const size_t useful_line_bytes = saturating_multiply_size(
          parent_batch, complex_bytes);
      const int use_direct =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
          direct_input && parent_batch >= lanes / 2;
#else
          0;
#endif
      if (use_direct) {
        /* The stage cost already includes one useful input read. Charge only
         * cache-line overfetch caused by its strided direct-input layout. */
        const size_t transferred = saturating_divide_by_utilization(
            data_bytes, useful_line_bytes, cache_line_bytes);
        extra_read_bytes = transferred > data_bytes
                               ? transferred - data_bytes : 0;
      } else {
        /* Materialization is a distinct pass before the matrix kernel. The
         * generated model uses measured layout efficiency when calibration
         * data is available, and a cache-line utilization bound otherwise. */
        const int blocked =
            (blocked_transpose_mask >> level) & UINT64_C(1);
        const size_t layout_working_set = saturating_multiply_size(
            data_bytes, 2);
        const double layout_cost = mofft_generated_transpose_cost(
            layout_working_set, data_bytes,
            (size_t)radix, parent_batch, complex_bytes, blocked);
        breakdown->layout_cost += layout_cost;
        breakdown->logical_read_bytes = saturating_add_size(
            breakdown->logical_read_bytes, data_bytes);
        breakdown->logical_write_bytes = saturating_add_size(
            breakdown->logical_write_bytes, data_bytes);
        const size_t useful_line_bytes = saturating_multiply_size(
            parent_batch, complex_bytes);
        const size_t layout_reads = blocked
            ? data_bytes : saturating_divide_by_utilization(
                               data_bytes, useful_line_bytes,
                               cache_line_bytes);
        breakdown->transferred_read_bytes = saturating_add_size(
            breakdown->transferred_read_bytes, layout_reads);
        breakdown->transferred_write_bytes = saturating_add_size(
            breakdown->transferred_write_bytes, data_bytes);
        if (layout_working_set > breakdown->peak_working_set_bytes)
          breakdown->peak_working_set_bytes = layout_working_set;
      }
    }
    mofft_stage_cost stage;
    if (!estimate_stage_cost(radix, precision_bits, remainder == 1,
                             length, n, extra_read_bytes, &stage))
      return MOFFT_INVALID_ARGUMENT;
    breakdown->kernel_cost += stage.kernel_cost;
    breakdown->memory_cost += stage.memory_cost;
    breakdown->total_cost += stage.total_cost;
    breakdown->logical_read_bytes = saturating_add_size(
        breakdown->logical_read_bytes, stage.logical_read_bytes);
    breakdown->logical_write_bytes = saturating_add_size(
        breakdown->logical_write_bytes, stage.logical_write_bytes);
    breakdown->transferred_read_bytes = saturating_add_size(
        breakdown->transferred_read_bytes,
        saturating_add_size(stage.logical_read_bytes, extra_read_bytes));
    breakdown->transferred_write_bytes = saturating_add_size(
        breakdown->transferred_write_bytes, stage.logical_write_bytes);
    if (stage.working_set_bytes > breakdown->peak_working_set_bytes)
      breakdown->peak_working_set_bytes = stage.working_set_bytes;
    n = remainder;
  }
  if (n != 1) return MOFFT_INVALID_ARGUMENT;
  breakdown->total_cost += breakdown->layout_cost;
  return MOFFT_SUCCESS;
}

double mofft_estimated_plan_cost_with_layout(
    size_t length, int precision_bits, const int *radices, size_t radix_count,
    uint64_t blocked_transpose_mask, int direct_input,
    mofft_stage_layout layout) {
  mofft_plan_cost_breakdown breakdown;
  return mofft_estimated_plan_cost_breakdown_with_layout(
             length, precision_bits, radices, radix_count,
             blocked_transpose_mask, direct_input, layout, &breakdown) ==
         MOFFT_SUCCESS
             ? breakdown.total_cost : DBL_MAX;
}

double mofft_estimated_plan_cost(size_t length, int precision_bits,
                                 const int *radices, size_t radix_count,
                                 uint64_t blocked_transpose_mask,
                                 int direct_input) {
  return mofft_estimated_plan_cost_with_layout(
      length, precision_bits, radices, radix_count, blocked_transpose_mask,
      direct_input, MOFFT_STAGE_LAYOUT_TRANSPOSE);
}

static double choose_factorization(size_t n, size_t transform_length,
                                   int precision_bits,
                                   double *memo, int *choice) {
  if (n == 1) return 0.0;
  if (memo[n] >= 0.0) return memo[n];
  double best = DBL_MAX;
  int best_radix = 0;
  for (size_t i = 0; i < sizeof(k_radices) / sizeof(k_radices[0]); ++i) {
    int radix = k_radices[i];
    if (n % (size_t)radix != 0) continue;
    size_t remainder = n / (size_t)radix;
    double suffix = choose_factorization(remainder, transform_length,
                                         precision_bits, memo, choice);
    if (suffix == DBL_MAX) continue;
    double score = mofft_estimated_stage_cost(
        radix, precision_bits, remainder == 1, transform_length, n) + suffix;
    if (score < best) {
      best = score;
      best_radix = radix;
    }
  }
  memo[n] = best;
  choice[n] = best_radix;
  return best;
}

static mofft_status make_factors(size_t length, int precision_bits,
                                 int *factors, size_t *lengths, int *depth,
                                 size_t *workspace_count,
                                 uint64_t *blocked_transpose_mask,
                                 int *direct_input,
                                 mofft_stage_layout *stage_layout) {
  if (length < 2 || length > SIZE_MAX / sizeof(double))
    return MOFFT_UNSUPPORTED_SIZE;
  int wisdom[MOFFT_MAX_DEPTH];
  uint64_t wisdom_transpose_mask = 0;
  int wisdom_direct_input =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
      1;
#else
      0;
#endif
  int wisdom_stage_layout = MOFFT_STAGE_LAYOUT_TRANSPOSE;
  size_t wisdom_count = mofft_generated_wisdom_lookup(
      length, precision_bits, wisdom, MOFFT_MAX_DEPTH, &wisdom_transpose_mask,
      &wisdom_direct_input, &wisdom_stage_layout);
  *blocked_transpose_mask = wisdom_transpose_mask;
  *direct_input = wisdom_direct_input;
  *stage_layout = (mofft_stage_layout)wisdom_stage_layout;
  if (wisdom_count != 0)
    return set_specific_factors(length, wisdom, wisdom_count, factors, lengths,
                                depth, workspace_count);
  double *memo = (double *)malloc((length + 1) * sizeof(double));
  int *choice = (int *)calloc(length + 1, sizeof(int));
  if (!memo || !choice) {
    free(memo); free(choice);
    return MOFFT_ALLOCATION_FAILURE;
  }
  for (size_t i = 0; i <= length; ++i) memo[i] = -1.0;
  memo[1] = 0.0;
  double score = choose_factorization(length, length, precision_bits,
                                      memo, choice);
  if (score == DBL_MAX || choice[length] == 0) {
    free(memo); free(choice);
    return MOFFT_UNSUPPORTED_SIZE;
  }
  size_t n = length;
  int d = 0;
  while (n > 1 && d < MOFFT_MAX_DEPTH) {
    int radix = choice[n];
    if (radix == 0 || n % (size_t)radix != 0) break;
    factors[d] = radix;
    lengths[d] = n;
    n /= (size_t)radix;
    ++d;
  }
  free(memo); free(choice);
  if (n != 1 || d == 0) return MOFFT_UNSUPPORTED_SIZE;
  if (length > SIZE_MAX / 2)
    return MOFFT_ALLOCATION_FAILURE;
  *depth = d;
  *workspace_count = d == 1 ? 0 : length * 2;
  return MOFFT_SUCCESS;
}

static int is_supported_radix(int radix) {
  for (size_t i = 0; i < sizeof(k_radices) / sizeof(k_radices[0]); ++i)
    if (k_radices[i] == radix) return 1;
  return 0;
}

static mofft_status set_specific_factors(size_t length, const int *radices,
                                         size_t radix_count, int *factors,
                                         size_t *lengths, int *depth,
                                         size_t *workspace_count) {
  if (!radices || radix_count == 0 || radix_count > MOFFT_MAX_DEPTH ||
      length < 2 || length > SIZE_MAX / 2)
    return MOFFT_INVALID_ARGUMENT;
  size_t remaining = length;
  for (size_t level = 0; level < radix_count; ++level) {
    int radix = radices[level];
    if (!is_supported_radix(radix) || remaining % (size_t)radix != 0)
      return MOFFT_INVALID_ARGUMENT;
    factors[level] = radix;
    lengths[level] = remaining;
    remaining /= (size_t)radix;
  }
  if (remaining != 1) return MOFFT_INVALID_ARGUMENT;
  *depth = (int)radix_count;
  *workspace_count = radix_count == 1 ? 0 : length * 2;
  return MOFFT_SUCCESS;
}

static void fill_twiddles_f32(mofft_plan_f32 *plan) {
  for (int level = 0; level < plan->depth - 1; ++level) {
    size_t n = plan->level_lengths[level];
    int radix = plan->factors[level];
    size_t m = n / (size_t)radix;
    size_t parent_batch = plan->length / n;
    size_t stage_batch = m * parent_batch;
    mofft_complex_f32 *table = plan->twiddles + plan->offsets[level];
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
    const int compact = parent_batch > 1;
#else
    const int compact = 0;
#endif
    for (int k1 = 0; k1 < radix; ++k1) {
      for (size_t b = 0; b < m; ++b) {
        double a = (double)plan->direction * 2.0 * MOFFT_PI
                 * (double)k1 * (double)b / (double)n;
        const size_t transform_count = compact ? 1 : parent_batch;
        for (size_t transform = 0; transform < transform_count; ++transform) {
          size_t index = compact ? (size_t)k1 * m + b
                                 : (size_t)k1 * stage_batch +
                                       b * parent_batch + transform;
          table[index].real = (float)cos(a);
          table[index].imag = (float)sin(a);
        }
      }
    }
  }
}

static void fill_twiddles_f64(mofft_plan_f64 *plan) {
  for (int level = 0; level < plan->depth - 1; ++level) {
    size_t n = plan->level_lengths[level];
    int radix = plan->factors[level];
    size_t m = n / (size_t)radix;
    size_t parent_batch = plan->length / n;
    size_t stage_batch = m * parent_batch;
    mofft_complex_f64 *table = plan->twiddles + plan->offsets[level];
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
    const int compact = parent_batch > 1;
#else
    const int compact = 0;
#endif
    for (int k1 = 0; k1 < radix; ++k1) {
      for (size_t b = 0; b < m; ++b) {
        double a = (double)plan->direction * 2.0 * MOFFT_PI
                 * (double)k1 * (double)b / (double)n;
        const size_t transform_count = compact ? 1 : parent_batch;
        for (size_t transform = 0; transform < transform_count; ++transform) {
          size_t index = compact ? (size_t)k1 * m + b
                                 : (size_t)k1 * stage_batch +
                                       b * parent_batch + transform;
          table[index].real = cos(a);
          table[index].imag = sin(a);
        }
      }
    }
  }
}

static mofft_status finish_plan_f32(mofft_plan_f32 *plan,
                                    mofft_plan_f32 **out) {
  if (plan->length > SIZE_MAX / (size_t)plan->depth) {
    free(plan);
    return MOFFT_ALLOCATION_FAILURE;
  }
  size_t twiddle_count = 0;
  for (int i = 0; i < plan->depth - 1; ++i) {
    plan->offsets[i] = twiddle_count;
    const size_t parent_batch = plan->length / plan->level_lengths[i];
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
    const int compact = parent_batch > 1;
#else
    const int compact = 0;
#endif
    const size_t stage_count = compact ? plan->level_lengths[i] : plan->length;
    if (twiddle_count > SIZE_MAX - stage_count) {
      free(plan); return MOFFT_ALLOCATION_FAILURE;
    }
    twiddle_count += stage_count;
  }
  plan->scratch = plan->workspace_count == 0 ? NULL : aligned_allocate(
      64, plan->workspace_count * sizeof(*plan->scratch));
  if (twiddle_count > SIZE_MAX / sizeof(*plan->twiddles)) {
    mofft_plan_destroy_f32(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->twiddles = twiddle_count == 0 ? NULL : aligned_allocate(
      64, twiddle_count * sizeof(*plan->twiddles));
  if ((plan->workspace_count != 0 && !plan->scratch) ||
      (twiddle_count != 0 && !plan->twiddles)) {
    mofft_plan_destroy_f32(plan);
    return MOFFT_ALLOCATION_FAILURE;
  }
  fill_twiddles_f32(plan);
  *out = plan;
  return MOFFT_SUCCESS;
}

static mofft_status finish_plan_f64(mofft_plan_f64 *plan,
                                    mofft_plan_f64 **out) {
  if (plan->length > SIZE_MAX / (size_t)plan->depth) {
    free(plan);
    return MOFFT_ALLOCATION_FAILURE;
  }
  size_t twiddle_count = 0;
  for (int i = 0; i < plan->depth - 1; ++i) {
    plan->offsets[i] = twiddle_count;
    const size_t parent_batch = plan->length / plan->level_lengths[i];
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
    const int compact = parent_batch > 1;
#else
    const int compact = 0;
#endif
    const size_t stage_count = compact ? plan->level_lengths[i] : plan->length;
    if (twiddle_count > SIZE_MAX - stage_count) {
      free(plan); return MOFFT_ALLOCATION_FAILURE;
    }
    twiddle_count += stage_count;
  }
  plan->scratch = plan->workspace_count == 0 ? NULL : aligned_allocate(
      64, plan->workspace_count * sizeof(*plan->scratch));
  if (twiddle_count > SIZE_MAX / sizeof(*plan->twiddles)) {
    mofft_plan_destroy_f64(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->twiddles = twiddle_count == 0 ? NULL : aligned_allocate(
      64, twiddle_count * sizeof(*plan->twiddles));
  if ((plan->workspace_count != 0 && !plan->scratch) ||
      (twiddle_count != 0 && !plan->twiddles)) {
    mofft_plan_destroy_f64(plan);
    return MOFFT_ALLOCATION_FAILURE;
  }
  fill_twiddles_f64(plan);
  *out = plan;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_create_f32(mofft_plan_f32 **out, size_t length,
                                   mofft_direction direction,
                                   mofft_placement placement) {
  if (!out || (direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  mofft_plan_f32 *plan = (mofft_plan_f32 *)calloc(1, sizeof(*plan));
  if (!plan) return MOFFT_ALLOCATION_FAILURE;
  plan->length = length; plan->direction = direction; plan->placement = placement;
  mofft_status status = make_factors(length, 32, plan->factors,
                                     plan->level_lengths, &plan->depth,
                                     &plan->workspace_count,
                                     &plan->blocked_transpose_mask,
                                     &plan->direct_input,
                                     &plan->stage_layout);
  if (status != MOFFT_SUCCESS) { free(plan); return status; }
  return finish_plan_f32(plan, out);
}

mofft_status mofft_plan_create_f64(mofft_plan_f64 **out, size_t length,
                                   mofft_direction direction,
                                   mofft_placement placement) {
  if (!out || (direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  mofft_plan_f64 *plan = (mofft_plan_f64 *)calloc(1, sizeof(*plan));
  if (!plan) return MOFFT_ALLOCATION_FAILURE;
  plan->length = length; plan->direction = direction; plan->placement = placement;
  mofft_status status = make_factors(length, 64, plan->factors,
                                     plan->level_lengths, &plan->depth,
                                     &plan->workspace_count,
                                     &plan->blocked_transpose_mask,
                                     &plan->direct_input,
                                     &plan->stage_layout);
  if (status != MOFFT_SUCCESS) { free(plan); return status; }
  return finish_plan_f64(plan, out);
}

mofft_status mofft_plan_create_with_radices_f32(
    mofft_plan_f32 **out, size_t length, mofft_direction direction,
    mofft_placement placement, const int *radices, size_t radix_count) {
  if (!out || (direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  mofft_plan_f32 *plan = (mofft_plan_f32 *)calloc(1, sizeof(*plan));
  if (!plan) return MOFFT_ALLOCATION_FAILURE;
  plan->length = length;
  plan->direction = direction;
  plan->placement = placement;
  plan->blocked_transpose_mask = 0;
  plan->stage_layout = MOFFT_STAGE_LAYOUT_TRANSPOSE;
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
  plan->direct_input = 1;
#endif
  mofft_status status = set_specific_factors(
      length, radices, radix_count, plan->factors, plan->level_lengths,
      &plan->depth, &plan->workspace_count);
  if (status != MOFFT_SUCCESS) {
    free(plan);
    return status;
  }
  return finish_plan_f32(plan, out);
}

mofft_status mofft_plan_create_with_radices_f64(
    mofft_plan_f64 **out, size_t length, mofft_direction direction,
    mofft_placement placement, const int *radices, size_t radix_count) {
  if (!out || (direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  mofft_plan_f64 *plan = (mofft_plan_f64 *)calloc(1, sizeof(*plan));
  if (!plan) return MOFFT_ALLOCATION_FAILURE;
  plan->length = length;
  plan->direction = direction;
  plan->placement = placement;
  plan->blocked_transpose_mask = 0;
  plan->stage_layout = MOFFT_STAGE_LAYOUT_TRANSPOSE;
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
  plan->direct_input = 1;
#endif
  mofft_status status = set_specific_factors(
      length, radices, radix_count, plan->factors, plan->level_lengths,
      &plan->depth, &plan->workspace_count);
  if (status != MOFFT_SUCCESS) {
    free(plan);
    return status;
  }
  return finish_plan_f64(plan, out);
}

mofft_status mofft_plan_set_transpose_strategy_f32(
    mofft_plan_f32 *plan, mofft_transpose_strategy strategy) {
  if (!plan || (strategy != MOFFT_TRANSPOSE_LINEAR &&
                strategy != MOFFT_TRANSPOSE_BLOCKED))
    return MOFFT_INVALID_ARGUMENT;
  plan->blocked_transpose_mask = strategy == MOFFT_TRANSPOSE_BLOCKED
      ? (plan->depth <= 1 ? 0 : (UINT64_C(1) << (plan->depth - 1)) - 1) : 0;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_transpose_strategy_f64(
    mofft_plan_f64 *plan, mofft_transpose_strategy strategy) {
  if (!plan || (strategy != MOFFT_TRANSPOSE_LINEAR &&
                strategy != MOFFT_TRANSPOSE_BLOCKED))
    return MOFFT_INVALID_ARGUMENT;
  plan->blocked_transpose_mask = strategy == MOFFT_TRANSPOSE_BLOCKED
      ? (plan->depth <= 1 ? 0 : (UINT64_C(1) << (plan->depth - 1)) - 1) : 0;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_blocked_transpose_mask_f32(
    mofft_plan_f32 *plan, uint64_t blocked_levels) {
  if (!plan) return MOFFT_INVALID_ARGUMENT;
  const int transpose_count = plan->depth > 1 ? plan->depth - 1 : 0;
  if (transpose_count < 64 && blocked_levels >> transpose_count)
    return MOFFT_INVALID_ARGUMENT;
  plan->blocked_transpose_mask = blocked_levels;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_blocked_transpose_mask_f64(
    mofft_plan_f64 *plan, uint64_t blocked_levels) {
  if (!plan) return MOFFT_INVALID_ARGUMENT;
  const int transpose_count = plan->depth > 1 ? plan->depth - 1 : 0;
  if (transpose_count < 64 && blocked_levels >> transpose_count)
    return MOFFT_INVALID_ARGUMENT;
  plan->blocked_transpose_mask = blocked_levels;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_direct_input_f32(mofft_plan_f32 *plan,
                                              int enabled) {
  if (!plan || (enabled != 0 && enabled != 1)) return MOFFT_INVALID_ARGUMENT;
  plan->direct_input = enabled;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_direct_input_f64(mofft_plan_f64 *plan,
                                              int enabled) {
  if (!plan || (enabled != 0 && enabled != 1)) return MOFFT_INVALID_ARGUMENT;
  plan->direct_input = enabled;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_stage_layout_f32(mofft_plan_f32 *plan,
                                              mofft_stage_layout layout) {
  if (!plan || (layout != MOFFT_STAGE_LAYOUT_TRANSPOSE &&
                layout != MOFFT_STAGE_LAYOUT_SECTION))
    return MOFFT_INVALID_ARGUMENT;
  plan->stage_layout = layout;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_stage_layout_f64(mofft_plan_f64 *plan,
                                              mofft_stage_layout layout) {
  if (!plan || (layout != MOFFT_STAGE_LAYOUT_TRANSPOSE &&
                layout != MOFFT_STAGE_LAYOUT_SECTION))
    return MOFFT_INVALID_ARGUMENT;
  plan->stage_layout = layout;
  return MOFFT_SUCCESS;
}

/*
 * Transpose the two plan-level axes while retaining the innermost batch axis:
 *
 *   source[k][j][b] -> destination[j][k][b].
 *
 * A complete j column can be much larger than cache.  The straightforward
 * loop then revisits every source cache line once per j.  Finishing an 8x8
 * tile before advancing k keeps those lines resident, which matters much more
 * than instruction count once a plan has three or more stages.  Keep this in
 * this helper always-inline: an out-of-line streaming call makes AppleClang
 * materialize an additional lazy ZA-save frame at every stage boundary.
 */
#define MOFFT_TRANSPOSE_TILE 8

__attribute__((target("sme"), always_inline))
static void transpose_stage_f32(const mofft_complex_f32 *source,
                                mofft_complex_f32 *destination,
                                size_t rows, int columns, size_t batch)
    __arm_streaming __arm_inout("za") {
  const size_t source_row = (size_t)columns * batch;
  const size_t destination_row = rows * batch;
  /* The planner measures this strategy independently from the radix chain. */
  for (size_t k0 = 0; k0 < rows; k0 += MOFFT_TRANSPOSE_TILE) {
    const size_t kend = k0 + MOFFT_TRANSPOSE_TILE < rows
                            ? k0 + MOFFT_TRANSPOSE_TILE : rows;
    for (int j0 = 0; j0 < columns; j0 += MOFFT_TRANSPOSE_TILE) {
      const int jend = j0 + MOFFT_TRANSPOSE_TILE < columns
                           ? j0 + MOFFT_TRANSPOSE_TILE : columns;
      for (int j = j0; j < jend; ++j)
        for (size_t k = k0; k < kend; ++k)
          for (size_t b = 0; b < batch; ++b)
            destination[(size_t)j * destination_row + k * batch + b] =
                source[k * source_row + (size_t)j * batch + b];
    }
  }
}

__attribute__((target("sme"), always_inline))
static void transpose_stage_f64(const mofft_complex_f64 *source,
                                mofft_complex_f64 *destination,
                                size_t rows, int columns, size_t batch)
    __arm_streaming __arm_inout("za") {
  const size_t source_row = (size_t)columns * batch;
  const size_t destination_row = rows * batch;
  /* The planner measures this strategy independently from the radix chain. */
  for (size_t k0 = 0; k0 < rows; k0 += MOFFT_TRANSPOSE_TILE) {
    const size_t kend = k0 + MOFFT_TRANSPOSE_TILE < rows
                            ? k0 + MOFFT_TRANSPOSE_TILE : rows;
    for (int j0 = 0; j0 < columns; j0 += MOFFT_TRANSPOSE_TILE) {
      const int jend = j0 + MOFFT_TRANSPOSE_TILE < columns
                           ? j0 + MOFFT_TRANSPOSE_TILE : columns;
      for (int j = j0; j < jend; ++j)
        for (size_t k = k0; k < kend; ++k)
          for (size_t b = 0; b < batch; ++b)
            destination[(size_t)j * destination_row + k * batch + b] =
                source[k * source_row + (size_t)j * batch + b];
    }
  }
}

__attribute__((target("sme")))
void mofft_execute_streaming_body_f32(const mofft_plan_f32 *plan,
                                      const mofft_complex_f32 *input,
                                      mofft_complex_f32 *output) __arm_streaming __arm_inout("za") {
  const int deepest = plan->depth - 1;
  const int first_radix = plan->factors[deepest];
  const size_t first_batch = plan->length / (size_t)first_radix;
  if (deepest == 0) {
    mofft_dispatch_first_fp32(first_radix, plan->direction, input, first_batch,
                              output, first_batch);
    return;
  }
  mofft_complex_f32 *current = plan->scratch;
  mofft_complex_f32 *spare = plan->scratch + plan->length;
  if (plan->stage_layout == MOFFT_STAGE_LAYOUT_SECTION) {
    mofft_dispatch_first_transposed_fp32(first_radix, plan->direction, input,
                                         first_batch, current, first_batch);
    for (int level = deepest - 1; level >= 0; --level) {
      const size_t n = plan->level_lengths[level];
      const int radix = plan->factors[level];
      const size_t butterfly_batch = n / (size_t)radix;
      const size_t section_count = plan->length / n;
      mofft_complex_f32 *stage_output = level == 0 ? output : spare;
      const mofft_complex_f32 *tw = plan->twiddles + plan->offsets[level];
      const size_t input_stride = butterfly_batch * section_count;
      for (size_t section = 0; section < section_count; ++section)
        mofft_dispatch_other_fp32(
            radix, plan->direction, current + section * butterfly_batch,
            input_stride, 0, tw, butterfly_batch, 1,
            stage_output + section * (size_t)radix * butterfly_batch,
            butterfly_batch);
      if (level != 0) {
        mofft_complex_f32 *temporary = current;
        current = spare;
        spare = temporary;
      }
    }
    return;
  }
  const int fuse_first_transpose =
#ifdef MOFFT_FUSE_FIRST_TRANSPOSE
      plan->depth == 2;
#else
      0;
#endif
  if (fuse_first_transpose) {
    mofft_dispatch_first_transposed_fp32(first_radix, plan->direction, input,
                                         first_batch, spare, first_batch);
  } else {
    mofft_dispatch_first_fp32(first_radix, plan->direction, input, first_batch,
                              current, first_batch);
  }
  for (int level = deepest - 1; level >= 0; --level) {
    const size_t n = plan->level_lengths[level];
    const int radix = plan->factors[level];
    const size_t m = n / (size_t)radix;
    const size_t parent_batch = plan->length / n;
    const size_t stage_batch = parent_batch * m;
    const int direct_input =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
        plan->direct_input && !fuse_first_transpose && parent_batch >= 8;
#else
        0;
#endif
    const mofft_complex_f32 *stage_input;
    size_t input_stride;
    size_t input_repeat;
    if (fuse_first_transpose && level == deepest - 1) {
      stage_input = spare;
      input_stride = stage_batch;
      input_repeat = 0;
    } else if (direct_input) {
      stage_input = current;
      input_stride = parent_batch;
      input_repeat = parent_batch;
    } else {
      if ((plan->blocked_transpose_mask >> level) & UINT64_C(1)) {
        transpose_stage_f32(current, spare, m, radix, parent_batch);
      } else {
        const size_t child_batch = parent_batch * (size_t)radix;
        for (int j = 0; j < radix; ++j)
          for (size_t k = 0; k < m; ++k)
            for (size_t b = 0; b < parent_batch; ++b)
              spare[(size_t)j * stage_batch + k * parent_batch + b] =
                  current[k * child_batch + (size_t)j * parent_batch + b];
      }
      stage_input = spare;
      input_stride = stage_batch;
      input_repeat = 0;
    }
    mofft_complex_f32 *stage_output = level == 0 ? output
        : (direct_input ? spare : current);
    const mofft_complex_f32 *tw = plan->twiddles + plan->offsets[level];
    const int compact_twiddles =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
        parent_batch > 1;
#else
        0;
#endif
    const size_t twiddle_repeat = compact_twiddles ? parent_batch : 1;
    const size_t twiddle_stride = compact_twiddles ? m : stage_batch;
    mofft_dispatch_other_fp32(radix, plan->direction, stage_input,
                              input_stride, input_repeat, tw, twiddle_stride,
                              twiddle_repeat, stage_output, stage_batch);
    if (level != 0 && direct_input) {
      mofft_complex_f32 *temporary = current;
      current = spare;
      spare = temporary;
    }
  }
}

__attribute__((target("sme")))
void mofft_execute_streaming_body_f64(const mofft_plan_f64 *plan,
                                      const mofft_complex_f64 *input,
                                      mofft_complex_f64 *output) __arm_streaming __arm_inout("za") {
  const int deepest = plan->depth - 1;
  const int first_radix = plan->factors[deepest];
  const size_t first_batch = plan->length / (size_t)first_radix;
  if (deepest == 0) {
    mofft_dispatch_first_fp64(first_radix, plan->direction, input, first_batch,
                              output, first_batch);
    return;
  }
  mofft_complex_f64 *current = plan->scratch;
  mofft_complex_f64 *spare = plan->scratch + plan->length;
  if (plan->stage_layout == MOFFT_STAGE_LAYOUT_SECTION) {
    mofft_dispatch_first_transposed_fp64(first_radix, plan->direction, input,
                                         first_batch, current, first_batch);
    for (int level = deepest - 1; level >= 0; --level) {
      const size_t n = plan->level_lengths[level];
      const int radix = plan->factors[level];
      const size_t butterfly_batch = n / (size_t)radix;
      const size_t section_count = plan->length / n;
      mofft_complex_f64 *stage_output = level == 0 ? output : spare;
      const mofft_complex_f64 *tw = plan->twiddles + plan->offsets[level];
      const size_t input_stride = butterfly_batch * section_count;
      for (size_t section = 0; section < section_count; ++section)
        mofft_dispatch_other_fp64(
            radix, plan->direction, current + section * butterfly_batch,
            input_stride, 0, tw, butterfly_batch, 1,
            stage_output + section * (size_t)radix * butterfly_batch,
            butterfly_batch);
      if (level != 0) {
        mofft_complex_f64 *temporary = current;
        current = spare;
        spare = temporary;
      }
    }
    return;
  }
  const int fuse_first_transpose =
#ifdef MOFFT_FUSE_FIRST_TRANSPOSE
      plan->depth == 2;
#else
      0;
#endif
  if (fuse_first_transpose) {
    mofft_dispatch_first_transposed_fp64(first_radix, plan->direction, input,
                                         first_batch, spare, first_batch);
  } else {
    mofft_dispatch_first_fp64(first_radix, plan->direction, input, first_batch,
                              current, first_batch);
  }
  for (int level = deepest - 1; level >= 0; --level) {
    const size_t n = plan->level_lengths[level];
    const int radix = plan->factors[level];
    const size_t m = n / (size_t)radix;
    const size_t parent_batch = plan->length / n;
    const size_t stage_batch = parent_batch * m;
    const int direct_input =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
        plan->direct_input && !fuse_first_transpose && parent_batch >= 4;
#else
        0;
#endif
    const mofft_complex_f64 *stage_input;
    size_t input_stride;
    size_t input_repeat;
    if (fuse_first_transpose && level == deepest - 1) {
      stage_input = spare;
      input_stride = stage_batch;
      input_repeat = 0;
    } else if (direct_input) {
      stage_input = current;
      input_stride = parent_batch;
      input_repeat = parent_batch;
    } else {
      if ((plan->blocked_transpose_mask >> level) & UINT64_C(1)) {
        transpose_stage_f64(current, spare, m, radix, parent_batch);
      } else {
        const size_t child_batch = parent_batch * (size_t)radix;
        for (int j = 0; j < radix; ++j)
          for (size_t k = 0; k < m; ++k)
            for (size_t b = 0; b < parent_batch; ++b)
              spare[(size_t)j * stage_batch + k * parent_batch + b] =
                  current[k * child_batch + (size_t)j * parent_batch + b];
      }
      stage_input = spare;
      input_stride = stage_batch;
      input_repeat = 0;
    }
    mofft_complex_f64 *stage_output = level == 0 ? output
        : (direct_input ? spare : current);
    const mofft_complex_f64 *tw = plan->twiddles + plan->offsets[level];
    const int compact_twiddles =
#ifdef MOFFT_BROADCAST_REPEATED_TWIDDLES
        parent_batch > 1;
#else
        0;
#endif
    const size_t twiddle_repeat = compact_twiddles ? parent_batch : 1;
    const size_t twiddle_stride = compact_twiddles ? m : stage_batch;
    mofft_dispatch_other_fp64(radix, plan->direction, stage_input,
                              input_stride, input_repeat, tw, twiddle_stride,
                              twiddle_repeat, stage_output, stage_batch);
    if (level != 0 && direct_input) {
      mofft_complex_f64 *temporary = current;
      current = spare;
      spare = temporary;
    }
  }
}

extern void mofft_streaming_entry_f32(const mofft_plan_f32 *,
                                      const mofft_complex_f32 *,
                                      mofft_complex_f32 *);
extern void mofft_streaming_entry_f64(const mofft_plan_f64 *,
                                      const mofft_complex_f64 *,
                                      mofft_complex_f64 *);

mofft_status mofft_execute_f32(const mofft_plan_f32 *plan,
                               const mofft_complex_f32 *input,
                               mofft_complex_f32 *output) {
  if (!plan || !input || !output) return MOFFT_INVALID_ARGUMENT;
  if ((plan->placement == MOFFT_IN_PLACE) != (input == output))
    return MOFFT_INVALID_ARGUMENT;
  mofft_streaming_entry_f32(plan, input, output);
  return MOFFT_SUCCESS;
}

mofft_status mofft_execute_f64(const mofft_plan_f64 *plan,
                               const mofft_complex_f64 *input,
                               mofft_complex_f64 *output) {
  if (!plan || !input || !output) return MOFFT_INVALID_ARGUMENT;
  if ((plan->placement == MOFFT_IN_PLACE) != (input == output))
    return MOFFT_INVALID_ARGUMENT;
  mofft_streaming_entry_f64(plan, input, output);
  return MOFFT_SUCCESS;
}

void mofft_plan_destroy_f32(mofft_plan_f32 *plan) {
  if (!plan) return;
  free(plan->scratch); free(plan->twiddles); free(plan);
}

void mofft_plan_destroy_f64(mofft_plan_f64 *plan) {
  if (!plan) return;
  free(plan->scratch); free(plan->twiddles); free(plan);
}

double mofft_estimated_radix_cost(int radix, int precision_bits,
                                  int first_stage) {
  return mofft_generated_radix_cost_stage(radix, precision_bits,
                                           first_stage != 0);
}

int mofft_wisdom_is_formal(void) {
  return mofft_generated_wisdom_is_formal();
}

const char *mofft_wisdom_source(void) {
  return mofft_generated_wisdom_source();
}

const char *mofft_status_string(mofft_status status) {
  switch (status) {
    case MOFFT_SUCCESS: return "success";
    case MOFFT_INVALID_ARGUMENT: return "invalid argument";
    case MOFFT_UNSUPPORTED_SIZE: return "unsupported size";
    case MOFFT_ALLOCATION_FAILURE: return "allocation failure";
    case MOFFT_UNSUPPORTED_HARDWARE: return "unsupported hardware";
    case MOFFT_INTERNAL_ERROR: return "internal error";
    default: return "unknown status";
  }
}

const char *mofft_version(void) { return "0.4.0"; }
