#include "mofft.h"
#include "mofft_generated_kernels.h"
#include "mofft_generated_wisdom.h"
#ifdef MOFFT_HAVE_KERNEL_PORTFOLIO
#include "mofft_kernel_portfolio.h"
#endif
#include <arm_sve.h>

#include <float.h>
#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/sysctl.h>

#define MOFFT_MAX_DEPTH MOFFT_MAX_PLAN_DEPTH
#define MOFFT_PI 3.141592653589793238462643383279502884
#define MOFFT_DEFAULT_SEARCH_CANDIDATES 16
#define MOFFT_DEFAULT_TIMING_SAMPLES 5
#define MOFFT_MAX_SEARCH_CANDIDATES 64
#define MOFFT_MAX_TIMING_SAMPLES 31

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
  const mofft_complex_f32 *bound_input;
  mofft_complex_f32 *bound_output;
  int has_bound_arrays;
  unsigned int kernel_bank;
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
  const mofft_complex_f64 *bound_input;
  mofft_complex_f64 *bound_output;
  int has_bound_arrays;
  unsigned int kernel_bank;
};

static void *aligned_allocate(size_t alignment, size_t bytes) {
  void *result = NULL;
  if (bytes == 0) return NULL;
  return posix_memalign(&result, alignment, bytes) == 0 ? result : NULL;
}

static int apple_feature_enabled(const char *name) {
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname(name, &value, &size, NULL, 0) == 0 &&
         size == sizeof(value) && value != 0;
}

static int mofft_hardware_supports(int precision_bits) {
  static atomic_int f32_support = ATOMIC_VAR_INIT(-1);
  static atomic_int f64_support = ATOMIC_VAR_INIT(-1);
  atomic_int *cached = precision_bits == 32 ? &f32_support : &f64_support;
  int value = atomic_load_explicit(cached, memory_order_acquire);
  if (value >= 0) return value;
  const int supports_sme = apple_feature_enabled("hw.optional.arm.FEAT_SME");
  const int supports_sme2 = apple_feature_enabled("hw.optional.arm.FEAT_SME2");
  int supported = supports_sme && supports_sme2;
  if (precision_bits == 64)
    supported = supported &&
        apple_feature_enabled("hw.optional.arm.FEAT_SME_F64F64");
  int expected = -1;
  atomic_compare_exchange_strong_explicit(
      cached, &expected, supported, memory_order_release,
      memory_order_relaxed);
  return atomic_load_explicit(cached, memory_order_acquire);
}

static unsigned int mofft_kernel_bank_count(void) {
#ifdef MOFFT_HAVE_KERNEL_PORTFOLIO
  const unsigned int count = mofft_generated_kernel_bank_count();
  return count == 0 ? 1 : count;
#else
  return 1;
#endif
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
                                 mofft_stage_layout *stage_layout,
                                 unsigned int *kernel_bank) {
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
  unsigned int wisdom_kernel_bank = 0;
  size_t wisdom_count = mofft_generated_wisdom_lookup(
      length, precision_bits, wisdom, MOFFT_MAX_DEPTH, &wisdom_transpose_mask,
      &wisdom_direct_input, &wisdom_stage_layout, &wisdom_kernel_bank);
  if (kernel_bank) *kernel_bank = 0;
  *blocked_transpose_mask = wisdom_transpose_mask;
  *direct_input = wisdom_direct_input;
  *stage_layout = (mofft_stage_layout)wisdom_stage_layout;
  if (wisdom_count != 0) {
    if (wisdom_kernel_bank >= mofft_kernel_bank_count())
      return MOFFT_WISDOM_NOT_FOUND;
    if (kernel_bank) *kernel_bank = wisdom_kernel_bank;
    return set_specific_factors(length, wisdom, wisdom_count, factors, lengths,
                                depth, workspace_count);
  }
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

typedef struct {
  int factors[MOFFT_MAX_DEPTH];
  size_t depth;
  double model_cost;
} mofft_factor_path;

typedef struct {
  mofft_factor_path *items;
  size_t count;
} mofft_factor_path_list;

typedef struct {
  size_t length;
  int precision_bits;
  size_t limit;
  mofft_factor_path_list **memo;
  int allocation_failed;
} mofft_factor_search;

typedef struct {
  int factors[MOFFT_MAX_DEPTH];
  size_t depth;
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
  unsigned int kernel_bank;
  double model_cost;
  double samples[MOFFT_MAX_TIMING_SAMPLES];
  double median_seconds;
  void *plan;
} mofft_measured_candidate;

typedef struct {
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
} mofft_plan_variant;

static int factor_path_compare(const mofft_factor_path *a,
                               const mofft_factor_path *b) {
  if (a->model_cost < b->model_cost) return -1;
  if (a->model_cost > b->model_cost) return 1;
  const size_t common = a->depth < b->depth ? a->depth : b->depth;
  for (size_t i = 0; i < common; ++i) {
    if (a->factors[i] < b->factors[i]) return -1;
    if (a->factors[i] > b->factors[i]) return 1;
  }
  if (a->depth < b->depth) return -1;
  if (a->depth > b->depth) return 1;
  return 0;
}

static void insert_factor_path(mofft_factor_path_list *list, size_t limit,
                               const mofft_factor_path *path) {
  size_t position = 0;
  for (; position < list->count; ++position) {
    mofft_factor_path *existing = &list->items[position];
    if (existing->depth == path->depth &&
        memcmp(existing->factors, path->factors,
               path->depth * sizeof(path->factors[0])) == 0) {
      if (path->model_cost < existing->model_cost) *existing = *path;
      while (position > 0 && factor_path_compare(existing,
                                                  &list->items[position - 1]) < 0) {
        mofft_factor_path temporary = list->items[position - 1];
        list->items[position - 1] = list->items[position];
        list->items[position] = temporary;
        --position;
      }
      return;
    }
    if (factor_path_compare(path, existing) < 0) break;
  }
  if (position >= limit) return;
  size_t new_count = list->count < limit ? list->count + 1 : list->count;
  if (new_count > position)
    memmove(&list->items[position + 1], &list->items[position],
            (new_count - position - 1) * sizeof(list->items[0]));
  list->items[position] = *path;
  list->count = new_count;
}

/* Keep only the best modeled suffixes at each divisor. This is a bounded
 * dynamic-programming beam: it explores alternative radix orders without
 * enumerating every permutation of a smooth length. */
static mofft_factor_path_list *factor_paths_for(mofft_factor_search *search,
                                                size_t n) {
  if (search->memo[n]) return search->memo[n];
  mofft_factor_path_list *list =
      (mofft_factor_path_list *)calloc(1, sizeof(*list));
  if (!list) {
    search->allocation_failed = 1;
    return NULL;
  }
  list->items = (mofft_factor_path *)calloc(search->limit, sizeof(*list->items));
  if (!list->items) {
    free(list);
    search->allocation_failed = 1;
    return NULL;
  }
  if (n == 1) {
    list->count = 1;
    list->items[0].model_cost = 0.0;
    search->memo[n] = list;
    return list;
  }
  for (size_t i = 0; i < sizeof(k_radices) / sizeof(k_radices[0]); ++i) {
    const int radix = k_radices[i];
    if (n % (size_t)radix != 0) continue;
    const size_t remainder = n / (size_t)radix;
    mofft_factor_path_list *suffixes = factor_paths_for(search, remainder);
    if (!suffixes) break;
    const double stage_cost = mofft_estimated_stage_cost(
        radix, search->precision_bits, remainder == 1, search->length, n);
    for (size_t j = 0; j < suffixes->count; ++j) {
      if (suffixes->items[j].depth >= MOFFT_MAX_DEPTH) continue;
      mofft_factor_path candidate = suffixes->items[j];
      memmove(&candidate.factors[1], &candidate.factors[0],
              candidate.depth * sizeof(candidate.factors[0]));
      candidate.factors[0] = radix;
      ++candidate.depth;
      candidate.model_cost += stage_cost;
      insert_factor_path(list, search->limit, &candidate);
    }
  }
  search->memo[n] = list;
  return list;
}

static void destroy_factor_search(mofft_factor_search *search) {
  if (!search || !search->memo) return;
  for (size_t n = 0; n <= search->length; ++n) {
    if (!search->memo[n]) continue;
    free(search->memo[n]->items);
    free(search->memo[n]);
  }
  free(search->memo);
  search->memo = NULL;
}

static size_t make_plan_variants(size_t depth, mofft_plan_variant *variants,
                                 size_t capacity) {
  if (depth == 0 || !variants || capacity == 0) return 0;
  const size_t transpose_count = depth - 1;
  const uint64_t all = transpose_count == 0
      ? 0 : (UINT64_C(1) << transpose_count) - UINT64_C(1);
  uint64_t even = 0, odd = 0;
  for (size_t level = 0; level < transpose_count; ++level) {
    if (level & 1) odd |= UINT64_C(1) << level;
    else even |= UINT64_C(1) << level;
  }
  const mofft_plan_variant proposed[] = {
      {0, 0, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {all, 0, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {even, 0, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {odd, 0, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {0, 1, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {all, 1, MOFFT_STAGE_LAYOUT_TRANSPOSE},
      {0, 0, MOFFT_STAGE_LAYOUT_SECTION}};
  size_t count = 0;
  for (size_t i = 0; i < sizeof(proposed) / sizeof(proposed[0]); ++i) {
    int duplicate = 0;
    for (size_t j = 0; j < count; ++j)
      if (variants[j].blocked_transpose_mask ==
              proposed[i].blocked_transpose_mask &&
          variants[j].direct_input == proposed[i].direct_input &&
          variants[j].stage_layout == proposed[i].stage_layout) {
        duplicate = 1;
        break;
      }
    if (!duplicate && count < capacity) variants[count++] = proposed[i];
  }
  return count;
}

static int candidate_same_plan(const mofft_measured_candidate *a,
                               const mofft_measured_candidate *b) {
  return a->depth == b->depth &&
         memcmp(a->factors, b->factors,
                a->depth * sizeof(a->factors[0])) == 0 &&
         a->blocked_transpose_mask == b->blocked_transpose_mask &&
         a->direct_input == b->direct_input &&
         a->stage_layout == b->stage_layout &&
         a->kernel_bank == b->kernel_bank;
}

static int candidate_model_compare(const void *left, const void *right) {
  const mofft_measured_candidate *a =
      (const mofft_measured_candidate *)left;
  const mofft_measured_candidate *b =
      (const mofft_measured_candidate *)right;
  if (a->model_cost < b->model_cost) return -1;
  if (a->model_cost > b->model_cost) return 1;
  const size_t common = a->depth < b->depth ? a->depth : b->depth;
  for (size_t i = 0; i < common; ++i) {
    if (a->factors[i] < b->factors[i]) return -1;
    if (a->factors[i] > b->factors[i]) return 1;
  }
  if (a->depth < b->depth) return -1;
  if (a->depth > b->depth) return 1;
  if (a->blocked_transpose_mask < b->blocked_transpose_mask) return -1;
  if (a->blocked_transpose_mask > b->blocked_transpose_mask) return 1;
  if (a->direct_input != b->direct_input)
    return a->direct_input < b->direct_input ? -1 : 1;
  if (a->stage_layout != b->stage_layout)
    return a->stage_layout < b->stage_layout ? -1 : 1;
  if (a->kernel_bank != b->kernel_bank)
    return a->kernel_bank < b->kernel_bank ? -1 : 1;
  return 0;
}

static int candidate_time_compare(const void *left, const void *right) {
  const mofft_measured_candidate *a =
      (const mofft_measured_candidate *)left;
  const mofft_measured_candidate *b =
      (const mofft_measured_candidate *)right;
  if (a->median_seconds < b->median_seconds) return -1;
  if (a->median_seconds > b->median_seconds) return 1;
  return candidate_model_compare(left, right);
}

static double scalar_median(double *values, size_t count) {
  if (count == 0) return DBL_MAX;
  for (size_t i = 1; i < count; ++i) {
    const double value = values[i];
    size_t j = i;
    while (j > 0 && value < values[j - 1]) {
      values[j] = values[j - 1];
      --j;
    }
    values[j] = value;
  }
  return count & 1 ? values[count / 2]
                   : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double monotonic_seconds(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1.0;
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static mofft_status execute_measured_candidate(
    void *plan, int precision_bits, const void *input, void *output) {
  if (precision_bits == 32)
    return mofft_execute_f32((const mofft_plan_f32 *)plan,
        (const mofft_complex_f32 *)input, (mofft_complex_f32 *)output);
  return mofft_execute_f64((const mofft_plan_f64 *)plan,
      (const mofft_complex_f64 *)input, (mofft_complex_f64 *)output);
}

static void destroy_measured_candidate(mofft_measured_candidate *candidate,
                                       int precision_bits) {
  if (!candidate || !candidate->plan) return;
  if (precision_bits == 32)
    mofft_plan_destroy_f32((mofft_plan_f32 *)candidate->plan);
  else
    mofft_plan_destroy_f64((mofft_plan_f64 *)candidate->plan);
  candidate->plan = NULL;
}

static mofft_status create_measured_candidate(
    mofft_measured_candidate *candidate, size_t length,
    mofft_direction direction, mofft_placement placement, int precision_bits) {
  mofft_status status;
  if (precision_bits == 32) {
    mofft_plan_f32 *plan = NULL;
    status = mofft_plan_create_with_radices_f32(
        &plan, length, direction, placement, candidate->factors,
        candidate->depth);
    if (status == MOFFT_SUCCESS)
      status = mofft_plan_set_blocked_transpose_mask_f32(
          plan, candidate->blocked_transpose_mask);
    if (status == MOFFT_SUCCESS)
      status = mofft_plan_set_direct_input_f32(plan, candidate->direct_input);
    if (status == MOFFT_SUCCESS)
      status = mofft_plan_set_stage_layout_f32(plan, candidate->stage_layout);
    if (status == MOFFT_SUCCESS)
      status = mofft_plan_set_kernel_bank_f32(plan, candidate->kernel_bank);
    if (status != MOFFT_SUCCESS) {
      mofft_plan_destroy_f32(plan);
      return status;
    }
    candidate->plan = plan;
    return MOFFT_SUCCESS;
  }
  mofft_plan_f64 *plan = NULL;
  status = mofft_plan_create_with_radices_f64(
      &plan, length, direction, placement, candidate->factors,
      candidate->depth);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_blocked_transpose_mask_f64(
        plan, candidate->blocked_transpose_mask);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_direct_input_f64(plan, candidate->direct_input);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_stage_layout_f64(plan, candidate->stage_layout);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_set_kernel_bank_f64(plan, candidate->kernel_bank);
  if (status != MOFFT_SUCCESS) {
    mofft_plan_destroy_f64(plan);
    return status;
  }
  candidate->plan = plan;
  return MOFFT_SUCCESS;
}

static int measured_output_is_finite(const void *output, size_t length,
                                     int precision_bits) {
  if (precision_bits == 32) {
    const mofft_complex_f32 *values = (const mofft_complex_f32 *)output;
    for (size_t i = 0; i < length; ++i)
      if (!isfinite(values[i].real) || !isfinite(values[i].imag)) return 0;
  } else {
    const mofft_complex_f64 *values = (const mofft_complex_f64 *)output;
    for (size_t i = 0; i < length; ++i)
      if (!isfinite(values[i].real) || !isfinite(values[i].imag)) return 0;
  }
  return 1;
}

static mofft_status validate_options(const mofft_plan_options *given,
                                     mofft_plan_options *resolved) {
  if (!resolved) return MOFFT_INVALID_ARGUMENT;
  if (given) *resolved = *given;
  else {
    resolved->mode = MOFFT_PLANNER_PRETUNED;
    resolved->flags = 0;
    resolved->candidate_limit = 0;
    resolved->timing_samples = 0;
  }
  if ((resolved->mode != MOFFT_PLANNER_PRETUNED &&
       resolved->mode != MOFFT_PLANNER_MEASURE) ||
      (resolved->flags & ~((uint32_t)MOFFT_PLAN_STRICT_WISDOM)) != 0 ||
      (resolved->mode == MOFFT_PLANNER_MEASURE && resolved->flags != 0))
    return MOFFT_INVALID_ARGUMENT;
  if (resolved->candidate_limit == 0)
    resolved->candidate_limit = MOFFT_DEFAULT_SEARCH_CANDIDATES;
  if (resolved->timing_samples == 0)
    resolved->timing_samples = MOFFT_DEFAULT_TIMING_SAMPLES;
  if (resolved->candidate_limit > MOFFT_MAX_SEARCH_CANDIDATES ||
      resolved->timing_samples > MOFFT_MAX_TIMING_SAMPLES)
    return MOFFT_INVALID_ARGUMENT;
  return MOFFT_SUCCESS;
}

/* Returns 1 for a compatible exact entry, 0 for a miss, and -1 for a stale
 * entry that names a generated bank absent from this binary. */
static int exact_wisdom_lookup(size_t length, int precision_bits,
                               unsigned int *selected_bank) {
  int factors[MOFFT_MAX_DEPTH];
  uint64_t transpose_mask = 0;
  int direct_input = 0, stage_layout = MOFFT_STAGE_LAYOUT_TRANSPOSE;
  unsigned int bank = 0;
  const size_t count = mofft_generated_wisdom_lookup(
      length, precision_bits, factors, MOFFT_MAX_DEPTH, &transpose_mask,
      &direct_input, &stage_layout, &bank);
  if (count == 0) return 0;
  if (bank >= mofft_kernel_bank_count()) return -1;
  if (selected_bank) *selected_bank = bank;
  return 1;
}

static void copy_plan_diagnostics(mofft_plan_diagnostics *diagnostics,
                                  const int *factors, size_t depth,
                                  uint64_t blocked_transpose_mask,
                                  int direct_input,
                                  mofft_stage_layout stage_layout,
                                  unsigned int kernel_bank) {
  if (!diagnostics) return;
  diagnostics->radix_count = depth;
  if (depth > 0)
    memcpy(diagnostics->radices, factors, depth * sizeof(factors[0]));
  diagnostics->blocked_transpose_mask = blocked_transpose_mask;
  diagnostics->direct_input = direct_input;
  diagnostics->stage_layout = stage_layout;
  diagnostics->kernel_bank = kernel_bank;
  diagnostics->kernel_bank_count = mofft_kernel_bank_count();
}

static mofft_status search_best_plan(
    void **out_plan, size_t length, mofft_direction direction,
    mofft_placement placement, int precision_bits,
    const mofft_plan_options *options,
    mofft_plan_diagnostics *diagnostics) {
  mofft_status status = MOFFT_SUCCESS;
  mofft_factor_search search;
  memset(&search, 0, sizeof(search));
  search.length = length;
  search.precision_bits = precision_bits;
  search.limit = options->candidate_limit;
  if (length > (SIZE_MAX / sizeof(*search.memo)) - 1)
    return MOFFT_ALLOCATION_FAILURE;
  search.memo = (mofft_factor_path_list **)calloc(
      length + 1, sizeof(*search.memo));
  if (!search.memo) return MOFFT_ALLOCATION_FAILURE;
  mofft_factor_path_list *paths = factor_paths_for(&search, length);
  if (search.allocation_failed) {
    destroy_factor_search(&search);
    return MOFFT_ALLOCATION_FAILURE;
  }
  if (!paths || paths->count == 0) {
    destroy_factor_search(&search);
    return MOFFT_UNSUPPORTED_SIZE;
  }

  /* Generate several empirically distinct data layouts for each modeled
   * factorization, then retain a bounded global shortlist. */
  const unsigned int bank_count = mofft_kernel_bank_count();
  if (bank_count > MOFFT_MAX_SEARCH_CANDIDATES) {
    destroy_factor_search(&search);
    return MOFFT_INTERNAL_ERROR;
  }
  const size_t pool_capacity = search.limit * 7 * bank_count;
  mofft_measured_candidate *pool = (mofft_measured_candidate *)calloc(
      pool_capacity, sizeof(*pool));
  if (!pool) {
    destroy_factor_search(&search);
    return MOFFT_ALLOCATION_FAILURE;
  }
  size_t pool_count = 0;
  for (size_t p = 0; p < paths->count; ++p) {
    mofft_plan_variant variants[7];
    const size_t variant_count = make_plan_variants(
        paths->items[p].depth, variants, sizeof(variants) / sizeof(variants[0]));
    for (size_t v = 0; v < variant_count; ++v) {
      for (unsigned int bank = 0; bank < bank_count; ++bank) {
        mofft_measured_candidate candidate;
        memset(&candidate, 0, sizeof(candidate));
        candidate.depth = paths->items[p].depth;
        memcpy(candidate.factors, paths->items[p].factors,
               candidate.depth * sizeof(candidate.factors[0]));
        candidate.blocked_transpose_mask = variants[v].blocked_transpose_mask;
        candidate.direct_input = variants[v].direct_input;
        candidate.stage_layout = variants[v].stage_layout;
        candidate.kernel_bank = bank;
        candidate.model_cost = mofft_estimated_plan_cost_with_layout(
            length, precision_bits, candidate.factors, candidate.depth,
            candidate.blocked_transpose_mask, candidate.direct_input,
            candidate.stage_layout);
        int duplicate = 0;
        for (size_t i = 0; i < pool_count; ++i)
          if (candidate_same_plan(&pool[i], &candidate)) {
            duplicate = 1;
            break;
          }
        if (!duplicate && pool_count < pool_capacity)
          pool[pool_count++] = candidate;
      }
    }
  }
  destroy_factor_search(&search);
  if (pool_count == 0) {
    free(pool);
    return MOFFT_UNSUPPORTED_SIZE;
  }
  qsort(pool, pool_count, sizeof(pool[0]), candidate_model_compare);

  /* Include at least one measured candidate from every compiled kernel bank
   * before spending the remaining budget on additional factorizations/layouts. */
  mofft_measured_candidate *selected = (mofft_measured_candidate *)calloc(
      options->candidate_limit, sizeof(*selected));
  if (!selected) {
    free(pool);
    return MOFFT_ALLOCATION_FAILURE;
  }
  size_t selected_count = 0;
  for (unsigned int bank = 0; bank < bank_count &&
       selected_count < options->candidate_limit; ++bank) {
    for (size_t i = 0; i < pool_count; ++i) {
      if (pool[i].kernel_bank == bank) {
        selected[selected_count++] = pool[i];
        break;
      }
    }
  }
  /* Then include one model-best candidate for each distinct radix chain in
   * each bank, so tuning can choose both its decomposition and kernel bank. */
  for (unsigned int bank = 0; bank < bank_count &&
       selected_count < options->candidate_limit; ++bank) {
    for (size_t i = 0; i < pool_count &&
         selected_count < options->candidate_limit; ++i) {
      if (pool[i].kernel_bank != bank) continue;
      int seen_factorization = 0;
      for (size_t j = 0; j < selected_count; ++j)
        if (selected[j].kernel_bank == bank &&
            selected[j].depth == pool[i].depth &&
            memcmp(selected[j].factors, pool[i].factors,
                   pool[i].depth * sizeof(pool[i].factors[0])) == 0) {
          seen_factorization = 1;
          break;
        }
      if (!seen_factorization) selected[selected_count++] = pool[i];
    }
  }
  for (size_t i = 0; i < pool_count &&
       selected_count < options->candidate_limit; ++i) {
    int seen = 0;
    for (size_t j = 0; j < selected_count; ++j)
      if (candidate_same_plan(&selected[j], &pool[i])) {
        seen = 1;
        break;
      }
    if (!seen) selected[selected_count++] = pool[i];
  }
  free(pool);
  if (selected_count == 0) {
    free(selected);
    return MOFFT_UNSUPPORTED_SIZE;
  }

  /* Vary measurement order so the first modeled candidate is not always
   * favored by startup/cache effects. Keep this deterministic for reproducible
   * plan choices on a stable machine. */
  uint32_t shuffle_state = (uint32_t)length ^
      ((uint32_t)precision_bits << 16) ^ UINT32_C(0x9e3779b9);
  for (size_t i = selected_count; i > 1; --i) {
    shuffle_state ^= shuffle_state << 13;
    shuffle_state ^= shuffle_state >> 17;
    shuffle_state ^= shuffle_state << 5;
    const size_t j = (size_t)shuffle_state % i;
    mofft_measured_candidate temporary = selected[i - 1];
    selected[i - 1] = selected[j];
    selected[j] = temporary;
  }

  const size_t complex_bytes = precision_bits == 32
      ? sizeof(mofft_complex_f32) : sizeof(mofft_complex_f64);
  if (length > SIZE_MAX / complex_bytes) {
    free(selected);
    return MOFFT_ALLOCATION_FAILURE;
  }
  const size_t bytes = length * complex_bytes;
  void *input = aligned_allocate(64, bytes);
  void *output = aligned_allocate(64, bytes);
  void *working = placement == MOFFT_IN_PLACE
      ? aligned_allocate(64, bytes) : NULL;
  if (!input || !output || (placement == MOFFT_IN_PLACE && !working)) {
    free(input); free(output); free(working); free(selected);
    return MOFFT_ALLOCATION_FAILURE;
  }
  uint32_t state = UINT32_C(0x7f4a7c15);
  for (size_t i = 0; i < length; ++i) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    const double real = ((double)(state & UINT32_C(0xffff)) / 32768.0) - 1.0;
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    const double imag = ((double)(state & UINT32_C(0xffff)) / 32768.0) - 1.0;
    if (precision_bits == 32) {
      ((mofft_complex_f32 *)input)[i].real = (float)real;
      ((mofft_complex_f32 *)input)[i].imag = (float)imag;
    } else {
      ((mofft_complex_f64 *)input)[i].real = real;
      ((mofft_complex_f64 *)input)[i].imag = imag;
    }
  }
  const uint32_t sample_count = options->timing_samples;
  const size_t iterations = length < 256 ? 8 : (length < 4096 ? 4 : 1);
  for (size_t i = 0; i < selected_count; ++i) {
    mofft_measured_candidate *candidate = &selected[i];
    status = create_measured_candidate(candidate, length, direction,
                                       placement, precision_bits);
    if (status != MOFFT_SUCCESS) goto cleanup_search;

    /* Warm the candidate once, outside the timed region. Only one full plan
     * workspace remains live at a time, so peak planner workspace is O(N),
     * independent of candidate_limit. */
    if (placement == MOFFT_IN_PLACE) memcpy(working, input, bytes);
    void *execution_input = placement == MOFFT_IN_PLACE ? working : input;
    void *execution_output = placement == MOFFT_IN_PLACE ? working : output;
    status = execute_measured_candidate(candidate->plan, precision_bits,
                                        execution_input, execution_output);
    if (status != MOFFT_SUCCESS) goto cleanup_search;
    if (!measured_output_is_finite(execution_output, length, precision_bits)) {
      status = MOFFT_INTERNAL_ERROR;
      goto cleanup_search;
    }

    for (uint32_t sample = 0; sample < sample_count; ++sample) {
      double execution_times[8];
      for (size_t iteration = 0; iteration < iterations; ++iteration) {
        /* Reset in-place input outside the timed interval, ensuring every
         * measurement runs the same deterministic transform. */
        if (placement == MOFFT_IN_PLACE) memcpy(working, input, bytes);
        const double start = monotonic_seconds();
        if (start < 0.0) { status = MOFFT_INTERNAL_ERROR; goto cleanup_search; }
        status = execute_measured_candidate(candidate->plan, precision_bits,
                                            execution_input, execution_output);
        const double finish = monotonic_seconds();
        if (status != MOFFT_SUCCESS) goto cleanup_search;
        if (finish < start) { status = MOFFT_INTERNAL_ERROR; goto cleanup_search; }
        execution_times[iteration] = finish - start;
      }
      if (!measured_output_is_finite(execution_output, length, precision_bits)) {
        status = MOFFT_INTERNAL_ERROR;
        goto cleanup_search;
      }
      candidate->samples[sample] = scalar_median(execution_times, iterations);
    }
    candidate->median_seconds = scalar_median(candidate->samples,
                                               sample_count);
    if (!isfinite(candidate->median_seconds)) {
      status = MOFFT_INTERNAL_ERROR;
      goto cleanup_search;
    }
    destroy_measured_candidate(candidate, precision_bits);
  }
  qsort(selected, selected_count, sizeof(selected[0]), candidate_time_compare);
  status = create_measured_candidate(&selected[0], length, direction,
                                     placement, precision_bits);
  if (status != MOFFT_SUCCESS) goto cleanup_search;
  *out_plan = selected[0].plan;
  selected[0].plan = NULL;
  if (diagnostics) {
    diagnostics->measured = 1;
    diagnostics->candidates_measured = selected_count;
    diagnostics->selected_median_seconds = selected[0].median_seconds;
    diagnostics->length = length;
    diagnostics->precision_bits = precision_bits;
    diagnostics->direction = direction;
    diagnostics->placement = placement;
    copy_plan_diagnostics(diagnostics, selected[0].factors,
                         selected[0].depth,
                         selected[0].blocked_transpose_mask,
                         selected[0].direct_input,
                         selected[0].stage_layout,
                         selected[0].kernel_bank);
  }
  status = MOFFT_SUCCESS;

cleanup_search:
  for (size_t i = 0; i < selected_count; ++i)
    destroy_measured_candidate(&selected[i], precision_bits);
  free(input); free(output); free(working); free(selected);
  return status;
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
  const size_t scratch_count = plan->workspace_count != 0
      ? plan->workspace_count
      : (plan->depth == 1 && plan->placement == MOFFT_IN_PLACE
             ? plan->length : 0);
  if (scratch_count > SIZE_MAX / sizeof(*plan->scratch)) {
    free(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->scratch = scratch_count == 0 ? NULL : aligned_allocate(
      64, scratch_count * sizeof(*plan->scratch));
  if (twiddle_count > SIZE_MAX / sizeof(*plan->twiddles)) {
    mofft_plan_destroy_f32(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->twiddles = twiddle_count == 0 ? NULL : aligned_allocate(
      64, twiddle_count * sizeof(*plan->twiddles));
  if ((scratch_count != 0 && !plan->scratch) ||
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
  const size_t scratch_count = plan->workspace_count != 0
      ? plan->workspace_count
      : (plan->depth == 1 && plan->placement == MOFFT_IN_PLACE
             ? plan->length : 0);
  if (scratch_count > SIZE_MAX / sizeof(*plan->scratch)) {
    free(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->scratch = scratch_count == 0 ? NULL : aligned_allocate(
      64, scratch_count * sizeof(*plan->scratch));
  if (twiddle_count > SIZE_MAX / sizeof(*plan->twiddles)) {
    mofft_plan_destroy_f64(plan); return MOFFT_ALLOCATION_FAILURE;
  }
  plan->twiddles = twiddle_count == 0 ? NULL : aligned_allocate(
      64, twiddle_count * sizeof(*plan->twiddles));
  if ((scratch_count != 0 && !plan->scratch) ||
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
                                     &plan->stage_layout,
                                     &plan->kernel_bank);
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
                                     &plan->stage_layout,
                                     &plan->kernel_bank);
  if (status != MOFFT_SUCCESS) { free(plan); return status; }
  return finish_plan_f64(plan, out);
}

void mofft_plan_options_init(mofft_plan_options *options) {
  if (!options) return;
  options->mode = MOFFT_PLANNER_PRETUNED;
  options->flags = 0;
  options->candidate_limit = 0;
  options->timing_samples = 0;
}

mofft_status mofft_plan_create_with_options_f32(
    mofft_plan_f32 **out, size_t length, mofft_direction direction,
    mofft_placement placement, const mofft_plan_options *given_options,
    mofft_plan_diagnostics *diagnostics) {
  if (!out) return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  if (diagnostics) memset(diagnostics, 0, sizeof(*diagnostics));
  if ((direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  mofft_plan_options options;
  mofft_status status = validate_options(given_options, &options);
  if (status != MOFFT_SUCCESS) return status;
  if (!mofft_hardware_supports(32)) return MOFFT_UNSUPPORTED_HARDWARE;
  const double start = monotonic_seconds();
  if (start < 0.0) return MOFFT_INTERNAL_ERROR;
  if (options.mode == MOFFT_PLANNER_MEASURE) {
    if (!mofft_hardware_supports(32)) return MOFFT_UNSUPPORTED_HARDWARE;
    void *measured = NULL;
    status = search_best_plan(&measured, length, direction, placement, 32,
                              &options, diagnostics);
    if (status != MOFFT_SUCCESS) return status;
    *out = (mofft_plan_f32 *)measured;
    if (diagnostics) {
      diagnostics->planning_seconds = monotonic_seconds() - start;
      diagnostics->kernel_bank_count = mofft_kernel_bank_count();
    }
    return MOFFT_SUCCESS;
  }
  const int wisdom_state = exact_wisdom_lookup(length, 32, NULL);
  if (wisdom_state < 0) return MOFFT_WISDOM_NOT_FOUND;
  const int has_wisdom = wisdom_state > 0;
  if (!has_wisdom && (options.flags & MOFFT_PLAN_STRICT_WISDOM))
    return MOFFT_WISDOM_NOT_FOUND;
  status = mofft_plan_create_f32(out, length, direction, placement);
  if (status == MOFFT_SUCCESS && diagnostics) {
    diagnostics->used_wisdom = has_wisdom;
    diagnostics->used_model_fallback = !has_wisdom;
    diagnostics->length = length;
    diagnostics->precision_bits = 32;
    diagnostics->direction = direction;
    diagnostics->placement = placement;
    copy_plan_diagnostics(diagnostics, (*out)->factors, (*out)->depth,
                          (*out)->blocked_transpose_mask,
                          (*out)->direct_input, (*out)->stage_layout,
                          (*out)->kernel_bank);
    diagnostics->planning_seconds = monotonic_seconds() - start;
  }
  return status;
}

mofft_status mofft_plan_create_with_options_f64(
    mofft_plan_f64 **out, size_t length, mofft_direction direction,
    mofft_placement placement, const mofft_plan_options *given_options,
    mofft_plan_diagnostics *diagnostics) {
  if (!out) return MOFFT_INVALID_ARGUMENT;
  *out = NULL;
  if (diagnostics) memset(diagnostics, 0, sizeof(*diagnostics));
  if ((direction != MOFFT_FORWARD && direction != MOFFT_BACKWARD) ||
      (placement != MOFFT_IN_PLACE && placement != MOFFT_OUT_OF_PLACE))
    return MOFFT_INVALID_ARGUMENT;
  mofft_plan_options options;
  mofft_status status = validate_options(given_options, &options);
  if (status != MOFFT_SUCCESS) return status;
  if (!mofft_hardware_supports(64)) return MOFFT_UNSUPPORTED_HARDWARE;
  const double start = monotonic_seconds();
  if (start < 0.0) return MOFFT_INTERNAL_ERROR;
  if (options.mode == MOFFT_PLANNER_MEASURE) {
    if (!mofft_hardware_supports(64)) return MOFFT_UNSUPPORTED_HARDWARE;
    void *measured = NULL;
    status = search_best_plan(&measured, length, direction, placement, 64,
                              &options, diagnostics);
    if (status != MOFFT_SUCCESS) return status;
    *out = (mofft_plan_f64 *)measured;
    if (diagnostics) {
      diagnostics->planning_seconds = monotonic_seconds() - start;
      diagnostics->kernel_bank_count = mofft_kernel_bank_count();
    }
    return MOFFT_SUCCESS;
  }
  const int wisdom_state = exact_wisdom_lookup(length, 64, NULL);
  if (wisdom_state < 0) return MOFFT_WISDOM_NOT_FOUND;
  const int has_wisdom = wisdom_state > 0;
  if (!has_wisdom && (options.flags & MOFFT_PLAN_STRICT_WISDOM))
    return MOFFT_WISDOM_NOT_FOUND;
  status = mofft_plan_create_f64(out, length, direction, placement);
  if (status == MOFFT_SUCCESS && diagnostics) {
    diagnostics->used_wisdom = has_wisdom;
    diagnostics->used_model_fallback = !has_wisdom;
    diagnostics->length = length;
    diagnostics->precision_bits = 64;
    diagnostics->direction = direction;
    diagnostics->placement = placement;
    copy_plan_diagnostics(diagnostics, (*out)->factors, (*out)->depth,
                          (*out)->blocked_transpose_mask,
                          (*out)->direct_input, (*out)->stage_layout,
                          (*out)->kernel_bank);
    diagnostics->planning_seconds = monotonic_seconds() - start;
  }
  return status;
}

static int options_from_bound_flags(uint32_t flags,
                                    mofft_plan_options *options) {
  if ((flags & ~((uint32_t)MOFFT_FLAG_MEASURE |
                 (uint32_t)MOFFT_FLAG_WISDOM_ONLY)) != 0 ||
      (flags & MOFFT_FLAG_MEASURE && flags & MOFFT_FLAG_WISDOM_ONLY))
    return 0;
  mofft_plan_options_init(options);
  options->mode = flags & MOFFT_FLAG_MEASURE
      ? MOFFT_PLANNER_MEASURE : MOFFT_PLANNER_PRETUNED;
  if (flags & MOFFT_FLAG_WISDOM_ONLY)
    options->flags |= MOFFT_PLAN_STRICT_WISDOM;
  return 1;
}

mofft_plan_f32 *mofft_plan_dft_1d_f32(
    size_t length, const mofft_complex_f32 *input,
    mofft_complex_f32 *output, mofft_direction direction, uint32_t flags) {
  if (!input || !output) return NULL;
  mofft_plan_options options;
  if (!options_from_bound_flags(flags, &options)) return NULL;
  mofft_plan_f32 *plan = NULL;
  const mofft_placement placement = input == output
      ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE;
  if (mofft_plan_create_with_options_f32(&plan, length, direction, placement,
                                         &options, NULL) != MOFFT_SUCCESS)
    return NULL;
  plan->bound_input = input;
  plan->bound_output = output;
  plan->has_bound_arrays = 1;
  return plan;
}

mofft_plan_f64 *mofft_plan_dft_1d_f64(
    size_t length, const mofft_complex_f64 *input,
    mofft_complex_f64 *output, mofft_direction direction, uint32_t flags) {
  if (!input || !output) return NULL;
  mofft_plan_options options;
  if (!options_from_bound_flags(flags, &options)) return NULL;
  mofft_plan_f64 *plan = NULL;
  const mofft_placement placement = input == output
      ? MOFFT_IN_PLACE : MOFFT_OUT_OF_PLACE;
  if (mofft_plan_create_with_options_f64(&plan, length, direction, placement,
                                         &options, NULL) != MOFFT_SUCCESS)
    return NULL;
  plan->bound_input = input;
  plan->bound_output = output;
  plan->has_bound_arrays = 1;
  return plan;
}

mofft_status mofft_execute_plan_f32(const mofft_plan_f32 *plan) {
  if (!plan || !plan->has_bound_arrays) return MOFFT_INVALID_ARGUMENT;
  return mofft_execute_f32(plan, plan->bound_input, plan->bound_output);
}

mofft_status mofft_execute_plan_f64(const mofft_plan_f64 *plan) {
  if (!plan || !plan->has_bound_arrays) return MOFFT_INVALID_ARGUMENT;
  return mofft_execute_f64(plan, plan->bound_input, plan->bound_output);
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

mofft_status mofft_plan_set_kernel_bank_f32(mofft_plan_f32 *plan,
                                             unsigned int bank) {
  if (!plan || bank >= mofft_kernel_bank_count())
    return MOFFT_INVALID_ARGUMENT;
  plan->kernel_bank = bank;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_set_kernel_bank_f64(mofft_plan_f64 *plan,
                                             unsigned int bank) {
  if (!plan || bank >= mofft_kernel_bank_count())
    return MOFFT_INVALID_ARGUMENT;
  plan->kernel_bank = bank;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_get_kernel_bank_f32(const mofft_plan_f32 *plan,
                                             unsigned int *bank) {
  if (!plan || !bank) return MOFFT_INVALID_ARGUMENT;
  *bank = plan->kernel_bank;
  return MOFFT_SUCCESS;
}

mofft_status mofft_plan_get_kernel_bank_f64(const mofft_plan_f64 *plan,
                                             unsigned int *bank) {
  if (!plan || !bank) return MOFFT_INVALID_ARGUMENT;
  *bank = plan->kernel_bank;
  return MOFFT_SUCCESS;
}

#ifndef MOFFT_BINARY_ID
#error "Build MoFFT with CMake to provide a content-derived MOFFT_BINARY_ID"
#endif

#define MOFFT_RECIPE_MAX_BYTES 1024

typedef struct {
  uint32_t cpu_family;
  size_t length;
  mofft_direction direction;
  mofft_placement placement;
  unsigned int kernel_bank;
  uint64_t blocked_transpose_mask;
  int direct_input;
  mofft_stage_layout stage_layout;
  size_t radix_count;
  int radices[MOFFT_MAX_DEPTH];
} mofft_recipe_fields;

const char *mofft_build_id(void) { return MOFFT_BINARY_ID; }

static int mofft_get_cpu_family(uint32_t *family) {
  if (!family) return 0;
  uint32_t value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname("hw.cpufamily", &value, &size, NULL, 0) != 0 ||
      size != sizeof(value) || value == 0)
    return 0;
  *family = value;
  return 1;
}

static int mofft_recipe_id_is_valid(const char *id) {
  if (!id || strlen(id) != 64) return 0;
  for (size_t i = 0; i < 64; ++i)
    if (!((id[i] >= '0' && id[i] <= '9') ||
          (id[i] >= 'a' && id[i] <= 'f')))
      return 0;
  return 1;
}

static int mofft_parse_uint_token(const char *token, int base,
                                 uint64_t *value) {
  if (!token || !*token || !value) return 0;
  for (const char *p = token; *p; ++p) {
    const int digit = (*p >= '0' && *p <= '9');
    const int lower_hex = base == 16 && *p >= 'a' && *p <= 'f';
    const int upper_hex = base == 16 && *p >= 'A' && *p <= 'F';
    if (!digit && !lower_hex && !upper_hex) return 0;
  }
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = strtoull(token, &end, base);
  if (errno == ERANGE || !end || *end != '\0') return 0;
  *value = (uint64_t)parsed;
  return 1;
}

static size_t mofft_split_recipe_tokens(char *text, char **tokens,
                                        size_t token_capacity) {
  size_t count = 0;
  char *cursor = text;
  while (*cursor) {
    if (count == token_capacity) return SIZE_MAX;
    tokens[count++] = cursor;
    while (*cursor && *cursor != ' ') ++cursor;
    if (*cursor == ' ') {
      *cursor = '\0';
      ++cursor;
      if (*cursor == ' ') return SIZE_MAX;
    }
  }
  return count;
}

static mofft_status mofft_parse_plan_recipe(const char *recipe,
                                             size_t recipe_size,
                                             int precision_bits,
                                             mofft_recipe_fields *fields) {
  if (!recipe || !fields || recipe_size < 2 ||
      recipe_size > MOFFT_RECIPE_MAX_BYTES || recipe[recipe_size - 1] != '\0' ||
      recipe[recipe_size - 2] != '\n' ||
      memchr(recipe, '\0', recipe_size - 1) != NULL)
    return MOFFT_INVALID_ARGUMENT;

  char text[MOFFT_RECIPE_MAX_BYTES];
  for (size_t i = 0; i < recipe_size - 2; ++i) {
    const unsigned char ch = (unsigned char)recipe[i];
    if (ch < 32 || ch > 126) return MOFFT_INVALID_ARGUMENT;
    text[i] = (char)ch;
  }
  text[recipe_size - 2] = '\0';
  if (recipe_size > 3 && text[recipe_size - 3] == ' ')
    return MOFFT_INVALID_ARGUMENT;

  char *tokens[12 + MOFFT_MAX_DEPTH];
  const size_t token_count = mofft_split_recipe_tokens(text, tokens,
                                                       12 + MOFFT_MAX_DEPTH);
  if (token_count == SIZE_MAX || token_count < 13 ||
      strcmp(tokens[0], "MOFFT_PLAN_V1") != 0 ||
      !mofft_recipe_id_is_valid(MOFFT_BINARY_ID) ||
      strlen(tokens[1]) != 64 || strcmp(tokens[1], MOFFT_BINARY_ID) != 0)
    return MOFFT_INCOMPATIBLE_RECIPE;
  if (strlen(tokens[2]) != 8 || strlen(tokens[8]) != 16)
    return MOFFT_INVALID_ARGUMENT;
  uint64_t cpu_family = 0, length = 0, placement = 0, bank = 0, mask = 0;
  uint64_t direct_input = 0, layout = 0, depth = 0;
  if (!mofft_parse_uint_token(tokens[2], 16, &cpu_family) ||
      cpu_family > UINT32_MAX ||
      !mofft_parse_uint_token(tokens[4], 10, &length) ||
      !mofft_parse_uint_token(tokens[6], 10, &placement) ||
      !mofft_parse_uint_token(tokens[7], 10, &bank) ||
      !mofft_parse_uint_token(tokens[8], 16, &mask) ||
      !mofft_parse_uint_token(tokens[9], 10, &direct_input) ||
      !mofft_parse_uint_token(tokens[10], 10, &layout) ||
      !mofft_parse_uint_token(tokens[11], 10, &depth))
    return MOFFT_INVALID_ARGUMENT;
  if ((precision_bits == 32 && strcmp(tokens[3], "f32") != 0) ||
      (precision_bits == 64 && strcmp(tokens[3], "f64") != 0))
    return MOFFT_INCOMPATIBLE_RECIPE;
  if ((strcmp(tokens[5], "-1") != 0 && strcmp(tokens[5], "1") != 0) ||
      (uint64_t)(size_t)length != length || length < 2 ||
      placement > MOFFT_IN_PLACE || bank >= mofft_kernel_bank_count() ||
      direct_input > 1 || layout > MOFFT_STAGE_LAYOUT_SECTION ||
      depth == 0 || depth > MOFFT_MAX_DEPTH ||
      token_count != 12 + (size_t)depth)
    return MOFFT_INVALID_ARGUMENT;
  if ((size_t)depth - 1 < 64 && (mask >> ((size_t)depth - 1)) != 0)
    return MOFFT_INVALID_ARGUMENT;

  fields->cpu_family = (uint32_t)cpu_family;
  fields->length = (size_t)length;
  fields->direction = strcmp(tokens[5], "-1") == 0
                          ? MOFFT_FORWARD : MOFFT_BACKWARD;
  fields->placement = (mofft_placement)placement;
  fields->kernel_bank = (unsigned int)bank;
  fields->blocked_transpose_mask = mask;
  fields->direct_input = (int)direct_input;
  fields->stage_layout = (mofft_stage_layout)layout;
  fields->radix_count = (size_t)depth;
  size_t remaining = fields->length;
  for (size_t i = 0; i < fields->radix_count; ++i) {
    uint64_t radix = 0;
    if (!mofft_parse_uint_token(tokens[12 + i], 10, &radix) ||
        radix > INT32_MAX || radix < 2 ||
        remaining % (size_t)radix != 0)
      return MOFFT_INVALID_ARGUMENT;
    fields->radices[i] = (int)radix;
    remaining /= (size_t)radix;
  }
  if (remaining != 1) return MOFFT_INVALID_ARGUMENT;
  uint32_t current_family = 0;
  if (!mofft_get_cpu_family(&current_family) ||
      current_family != fields->cpu_family)
    return MOFFT_INCOMPATIBLE_RECIPE;
  if (!mofft_hardware_supports(precision_bits))
    return MOFFT_UNSUPPORTED_HARDWARE;
  return MOFFT_SUCCESS;
}

#define MOFFT_DEFINE_RECIPE_API(SUFFIX, PLAN_TYPE, PRECISION, COMPLEX_TYPE)    \
  static mofft_status export_plan_##SUFFIX(                                  \
      const PLAN_TYPE *plan, char *buffer, size_t capacity, size_t *required) { \
    if (!plan || !required) return MOFFT_INVALID_ARGUMENT;                   \
    uint32_t cpu_family = 0;                                                  \
    if (!mofft_get_cpu_family(&cpu_family) ||                                \
        !mofft_recipe_id_is_valid(MOFFT_BINARY_ID))                           \
      return MOFFT_UNSUPPORTED_HARDWARE;                                      \
    char recipe[MOFFT_RECIPE_MAX_BYTES];                                      \
    int written = snprintf(                                                   \
        recipe, sizeof(recipe),                                               \
        "MOFFT_PLAN_V1 %s %08x f" #PRECISION " %zu %d %d %u %016llx %d %d %d", \
        MOFFT_BINARY_ID, cpu_family, plan->length, plan->direction,           \
        plan->placement, plan->kernel_bank,                                   \
        (unsigned long long)plan->blocked_transpose_mask, plan->direct_input, \
        plan->stage_layout, plan->depth);                                     \
    if (written < 0 || (size_t)written >= sizeof(recipe))                     \
      return MOFFT_INTERNAL_ERROR;                                            \
    size_t used = (size_t)written;                                            \
    for (int i = 0; i < plan->depth; ++i) {                                   \
      written = snprintf(recipe + used, sizeof(recipe) - used, " %d",         \
                         plan->factors[i]);                                   \
      if (written < 0 || (size_t)written >= sizeof(recipe) - used)            \
        return MOFFT_INTERNAL_ERROR;                                          \
      used += (size_t)written;                                                \
    }                                                                         \
    if (used + 2 > sizeof(recipe)) return MOFFT_INTERNAL_ERROR;               \
    recipe[used++] = '\n';                                                    \
    recipe[used] = '\0';                                                     \
    *required = used + 1;                                                     \
    if (!buffer) return capacity == 0 ? MOFFT_SUCCESS : MOFFT_INVALID_ARGUMENT; \
    if (capacity < *required) return MOFFT_BUFFER_TOO_SMALL;                  \
    memcpy(buffer, recipe, *required);                                        \
    return MOFFT_SUCCESS;                                                     \
  }                                                                           \
  mofft_status mofft_plan_export_##SUFFIX(                                   \
      const PLAN_TYPE *plan, char *buffer, size_t capacity, size_t *required) { \
    return export_plan_##SUFFIX(plan, buffer, capacity, required);            \
  }                                                                           \
  mofft_status mofft_plan_import_##SUFFIX(                                    \
      PLAN_TYPE **out, const char *recipe, size_t recipe_size,                \
      mofft_plan_diagnostics *diagnostics) {                                  \
    if (!out) return MOFFT_INVALID_ARGUMENT;                                  \
    *out = NULL;                                                              \
    if (diagnostics) memset(diagnostics, 0, sizeof(*diagnostics));             \
    mofft_recipe_fields fields;                                               \
    mofft_status status = mofft_parse_plan_recipe(recipe, recipe_size,        \
                                                   PRECISION, &fields);       \
    if (status != MOFFT_SUCCESS) return status;                               \
    status = mofft_plan_create_with_radices_##SUFFIX(                         \
        out, fields.length, fields.direction, fields.placement,               \
        fields.radices, fields.radix_count);                                  \
    if (status != MOFFT_SUCCESS) return status;                               \
    status = mofft_plan_set_blocked_transpose_mask_##SUFFIX(                  \
        *out, fields.blocked_transpose_mask);                                 \
    if (status == MOFFT_SUCCESS)                                              \
      status = mofft_plan_set_direct_input_##SUFFIX(*out, fields.direct_input); \
    if (status == MOFFT_SUCCESS)                                              \
      status = mofft_plan_set_stage_layout_##SUFFIX(*out, fields.stage_layout); \
    if (status == MOFFT_SUCCESS)                                              \
      status = mofft_plan_set_kernel_bank_##SUFFIX(*out, fields.kernel_bank); \
    if (status != MOFFT_SUCCESS) {                                            \
      mofft_plan_destroy_##SUFFIX(*out);                                     \
      *out = NULL;                                                            \
      return status;                                                          \
    }                                                                         \
    if (diagnostics) {                                                        \
      diagnostics->used_wisdom = 1;                                          \
      diagnostics->length = (*out)->length;                                   \
      diagnostics->precision_bits = PRECISION;                               \
      diagnostics->direction = (mofft_direction)(*out)->direction;            \
      diagnostics->placement = (*out)->placement;                             \
      copy_plan_diagnostics(diagnostics, (*out)->factors, (*out)->depth,      \
                            (*out)->blocked_transpose_mask,                  \
                            (*out)->direct_input, (*out)->stage_layout,       \
                            (*out)->kernel_bank);                            \
    }                                                                         \
    return MOFFT_SUCCESS;                                                     \
  }

MOFFT_DEFINE_RECIPE_API(f32, mofft_plan_f32, 32, mofft_complex_f32)
MOFFT_DEFINE_RECIPE_API(f64, mofft_plan_f64, 64, mofft_complex_f64)
#undef MOFFT_DEFINE_RECIPE_API

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

#ifdef MOFFT_HAVE_KERNEL_PORTFOLIO
#define MOFFT_DISPATCH_FIRST_FP32(plan, ...) \
  mofft_portfolio_dispatch_first_fp32((plan)->kernel_bank, __VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_TRANSPOSED_FP32(plan, ...) \
  mofft_portfolio_dispatch_first_transposed_fp32((plan)->kernel_bank, __VA_ARGS__)
#define MOFFT_DISPATCH_OTHER_FP32(plan, ...) \
  mofft_portfolio_dispatch_other_fp32((plan)->kernel_bank, __VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_FP64(plan, ...) \
  mofft_portfolio_dispatch_first_fp64((plan)->kernel_bank, __VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_TRANSPOSED_FP64(plan, ...) \
  mofft_portfolio_dispatch_first_transposed_fp64((plan)->kernel_bank, __VA_ARGS__)
#define MOFFT_DISPATCH_OTHER_FP64(plan, ...) \
  mofft_portfolio_dispatch_other_fp64((plan)->kernel_bank, __VA_ARGS__)
#else
#define MOFFT_DISPATCH_FIRST_FP32(plan, ...) \
  mofft_dispatch_first_fp32(__VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_TRANSPOSED_FP32(plan, ...) \
  mofft_dispatch_first_transposed_fp32(__VA_ARGS__)
#define MOFFT_DISPATCH_OTHER_FP32(plan, ...) \
  mofft_dispatch_other_fp32(__VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_FP64(plan, ...) \
  mofft_dispatch_first_fp64(__VA_ARGS__)
#define MOFFT_DISPATCH_FIRST_TRANSPOSED_FP64(plan, ...) \
  mofft_dispatch_first_transposed_fp64(__VA_ARGS__)
#define MOFFT_DISPATCH_OTHER_FP64(plan, ...) \
  mofft_dispatch_other_fp64(__VA_ARGS__)
#endif

__attribute__((target("sme")))
void mofft_execute_streaming_body_f32(const mofft_plan_f32 *plan,
                                      const mofft_complex_f32 *input,
                                      mofft_complex_f32 *output) __arm_streaming __arm_inout("za") {
  const int deepest = plan->depth - 1;
  const int first_radix = plan->factors[deepest];
  const size_t first_batch = plan->length / (size_t)first_radix;
  if (deepest == 0) {
    MOFFT_DISPATCH_FIRST_FP32(plan, first_radix, plan->direction, input,
                              first_batch, output, first_batch);
    return;
  }
  mofft_complex_f32 *current = plan->scratch;
  mofft_complex_f32 *spare = plan->scratch + plan->length;
  if (plan->stage_layout == MOFFT_STAGE_LAYOUT_SECTION) {
    MOFFT_DISPATCH_FIRST_TRANSPOSED_FP32(
        plan, first_radix, plan->direction, input, first_batch, current,
        first_batch);
    for (int level = deepest - 1; level >= 0; --level) {
      const size_t n = plan->level_lengths[level];
      const int radix = plan->factors[level];
      const size_t butterfly_batch = n / (size_t)radix;
      const size_t section_count = plan->length / n;
      mofft_complex_f32 *stage_output = level == 0 ? output : spare;
      const mofft_complex_f32 *tw = plan->twiddles + plan->offsets[level];
      const size_t input_stride = butterfly_batch * section_count;
      for (size_t section = 0; section < section_count; ++section)
        MOFFT_DISPATCH_OTHER_FP32(
            plan,
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
    MOFFT_DISPATCH_FIRST_TRANSPOSED_FP32(
        plan, first_radix, plan->direction, input, first_batch, spare,
        first_batch);
  } else {
    MOFFT_DISPATCH_FIRST_FP32(plan, first_radix, plan->direction, input,
                              first_batch, current, first_batch);
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
    MOFFT_DISPATCH_OTHER_FP32(plan, radix, plan->direction, stage_input,
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
    MOFFT_DISPATCH_FIRST_FP64(plan, first_radix, plan->direction, input,
                              first_batch, output, first_batch);
    return;
  }
  mofft_complex_f64 *current = plan->scratch;
  mofft_complex_f64 *spare = plan->scratch + plan->length;
  if (plan->stage_layout == MOFFT_STAGE_LAYOUT_SECTION) {
    MOFFT_DISPATCH_FIRST_TRANSPOSED_FP64(
        plan, first_radix, plan->direction, input, first_batch, current,
        first_batch);
    for (int level = deepest - 1; level >= 0; --level) {
      const size_t n = plan->level_lengths[level];
      const int radix = plan->factors[level];
      const size_t butterfly_batch = n / (size_t)radix;
      const size_t section_count = plan->length / n;
      mofft_complex_f64 *stage_output = level == 0 ? output : spare;
      const mofft_complex_f64 *tw = plan->twiddles + plan->offsets[level];
      const size_t input_stride = butterfly_batch * section_count;
      for (size_t section = 0; section < section_count; ++section)
        MOFFT_DISPATCH_OTHER_FP64(
            plan,
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
    MOFFT_DISPATCH_FIRST_TRANSPOSED_FP64(
        plan, first_radix, plan->direction, input, first_batch, spare,
        first_batch);
  } else {
    MOFFT_DISPATCH_FIRST_FP64(plan, first_radix, plan->direction, input,
                              first_batch, current, first_batch);
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
    MOFFT_DISPATCH_OTHER_FP64(plan, radix, plan->direction, stage_input,
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
  if (!mofft_hardware_supports(32)) return MOFFT_UNSUPPORTED_HARDWARE;
  if (plan->depth == 1 && input == output) {
    if (!plan->scratch) return MOFFT_INTERNAL_ERROR;
    memcpy(plan->scratch, input, plan->length * sizeof(*plan->scratch));
    mofft_streaming_entry_f32(plan, plan->scratch, output);
  } else {
    mofft_streaming_entry_f32(plan, input, output);
  }
  return MOFFT_SUCCESS;
}

mofft_status mofft_execute_f64(const mofft_plan_f64 *plan,
                               const mofft_complex_f64 *input,
                               mofft_complex_f64 *output) {
  if (!plan || !input || !output) return MOFFT_INVALID_ARGUMENT;
  if ((plan->placement == MOFFT_IN_PLACE) != (input == output))
    return MOFFT_INVALID_ARGUMENT;
  if (!mofft_hardware_supports(64)) return MOFFT_UNSUPPORTED_HARDWARE;
  if (plan->depth == 1 && input == output) {
    if (!plan->scratch) return MOFFT_INTERNAL_ERROR;
    memcpy(plan->scratch, input, plan->length * sizeof(*plan->scratch));
    mofft_streaming_entry_f64(plan, plan->scratch, output);
  } else {
    mofft_streaming_entry_f64(plan, input, output);
  }
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
    case MOFFT_WISDOM_NOT_FOUND: return "compiled wisdom not found";
    case MOFFT_INCOMPATIBLE_RECIPE: return "incompatible plan recipe";
    case MOFFT_BUFFER_TOO_SMALL: return "output buffer too small";
    default: return "unknown status";
  }
}

const char *mofft_version(void) { return "0.4.0"; }
