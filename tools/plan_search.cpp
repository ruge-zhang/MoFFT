#include "mofft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/sysctl.h>
#include <tuple>
#include <vector>
#include <pthread/qos.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifndef MOFFT_MANIFEST_PATH
#define MOFFT_MANIFEST_PATH ""
#endif

using Clock = std::chrono::steady_clock;
static constexpr int radices[] = {64, 32, 16, 15, 14, 13, 12, 11, 10,
                                  9, 8, 7, 6, 5, 4, 3, 2};

struct Candidate {
  std::vector<int> factors;
  uint64_t blocked_transpose_mask = 0;
  bool direct_input = true;
  bool section_layout = false;
  double model_cost = 0.0;
  mofft_plan_cost_breakdown model_breakdown{};
  std::vector<double> samples;
  std::vector<double> relative_samples;
  bool finalist = false;
  bool steady_finalist = false;
};

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() & 1 ? values[middle]
                           : .5 * (values[middle - 1] + values[middle]);
}

static std::string sysstr(const char *key) {
  size_t size = 0;
  if (sysctlbyname(key, nullptr, &size, nullptr, 0)) return "unknown";
  std::string value(size, '\0');
  if (sysctlbyname(key, value.data(), &size, nullptr, 0)) return "unknown";
  if (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

static long long sysint(const char *key) {
  int value = -1;
  size_t size = sizeof(value);
  return sysctlbyname(key, &value, &size, nullptr, 0) ? -1 : value;
}

static std::string json_string_field(const std::string &text,
                                     const std::string &field,
                                     size_t start = 0) {
  const std::string needle = "\"" + field + "\"";
  size_t position = text.find(needle, start);
  if (position == std::string::npos) return "unknown";
  position = text.find(':', position + needle.size());
  position = text.find('"', position + 1);
  if (position == std::string::npos) return "unknown";
  size_t end = text.find('"', position + 1);
  return end == std::string::npos ? "unknown"
                                  : text.substr(position + 1, end - position - 1);
}

static std::string json_escape(const std::string &value) {
  std::string result;
  for (char c : value) {
    if (c == '\\' || c == '"') result.push_back('\\');
    result.push_back(c);
  }
  return result;
}

static std::string factor_key(const std::vector<int> &factors) {
  std::ostringstream out;
  for (int factor : factors) out << factor << ',';
  return out.str();
}

static std::vector<Candidate> model_candidates(size_t length,
                                                int precision_bits,
                                                size_t limit) {
  std::map<size_t, std::vector<Candidate>> memo;
  memo[1] = {Candidate{}};
  std::function<const std::vector<Candidate> &(size_t)> solve =
      [&](size_t n) -> const std::vector<Candidate> & {
    auto found = memo.find(n);
    if (found != memo.end()) return found->second;
    std::vector<Candidate> choices;
    for (int radix : radices) {
      if (n % (size_t)radix != 0) continue;
      size_t remainder = n / (size_t)radix;
      const auto &suffixes = solve(remainder);
      double stage = mofft_estimated_stage_cost(
          radix, precision_bits, remainder == 1, length, n);
      for (const auto &suffix : suffixes) {
        Candidate candidate;
        candidate.factors.push_back(radix);
        candidate.factors.insert(candidate.factors.end(),
                                 suffix.factors.begin(),
                                 suffix.factors.end());
        candidate.model_cost = stage + suffix.model_cost;
        choices.push_back(std::move(candidate));
      }
    }
    std::sort(choices.begin(), choices.end(), [](const Candidate &a,
                                                 const Candidate &b) {
      if (a.model_cost != b.model_cost) return a.model_cost < b.model_cost;
      return a.factors < b.factors;
    });
    choices.erase(std::unique(choices.begin(), choices.end(),
                              [](const Candidate &a, const Candidate &b) {
                                return a.factors == b.factors;
                              }), choices.end());
    if (choices.size() > limit) {
      std::vector<Candidate> retained;
      std::set<std::string> retained_keys;
      std::set<std::tuple<size_t, int, int>> shapes;
      const size_t model_prefix = std::max<size_t>(1, limit / 2);
      auto retain = [&](const Candidate &candidate) {
        if (retained.size() >= limit) return;
        std::string key = factor_key(candidate.factors);
        if (!retained_keys.insert(key).second) return;
        shapes.emplace(candidate.factors.size(), candidate.factors.front(),
                       candidate.factors.back());
        retained.push_back(candidate);
      };
      /* Preserve the best representative of every stage depth before the
       * fixed-width model prefix. A slightly optimistic cache/bandwidth term
       * must not remove an entire shallow-plan family such as 32x64. */
      std::set<size_t> depths;
      for (const auto &candidate : choices)
        if (depths.insert(candidate.factors.size()).second) retain(candidate);
      for (size_t i = 0; i < model_prefix; ++i) retain(choices[i]);
      for (const auto &candidate : choices) {
        auto shape = std::make_tuple(candidate.factors.size(),
                                     candidate.factors.front(),
                                     candidate.factors.back());
        if (shapes.count(shape) == 0) retain(candidate);
        if (retained.size() == limit) break;
      }
      for (const auto &candidate : choices) {
        if (retained.size() == limit) break;
        retain(candidate);
      }
      std::sort(retained.begin(), retained.end(), [](const Candidate &a,
                                                     const Candidate &b) {
        if (a.model_cost != b.model_cost) return a.model_cost < b.model_cost;
        return a.factors < b.factors;
      });
      choices = std::move(retained);
    }
    return memo.emplace(n, std::move(choices)).first->second;
  };
  return solve(length);
}

static std::set<uint64_t> transpose_masks(size_t transpose_count) {
  std::set<uint64_t> masks{0};
  if (transpose_count < 2) return masks;
  const uint64_t all = (UINT64_C(1) << transpose_count) - 1;
  masks.insert(all);
  uint64_t even = 0, odd = 0;
  for (size_t level = 0; level < transpose_count; ++level)
    (level & 1 ? odd : even) |= UINT64_C(1) << level;
  masks.insert(even);
  masks.insert(odd);
  // Per-bit single-stage blocking is a refinement that adds two masks per
  // stage. Deep plans use the representative all/none/alternating families
  // to keep the device measurement budget bounded.
  if (transpose_count <= 3) {
    for (size_t level = 0; level < transpose_count; ++level) {
      const uint64_t bit = UINT64_C(1) << level;
      masks.insert(bit);
      masks.insert(all ^ bit);
    }
  }
  return masks;
}

static std::vector<std::tuple<uint64_t, bool, bool>> plan_variants(
    const std::vector<int> &factors, int precision_bits) {
  const size_t transpose_count = factors.size() - 1;
  std::set<std::tuple<uint64_t, bool, bool>> variants;
  variants.emplace(0, false, true);
  for (uint64_t mask : transpose_masks(transpose_count)) {
    for (bool direct_input : {false, true}) {
      uint64_t effective_mask = mask;
      if (direct_input) {
        size_t parent_batch = 1;
        const size_t lanes = precision_bits == 32 ? 16 : 8;
        for (size_t level = 0; level < transpose_count; ++level) {
          if (parent_batch >= lanes / 2)
            effective_mask &= ~(UINT64_C(1) << level);
          parent_batch *= (size_t)factors[level];
        }
      }
      variants.emplace(effective_mask, direct_input, false);
    }
  }
  return {variants.begin(), variants.end()};
}

static double best_plan_model_cost(size_t length, int precision_bits,
                                   const std::vector<int> &factors) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto &[mask, direct_input, section_layout] :
       plan_variants(factors, precision_bits))
    best = std::min(best, mofft_estimated_plan_cost_with_layout(
                              length, precision_bits, factors.data(),
                              factors.size(), mask, direct_input,
                              section_layout ? MOFFT_STAGE_LAYOUT_SECTION
                                             : MOFFT_STAGE_LAYOUT_TRANSPOSE));
  return best;
}

static std::vector<Candidate> select_factorizations(
    size_t length, int precision_bits, size_t budget,
    std::vector<Candidate> pool, const std::set<std::string> &anchor_keys) {
  for (Candidate &candidate : pool)
    candidate.model_cost = best_plan_model_cost(
        length, precision_bits, candidate.factors);
  std::sort(pool.begin(), pool.end(), [](const Candidate &a,
                                         const Candidate &b) {
    if (a.model_cost != b.model_cost) return a.model_cost < b.model_cost;
    return a.factors < b.factors;
  });
  pool.erase(std::unique(pool.begin(), pool.end(),
                         [](const Candidate &a, const Candidate &b) {
                           return a.factors == b.factors;
                         }),
             pool.end());

  std::vector<Candidate> selected;
  std::set<std::string> selected_keys;
  auto retain = [&](const Candidate &candidate) {
    if (selected.size() >= budget) return;
    const std::string key = factor_key(candidate.factors);
    if (selected_keys.insert(key).second) selected.push_back(candidate);
  };
  // Structural anchors and the best member of every stage depth protect the
  // measurement shortlist from inevitable model error. The remaining slots
  // are selected by the complete calibrated plan cost, including cache tier,
  // transpose efficiency, and cache-line overfetch.
  for (const Candidate &candidate : pool)
    if (anchor_keys.count(factor_key(candidate.factors))) retain(candidate);
  std::set<size_t> depths;
  for (const Candidate &candidate : pool)
    if (depths.insert(candidate.factors.size()).second) retain(candidate);
  for (const Candidate &candidate : pool) retain(candidate);
  std::sort(selected.begin(), selected.end(), [](const Candidate &a,
                                                  const Candidate &b) {
    if (a.model_cost != b.model_cost) return a.model_cost < b.model_cost;
    return a.factors < b.factors;
  });
  return selected;
}

static double score_factorization(size_t length, int precision_bits,
                                  const std::vector<int> &factors) {
  size_t n = length;
  double cost = 0.0;
  for (int radix : factors) {
    if (radix < 2 || n % (size_t)radix != 0)
      return std::numeric_limits<double>::infinity();
    const size_t remainder = n / (size_t)radix;
    cost += mofft_estimated_stage_cost(
        radix, precision_bits, remainder == 1, length, n);
    n = remainder;
  }
  return n == 1 ? cost : std::numeric_limits<double>::infinity();
}

static std::vector<Candidate> power_of_two_structure_anchors(
    size_t length, int precision_bits) {
  if (length == 0 || (length & (length - 1)) != 0) return {};
  static constexpr int first_radices[] = {64, 32, 16, 8, 4, 2};
  static constexpr int middle_radices[] = {16, 8, 4, 2};
  std::vector<Candidate> result;
  std::set<std::string> keys;
  /* Retain one shallow, regular execution-order family for each possible
   * first-stage radix. These candidates share the normal search budget: the
   * safeguard prevents an imperfect locality model from deleting an entire
   * family without increasing the iOS measurement workload. */
  for (int first_radix : first_radices) {
    if (length % (size_t)first_radix != 0) continue;
    size_t remaining = length / (size_t)first_radix;
    std::vector<int> execution_order{first_radix};
    while (remaining > 1) {
      int selected = 0;
      for (int radix : middle_radices) {
        if (remaining % (size_t)radix == 0) {
          selected = radix;
          break;
        }
      }
      if (selected == 0) break;
      execution_order.push_back(selected);
      remaining /= (size_t)selected;
    }
    if (remaining != 1) continue;
    Candidate candidate;
    candidate.factors.assign(execution_order.rbegin(),
                             execution_order.rend());
    if (!keys.insert(factor_key(candidate.factors)).second) continue;
    candidate.model_cost = score_factorization(
        length, precision_bits, candidate.factors);
    result.push_back(std::move(candidate));
  }
  return result;
}

template <class Complex> struct Api;
template <> struct Api<mofft_complex_f32> {
  using Plan = mofft_plan_f32;
  static mofft_status create(Plan **plan, size_t n,
                             const std::vector<int> &factors) {
    return mofft_plan_create_with_radices_f32(
        plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, factors.data(),
        factors.size());
  }
  static mofft_status run(Plan *plan, const mofft_complex_f32 *input,
                          mofft_complex_f32 *output) {
    return mofft_execute_f32(plan, input, output);
  }
  static mofft_status set_transpose(Plan *plan, uint64_t mask) {
    return mofft_plan_set_blocked_transpose_mask_f32(plan, mask);
  }
  static mofft_status set_direct_input(Plan *plan, bool enabled) {
    return mofft_plan_set_direct_input_f32(plan, enabled ? 1 : 0);
  }
  static mofft_status set_stage_layout(Plan *plan, bool section) {
    return mofft_plan_set_stage_layout_f32(
        plan, section ? MOFFT_STAGE_LAYOUT_SECTION
                      : MOFFT_STAGE_LAYOUT_TRANSPOSE);
  }
  static void destroy(Plan *plan) { mofft_plan_destroy_f32(plan); }
};
template <> struct Api<mofft_complex_f64> {
  using Plan = mofft_plan_f64;
  static mofft_status create(Plan **plan, size_t n,
                             const std::vector<int> &factors) {
    return mofft_plan_create_with_radices_f64(
        plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE, factors.data(),
        factors.size());
  }
  static mofft_status run(Plan *plan, const mofft_complex_f64 *input,
                          mofft_complex_f64 *output) {
    return mofft_execute_f64(plan, input, output);
  }
  static mofft_status set_transpose(Plan *plan, uint64_t mask) {
    return mofft_plan_set_blocked_transpose_mask_f64(plan, mask);
  }
  static mofft_status set_direct_input(Plan *plan, bool enabled) {
    return mofft_plan_set_direct_input_f64(plan, enabled ? 1 : 0);
  }
  static mofft_status set_stage_layout(Plan *plan, bool section) {
    return mofft_plan_set_stage_layout_f64(
        plan, section ? MOFFT_STAGE_LAYOUT_SECTION
                      : MOFFT_STAGE_LAYOUT_TRANSPOSE);
  }
  static void destroy(Plan *plan) { mofft_plan_destroy_f64(plan); }
};

template <class Complex>
static void measure_finalists(size_t length, int sample_count,
                              size_t warmup_executions,
                              std::vector<Candidate *> finalists,
                              bool steady_state) {
  using Plan = typename Api<Complex>::Plan;
  std::vector<Plan *> plans(finalists.size(), nullptr);
  std::vector<size_t> repetitions(finalists.size(), 1);
  std::vector<Complex> input(length), output(length);
  for (size_t i = 0; i < length; ++i) {
    input[i].real = std::sin(.017 * (double)i);
    input[i].imag = std::cos(.013 * (double)i);
  }
  for (size_t index = 0; index < finalists.size(); ++index) {
    mofft_status status = Api<Complex>::create(
        &plans[index], length, finalists[index]->factors);
    if (status != MOFFT_SUCCESS)
      throw std::runtime_error("finalist plan creation failed");
    status = Api<Complex>::set_transpose(
        plans[index], finalists[index]->blocked_transpose_mask);
    if (status == MOFFT_SUCCESS)
      status = Api<Complex>::set_direct_input(
          plans[index], finalists[index]->direct_input);
    if (status == MOFFT_SUCCESS)
      status = Api<Complex>::set_stage_layout(
          plans[index], finalists[index]->section_layout);
    if (status != MOFFT_SUCCESS)
      throw std::runtime_error("finalist transpose strategy failed");
    finalists[index]->samples.clear();
    finalists[index]->relative_samples.clear();
    finalists[index]->finalist = true;
    finalists[index]->steady_finalist = steady_state;
    for (size_t warmup = 0; warmup < warmup_executions; ++warmup)
      Api<Complex>::run(plans[index], input.data(), output.data());
    auto start = Clock::now();
    Api<Complex>::run(plans[index], input.data(), output.data());
    auto stop = Clock::now();
    double pilot = std::chrono::duration<double>(stop - start).count();
    repetitions[index] = std::min<size_t>(1000000, std::max<size_t>(
        1, (size_t)std::ceil(.002 / pilot)));
  }
  auto measure = [&](size_t index) {
    auto start = Clock::now();
    for (size_t repetition = 0; repetition < repetitions[index];
         ++repetition)
      Api<Complex>::run(plans[index], input.data(), output.data());
    auto stop = Clock::now();
    return std::chrono::duration<double>(stop - start).count() /
           repetitions[index];
  };
  for (int sample = 0; sample < sample_count; ++sample) {
    for (size_t position = 0; position < finalists.size(); ++position) {
      size_t index = (size_t(sample) + position) % finalists.size();
      if (sample & 1) index = finalists.size() - 1 - index;
      if (index == 0) {
        finalists[index]->samples.push_back(measure(index));
        finalists[index]->relative_samples.push_back(1.0);
      } else {
        const double anchor_before = measure(0);
        const double candidate = measure(index);
        const double anchor_after = measure(0);
        finalists[index]->samples.push_back(candidate);
        finalists[index]->relative_samples.push_back(
            candidate / (.5 * (anchor_before + anchor_after)));
      }
    }
  }
  for (Plan *plan : plans) Api<Complex>::destroy(plan);
}

template <class Complex>
static void measure_coarse_candidates(size_t length,
                                      std::vector<Candidate> &candidates) {
  using Plan = typename Api<Complex>::Plan;
  std::vector<size_t> repetitions(candidates.size(), 0);
  std::vector<Complex> input(length), output(length);
  for (size_t i = 0; i < length; ++i) {
    input[i].real = std::sin(.017 * (double)i);
    input[i].imag = std::cos(.013 * (double)i);
  }
  for (int sample = 0; sample < 3; ++sample) {
    for (size_t position = 0; position < candidates.size(); ++position) {
      size_t index = (size_t(sample) + position) % candidates.size();
      if (sample & 1) index = candidates.size() - 1 - index;
      Candidate &candidate = candidates[index];
      Plan *plan = nullptr;
      mofft_status status = Api<Complex>::create(
          &plan, length, candidate.factors);
      if (status == MOFFT_SUCCESS)
        status = Api<Complex>::set_transpose(
            plan, candidate.blocked_transpose_mask);
      if (status == MOFFT_SUCCESS)
        status = Api<Complex>::set_direct_input(plan, candidate.direct_input);
      if (status == MOFFT_SUCCESS)
        status = Api<Complex>::set_stage_layout(plan,
                                                candidate.section_layout);
      if (status != MOFFT_SUCCESS) {
        Api<Complex>::destroy(plan);
        throw std::runtime_error("coarse candidate plan creation failed");
      }
      for (int warmup = 0; warmup < 2; ++warmup)
        Api<Complex>::run(plan, input.data(), output.data());
      if (repetitions[index] == 0) {
        auto start = Clock::now();
        Api<Complex>::run(plan, input.data(), output.data());
        auto stop = Clock::now();
        double pilot = std::chrono::duration<double>(stop - start).count();
        repetitions[index] = std::min<size_t>(1000000, std::max<size_t>(
            1, (size_t)std::ceil(.001 / pilot)));
      }
      auto start = Clock::now();
      for (size_t repetition = 0; repetition < repetitions[index];
           ++repetition)
        Api<Complex>::run(plan, input.data(), output.data());
      auto stop = Clock::now();
      candidate.samples.push_back(
          std::chrono::duration<double>(stop - start).count() /
          repetitions[index]);
      Api<Complex>::destroy(plan);
    }
  }
}

struct SearchResult {
  size_t length;
  int precision_bits;
  std::vector<Candidate> candidates;
};

template <class Complex>
static SearchResult search(size_t length, int precision_bits,
                           size_t candidate_limit, int samples,
                           size_t steady_finalist_limit,
                           size_t steady_warmup_executions) {
  const size_t layout_limit = std::max<size_t>(2, candidate_limit / 2);
  const size_t modeled_budget = candidate_limit + layout_limit;
  const size_t pool_limit = modeled_budget > SIZE_MAX / 4
                                ? modeled_budget
                                : modeled_budget * 4;
  std::vector<Candidate> modeled = model_candidates(
      length, precision_bits, pool_limit);
  std::set<std::string> modeled_keys;
  for (const Candidate &candidate : modeled)
    modeled_keys.insert(factor_key(candidate.factors));
  std::set<std::string> anchor_keys;
  for (Candidate &candidate : power_of_two_structure_anchors(
           length, precision_bits)) {
    const std::string key = factor_key(candidate.factors);
    anchor_keys.insert(key);
    if (modeled_keys.insert(key).second) modeled.push_back(std::move(candidate));
  }
  modeled = select_factorizations(length, precision_bits, modeled_budget,
                                  std::move(modeled), anchor_keys);
  std::vector<Candidate> expanded;
  for (const Candidate &candidate : modeled) {
    for (const auto &[effective_mask, direct_input, section_layout] :
         plan_variants(candidate.factors, precision_bits)) {
      Candidate variant = candidate;
      variant.blocked_transpose_mask = effective_mask;
      variant.direct_input = direct_input;
      variant.section_layout = section_layout;
      const mofft_status model_status =
          mofft_estimated_plan_cost_breakdown_with_layout(
              length, precision_bits, variant.factors.data(),
              variant.factors.size(), effective_mask, direct_input,
              section_layout ? MOFFT_STAGE_LAYOUT_SECTION
                             : MOFFT_STAGE_LAYOUT_TRANSPOSE,
              &variant.model_breakdown);
      variant.model_cost = model_status == MOFFT_SUCCESS
                               ? variant.model_breakdown.total_cost
                               : std::numeric_limits<double>::infinity();
      expanded.push_back(std::move(variant));
    }
  }
  SearchResult result{length, precision_bits, std::move(expanded)};
  if (result.candidates.empty())
    throw std::runtime_error("unsupported transform length");
  measure_coarse_candidates<Complex>(length, result.candidates);
  std::sort(result.candidates.begin(), result.candidates.end(),
            [](const Candidate &a, const Candidate &b) {
    double am = median(a.samples), bm = median(b.samples);
    if (am != bm) return am < bm;
    if (a.factors != b.factors) return a.factors < b.factors;
    if (a.blocked_transpose_mask != b.blocked_transpose_mask)
      return a.blocked_transpose_mask < b.blocked_transpose_mask;
    if (a.direct_input != b.direct_input)
      return a.direct_input < b.direct_input;
    return a.section_layout < b.section_layout;
  });
  /*
   * A short coarse pass is deliberately cheap, but it is also the part most
   * vulnerable to unrelated macOS activity.  Keep a wider refinement pool so
   * one disturbed sample cannot discard a materially different radix order.
   * The final pass is still interleaved and is the only timing used to choose
   * the winner.
  */
  std::vector<Candidate *> finalists;
  std::set<std::string> finalist_factorizations;
  for (size_t i = 0; i < std::min<size_t>(16, result.candidates.size()); ++i) {
    finalists.push_back(&result.candidates[i]);
    finalist_factorizations.insert(factor_key(result.candidates[i].factors));
  }
  /* Radix order changes which later stages can use direct input.  Preserve the
   * best coarse variant of every modeled order even when a load spike moves
   * all of its variants below the fixed-width coarse cutoff. */
  for (Candidate &candidate : result.candidates) {
    if (finalist_factorizations.insert(factor_key(candidate.factors)).second)
      finalists.push_back(&candidate);
  }
  measure_finalists<Complex>(length, samples, 3, std::move(finalists), false);
  std::sort(result.candidates.begin(), result.candidates.end(),
            [](const Candidate &a, const Candidate &b) {
    if (a.finalist != b.finalist) return a.finalist > b.finalist;
    double am = median(a.finalist ? a.relative_samples : a.samples);
    double bm = median(b.finalist ? b.relative_samples : b.samples);
    if (am != bm) return am < bm;
    if (a.factors != b.factors) return a.factors < b.factors;
    if (a.blocked_transpose_mask != b.blocked_transpose_mask)
      return a.blocked_transpose_mask < b.blocked_transpose_mask;
    if (a.direct_input != b.direct_input)
      return a.direct_input < b.direct_input;
    return a.section_layout < b.section_layout;
  });
  /* Short candidate timings identify a broad competitive region. Re-measure
   * the best distinct radix chains after the same execution-count warmup used
   * by the runtime benchmark. This prevents streaming-mode and code-state
   * transients from choosing a plan that is only a short-burst winner. */
  std::vector<Candidate *> steady_finalists;
  std::set<std::string> steady_factorizations;
  for (Candidate &candidate : result.candidates) {
    if (!candidate.finalist) continue;
    if (!steady_factorizations.insert(factor_key(candidate.factors)).second)
      continue;
    steady_finalists.push_back(&candidate);
    if (steady_finalists.size() == steady_finalist_limit) break;
  }
  measure_finalists<Complex>(length, samples, steady_warmup_executions,
                              std::move(steady_finalists), true);
  std::sort(result.candidates.begin(), result.candidates.end(),
            [](const Candidate &a, const Candidate &b) {
    if (a.steady_finalist != b.steady_finalist)
      return a.steady_finalist > b.steady_finalist;
    if (a.finalist != b.finalist) return a.finalist > b.finalist;
    double am = median(a.finalist ? a.relative_samples : a.samples);
    double bm = median(b.finalist ? b.relative_samples : b.samples);
    if (am != bm) return am < bm;
    if (a.factors != b.factors) return a.factors < b.factors;
    if (a.blocked_transpose_mask != b.blocked_transpose_mask)
      return a.blocked_transpose_mask < b.blocked_transpose_mask;
    if (a.direct_input != b.direct_input)
      return a.direct_input < b.direct_input;
    return a.section_layout < b.section_layout;
  });
  return result;
}

static std::vector<size_t> parse_sizes(const std::string &text) {
  std::vector<size_t> sizes;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t value = (size_t)std::stoull(item);
    if (value < 2) throw std::runtime_error("length must be at least 2");
    sizes.push_back(value);
  }
  if (sizes.empty()) throw std::runtime_error("no lengths specified");
  return sizes;
}

static std::string default_manifest_path() {
  if (std::strlen(MOFFT_MANIFEST_PATH)) return MOFFT_MANIFEST_PATH;
#ifdef __APPLE__
  char executable[4096];
  uint32_t size = sizeof(executable);
  if (_NSGetExecutablePath(executable, &size) == 0) {
    std::string path(executable);
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos)
      return path.substr(0, slash + 1) + "manifest.json";
  }
#endif
  return "";
}

int main(int argc, char **argv) {
  std::string size_text, precision = "both", output = "mofft-wisdom.json";
  std::string manifest_path = default_manifest_path();
  size_t candidate_limit = 32;
  size_t steady_finalist_limit = 4;
  size_t steady_warmup_executions = 1000;
  int samples = 7;
  bool formal = false, confirmed = false, compact = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc)
      size_text = argv[++i];
    else if (!std::strcmp(argv[i], "--precision") && i + 1 < argc)
      precision = argv[++i];
    else if (!std::strcmp(argv[i], "--candidates") && i + 1 < argc)
      candidate_limit = (size_t)std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--samples") && i + 1 < argc)
      samples = std::stoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--steady-finalists") && i + 1 < argc)
      steady_finalist_limit = (size_t)std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--steady-warmup") && i + 1 < argc)
      steady_warmup_executions = (size_t)std::stoull(argv[++i]);
    else if (!std::strcmp(argv[i], "--output") && i + 1 < argc)
      output = argv[++i];
    else if (!std::strcmp(argv[i], "--manifest") && i + 1 < argc)
      manifest_path = argv[++i];
    else if (!std::strcmp(argv[i], "--formal")) formal = true;
    else if (!std::strcmp(argv[i], "--exclusive-confirmed")) confirmed = true;
    else if (!std::strcmp(argv[i], "--compact")) compact = true;
    else {
      std::cerr << "usage: mofft-plan-search --sizes N[,N...] "
                   "[--precision fp32|fp64|both] [--candidates K] "
                   "[--samples S] [--steady-finalists K] "
                   "[--steady-warmup N] [--output FILE] [--formal "
                   "--exclusive-confirmed] [--compact]\n";
      return 2;
    }
  }
  if (size_text.empty() || candidate_limit == 0 || steady_finalist_limit == 0 ||
      steady_warmup_executions == 0 || samples < 3 ||
      (precision != "fp32" && precision != "fp64" && precision != "both")) {
    std::cerr << "invalid search arguments\n";
    return 2;
  }
  if (formal && !confirmed) {
    std::cerr << "Refusing formal plan search without an explicitly confirmed "
                 "exclusive window.\n";
    return 2;
  }
  if (formal && samples < 15) samples = 15;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  double load[3] = {};
  getloadavg(load, 3);
  if (formal && load[0] > 2.0) {
    std::cerr << "Refusing formal plan search: one-minute load average is "
              << load[0] << ".\n";
    return 2;
  }
  if (formal && sysint("kern.thermal_level") > 0) {
    std::cerr << "Refusing formal plan search because macOS reports thermal "
                 "pressure.\n";
    return 2;
  }
  std::vector<SearchResult> results;
  try {
    for (size_t length : parse_sizes(size_text)) {
      if (precision == "fp32" || precision == "both")
        results.push_back(search<mofft_complex_f32>(
            length, 32, candidate_limit, samples, steady_finalist_limit,
            steady_warmup_executions));
      if (precision == "fp64" || precision == "both")
        results.push_back(search<mofft_complex_f64>(
            length, 64, candidate_limit, samples, steady_finalist_limit,
            steady_warmup_executions));
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  double ending_load[3] = {};
  getloadavg(ending_load, 3);
  // A long single-threaded search contributes to macOS's load average itself.
  // Record the ending value for provenance; reject interference using the
  // pre-run load, thermal state, and measured winner CV.
  if (formal) {
    for (const auto &result : results) {
      const auto &best = result.candidates.front();
      const auto &samples_for_best =
          best.steady_finalist ? best.relative_samples : best.samples;
      double mean = std::accumulate(samples_for_best.begin(),
                                    samples_for_best.end(), 0.0) /
                    samples_for_best.size();
      double squared = 0.0;
      for (double value : samples_for_best)
        squared += (value - mean) * (value - mean);
      double cv = std::sqrt(squared / samples_for_best.size()) / mean;
      if (cv > .05) {
        std::cerr << "Discarding formal plan search: winner CV is " << cv
                  << " for fp" << result.precision_bits << " n="
                  << result.length << ".\n";
        return 2;
      }
    }
  }
  std::string manifest_text;
  if (!manifest_path.empty()) {
    std::ifstream manifest(manifest_path);
    manifest_text.assign(std::istreambuf_iterator<char>(manifest),
                         std::istreambuf_iterator<char>());
  }
  size_t profile_position = manifest_text.find("\"profile\"");
  std::string profile_name = json_string_field(
      manifest_text, "name", profile_position == std::string::npos
                                  ? 0 : profile_position);
  std::ostringstream out;
  out << std::setprecision(12)
      << "{\n  \"schema_version\":1,\n  \"formal_measurement\":"
      << (formal ? "true" : "false") << ",\n  \"machine\":\""
      << json_escape(sysstr("hw.model")) << "\",\n  \"generator_version\":\""
      << mofft_version() << "\",\n  \"profile\":\""
      << json_escape(profile_name) << "\",\n  \"kernel_input_sha256\":\""
      << json_escape(json_string_field(manifest_text, "input_sha256"))
      << "\",\n  \"model_cost_unit\":\""
      << json_escape(json_string_field(manifest_text, "cost_unit"))
      << "\",\n  \"starting_load_average\":" << load[0]
      << ",\n  \"ending_load_average\":" << ending_load[0]
      << ",\n  \"steady_finalists\":" << steady_finalist_limit
      << ",\n  \"steady_warmup_executions\":" << steady_warmup_executions
      << ",\n  \"entries\":[\n";
  for (size_t entry = 0; entry < results.size(); ++entry) {
    const auto &result = results[entry];
    const auto &best = result.candidates.front();
    out << "    {\"length\":" << result.length << ",\"precision\":\"fp"
        << result.precision_bits << "\",\"radices\":[";
    for (size_t i = 0; i < best.factors.size(); ++i) {
      if (i) out << ',';
      out << best.factors[i];
    }
    out << "],\"blocked_transpose_mask\":"
        << best.blocked_transpose_mask
        << ",\"direct_input\":" << (best.direct_input ? "true" : "false")
        << ",\"stage_layout\":\""
        << (best.section_layout ? "section" : "transpose") << "\""
        << ",\"transpose_strategy\":\""
        << (best.blocked_transpose_mask == 0 ? "linear" : "mixed")
        << "\",\"median_seconds\":" << median(best.samples)
        << ",\"model_cost\":" << best.model_cost
        << ",\"model_kernel_cost\":" << best.model_breakdown.kernel_cost
        << ",\"model_memory_cost\":" << best.model_breakdown.memory_cost
        << ",\"model_layout_cost\":" << best.model_breakdown.layout_cost
        << ",\"model_logical_read_bytes\":"
        << best.model_breakdown.logical_read_bytes
        << ",\"model_logical_write_bytes\":"
        << best.model_breakdown.logical_write_bytes
        << ",\"model_transferred_read_bytes\":"
        << best.model_breakdown.transferred_read_bytes
        << ",\"model_transferred_write_bytes\":"
        << best.model_breakdown.transferred_write_bytes
        << ",\"model_peak_working_set_bytes\":"
        << best.model_breakdown.peak_working_set_bytes;
    if (!compact) out << ",\"candidates\":[";
    for (size_t i = 0; !compact && i < result.candidates.size(); ++i) {
      if (i) out << ',';
      const auto &candidate = result.candidates[i];
      out << "{\"radices\":[";
      for (size_t j = 0; j < candidate.factors.size(); ++j) {
        if (j) out << ',';
        out << candidate.factors[j];
      }
      out << "],\"blocked_transpose_mask\":"
          << candidate.blocked_transpose_mask
          << ",\"direct_input\":"
          << (candidate.direct_input ? "true" : "false")
          << ",\"stage_layout\":\""
          << (candidate.section_layout ? "section" : "transpose") << "\""
          << ",\"transpose_strategy\":\""
          << (candidate.blocked_transpose_mask == 0 ? "linear" : "mixed")
          << "\",\"model_cost\":" << candidate.model_cost
          << ",\"model_kernel_cost\":"
          << candidate.model_breakdown.kernel_cost
          << ",\"model_memory_cost\":"
          << candidate.model_breakdown.memory_cost
          << ",\"model_layout_cost\":"
          << candidate.model_breakdown.layout_cost
          << ",\"model_logical_read_bytes\":"
          << candidate.model_breakdown.logical_read_bytes
          << ",\"model_logical_write_bytes\":"
          << candidate.model_breakdown.logical_write_bytes
          << ",\"model_transferred_read_bytes\":"
          << candidate.model_breakdown.transferred_read_bytes
          << ",\"model_transferred_write_bytes\":"
          << candidate.model_breakdown.transferred_write_bytes
          << ",\"model_peak_working_set_bytes\":"
          << candidate.model_breakdown.peak_working_set_bytes
          << ",\"finalist\":" << (candidate.finalist ? "true" : "false")
          << ",\"steady_finalist\":"
          << (candidate.steady_finalist ? "true" : "false")
          << ",\"median_seconds\":" << median(candidate.samples)
          << ",\"median_relative_to_anchor\":"
          << (candidate.finalist ? median(candidate.relative_samples) : 0.0)
          << ",\"raw_seconds\":[";
      for (size_t j = 0; j < candidate.samples.size(); ++j) {
        if (j) out << ',';
        out << candidate.samples[j];
      }
      out << "]}";
    }
    if (!compact) out << ']';
    out << "}" << (entry + 1 == results.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";
  // Dump the wisdom JSON to stdout so a remote iOS run (captured by devicectl
  // --console) can retrieve it without reading a sandboxed output file.
  std::cout << "WISDOM_JSON_BEGIN\n" << out.str() << "\nWISDOM_JSON_END\n";
  { std::ofstream file(output); file << out.str(); }
  std::cout << "wrote " << output << " ("
            << (formal ? "formal" : "nonformal") << ")\n";
  return 0;
}
