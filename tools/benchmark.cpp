#include "mofft.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <pthread/qos.h>
#include <random>
#include <sstream>
#include <string>
#include <sys/sysctl.h>
#include <thread>
#include <vector>
#ifdef MOFFT_HAVE_FFTW
#include <fftw3.h>
#endif
#ifndef MOFFT_COMPILER_INFO
#define MOFFT_COMPILER_INFO "unknown"
#endif

using clock_type = std::chrono::steady_clock;
static const size_t extended_sizes[] = {
    256,    512,    1024,  2048,  4096,  8192,   16384,  32768,  65536,
    131072, 262144, 144,   169,   196,   225,    1728,   2197,   2744,
    3375,   20736,  28561, 38416, 50625, 248832, 371293, 537824, 759375};

// This is the non-power-of-two and power-of-two suite used by the public
// overall-performance figure.  The tiny N=144 point is intentionally not
// part of that figure because its fixed streaming-entry overhead dominates
// the transform itself.
static const size_t overall_sizes[] = {
    256,    512,    1024,   2048,   4096,   8192,   16384, 32768,
    65536,  131072, 262144, 169,    196,    225,    1728,  2197,
    2744,   3375,   20736,  28561,  38416,  50625, 248832, 371293,
    537824, 759375};

static std::vector<size_t> parse_sizes(const std::string &text) {
  std::vector<size_t> result;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t value = (size_t)std::stoull(item);
    if (value < 2)
      throw std::runtime_error("benchmark size must be at least 2");
    result.push_back(value);
  }
  if (result.empty())
    throw std::runtime_error("empty benchmark size list");
  return result;
}

struct Result {
  std::string precision, implementation;
  size_t n, repetitions;
  std::vector<double> seconds;
  size_t warmup_executions = 0;
  double warmup_seconds = 0.0;
};
static constexpr double target_batch_seconds = .002;
static constexpr double target_warmup_seconds = .250;
static constexpr size_t minimum_warmup_executions = 1000;
static constexpr size_t maximum_warmup_executions = 1000000;

struct Warmup {
  size_t executions;
  double seconds;
};

#ifdef MOFFT_HAVE_FFTW
static constexpr const char *formal_fftw_version = "fftw-3.3.11";

static std::string fftw_label(const char *version) {
  if (!version || !*version)
    return "fftw";
  const char *end = std::strchr(version, ' ');
  return std::string(version, end ? end : version + std::strlen(version));
}
#endif

template <class Function> static Warmup warmup_for(Function &&run) {
  const auto start = clock_type::now();
  size_t executions = 0;
  double elapsed;
  do {
    run();
    ++executions;
    elapsed = std::chrono::duration<double>(clock_type::now() - start).count();
  } while (executions < maximum_warmup_executions &&
           (executions < minimum_warmup_executions ||
            elapsed < target_warmup_seconds));
  return {executions, elapsed};
}

static size_t repetitions_for(double fastest_seconds) {
  if (!(fastest_seconds > 0.0))
    return 1;
  return std::min<size_t>(
      1000000, std::max<size_t>(1, (size_t)std::ceil(target_batch_seconds /
                                                     fastest_seconds)));
}
static double quantile(std::vector<double> x, double q) {
  std::sort(x.begin(), x.end());
  double p = q * (x.size() - 1), f = std::floor(p);
  size_t i = (size_t)f;
  return x[i] + (p - f) * (x[std::min(i + 1, x.size() - 1)] - x[i]);
}
static double med(const std::vector<double> &x) { return quantile(x, .5); }
static double coefficient_of_variation(const std::vector<double> &x) {
  double mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size(), sq = 0;
  for (double value : x)
    sq += (value - mean) * (value - mean);
  return std::sqrt(sq / x.size()) / mean;
}
static std::pair<double, double> bootstrap_ci(const std::vector<double> &x) {
  std::mt19937_64 rng(0x4d6f464654ULL + x.size());
  std::uniform_int_distribution<size_t> pick(0, x.size() - 1);
  std::vector<double> bs(2000), sample(x.size());
  for (double &v : bs) {
    for (double &s : sample)
      s = x[pick(rng)];
    v = med(sample);
  }
  return {quantile(bs, .025), quantile(bs, .975)};
}
static std::string sysstr(const char *key) {
  size_t n = 0;
  if (sysctlbyname(key, nullptr, &n, nullptr, 0))
    return "unknown";
  std::string s(n, '\0');
  if (sysctlbyname(key, s.data(), &n, nullptr, 0))
    return "unknown";
  if (!s.empty() && s.back() == '\0')
    s.pop_back();
  return s;
}
static long long sysint(const char *key) {
  int value = -1;
  size_t n = sizeof(value);
  if (sysctlbyname(key, &value, &n, nullptr, 0))
    return -1;
  return value;
}

template <class T> struct Api;
template <> struct Api<mofft_complex_f32> {
  using Plan = mofft_plan_f32;
  static mofft_status create(Plan **p, size_t n) {
    return mofft_plan_create_f32(p, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  }
  static mofft_status run(Plan *p, const mofft_complex_f32 *i,
                          mofft_complex_f32 *o) {
    return mofft_execute_f32(p, i, o);
  }
  static void destroy(Plan *p) { mofft_plan_destroy_f32(p); }
  static const char *name() { return "fp32"; }
};
template <> struct Api<mofft_complex_f64> {
  using Plan = mofft_plan_f64;
  static mofft_status create(Plan **p, size_t n) {
    return mofft_plan_create_f64(p, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  }
  static mofft_status run(Plan *p, const mofft_complex_f64 *i,
                          mofft_complex_f64 *o) {
    return mofft_execute_f64(p, i, o);
  }
  static void destroy(Plan *p) { mofft_plan_destroy_f64(p); }
  static const char *name() { return "fp64"; }
};

template <class T> Result run_mofft(size_t n, int batches) {
  typename Api<T>::Plan *plan = nullptr;
  auto st = Api<T>::create(&plan, n);
  if (st != MOFFT_SUCCESS)
    throw std::runtime_error(mofft_status_string(st));
  std::vector<T> in(n), out(n);
  for (size_t i = 0; i < n; i++) {
    in[i].real = (decltype(T::real))std::sin(.01 * i);
    in[i].imag = (decltype(T::imag))std::cos(.013 * i);
  }
  const Warmup warmup =
      warmup_for([&] { st = Api<T>::run(plan, in.data(), out.data()); });
  if (st != MOFFT_SUCCESS)
    throw std::runtime_error(mofft_status_string(st));
  auto a0 = clock_type::now();
  st = Api<T>::run(plan, in.data(), out.data());
  auto b0 = clock_type::now();
  size_t repetitions =
      repetitions_for(std::chrono::duration<double>(b0 - a0).count());
  Result r{Api<T>::name(),    "mofft",       n, repetitions, {},
           warmup.executions, warmup.seconds};
  r.seconds.reserve(batches);
  for (int i = 0; i < batches; i++) {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      st = Api<T>::run(plan, in.data(), out.data());
    auto b = clock_type::now();
    if (st != MOFFT_SUCCESS)
      throw std::runtime_error(mofft_status_string(st));
    r.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                        repetitions);
  }
  Api<T>::destroy(plan);
  return r;
}

#ifdef MOFFT_HAVE_FFTW
template <class T> Result run_fftw(size_t, int);
template <> Result run_fftw<mofft_complex_f32>(size_t n, int batches) {
  auto *in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n),
       *out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n);
  auto p = fftwf_plan_dft_1d((int)n, in, out, FFTW_FORWARD, FFTW_MEASURE);
  for (size_t i = 0; i < n; i++) {
    in[i][0] = std::sin(.01 * i);
    in[i][1] = std::cos(.013 * i);
  }
  const Warmup warmup = warmup_for([&] { fftwf_execute(p); });
  auto a0 = clock_type::now();
  fftwf_execute(p);
  auto b0 = clock_type::now();
  size_t repetitions =
      repetitions_for(std::chrono::duration<double>(b0 - a0).count());
  Result r{"fp32", fftw_label(fftwf_version), n, repetitions, {},
           warmup.executions, warmup.seconds};
  for (int i = 0; i < batches; i++) {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      fftwf_execute(p);
    auto b = clock_type::now();
    r.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                        repetitions);
  }
  fftwf_destroy_plan(p);
  fftwf_free(in);
  fftwf_free(out);
  return r;
}
template <> Result run_fftw<mofft_complex_f64>(size_t n, int batches) {
  auto *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n),
       *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
  auto p = fftw_plan_dft_1d((int)n, in, out, FFTW_FORWARD, FFTW_MEASURE);
  for (size_t i = 0; i < n; i++) {
    in[i][0] = std::sin(.01 * i);
    in[i][1] = std::cos(.013 * i);
  }
  const Warmup warmup = warmup_for([&] { fftw_execute(p); });
  auto a0 = clock_type::now();
  fftw_execute(p);
  auto b0 = clock_type::now();
  size_t repetitions =
      repetitions_for(std::chrono::duration<double>(b0 - a0).count());
  Result r{"fp64", fftw_label(fftw_version), n, repetitions, {},
           warmup.executions, warmup.seconds};
  for (int i = 0; i < batches; i++) {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      fftw_execute(p);
    auto b = clock_type::now();
    r.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                        repetitions);
  }
  fftw_destroy_plan(p);
  fftw_free(in);
  fftw_free(out);
  return r;
}

static std::pair<Result, Result> run_pair_f32(size_t n, int batches) {
  mofft_plan_f32 *mp = nullptr;
  if (mofft_plan_create_f32(&mp, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE) !=
      MOFFT_SUCCESS)
    throw std::runtime_error("MoFFT plan failed");
  std::vector<mofft_complex_f32> mi(n), mo(n);
  auto *fi = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n),
       *fo = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * n);
  auto fp = fftwf_plan_dft_1d((int)n, fi, fo, FFTW_FORWARD, FFTW_MEASURE);
  for (size_t i = 0; i < n; i++) {
    mi[i].real = fi[i][0] = std::sin(.01 * i);
    mi[i].imag = fi[i][1] = std::cos(.013 * i);
  }
  const Warmup fftw_warmup = warmup_for([&] { fftwf_execute(fp); });
  mofft_status warmup_status = MOFFT_SUCCESS;
  const Warmup mofft_warmup = warmup_for(
      [&] { warmup_status = mofft_execute_f32(mp, mi.data(), mo.data()); });
  if (warmup_status != MOFFT_SUCCESS)
    throw std::runtime_error(mofft_status_string(warmup_status));
  auto ma = clock_type::now();
  mofft_execute_f32(mp, mi.data(), mo.data());
  auto mb = clock_type::now();
  auto fa = clock_type::now();
  fftwf_execute(fp);
  auto fb = clock_type::now();
  size_t repetitions =
      repetitions_for(std::min(std::chrono::duration<double>(mb - ma).count(),
                               std::chrono::duration<double>(fb - fa).count()));
  Result mr{"fp32",
            "mofft",
            n,
            repetitions,
            {},
            mofft_warmup.executions,
            mofft_warmup.seconds},
      fr{"fp32",
         fftw_label(fftwf_version),
         n,
         repetitions,
         {},
         fftw_warmup.executions,
         fftw_warmup.seconds};
  auto mt = [&] {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      mofft_execute_f32(mp, mi.data(), mo.data());
    auto b = clock_type::now();
    mr.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                         repetitions);
  };
  auto ft = [&] {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      fftwf_execute(fp);
    auto b = clock_type::now();
    fr.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                         repetitions);
  };
  for (int i = 0; i < batches; i++) {
    if (i & 1) {
      ft();
      mt();
    } else {
      mt();
      ft();
    }
  }
  mofft_plan_destroy_f32(mp);
  fftwf_destroy_plan(fp);
  fftwf_free(fi);
  fftwf_free(fo);
  return {std::move(mr), std::move(fr)};
}
static std::pair<Result, Result> run_pair_f64(size_t n, int batches) {
  mofft_plan_f64 *mp = nullptr;
  if (mofft_plan_create_f64(&mp, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE) !=
      MOFFT_SUCCESS)
    throw std::runtime_error("MoFFT plan failed");
  std::vector<mofft_complex_f64> mi(n), mo(n);
  auto *fi = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n),
       *fo = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
  auto fp = fftw_plan_dft_1d((int)n, fi, fo, FFTW_FORWARD, FFTW_MEASURE);
  for (size_t i = 0; i < n; i++) {
    mi[i].real = fi[i][0] = std::sin(.01 * i);
    mi[i].imag = fi[i][1] = std::cos(.013 * i);
  }
  const Warmup fftw_warmup = warmup_for([&] { fftw_execute(fp); });
  mofft_status warmup_status = MOFFT_SUCCESS;
  const Warmup mofft_warmup = warmup_for(
      [&] { warmup_status = mofft_execute_f64(mp, mi.data(), mo.data()); });
  if (warmup_status != MOFFT_SUCCESS)
    throw std::runtime_error(mofft_status_string(warmup_status));
  auto ma = clock_type::now();
  mofft_execute_f64(mp, mi.data(), mo.data());
  auto mb = clock_type::now();
  auto fa = clock_type::now();
  fftw_execute(fp);
  auto fb = clock_type::now();
  size_t repetitions =
      repetitions_for(std::min(std::chrono::duration<double>(mb - ma).count(),
                               std::chrono::duration<double>(fb - fa).count()));
  Result mr{"fp64",
            "mofft",
            n,
            repetitions,
            {},
            mofft_warmup.executions,
            mofft_warmup.seconds},
      fr{"fp64",
         fftw_label(fftw_version),
         n,
         repetitions,
         {},
         fftw_warmup.executions,
         fftw_warmup.seconds};
  auto mt = [&] {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      mofft_execute_f64(mp, mi.data(), mo.data());
    auto b = clock_type::now();
    mr.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                         repetitions);
  };
  auto ft = [&] {
    auto a = clock_type::now();
    for (size_t q = 0; q < repetitions; q++)
      fftw_execute(fp);
    auto b = clock_type::now();
    fr.seconds.push_back(std::chrono::duration<double>(b - a).count() /
                         repetitions);
  };
  for (int i = 0; i < batches; i++) {
    if (i & 1) {
      ft();
      mt();
    } else {
      mt();
      ft();
    }
  }
  mofft_plan_destroy_f64(mp);
  fftw_destroy_plan(fp);
  fftw_free(fi);
  fftw_free(fo);
  return {std::move(mr), std::move(fr)};
}
#endif

static double correctness_reference_tolerance(size_t n, double epsilon) {
  // FFT rounding error grows with the number of stages, not with the output
  // length itself.  This is a relative sanity bound, not a performance
  // criterion.
  return 256.0 * epsilon *
         std::max(1.0, std::log2(static_cast<double>(n)));
}

static double correctness_roundtrip_tolerance(size_t n, double epsilon) {
  return 512.0 * epsilon *
         std::max(1.0, std::log2(static_cast<double>(n)));
}

static void fill_correctness_input(std::vector<mofft_complex_f32> &input) {
  for (size_t i = 0; i < input.size(); ++i) {
    input[i].real = static_cast<float>(std::sin(.01 * i));
    input[i].imag = static_cast<float>(std::cos(.013 * i));
  }
}

static void fill_correctness_input(std::vector<mofft_complex_f64> &input) {
  for (size_t i = 0; i < input.size(); ++i) {
    input[i].real = std::sin(.01 * i);
    input[i].imag = std::cos(.013 * i);
  }
}

static int check_overall_roundtrip_f32(size_t n) {
  std::vector<mofft_complex_f32> input(n), forward(n), recovered(n);
  fill_correctness_input(input);
  mofft_plan_f32 *forward_plan = nullptr, *backward_plan = nullptr;
  mofft_status status = mofft_plan_create_f32(
      &forward_plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_create_f32(
        &backward_plan, n, MOFFT_BACKWARD, MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f32(forward_plan, input.data(), forward.data());
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f32(backward_plan, forward.data(), recovered.data());
  double max_error = 0.0, max_reference = 0.0;
  if (status == MOFFT_SUCCESS) {
    for (size_t i = 0; i < n; ++i) {
      max_reference = std::max(
          max_reference,
          std::hypot(static_cast<double>(input[i].real),
                     static_cast<double>(input[i].imag)));
      max_error = std::max(
          max_error,
          std::hypot(static_cast<double>(recovered[i].real) / n - input[i].real,
                     static_cast<double>(recovered[i].imag) / n - input[i].imag));
    }
  }
  const double relative_error = max_error / std::max(1.0, max_reference);
  const double tolerance = correctness_roundtrip_tolerance(n, FLT_EPSILON);
  if (status != MOFFT_SUCCESS || relative_error > tolerance)
    std::cerr << "correctness fp32 n=" << n << " reference=roundtrip error="
              << relative_error << " (absolute " << max_error
              << ") tolerance=" << tolerance << " status="
              << mofft_status_string(status) << "\n";
  if (forward_plan) mofft_plan_destroy_f32(forward_plan);
  if (backward_plan) mofft_plan_destroy_f32(backward_plan);
  return status != MOFFT_SUCCESS || relative_error > tolerance;
}

static int check_overall_roundtrip_f64(size_t n) {
  std::vector<mofft_complex_f64> input(n), forward(n), recovered(n);
  fill_correctness_input(input);
  mofft_plan_f64 *forward_plan = nullptr, *backward_plan = nullptr;
  mofft_status status = mofft_plan_create_f64(
      &forward_plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_plan_create_f64(
        &backward_plan, n, MOFFT_BACKWARD, MOFFT_OUT_OF_PLACE);
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f64(forward_plan, input.data(), forward.data());
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f64(backward_plan, forward.data(), recovered.data());
  double max_error = 0.0, max_reference = 0.0;
  if (status == MOFFT_SUCCESS) {
    for (size_t i = 0; i < n; ++i) {
      max_reference = std::max(
          max_reference, std::hypot(input[i].real, input[i].imag));
      max_error = std::max(
          max_error,
          std::hypot(recovered[i].real / n - input[i].real,
                     recovered[i].imag / n - input[i].imag));
    }
  }
  const double relative_error = max_error / std::max(1.0, max_reference);
  const double tolerance = correctness_roundtrip_tolerance(n, DBL_EPSILON);
  if (status != MOFFT_SUCCESS || relative_error > tolerance)
    std::cerr << "correctness fp64 n=" << n << " reference=roundtrip error="
              << relative_error << " (absolute " << max_error
              << ") tolerance=" << tolerance << " status="
              << mofft_status_string(status) << "\n";
  if (forward_plan) mofft_plan_destroy_f64(forward_plan);
  if (backward_plan) mofft_plan_destroy_f64(backward_plan);
  return status != MOFFT_SUCCESS || relative_error > tolerance;
}

#ifdef MOFFT_HAVE_FFTW
static int check_overall_fftw_f32(size_t n) {
  std::vector<mofft_complex_f32> input(n), output(n);
  fill_correctness_input(input);
  auto *reference_input = static_cast<fftwf_complex *>(
      fftwf_malloc(sizeof(fftwf_complex) * n));
  auto *reference_output = static_cast<fftwf_complex *>(
      fftwf_malloc(sizeof(fftwf_complex) * n));
  if (!reference_input || !reference_output) {
    std::cerr << "correctness fp32 n=" << n << " allocation failed\n";
    fftwf_free(reference_input);
    fftwf_free(reference_output);
    return 1;
  }
  for (size_t i = 0; i < n; ++i) {
    reference_input[i][0] = input[i].real;
    reference_input[i][1] = input[i].imag;
  }
  fftwf_plan reference_plan = fftwf_plan_dft_1d(
      static_cast<int>(n), reference_input, reference_output,
      FFTW_FORWARD, FFTW_ESTIMATE);
  mofft_plan_f32 *plan = nullptr;
  mofft_status status = mofft_plan_create_f32(
      &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  if (!reference_plan)
    status = MOFFT_INTERNAL_ERROR;
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f32(plan, input.data(), output.data());
  double max_error = 0.0, max_reference = 0.0;
  if (status == MOFFT_SUCCESS) {
    fftwf_execute(reference_plan);
    for (size_t i = 0; i < n; ++i) {
      max_reference = std::max(
          max_reference,
          std::hypot(static_cast<double>(reference_output[i][0]),
                     static_cast<double>(reference_output[i][1])));
      max_error = std::max(
          max_error,
          std::hypot(static_cast<double>(output[i].real) - reference_output[i][0],
                     static_cast<double>(output[i].imag) - reference_output[i][1]));
    }
  }
  const double relative_error = max_error / std::max(1.0, max_reference);
  const double tolerance = correctness_reference_tolerance(n, FLT_EPSILON);
  if (status != MOFFT_SUCCESS || relative_error > tolerance)
    std::cerr << "correctness fp32 n=" << n << " reference=fftw error="
              << relative_error << " (absolute " << max_error
              << ") tolerance=" << tolerance << " status="
              << mofft_status_string(status) << "\n";
  if (plan) mofft_plan_destroy_f32(plan);
  if (reference_plan) fftwf_destroy_plan(reference_plan);
  fftwf_free(reference_input);
  fftwf_free(reference_output);
  return status != MOFFT_SUCCESS || relative_error > tolerance;
}

static int check_overall_fftw_f64(size_t n) {
  std::vector<mofft_complex_f64> input(n), output(n);
  fill_correctness_input(input);
  auto *reference_input = static_cast<fftw_complex *>(
      fftw_malloc(sizeof(fftw_complex) * n));
  auto *reference_output = static_cast<fftw_complex *>(
      fftw_malloc(sizeof(fftw_complex) * n));
  if (!reference_input || !reference_output) {
    std::cerr << "correctness fp64 n=" << n << " allocation failed\n";
    fftw_free(reference_input);
    fftw_free(reference_output);
    return 1;
  }
  for (size_t i = 0; i < n; ++i) {
    reference_input[i][0] = input[i].real;
    reference_input[i][1] = input[i].imag;
  }
  fftw_plan reference_plan = fftw_plan_dft_1d(
      static_cast<int>(n), reference_input, reference_output,
      FFTW_FORWARD, FFTW_ESTIMATE);
  mofft_plan_f64 *plan = nullptr;
  mofft_status status = mofft_plan_create_f64(
      &plan, n, MOFFT_FORWARD, MOFFT_OUT_OF_PLACE);
  if (!reference_plan)
    status = MOFFT_INTERNAL_ERROR;
  if (status == MOFFT_SUCCESS)
    status = mofft_execute_f64(plan, input.data(), output.data());
  double max_error = 0.0, max_reference = 0.0;
  if (status == MOFFT_SUCCESS) {
    fftw_execute(reference_plan);
    for (size_t i = 0; i < n; ++i) {
      max_reference = std::max(
          max_reference,
          std::hypot(reference_output[i][0], reference_output[i][1]));
      max_error = std::max(
          max_error,
          std::hypot(output[i].real - reference_output[i][0],
                     output[i].imag - reference_output[i][1]));
    }
  }
  const double relative_error = max_error / std::max(1.0, max_reference);
  const double tolerance = correctness_reference_tolerance(n, DBL_EPSILON);
  if (status != MOFFT_SUCCESS || relative_error > tolerance)
    std::cerr << "correctness fp64 n=" << n << " reference=fftw error="
              << relative_error << " (absolute " << max_error
              << ") tolerance=" << tolerance << " status="
              << mofft_status_string(status) << "\n";
  if (plan) mofft_plan_destroy_f64(plan);
  if (reference_plan) fftw_destroy_plan(reference_plan);
  fftw_free(reference_input);
  fftw_free(reference_output);
  return status != MOFFT_SUCCESS || relative_error > tolerance;
}
#endif

static int run_overall_correctness(const std::vector<size_t> &sizes,
                                   const std::string &precision) {
  size_t cases = 0, failures = 0;
  for (size_t n : sizes) {
    if (precision == "fp32" || precision == "both") {
      ++cases;
#ifdef MOFFT_HAVE_FFTW
      failures += check_overall_fftw_f32(n);
#else
      failures += check_overall_roundtrip_f32(n);
#endif
    }
    if (precision == "fp64" || precision == "both") {
      ++cases;
#ifdef MOFFT_HAVE_FFTW
      failures += check_overall_fftw_f64(n);
#else
      failures += check_overall_roundtrip_f64(n);
#endif
    }
  }
  if (failures) {
    std::cerr << "overall correctness failed: " << failures << "/" << cases
              << " cases\n";
    return 1;
  }
#ifdef MOFFT_HAVE_FFTW
  std::cout << "overall correctness passed: " << cases
            << " FP32/FP64 forward out-of-place cases compared with FFTW\n";
#else
  std::cout << "overall correctness passed: " << cases
            << " FP32/FP64 forward/backward roundtrip cases\n";
#endif
  return 0;
}

static void write_json(const std::string &path,
                       const std::vector<Result> &results, bool formal,
                       double load[3], double ending_load[3]) {
  std::ofstream o(path);
  o << std::setprecision(12) << "{\n  \"schema_version\":2,\n  \"formal_run\":"
    << (formal ? "true" : "false") << ",\n  \"machine\":\""
    << sysstr("hw.model") << "\",\n  \"os\":\""
    << sysstr("kern.osproductversion") << "\",\n  \"compiler\":\""
    << MOFFT_COMPILER_INFO
    << "\",\n  \"qos\":\"user-interactive\",\n  \"affinity_claimed\":false,\n  "
       "\"warmup_target_seconds\":"
    << target_warmup_seconds
    << ",\n  \"warmup_minimum_executions\":"
    << minimum_warmup_executions
    << ",\n  \"thermal_level_sysctl\":" << sysint("kern.thermal_level")
#ifdef MOFFT_HAVE_FFTW
    << ",\n  \"fftw_fp32_version\":\"" << fftwf_version
    << "\",\n  \"fftw_fp64_version\":\"" << fftw_version << "\""
#endif
    << ",\n  \"wisdom_formal\":"
    << (mofft_wisdom_is_formal() ? "true" : "false")
    << ",\n  \"wisdom_source\":\"" << mofft_wisdom_source()
    << "\",\n  \"load_average\":[" << load[0] << "," << load[1] << ","
    << load[2] << "],\n  \"ending_load_average\":[" << ending_load[0]
    << "," << ending_load[1] << "," << ending_load[2]
    << "],\n  \"results\":[\n";
  for (size_t k = 0; k < results.size(); k++) {
    auto &r = results[k];
    double m = med(r.seconds);
    std::vector<double> dev;
    for (double x : r.seconds)
      dev.push_back(std::abs(x - m));
    double mad = med(dev),
           mean = std::accumulate(r.seconds.begin(), r.seconds.end(), 0.0) /
                  r.seconds.size(),
           sq = 0;
    for (double x : r.seconds)
      sq += (x - mean) * (x - mean);
    double cv = std::sqrt(sq / r.seconds.size()) / mean;
    auto ci = bootstrap_ci(r.seconds);
    double gflops = 5.0 * r.n * std::log2((double)r.n) / m / 1e9, speedup = 1.0;
    for (auto &q : results)
      if (q.n == r.n && q.precision == r.precision &&
          q.implementation != r.implementation) {
        double fm = q.implementation == "mofft" ? med(q.seconds) : m,
               ff = q.implementation == "mofft" ? m : med(q.seconds);
        speedup = ff / fm;
      }
    o << "    {\"precision\":\"" << r.precision << "\",\"implementation\":\""
      << r.implementation << "\",\"n\":" << r.n
      << ",\"warmup_executions\":" << r.warmup_executions
      << ",\"warmup_seconds\":" << r.warmup_seconds
      << ",\"repetitions_per_sample\":" << r.repetitions
      << ",\"median_seconds\":" << m << ",\"ci95\":[" << ci.first << ","
      << ci.second << "],\"mad_seconds\":" << mad << ",\"cv\":" << cv
      << ",\"gflops\":" << gflops << ",\"mofft_speedup_vs_fftw\":" << speedup
      << ",\"raw_seconds\":[";
    for (size_t i = 0; i < r.seconds.size(); i++) {
      if (i)
        o << ",";
      o << r.seconds[i];
    }
    o << "]}" << (k + 1 < results.size() ? "," : "") << "\n";
  }
  o << "  ]\n}\n";
}
static void write_csv(const std::string &json_path,
                      const std::vector<Result> &results) {
  std::string path = json_path;
  auto dot = path.rfind('.');
  if (dot != std::string::npos)
    path.resize(dot);
  path += ".csv";
  std::ofstream o(path);
  o << "precision,implementation,n,median_seconds,ci95_low,ci95_high,mad_"
       "seconds,cv,gflops\n"
    << std::setprecision(12);
  for (auto &r : results) {
    double m = med(r.seconds);
    std::vector<double> d;
    for (double x : r.seconds)
      d.push_back(std::abs(x - m));
    double mean = std::accumulate(r.seconds.begin(), r.seconds.end(), 0.0) /
                  r.seconds.size(),
           sq = 0;
    for (double x : r.seconds)
      sq += (x - mean) * (x - mean);
    auto ci = bootstrap_ci(r.seconds);
    o << r.precision << "," << r.implementation << "," << r.n << "," << m << ","
      << ci.first << "," << ci.second << "," << med(d) << ","
      << std::sqrt(sq / r.seconds.size()) / mean << ","
      << 5.0 * r.n * std::log2((double)r.n) / m / 1e9 << "\n";
  }
}
int main(int argc, char **argv) {
  bool formal = false, full_dry_run = false, correctness = false,
       confirmed = false;
  int batches = 7;
  std::string output = "mofft-smoke.json", size_text, precision = "both";
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--formal"))
      formal = true;
    else if (!strcmp(argv[i], "--full-dry-run"))
      full_dry_run = true;
    else if (!strcmp(argv[i], "--correctness"))
      correctness = true;
    else if (!strcmp(argv[i], "--exclusive-confirmed"))
      confirmed = true;
    else if (!strcmp(argv[i], "--batches") && i + 1 < argc)
      batches = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--output") && i + 1 < argc)
      output = argv[++i];
    else if (!strcmp(argv[i], "--sizes") && i + 1 < argc)
      size_text = argv[++i];
    else if (!strcmp(argv[i], "--precision") && i + 1 < argc)
      precision = argv[++i];
    else {
      std::cerr << "usage: mofft_benchmark [--sizes N[,N...]] [--precision "
                   "fp32|fp64|both] [--batches B] [--output FILE] "
                   "[--correctness | --full-dry-run | "
                   "--formal --exclusive-confirmed]\n";
      return 2;
    }
  }
  if ((formal && full_dry_run) || correctness && (formal || full_dry_run)) {
    std::cerr << "Choose one of --correctness, --full-dry-run, or --formal.\n";
    return 2;
  }
  if (precision != "fp32" && precision != "fp64" && precision != "both") {
    std::cerr << "invalid precision\n";
    return 2;
  }
  if (formal && !size_text.empty()) {
    std::cerr << "Formal runs use the fixed extended suite; --sizes is for "
                 "diagnostics.\n";
    return 2;
  }
  if (formal && !confirmed) {
    std::cerr
        << "Refusing formal run: pass --exclusive-confirmed only after the "
           "user grants an exclusive, powered, thermally stable window.\n";
    return 2;
  }
  if (formal && batches < 30)
    batches = 30;
  if (formal && !mofft_wisdom_is_formal()) {
    std::cerr << "Refusing formal run: compile a wisdom file produced by a "
                 "formal plan search first.\n";
    return 2;
  }
#ifndef MOFFT_HAVE_FFTW
  if (formal) {
    std::cerr << "Refusing formal run: rebuild with the exact FFTW 3.3.11 "
                 "baseline.\n";
    return 2;
  }
#else
  if (formal && (!std::strstr(fftw_version, formal_fftw_version) ||
                 !std::strstr(fftwf_version, formal_fftw_version))) {
    std::cerr << "Refusing formal run: linked FFTW is not 3.3.11.\n";
    return 2;
  }
#endif
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  double load[3] = {};
  getloadavg(load, 3);
  if (formal && load[0] > 2.0) {
    std::cerr << "Refusing formal run: one-minute load average is " << load[0]
              << ", so the machine is not in an exclusive window.\n";
    return 2;
  }
  long long thermal_level = sysint("kern.thermal_level");
  if (formal && thermal_level > 0) {
    std::cerr << "Refusing formal run: macOS reports thermal level "
              << thermal_level << ".\n";
    return 2;
  }
  std::vector<size_t> sizes;
  try {
    if (!size_text.empty())
      sizes = parse_sizes(size_text);
    else if (correctness)
      sizes.assign(std::begin(overall_sizes), std::end(overall_sizes));
    else if (formal || full_dry_run)
      sizes.assign(std::begin(extended_sizes), std::end(extended_sizes));
    else
      sizes = {144, 256, 1024};
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 2;
  }
  if (correctness)
    return run_overall_correctness(sizes, precision);
  std::vector<Result> results;
  try {
    for (size_t n : sizes) {
#ifdef MOFFT_HAVE_FFTW
      if (precision == "fp32" || precision == "both") {
        auto p32 = run_pair_f32(n, batches);
        results.push_back(std::move(p32.first));
        results.push_back(std::move(p32.second));
      }
#else
      if (precision == "fp32" || precision == "both")
        results.push_back(run_mofft<mofft_complex_f32>(n, batches));
#endif
#ifdef MOFFT_HAVE_FFTW
      if (precision == "fp64" || precision == "both") {
        auto p64 = run_pair_f64(n, batches);
        results.push_back(std::move(p64.first));
        results.push_back(std::move(p64.second));
      }
#else
      if (precision == "fp64" || precision == "both")
        results.push_back(run_mofft<mofft_complex_f64>(n, batches));
#endif
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  double ending_load[3] = {};
  getloadavg(ending_load, 3);
  if (formal) {
    // The benchmark itself raises macOS's load average on long runs. Record
    // it for provenance; reject interference using the pre-run load, thermal
    // state, and per-result CV.
    for (const auto &result : results)
      if (coefficient_of_variation(result.seconds) > .05) {
        std::cerr << "Discarding formal round: unstable batch CV for "
                  << result.precision << " " << result.implementation
                  << " n=" << result.n << ".\n";
        return 2;
      }
  }
  write_json(output, results, formal, load, ending_load);
  write_csv(output, results);
  // Compact per-point GFLOPS to stdout so remote runs (e.g. devicectl on an
  // iPad) can capture the numbers without reading a sandboxed output file.
  for (const auto &r : results)
    std::cout << "GFLOPS " << r.precision << " " << r.implementation
              << " n=" << r.n << " gflops="
              << (5.0 * r.n * std::log2((double)r.n) / med(r.seconds) / 1e9)
              << "\n";
  std::cout << "wrote " << output << " and CSV ("
            << (formal ? "formal" : (full_dry_run ? "full-dry-run" : "smoke"))
            << ")\n";
  return 0;
}
