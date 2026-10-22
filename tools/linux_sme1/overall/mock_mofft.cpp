// Login-only protocol stand-in backed by FFTW. Never numerical SME evidence.
#include "mofft.h"
#include <fftw3.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

struct mofft_plan_f32 {
  size_t n;
  std::vector<mofft_complex_f32> in, out;
  fftwf_plan fft;
  mofft_plan_f32(size_t length, int sign): n(length), in(n), out(n) {
    fft = fftwf_plan_dft_1d(n, reinterpret_cast<fftwf_complex*>(in.data()),
      reinterpret_cast<fftwf_complex*>(out.data()), sign, FFTW_ESTIMATE|FFTW_UNALIGNED);
  }
};
struct mofft_plan_f64 {
  size_t n;
  std::vector<mofft_complex_f64> in, out;
  fftw_plan fft;
  mofft_plan_f64(size_t length, int sign): n(length), in(n), out(n) {
    fft = fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex*>(in.data()),
      reinterpret_cast<fftw_complex*>(out.data()), sign, FFTW_ESTIMATE|FFTW_UNALIGNED);
  }
};
static bool fault(const char *name) {
  const char *value = std::getenv("MOFFT_MOCK_FAULT");
  return value && !std::strcmp(name, value);
}
extern "C" {
mofft_status mofft_plan_create_f32(mofft_plan_f32 **p, size_t n,
                                  mofft_direction d, mofft_placement) {
  *p = nullptr;
  if (fault("allocation")) return MOFFT_ALLOCATION_FAILURE;
  try { *p = new mofft_plan_f32(n, d); }
  catch (...) { return MOFFT_ALLOCATION_FAILURE; }
  return (*p)->fft ? MOFFT_SUCCESS : MOFFT_INTERNAL_ERROR;
}
mofft_status mofft_plan_create_f64(mofft_plan_f64 **p, size_t n,
                                  mofft_direction d, mofft_placement) {
  *p = nullptr;
  if (fault("allocation")) return MOFFT_ALLOCATION_FAILURE;
  try { *p = new mofft_plan_f64(n, d); }
  catch (...) { return MOFFT_ALLOCATION_FAILURE; }
  return (*p)->fft ? MOFFT_SUCCESS : MOFFT_INTERNAL_ERROR;
}
mofft_status mofft_execute_f32(const mofft_plan_f32 *p,
    const mofft_complex_f32 *in, mofft_complex_f32 *out) {
  if (fault("status")) return MOFFT_INTERNAL_ERROR;
  fftwf_execute_dft(p->fft, reinterpret_cast<fftwf_complex*>(const_cast<mofft_complex_f32*>(in)),
                   reinterpret_cast<fftwf_complex*>(out));
  if (fault("nan")) out[0].real = std::numeric_limits<float>::quiet_NaN();
  return MOFFT_SUCCESS;
}
mofft_status mofft_execute_f64(const mofft_plan_f64 *p,
    const mofft_complex_f64 *in, mofft_complex_f64 *out) {
  if (fault("status")) return MOFFT_INTERNAL_ERROR;
  fftw_execute_dft(p->fft, reinterpret_cast<fftw_complex*>(const_cast<mofft_complex_f64*>(in)),
                  reinterpret_cast<fftw_complex*>(out));
  if (fault("nan")) out[0].real = std::numeric_limits<double>::quiet_NaN();
  return MOFFT_SUCCESS;
}
void mofft_plan_destroy_f32(mofft_plan_f32 *p) {
  if (p) { if (p->fft) fftwf_destroy_plan(p->fft); delete p; }
}
void mofft_plan_destroy_f64(mofft_plan_f64 *p) {
  if (p) { if (p->fft) fftw_destroy_plan(p->fft); delete p; }
}
int mofft_wisdom_is_formal(void) { return 0; }
const char *mofft_wisdom_source(void) { return "LOGIN_MOCK_NOT_A_MEASURED_PLAN"; }
const char *mofft_status_string(mofft_status s) {
  return s == MOFFT_SUCCESS ? "MOCK_SUCCESS" : "MOCK_INJECTED_OR_ALLOCATION_FAILURE";
}
const char *mofft_version(void) { return "LOGIN_MOCK_NOT_MOFFT"; }
}
