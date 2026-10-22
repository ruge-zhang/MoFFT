// Small direct-DFT check of the FFTW baseline, not a MoFFT execution test.
#include <fftw3.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

template<class T> bool check(size_t n);
#define CHECK(T, COMPLEX, PLAN, PREFIX, EPS, LABEL) \
template<> bool check<T>(size_t n) { \
  auto *in = static_cast<COMPLEX*>(PREFIX##_malloc(sizeof(COMPLEX)*n)); \
  auto *out = static_cast<COMPLEX*>(PREFIX##_malloc(sizeof(COMPLEX)*n)); \
  if (!in || !out) return false; \
  for (size_t j=0;j<n;++j) { in[j][0]=T(std::sin(.01*j)); in[j][1]=T(std::cos(.013*j)); } \
  PLAN p=PREFIX##_plan_dft_1d(n,in,out,FFTW_FORWARD,FFTW_ESTIMATE); \
  if (!p) return false; \
  PREFIX##_execute(p); \
  long double max_error=0,max_reference=0; \
  for(size_t k=0;k<n;++k) { \
    long double rr=0,ii=0; \
    for(size_t j=0;j<n;++j) { \
      const long double angle=-2*acosl(-1.L)*j*k/n; \
      rr+=(long double)in[j][0]*cosl(angle)-(long double)in[j][1]*sinl(angle); \
      ii+=(long double)in[j][0]*sinl(angle)+(long double)in[j][1]*cosl(angle); \
    } \
    if(!std::isfinite(out[k][0]) || !std::isfinite(out[k][1])) return false; \
    max_reference=std::max(max_reference,hypotl(rr,ii)); \
    max_error=std::max(max_error,hypotl((long double)out[k][0]-rr,(long double)out[k][1]-ii)); \
  } \
  const long double relative=max_error/std::max(1.L,max_reference); \
  const long double tolerance=256.L*EPS*std::max(1.,std::log2(double(n))); \
  std::cout<<LABEL<<","<<n<<","<<std::setprecision(18)<<double(relative)<<","<<double(tolerance)<<"\n"; \
  PREFIX##_destroy_plan(p); PREFIX##_free(in); PREFIX##_free(out); \
  return std::isfinite(relative) && relative <= tolerance; \
}
CHECK(float, fftwf_complex, fftwf_plan, fftwf, FLT_EPSILON, "fp32")
CHECK(double, fftw_complex, fftw_plan, fftw, DBL_EPSILON, "fp64")
int main() {
  bool ok=true;
  std::cout<<"#fp32_version="<<fftwf_version<<"\n#fp64_version="<<fftw_version<<"\n";
  std::cout<<"precision,n,relative_error,tolerance\n";
  for (size_t n:{8,16,169,256}) { ok=check<float>(n)&&ok; ok=check<double>(n)&&ok; }
  return ok?0:1;
}
