#include "linux_platform.h"
#include "mofft.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static unsigned long svcr() {
  unsigned long x;
  asm volatile("mrs %0, S3_3_C4_C2_2" : "=r"(x));
  return x;
}
static void unchanged(unsigned long a, const char *operation, size_t n, int bits) {
  unsigned long b=svcr();
  if ((a&3)!=(b&3)) throw std::runtime_error("SVCR_PUBLIC_API_CHANGED");
  std::printf("state,%d,%zu,%s,%lu,%lu\n",bits,n,operation,a,b);
}
static const size_t sizes[] = {
  256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,
  144,169,196,225,1728,2197,2744,3375,20736,28561,38416,50625,
  248832,371293,537824,759375};

int main() {
  if (!linux_capture_admission()) return 2;
  try {
    for(size_t n:sizes) {
      mofft_plan_f32 *p=nullptr;
      std::vector<mofft_complex_f32> in(n), a(n), b(n);
      for(size_t i=0;i<n;++i) in[i]={(float)std::sin(.01*i),(float)std::cos(.013*i)};
      unsigned long s=svcr();
      if(mofft_plan_create_f32(&p,n,MOFFT_FORWARD,MOFFT_OUT_OF_PLACE)!=MOFFT_SUCCESS)
        throw std::runtime_error("CREATE32_FAILED");
      unchanged(s,"create",n,32);
      s=svcr();if(mofft_execute_f32(p,in.data(),a.data())!=MOFFT_SUCCESS)
        throw std::runtime_error("EXECUTE32_FAILED");
      unchanged(s,"execute1",n,32);
      s=svcr();if(mofft_execute_f32(p,in.data(),b.data())!=MOFFT_SUCCESS)
        throw std::runtime_error("EXECUTE32_FAILED");
      unchanged(s,"execute2",n,32);
      for(size_t i=0;i<n;++i)
        if(!std::isfinite(a[i].real)||!std::isfinite(a[i].imag)||
           std::memcmp(&a[i],&b[i],sizeof(a[i])))
          throw std::runtime_error("FINITE_OR_REPEAT32_FAILED");
      s=svcr();mofft_plan_destroy_f32(p);unchanged(s,"destroy",n,32);
    }
    for(size_t n:sizes) {
      mofft_plan_f64 *p=nullptr;
      std::vector<mofft_complex_f64> in(n), a(n), b(n);
      for(size_t i=0;i<n;++i) in[i]={std::sin(.01*i),std::cos(.013*i)};
      unsigned long s=svcr();
      if(mofft_plan_create_f64(&p,n,MOFFT_FORWARD,MOFFT_OUT_OF_PLACE)!=MOFFT_SUCCESS)
        throw std::runtime_error("CREATE64_FAILED");
      unchanged(s,"create",n,64);
      s=svcr();if(mofft_execute_f64(p,in.data(),a.data())!=MOFFT_SUCCESS)
        throw std::runtime_error("EXECUTE64_FAILED");
      unchanged(s,"execute1",n,64);
      s=svcr();if(mofft_execute_f64(p,in.data(),b.data())!=MOFFT_SUCCESS)
        throw std::runtime_error("EXECUTE64_FAILED");
      unchanged(s,"execute2",n,64);
      for(size_t i=0;i<n;++i)
        if(!std::isfinite(a[i].real)||!std::isfinite(a[i].imag)||
           std::memcmp(&a[i],&b[i],sizeof(a[i])))
          throw std::runtime_error("FINITE_OR_REPEAT64_FAILED");
      s=svcr();mofft_plan_destroy_f64(p);unchanged(s,"destroy",n,64);
    }
    std::puts("PUBLIC_API_LIFECYCLE_PASS:54_CONDITIONS_216_TRANSITIONS");
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what());return 1;
  }
}
