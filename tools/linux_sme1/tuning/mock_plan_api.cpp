// Explicitly fictitious cost model and FFTW-backed plans: login protocol only.
#include "mofft.h"
#include <cmath>
extern "C" {
double mofft_estimated_stage_cost(int r,int,int,size_t n,size_t) {
  return double(n)*r;
}
double mofft_estimated_plan_cost_with_layout(size_t n,int,const int *r,size_t k,
    uint64_t,int,mofft_stage_layout) {
  double cost=0;for(size_t i=0;i<k;++i)cost+=r[i]*double(n);return cost;
}
mofft_status mofft_estimated_plan_cost_breakdown_with_layout(size_t n,int bits,
    const int *r,size_t k,uint64_t mask,int direct,mofft_stage_layout layout,
    mofft_plan_cost_breakdown *b) {
  *b={};b->total_cost=b->kernel_cost=mofft_estimated_plan_cost_with_layout(n,bits,r,k,mask,direct,layout);
  return MOFFT_SUCCESS;
}
mofft_status mofft_plan_create_with_radices_f32(mofft_plan_f32 **p,size_t n,
    mofft_direction d,mofft_placement place,const int *r,size_t k) {
  size_t product=1;for(size_t i=0;i<k;++i) product*=r[i];
  if(product!=n)return MOFFT_INVALID_ARGUMENT;return mofft_plan_create_f32(p,n,d,place);
}
mofft_status mofft_plan_create_with_radices_f64(mofft_plan_f64 **p,size_t n,
    mofft_direction d,mofft_placement place,const int *r,size_t k) {
  size_t product=1;for(size_t i=0;i<k;++i) product*=r[i];
  if(product!=n)return MOFFT_INVALID_ARGUMENT;return mofft_plan_create_f64(p,n,d,place);
}
#define SETTERS(P) \
mofft_status mofft_plan_set_blocked_transpose_mask_##P(mofft_plan_##P*,uint64_t){return MOFFT_SUCCESS;} \
mofft_status mofft_plan_set_direct_input_##P(mofft_plan_##P*,int){return MOFFT_SUCCESS;} \
mofft_status mofft_plan_set_stage_layout_##P(mofft_plan_##P*,mofft_stage_layout){return MOFFT_SUCCESS;}
SETTERS(f32)
SETTERS(f64)
}
