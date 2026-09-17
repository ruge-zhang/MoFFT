#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <time.h>

typedef unsigned long long (*kernel_fn)(unsigned long long);
#define DECL(op, p, i) extern unsigned long long mofft_mb_##op##_##p##_i##i(unsigned long long)
DECL(add, fp32, 1); DECL(add, fp32, 2); DECL(add, fp32, 4); DECL(add, fp32, 8);
DECL(sub, fp32, 1); DECL(sub, fp32, 2); DECL(sub, fp32, 4); DECL(sub, fp32, 8);
DECL(fmla_v, fp32, 1); DECL(fmla_v, fp32, 2); DECL(fmla_v, fp32, 4); DECL(fmla_v, fp32, 8);
DECL(fmopa, fp32, 1); DECL(fmopa, fp32, 2); DECL(fmopa, fp32, 4);
DECL(add, fp64, 1); DECL(add, fp64, 2); DECL(add, fp64, 4); DECL(add, fp64, 8);
DECL(sub, fp64, 1); DECL(sub, fp64, 2); DECL(sub, fp64, 4); DECL(sub, fp64, 8);
DECL(fmla_v, fp64, 1); DECL(fmla_v, fp64, 2); DECL(fmla_v, fp64, 4); DECL(fmla_v, fp64, 8);
DECL(fmopa, fp64, 1); DECL(fmopa, fp64, 2); DECL(fmopa, fp64, 4); DECL(fmopa, fp64, 8);
DECL(fmla_m_vg2, fp32, 1); DECL(fmla_m_vg2, fp32, 2); DECL(fmla_m_vg2, fp32, 4);
DECL(fmla_m_vg4, fp32, 1); DECL(fmla_m_vg4, fp32, 2); DECL(fmla_m_vg4, fp32, 4);
DECL(fmla_m_vg2, fp64, 1); DECL(fmla_m_vg2, fp64, 2); DECL(fmla_m_vg2, fp64, 4);
DECL(fmla_m_vg4, fp64, 1); DECL(fmla_m_vg4, fp64, 2); DECL(fmla_m_vg4, fp64, 4);
DECL(za_slice, fp32, 1); DECL(za_slice, fp32, 2); DECL(za_slice, fp32, 4);
DECL(za_slice, fp64, 1); DECL(za_slice, fp64, 2); DECL(za_slice, fp64, 4);
DECL(mixed_add_fmopa, fp32, 4); DECL(mixed_add_fmopa, fp64, 4);
extern unsigned long long mofft_mb_transition(unsigned long long);

struct item { const char *op, *precision; int ilp; kernel_fn fn; };
#define ITEM(op,p,i) {#op, #p, i, mofft_mb_##op##_##p##_i##i}
static const struct item items[] = {
  ITEM(add,fp32,1), ITEM(add,fp32,2), ITEM(add,fp32,4), ITEM(add,fp32,8),
  ITEM(sub,fp32,1), ITEM(sub,fp32,2), ITEM(sub,fp32,4), ITEM(sub,fp32,8),
  ITEM(fmla_v,fp32,1), ITEM(fmla_v,fp32,2), ITEM(fmla_v,fp32,4), ITEM(fmla_v,fp32,8),
  ITEM(fmopa,fp32,1), ITEM(fmopa,fp32,2), ITEM(fmopa,fp32,4),
  ITEM(add,fp64,1), ITEM(add,fp64,2), ITEM(add,fp64,4), ITEM(add,fp64,8),
  ITEM(sub,fp64,1), ITEM(sub,fp64,2), ITEM(sub,fp64,4), ITEM(sub,fp64,8),
  ITEM(fmla_v,fp64,1), ITEM(fmla_v,fp64,2), ITEM(fmla_v,fp64,4), ITEM(fmla_v,fp64,8),
  ITEM(fmopa,fp64,1), ITEM(fmopa,fp64,2), ITEM(fmopa,fp64,4), ITEM(fmopa,fp64,8),
  ITEM(fmla_m_vg2,fp32,1), ITEM(fmla_m_vg2,fp32,2), ITEM(fmla_m_vg2,fp32,4),
  ITEM(fmla_m_vg4,fp32,1), ITEM(fmla_m_vg4,fp32,2), ITEM(fmla_m_vg4,fp32,4),
  ITEM(fmla_m_vg2,fp64,1), ITEM(fmla_m_vg2,fp64,2), ITEM(fmla_m_vg2,fp64,4),
  ITEM(fmla_m_vg4,fp64,1), ITEM(fmla_m_vg4,fp64,2), ITEM(fmla_m_vg4,fp64,4),
  ITEM(za_slice,fp32,1), ITEM(za_slice,fp32,2), ITEM(za_slice,fp32,4),
  ITEM(za_slice,fp64,1), ITEM(za_slice,fp64,2), ITEM(za_slice,fp64,4),
  ITEM(mixed_add_fmopa,fp32,4), ITEM(mixed_add_fmopa,fp64,4)
};

__attribute__((noinline,optnone)) static unsigned long long now_ns(void) {
  struct timespec timestamp;
  if(clock_gettime(CLOCK_MONOTONIC_RAW,&timestamp)!=0 &&
     clock_gettime(CLOCK_MONOTONIC,&timestamp)!=0) return 0;
  unsigned long long value=(unsigned long long)timestamp.tv_sec*1000000000ULL+
                           (unsigned long long)timestamp.tv_nsec;
  __asm__ volatile("" : "+r"(value) : : "memory");
  return value;
}
static double seconds(unsigned long long nanoseconds) {
  return (double)nanoseconds * 1e-9;
}
static double measurable_seconds(double value) {
  return value > 0.0 ? value : 1e-9;
}
static int cmp_double(const void *a, const void *b) {
  double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y);
}
static void sysctl_string(const char *key, char *out, size_t cap) {
  size_t n=cap; if (sysctlbyname(key,out,&n,NULL,0)!=0) snprintf(out,cap,"unknown");
}
static uint64_t sysctl_u64(const char *key, uint64_t fallback) {
  uint64_t value=0; size_t n=sizeof value;
  return sysctlbyname(key,&value,&n,NULL,0)==0 ? value : fallback;
}
static void memory_read(const double *data, size_t count, size_t passes) {
  for(size_t pass=0;pass<passes;++pass)
    for(size_t i=0;i<count;i+=16) {
      const char *address=(const char *)(data+i);
      __asm__ volatile(
          "ldp q0, q1, [%0, #0]\n"
          "ldp q2, q3, [%0, #32]\n"
          "ldp q4, q5, [%0, #64]\n"
          "ldp q6, q7, [%0, #96]\n"
          : : "r"(address)
          : "v0","v1","v2","v3","v4","v5","v6","v7","memory");
    }
}
static void memory_write(double *data, size_t count, size_t passes) {
  __asm__ volatile(
      "movi v0.2d, #0\n" "movi v1.2d, #0\n"
      "movi v2.2d, #0\n" "movi v3.2d, #0\n"
      "movi v4.2d, #0\n" "movi v5.2d, #0\n"
      "movi v6.2d, #0\n" "movi v7.2d, #0\n"
      : : : "v0","v1","v2","v3","v4","v5","v6","v7");
  for(size_t pass=0;pass<passes;++pass)
    for(size_t i=0;i<count;i+=16) {
      char *address=(char *)(data+i);
      __asm__ volatile(
          "stp q0, q1, [%0, #0]\n"
          "stp q2, q3, [%0, #32]\n"
          "stp q4, q5, [%0, #64]\n"
          "stp q6, q7, [%0, #96]\n"
          : : "r"(address) : "memory");
    }
}
static void memory_copy(const double *source, double *destination,
                        size_t count, size_t passes) {
  for(size_t pass=0;pass<passes;++pass)
    for(size_t i=0;i<count;i+=16) {
      const char *src=(const char *)(source+i);
      char *dst=(char *)(destination+i);
      __asm__ volatile(
          "ldp q0, q1, [%0, #0]\n" "ldp q2, q3, [%0, #32]\n"
          "ldp q4, q5, [%0, #64]\n" "ldp q6, q7, [%0, #96]\n"
          "stp q0, q1, [%1, #0]\n" "stp q2, q3, [%1, #32]\n"
          "stp q4, q5, [%1, #64]\n" "stp q6, q7, [%1, #96]\n"
          : : "r"(src), "r"(dst)
          : "v0","v1","v2","v3","v4","v5","v6","v7","memory");
    }
}

typedef struct {
  double real;
  double imag;
} complex64;

__attribute__((noinline))
static void transpose_linear(const complex64 *source, complex64 *destination,
                             size_t rows, size_t columns, size_t batch,
                             size_t passes) {
  const size_t source_row = columns * batch;
  const size_t destination_row = rows * batch;
  for (size_t pass = 0; pass < passes; ++pass)
    for (size_t j = 0; j < columns; ++j)
      for (size_t k = 0; k < rows; ++k)
        for (size_t b = 0; b < batch; ++b)
          destination[j * destination_row + k * batch + b] =
              source[k * source_row + j * batch + b];
  __asm__ volatile("" : : "r"(destination) : "memory");
}

__attribute__((noinline))
static void transpose_blocked(const complex64 *source, complex64 *destination,
                              size_t rows, size_t columns, size_t batch,
                              size_t passes) {
  const size_t tile = 8;
  const size_t source_row = columns * batch;
  const size_t destination_row = rows * batch;
  for (size_t pass = 0; pass < passes; ++pass)
    for (size_t k0 = 0; k0 < rows; k0 += tile) {
      const size_t kend = k0 + tile < rows ? k0 + tile : rows;
      for (size_t j0 = 0; j0 < columns; j0 += tile) {
        const size_t jend = j0 + tile < columns ? j0 + tile : columns;
        for (size_t j = j0; j < jend; ++j)
          for (size_t k = k0; k < kend; ++k)
            for (size_t b = 0; b < batch; ++b)
              destination[j * destination_row + k * batch + b] =
                  source[k * source_row + j * batch + b];
      }
    }
  __asm__ volatile("" : : "r"(destination) : "memory");
}

int main(int argc, char **argv) {
  unsigned long long iterations=200000; int samples=9, formal=0, confirmed=0;
  for (int i=1;i<argc;i++) {
    if (!strcmp(argv[i],"--iterations") && i+1<argc) iterations=strtoull(argv[++i],0,10);
    else if (!strcmp(argv[i],"--samples") && i+1<argc) samples=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--formal")) formal=1;
    else if (!strcmp(argv[i],"--exclusive-confirmed")) confirmed=1;
  }
  if (iterations<1 || samples<3 || samples>99) { fprintf(stderr,"invalid arguments\n"); return 2; }
  double load[3]={0}; getloadavg(load,3);
  if (formal && !confirmed) { fprintf(stderr,"formal calibration requires --exclusive-confirmed\n"); return 2; }
  if (formal && (iterations<200000 || samples<9)) { fprintf(stderr,"formal calibration requires at least 200000 iterations and 9 samples\n"); return 2; }
  if (formal && load[0]>2.0) { fprintf(stderr,"formal calibration refused: load average %.3f\n",load[0]); return 2; }
  char model[128], os[128]; sysctl_string("hw.model",model,sizeof model); sysctl_string("kern.osproductversion",os,sizeof os);
  uint64_t line_bytes=sysctl_u64("hw.cachelinesize",128);
  uint64_t l1d_bytes=sysctl_u64("hw.perflevel0.l1dcachesize",
                       sysctl_u64("hw.l1dcachesize",64*1024));
  uint64_t l2_bytes=sysctl_u64("hw.perflevel0.l2cachesize",
                      sysctl_u64("hw.l2cachesize",4*1024*1024));
  printf("{\n  \"schema_version\": 1,\n  \"machine\": \"%s\",\n  \"os\": \"%s\",\n",model,os);
  printf("  \"formal_measurement\": %s,\n  \"load_average\": [%.6g, %.6g, %.6g],\n",formal?"true":"false",load[0],load[1],load[2]);
  printf("  \"iterations\": %llu,\n  \"samples_per_case\": %d,\n  \"cases\": [\n",iterations,samples);
  int first=1;
  for (size_t k=0;k<sizeof(items)/sizeof(items[0]);k++) {
    double values[99]; items[k].fn(2000);
    for (int s=0;s<samples;s++) { unsigned long long t0=now_ns(); items[k].fn(iterations); unsigned long long t1=now_ns(); values[s]=seconds(t1-t0); }
    qsort(values,samples,sizeof(double),cmp_double);
    double med=measurable_seconds(values[samples/2]);
    if (!first) printf(",\n"); first=0;
    printf("    {\"operation\":\"%s\",\"precision\":\"%s\",\"ilp\":%d,\"instructions\":%llu,\"median_seconds\":%.12g,\"instructions_per_second\":%.12g}",items[k].op,items[k].precision,items[k].ilp,iterations*8,med,(double)(iterations*8)/med);
  }
  double transitions[99]; mofft_mb_transition(100);
  for(int s=0;s<samples;s++){ unsigned long long t0=now_ns(); mofft_mb_transition(iterations); unsigned long long t1=now_ns(); transitions[s]=seconds(t1-t0); }
  qsort(transitions,samples,sizeof(double),cmp_double);
  transitions[samples/2]=measurable_seconds(transitions[samples/2]);
  printf(",\n    {\"operation\":\"streaming_transition\",\"precision\":\"mode\",\"ilp\":1,\"instructions\":%llu,\"median_seconds\":%.12g,\"instructions_per_second\":%.12g}\n",iterations,transitions[samples/2],(double)iterations/transitions[samples/2]);
  printf("  ],\n  \"cache_info\": {\"line_bytes\":%llu,\"l1d_bytes\":%llu,\"l2_bytes\":%llu},\n",
         (unsigned long long)line_bytes,(unsigned long long)l1d_bytes,
         (unsigned long long)l2_bytes);
  const size_t memory_sizes[]={32*1024,256*1024,4*1024*1024,64*1024*1024};
  const size_t largest=memory_sizes[sizeof(memory_sizes)/sizeof(memory_sizes[0])-1];
  double *memory=NULL, *copy_destination=NULL;
  if(posix_memalign((void **)&memory,128,largest)!=0) return 1;
  if(posix_memalign((void **)&copy_destination,128,largest)!=0) {
    free(memory); return 1;
  }
  for(size_t i=0;i<largest/sizeof(double);++i) memory[i]=(double)(i&255)*.001;
  printf("  \"memory_cases\": [\n");
  int memory_first=1;
  for(size_t m=0;m<sizeof(memory_sizes)/sizeof(memory_sizes[0]);++m) {
    size_t bytes=memory_sizes[m], count=bytes/sizeof(double);
    size_t passes=(256ULL*1024*1024+bytes-1)/bytes;
    for(int operation=0;operation<3;++operation) {
      double values[99];
      if(operation==0) memory_read(memory,count,1);
      else if(operation==1) memory_write(memory,count,1);
      else memory_copy(memory,copy_destination,count,1);
      for(int s=0;s<samples;++s) {
        unsigned long long t0=now_ns();
        if(operation==0) memory_read(memory,count,passes);
        else if(operation==1) memory_write(memory,count,passes);
        else memory_copy(memory,copy_destination,count,passes);
        unsigned long long t1=now_ns(); values[s]=measurable_seconds(seconds(t1-t0));
      }
      qsort(values,samples,sizeof(double),cmp_double);
      double processed=(double)bytes*passes*(operation==2?2.0:1.0);
      size_t working_set=bytes*(operation==2?2:1);
      if(!memory_first) printf(",\n"); memory_first=0;
      printf("    {\"operation\":\"%s\",\"working_set_bytes\":%zu,\"bytes\":%.0f,\"median_seconds\":%.12g,\"bytes_per_second\":%.12g}",
             operation==0?"read":operation==1?"write":"copy",working_set,processed,
             values[samples/2],processed/values[samples/2]);
    }
  }
  printf("\n  ],\n  \"layout_cases\": [\n");
  const size_t layout_elements = 262144;
  const size_t layout_columns[] = {8, 16, 32, 64};
  const size_t layout_batches[] = {1, 2, 4, 8};
  const size_t layout_bytes = layout_elements * sizeof(complex64);
  const size_t layout_passes =
      (64ULL * 1024 * 1024 + 2 * layout_bytes - 1) / (2 * layout_bytes);
  complex64 *layout_source = (complex64 *)memory;
  complex64 *layout_destination = (complex64 *)copy_destination;
  int layout_first = 1;
  for (size_t c = 0; c < sizeof(layout_columns) / sizeof(layout_columns[0]);
       ++c) {
    const size_t columns = layout_columns[c];
    for (size_t b = 0; b < sizeof(layout_batches) / sizeof(layout_batches[0]);
         ++b) {
      const size_t batch = layout_batches[b];
      const size_t rows = layout_elements / (columns * batch);
      for (int blocked = 0; blocked < 2; ++blocked) {
        double values[99];
        if (blocked)
          transpose_blocked(layout_source, layout_destination, rows, columns,
                            batch, 1);
        else
          transpose_linear(layout_source, layout_destination, rows, columns,
                           batch, 1);
        for (int s = 0; s < samples; ++s) {
          unsigned long long t0 = now_ns();
          if (blocked)
            transpose_blocked(layout_source, layout_destination, rows, columns,
                              batch, layout_passes);
          else
            transpose_linear(layout_source, layout_destination, rows, columns,
                             batch, layout_passes);
          unsigned long long t1 = now_ns();
          values[s] = measurable_seconds(seconds(t1 - t0));
        }
        qsort(values, samples, sizeof(double), cmp_double);
        const double processed = (double)(2 * layout_bytes) * layout_passes;
        if (!layout_first) printf(",\n");
        layout_first = 0;
        printf("    {\"operation\":\"%s\",\"complex_bytes\":%zu,"
               "\"transform_elements\":%zu,\"rows\":%zu,"
               "\"columns\":%zu,\"batch\":%zu,"
               "\"working_set_bytes\":%zu,\"bytes\":%.0f,"
               "\"median_seconds\":%.12g,\"bytes_per_second\":%.12g}",
               blocked ? "blocked_transpose" : "linear_transpose",
               sizeof(complex64), layout_elements, rows, columns, batch,
               2 * layout_bytes, processed, values[samples / 2],
               processed / values[samples / 2]);
      }
    }
  }
  free(memory);
  free(copy_destination);
  printf("\n  ],\n  \"claims\": {\"port_numbers\":\"not measured\",\"resource_relations\":\"inferred from throughput only\",\"memory_bandwidth\":\"measured by dependency-free 128-byte read, write, and copy streams\",\"layout_bandwidth\":\"measured by the runtime linear and 8x8 blocked transpose loop nests\",\"cache_capacities\":\"reported by sysctl when available\"}\n}\n");
  return 0;
}
