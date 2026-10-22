#ifndef MOFFT_CAL_TARGET_NODE
#define MOFFT_CAL_TARGET_NODE "not-configured"
#endif
#define _GNU_SOURCE
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <asm/hwcap.h>
#ifndef PR_SME_GET_VL
#define PR_SME_GET_VL 64
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME (1UL<<23)
#define HWCAP2_SME_F64F64 (1UL<<25)
#endif
#ifndef HWCAP2_SME_F32F32
#define HWCAP2_SME_F32F32 (1UL<<29)
#endif
static void fail(const char *s){fprintf(stderr,"CAL_ERROR:%s\n",s);exit(2);}
static uint64_t now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC_RAW,&t))fail("clock");return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static void guard(void){
 const char *job=getenv("CCS_JOB_ID");cpu_set_t mask;char host[256];
 if(!job || !*job || strspn(job,"0123456789")!=strlen(job))fail("job");
 if(sched_getaffinity(0,sizeof(mask),&mask)||CPU_COUNT(&mask)!=1)fail("affinity");
 if(gethostname(host,sizeof(host))||strcmp(host,MOFFT_CAL_TARGET_NODE))fail("hostname");
 if(!CPU_ISSET(0,&mask)||sched_getcpu()!=0)fail("cpu_binding");
 unsigned long a=getauxval(AT_HWCAP),b=getauxval(AT_HWCAP2);
 int v=prctl(PR_SVE_GET_VL,0,0,0,0),s=prctl(PR_SME_GET_VL,0,0,0,0);
 if(!(a&HWCAP_SVE)||!(b&HWCAP2_SME)||!(b&HWCAP2_SME_F64F64)||!(b&HWCAP2_SME_F32F32)||v<0||s<0||(v&65535)!=64||(s&65535)!=64)fail("capability");
 printf("{\"kind\":\"identity\",\"job\":\"%s\",\"host\":\"%s\",\"cpu\":%d,\"sve_bytes\":%d,\"sme_bytes\":%d,\"hwcap\":%lu,\"hwcap2\":%lu}\n",job,host,sched_getcpu(),v&65535,s&65535,a,b);fflush(stdout);
}
typedef struct {double real,imag;} complex64;
extern void memory_op(unsigned char*,unsigned char*,size_t,size_t,int);
extern void transpose(complex64*,complex64*,size_t,size_t,size_t,size_t,int);
#include "observation.h"

#define RING 8192
#define PAYLOAD (RING+1024)
static unsigned char arena[PAYLOAD+256] __attribute__((aligned(128))), expected[PAYLOAD];
static void scalar_value(unsigned char *p,int fp32,double v){if(fp32){float f=(float)v;memcpy(p,&f,4);}else memcpy(p,&v,8);}
static void prepare_instruction(const struct item *x,size_t loops){
 memset(arena,0xa5,sizeof(arena));memset(arena+128,0xcc,PAYLOAD);
 int fp32=!strcmp(x->precision,"fp32"),width=fp32?4:8,pair=strstr(x->op,"2_seq")!=NULL;
 int load=!strncmp(x->op,"load",4),seq=strstr(x->op,"_seq")!=NULL;
 if(load)for(int k=0;k<RING;k+=width)scalar_value(arena+128+k,fp32,(double)((k/width)%1021+1)/1024.0);
 memcpy(expected,arena+128,PAYLOAD);
 if(seq && !load){for(int k=0;k<RING;k+=width)scalar_value(expected+k,fp32,pair && (k/width)%2 ? 1.0 : .5);}
 else if(!strcmp(x->op,"store_repeat_diagnostic")){for(int k=0;k<x->ilp*64;k+=width)scalar_value(expected+k,fp32,.5);}
 else if(load){
  int bytes=pair?128:64,lanes=64/width;
  for(int d=0;d<x->ilp;d++){
   int last=31-((31-d)%x->ilp);size_t address=(((loops-1)*32+(size_t)last)*(size_t)bytes)%RING;
   for(int v=0;v<(pair?2:1);v++)for(int lane=0;lane<lanes;lane++)
    memcpy(expected+RING+(d*(pair?2:1)+v)*64+lane*width,arena+128+address+(lane*(pair?2:1)+v)*width,width);
  }
 }
}
static int verify_instruction(void){
 for(int i=0;i<128;i++)if(arena[i]!=0xa5 || arena[128+PAYLOAD+i]!=0xa5)return 0;
 return !memcmp(arena+128,expected,PAYLOAD);
}

int main(int argc,char **argv){
 const size_t nk=sizeof(items)/sizeof(items[0]), nm=24,nl=32,total=nk+nm+nl;
 if(argc==2&&!strcmp(argv[1],"--list")){printf("{\"kernels\":%zu,\"memory\":%zu,\"layout\":%zu,\"cases\":%zu}\n",nk,nm,nl,total);return 0;}

#ifdef CAL_SCALAR
 if(argc!=2||strcmp(argv[1],"--selfcheck"))fail("scalar_capture_forbidden");
 for(size_t id=0;id<nk;id++){const struct item *x=&items[id];prepare_instruction(x,4);x->fn(4,arena+128);if(!verify_instruction())fail("scalar_model_result");}
 arena[0]^=1;if(verify_instruction())fail("prefix_not_detected");arena[0]^=1;
 arena[sizeof(arena)-1]^=1;if(verify_instruction())fail("suffix_not_detected");arena[sizeof(arena)-1]^=1;
 arena[128]^=1;if(verify_instruction())fail("payload_not_detected");arena[128]^=1;
 arena[128+RING]^=1;if(verify_instruction())fail("dump_not_detected");
 puts("SCALAR_BUFFER_CHECKS_PASS_NO_SME");return 0;
#else
 if(argc!=2||strcmp(argv[1],"--capture"))fail("arguments");guard();
#endif
 int bound_cpu=sched_getcpu();
 unsigned char *a=NULL,*b=NULL;const size_t maxbytes=256ULL*1024*1024;
 if(posix_memalign((void**)&a,128,maxbytes)||posix_memalign((void**)&b,128,maxbytes))fail("allocation");
 for(size_t i=0;i<maxbytes;i++){a[i]=(unsigned char)(i%251+1);b[i]=0;}
 static const size_t sizes[]={16384,65536,262144,1048576,4194304,16777216,67108864,268435456};

 for(int block=0;block<3;block++)for(size_t pos=0;pos<total;pos++){
  size_t id=(pos+block*37)%total;
  for(int sample=-1;sample<9;sample++){
   char observation[1024]="null";uint64_t t0,t1;size_t loops=sample<0?2000:200000,traffic=0,working=0;
   const char *kind,*op,*precision;int ilp=0,count=0,cols=0,batch=0;
   if(id<nk){const struct item *x=&items[id];kind="instruction";op=x->op;precision=x->precision;ilp=x->ilp;count=x->count;prepare_instruction(x,loops);t0=now();x->fn(loops,arena+128);t1=now();if(!verify_instruction())fail("instruction_result_or_guard");}
   else if(id<nk+nm){size_t k=id-nk,bytes=sizes[k/3];int mode=k%3;kind="memory";op=mode==0?"read":mode==1?"write":"copy";precision="bytes";working=bytes*(mode==2?2:1);loops=sample<0?1:((64ULL*1024*1024+bytes-1)/bytes);traffic=bytes*loops*(mode==2?2:1);t0=now();memory_op(a,b,bytes,loops,mode);t1=now();
    if(mode==1){for(size_t z=0;z<bytes;z++)if(b[z])fail("write_check");}
    if(mode==2 && memcmp(a,b,bytes))fail("copy_check");
   }else{size_t k=id-nk-nm;int blocked=k%2;batch=1<<((k/2)%4);cols=8<<((k/8)%4);size_t rows=262144/(cols*batch);kind="layout";op=blocked?"blocked_transpose":"linear_transpose";precision="complex64";working=2*262144*16;loops=sample<0?1:8;traffic=working*loops;
    struct observed v0=observe_before();
    t0=now();transpose((complex64*)a,(complex64*)b,rows,cols,batch,loops,blocked);t1=now();
    struct observed v1=observe_after();format_observation(observation,sizeof(observation),v0,v1,a,b,rows);
    for(size_t j=0;j<(size_t)cols;j++)for(size_t k=0;k<rows;k++)for(size_t z=0;z<(size_t)batch;z++)if(memcmp(b+16*(j*rows*batch+k*batch+z),a+16*(k*cols*batch+j*batch+z),16))fail("transpose_check");
   }
   if(t1<=t0)fail("nonpositive_time");if(sched_getcpu()!=bound_cpu)fail("cpu_changed");
   printf("{\"kind\":\"%s\",\"case\":%zu,\"block\":%d,\"position\":%zu,\"sample\":%d,\"operation\":\"%s\",\"precision\":\"%s\",\"ilp\":%d,\"loops\":%zu,\"instructions_per_loop\":%d,\"elapsed_ns\":%llu,\"traffic_bytes\":%zu,\"working_set_bytes\":%zu,\"columns\":%d,\"batch\":%d,\"guard_ok\":true,\"cpu\":%d,\"start_ns\":%llu,\"end_ns\":%llu,\"observation\":%s}\n",kind,id,block,pos,sample,op,precision,ilp,loops,count,(unsigned long long)(t1-t0),traffic,working,cols,batch,bound_cpu,(unsigned long long)t0,(unsigned long long)t1,observation);fflush(stdout);
  }
 }
 free(a);free(b);puts("{\"kind\":\"complete\"}");return 0;
}
