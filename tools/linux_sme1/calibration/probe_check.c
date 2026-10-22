#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
typedef struct {double real,imag;} complex64;
extern void transpose(complex64*,complex64*,size_t,size_t,size_t,size_t,int);
extern void memory_op(unsigned char*,unsigned char*,size_t,size_t,int);
static void require(int ok,const char *s){if(!ok){fprintf(stderr,"PROBE_CHECK:%s\n",s);exit(2);}}
static int guards(unsigned char *p,size_t n){for(size_t i=0;i<128;i++)if(p[i]!=0xa5||p[128+n+i]!=0xa5)return 0;return 1;}
static int layout_valid(unsigned char *a,unsigned char *b,size_t r,size_t c,size_t z){
 for(size_t j=0;j<c;j++)for(size_t k=0;k<r;k++)for(size_t q=0;q<z;q++)
  if(memcmp(b+16*(j*r*z+k*z+q),a+16*(k*c*z+j*z+q),16))return 0;
 return 1;
}
int main(int argc,char **argv){
 char host[256];require(!gethostname(host,sizeof(host))&&!strncmp(host,"login",5)&&!getenv("CCS_JOB_ID"),"login_only");
 require(argc==2&&!strcmp(argv[1],"--check"),"arguments");
 size_t n=256ULL*1024*1024;unsigned char *pa=NULL,*pb=NULL;
 require(!posix_memalign((void**)&pa,128,n+256)&&!posix_memalign((void**)&pb,128,n+256),"allocate");
 memset(pa,0xa5,n+256);memset(pb,0xa5,n+256);unsigned char *a=pa+128,*b=pb+128;
 for(size_t i=0;i<n;i++)a[i]=(unsigned char)(i%251+1);
 int layouts=0,memories=0;
 for(int k=0;k<32;k++){
  size_t c=8<<((k/8)%4),z=1<<((k/2)%4),r=262144/(c*z),used=262144*16;
  for(int pass_index=0;pass_index<2;pass_index++){
   size_t passes=pass_index?8:1;memset(b,0,used);memset(b+used,0xa5,128);
   transpose((complex64*)a,(complex64*)b,r,c,z,passes,k%2);
   require(layout_valid(a,b,r,c,z),"layout_output");
   for(size_t j=0;j<128;j++)require(b[used+j]==0xa5,"layout_extent");
   require(guards(pa,n)&&guards(pb,n),"layout_guards");layouts++;
  }
 }
 b[7]^=1;require(!layout_valid(a,b,512,64,8),"corruption_detection");b[7]^=1;
 pa[0]^=1;require(!guards(pa,n),"prefix_detection");pa[0]^=1;
 pb[n+128]^=1;require(!guards(pb,n),"suffix_detection");pb[n+128]^=1;
 const size_t sizes[]={16384,65536,262144,1048576,4194304,16777216,67108864,268435456};
 for(int k=0;k<24;k++){
  size_t used=sizes[k/3];int mode=k%3;
  for(int pass_index=0;pass_index<2;pass_index++){
   size_t passes=pass_index?(67108864+used-1)/used:1;memset(b,0xcc,used);
   if(used<n)memset(b+used,0xa5,128);
   memory_op(a,b,used,passes,mode);
   for(size_t j=0;j<used;j++)require(b[j]==(mode==0?0xcc:mode==1?0:a[j]),"memory_output");
   for(size_t j=0;j<n;j++)require(a[j]==(unsigned char)(j%251+1),"input_unchanged");
   if(used<n)for(size_t j=0;j<128;j++)require(b[used+j]==0xa5,"memory_extent");
   require(guards(pa,n)&&guards(pb,n),"memory_guards");memories++;
  }
 }
 printf("{\"status\":\"PRESERVED_PROBE_OUTPUT_CHECK_PASS\",\"layout_checks\":%d,\"memory_checks\":%d,\"corruption_checks\":3,\"SME_executed\":false}\n",layouts,memories);
 free(pa);free(pb);return 0;
}
