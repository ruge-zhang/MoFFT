/* Runs before any generated SME candidate or SVCR read. */
#include <sched.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <unistd.h>
static int candidate_admission(void) {
  char host[256]={0}; const char *job=getenv("CCS_JOB_ID");
  cpu_set_t cpus; CPU_ZERO(&cpus);
  if(gethostname(host,sizeof(host)-1)||strncmp(host,"cn",2)||!job||!*job||
     strspn(job,"0123456789")!=strlen(job)) {
    fputs("CANDIDATES_REQUIRE_COMPUTE_JOB_AND_CN_HOST\n",stderr);return 0;
  }
  if(sched_getaffinity(0,sizeof(cpus),&cpus)||CPU_COUNT(&cpus)!=1||!CPU_ISSET(0,&cpus)) {
    fputs("CANDIDATES_REQUIRE_CPU0_SINGLE_AFFINITY\n",stderr);return 0;
  }
  unsigned long required=(1UL<<23)|(1UL<<25)|(1UL<<29);
  long sve=prctl(51,0,0,0,0),sme=prctl(64,0,0,0,0);
  if(!(getauxval(AT_HWCAP)&(1UL<<22))||(getauxval(AT_HWCAP2)&required)!=required||
     sve<0||sme<0||(sve&65535)!=64||(sme&65535)!=64) {
    fputs("CANDIDATES_REQUIRE_SME_FP64_AND_64_BYTE_VL\n",stderr);return 0;
  }
  return 1;
}
