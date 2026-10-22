#include <errno.h>
#include <sys/resource.h>
struct observed {uint64_t cpu_ns;struct rusage ru;long frequency;int error;};
static long read_frequency(int *error){
 FILE *f=fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq","r");long v=-1;
 if(!f){*error=errno;return -1;}int n=fscanf(f,"%ld",&v),e=ferror(f)?EIO:EINVAL;fclose(f);
 if(n!=1||v<=0){*error=e;return -1;}*error=0;return v;
}
static uint64_t thread_stamp(void){struct timespec t;if(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t))fail("thread_clock");return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static struct observed observe_before(void){struct observed v;v.frequency=read_frequency(&v.error);if(getrusage(RUSAGE_THREAD,&v.ru))fail("rusage_before");v.cpu_ns=thread_stamp();return v;}
static struct observed observe_after(void){struct observed v;v.cpu_ns=thread_stamp();if(getrusage(RUSAGE_THREAD,&v.ru))fail("rusage_after");v.frequency=read_frequency(&v.error);return v;}
static void format_observation(char *out,size_t n,struct observed a,struct observed b,unsigned char *src,unsigned char *dst,size_t rows){
 if(b.cpu_ns<=a.cpu_ns||b.ru.ru_nvcsw<a.ru.ru_nvcsw||b.ru.ru_nivcsw<a.ru.ru_nivcsw||b.ru.ru_minflt<a.ru.ru_minflt||b.ru.ru_majflt<a.ru.ru_majflt)fail("observation_counter");
 char f0[32],f1[32];if(a.error)strcpy(f0,"null");else snprintf(f0,sizeof(f0),"%ld",a.frequency);if(b.error)strcpy(f1,"null");else snprintf(f1,sizeof(f1),"%ld",b.frequency);
 int written=snprintf(out,n,"{\"thread_start_ns\":%llu,\"thread_end_ns\":%llu,\"thread_cpu_ns\":%llu,\"freq_before_khz\":%s,\"freq_after_khz\":%s,\"freq_before_errno\":%d,\"freq_after_errno\":%d,\"nvcsw\":%ld,\"nivcsw\":%ld,\"minflt\":%ld,\"majflt\":%ld,\"source_address\":%llu,\"destination_address\":%llu,\"rows\":%zu}",(unsigned long long)a.cpu_ns,(unsigned long long)b.cpu_ns,(unsigned long long)(b.cpu_ns-a.cpu_ns),f0,f1,a.error,b.error,b.ru.ru_nvcsw-a.ru.ru_nvcsw,b.ru.ru_nivcsw-a.ru.ru_nivcsw,b.ru.ru_minflt-a.ru.ru_minflt,b.ru.ru_majflt-a.ru.ru_majflt,(unsigned long long)(uintptr_t)src,(unsigned long long)(uintptr_t)dst,rows);
 if(written<0||(size_t)written>=n)fail("observation_truncated");
}
