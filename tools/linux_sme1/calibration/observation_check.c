#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static void fail(const char *s){fprintf(stderr,"OBSERVATION_CHECK:%s\n",s);exit(2);}
#include "observation.h"
int main(void){
 char out[1024];struct observed a=observe_before(),b;
 volatile unsigned long value=0;for(unsigned long k=0;k<10000;k++)value+=k;
 b=observe_after();format_observation(out,sizeof(out),a,b,(unsigned char*)1048576,(unsigned char*)16777216,512);puts(out);
 a.error=2;b.error=2;format_observation(out,sizeof(out),a,b,(unsigned char*)1048576,(unsigned char*)16777216,512);puts(out);
 puts("OBSERVATION_CHECK_PASS");return value==0;
}
