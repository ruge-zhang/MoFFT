"""Login-only scalar instruction simulator for buffer/checker validation."""
import json
from pathlib import Path
import sys


def main(root):
    root = Path(root)
    rows = json.loads((root/'kernel_manifest.json').read_text())
    code = ['#include <stdint.h>', '#include <string.h>', '#include <stddef.h>', r'''
static void simulate(uint64_t n,void *vp,int bits,int op,int streams){
 unsigned char *p=vp,regs[16][64];int width=bits/8,lanes=64/width;
 int pair=op==2||op==4,load=op==1||op==2;
 if(op==5){for(int i=0;i<streams*64;i+=width){if(bits==32){float v=.5f;memcpy(p+i,&v,4);}else {double v=.5;memcpy(p+i,&v,8);}}return;}
 if(!op)return;
 memset(regs,0,sizeof(regs));size_t cursor=0;
 for(uint64_t loop=0;loop<n;loop++)for(int i=0;i<32;i++){
  int reg=(i%streams)*(pair?2:1);
  for(int lane=0;lane<lanes;lane++)for(int component=0;component<(pair?2:1);component++){
   unsigned char *address=p+cursor+(lane*(pair?2:1)+component)*width;
   if(load)memcpy(regs[reg+component]+lane*width,address,width);
   else if(bits==32){float v=component?1.f:.5f;memcpy(address,&v,4);}
   else {double v=component?1.:.5;memcpy(address,&v,8);}
  }
  cursor=(cursor+(pair?128:64))%8192;
 }
 if(load)memcpy(p+8192,regs,streams*(pair?2:1)*64);
}
''']
    mapping={'load1_seq':1,'load2_seq':2,'store1_seq':3,'store2_seq':4,'store_repeat_diagnostic':5}
    for x in rows:
        code.append(f'void {x["name"]}(uint64_t n,void *p){{simulate(n,p,{32 if x["precision"]=="fp32" else 64},{mapping.get(x["operation"],0)},{x["ilp"]});}}')
    (root/'build/mock.c').write_text('\n'.join(code)+'\n')


if __name__=='__main__':
    main(sys.argv[1])
