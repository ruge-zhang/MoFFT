"""Generate isolated Linux SME1 microkernels; never executes target code."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def generate(out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    rows = []
    text = ['#include <stdint.h>', '#include <arm_sme.h>']
    for precision, lane in [('fp32', 's'), ('fp64', 'd')]:
        for op in ['add', 'fmla', 'fmopa', 'za_h', 'za_v', 'load', 'store']:
            for ilp in ([1, 2, 4] if op == 'fmopa' and precision == 'fp32' else [1, 2, 4, 8]):
                name = f'cal_{op}_{precision}_{ilp}'
                pre = ['ptrue p0.b', 'fmov z16.'+lane+', #0.5', 'fmov z17.'+lane+', #0.5']
                pre += [f'fmov z{i}.{lane}, #0.5' for i in range(8)]
                pre += ['zero {za}', 'mov w12, #0']
                ops = []
                for i in range(32):
                    d = i % ilp
                    if op == 'add': inst = f'fadd z{d}.{lane}, p0/m, z{d}.{lane}, z16.{lane}'
                    elif op == 'fmla': inst = f'fmla z{d}.{lane}, p0/m, z16.{lane}, z17.{lane}'
                    elif op == 'fmopa': inst = f'fmopa za{d}.{lane}, p0/m, p0/m, z16.{lane}, z17.{lane}'
                    elif op in ['za_h', 'za_v']:
                        inst = f'mova z{d}.{lane}, p0/m, za0{op[-1]}.{lane}[w12, 0]'
                    elif op == 'load': inst = f'ld1{ "w" if lane == "s" else "d" } {{z{d}.{lane}}}, p0/z, [%[p], #{d}, mul vl]'
                    else: inst = f'st1{ "w" if lane == "s" else "d" } {{z{d}.{lane}}}, p0, [%[p], #{d}, mul vl]'
                    ops.append(inst)
                lines = pre + ['1:'] + ops + ['subs %[n], %[n], #1', 'b.ne 1b']
                asm = '\n'.join('        '+json.dumps(x+'\n') for x in lines)
                text += [f'__attribute__((noinline,target("sme"))) __arm_new("za") __arm_locally_streaming\nvoid {name}(uint64_t n, void *p) {{',
                    '    __asm__ volatile(', asm,
                    '        : [n] "+&r"(n) : [p] "r"(p)',
                    '        : "cc", "memory", "p0", "w12", "za", '+', '.join(f'"z{i}"' for i in list(range(8))+[16,17])+');', '}']
                rows.append(dict(name=name,operation=op,precision=precision,ilp=ilp,instructions_per_loop=32,loop_instructions=ops))
    text += ['__attribute__((noinline,target("sme"))) __arm_new("za") __arm_locally_streaming',
             'void cal_transition_once(void) { __asm__ volatile("zero {za}" ::: "za", "memory"); }',
             '__attribute__((noinline)) void cal_transition(uint64_t n, void *p) { (void)p; while(n--) cal_transition_once(); }',
             '__attribute__((noinline)) void cal_empty(uint64_t n, void *p) { (void)p; __asm__ volatile("1: subs %0, %0, #1\\n b.ne 1b" : "+r"(n) :: "cc", "memory"); }']
    rows += [dict(name='cal_transition',operation='transition',precision='mode',ilp=1,instructions_per_loop=1),dict(name='cal_empty',operation='empty',precision='mode',ilp=1,instructions_per_loop=1)]
    (out/'kernels.c').write_text('\n'.join(text)+'\n',encoding='utf-8')
    declarations=['#include <stdint.h>', 'typedef void (*cal_fn)(uint64_t,void*);']
    declarations += [f'void {x["name"]}(uint64_t,void*);' for x in rows]
    declarations += ['struct item { const char *name, *op, *precision; int ilp, count; cal_fn fn; };', 'static const struct item items[] = {']
    declarations += [f'{{"{x["name"]}","{x["operation"]}","{x["precision"]}",{x["ilp"]},{x["instructions_per_loop"]},{x["name"]}}},' for x in rows]
    declarations += ['};']
    (out/'kernels.h').write_text('\n'.join(declarations)+'\n',encoding='utf-8')
    (out/'kernel_manifest.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    return rows

if __name__ == '__main__':
    import sys
    print(json.dumps(dict(kernel_count=len(generate(sys.argv[1])))))
