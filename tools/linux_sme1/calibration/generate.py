"""Separate sequential LD1/LD2/ST1/ST2 calibration; retain old store diagnostics."""
import json
from pathlib import Path
from legacy_generate import generate as legacy_generate


def generate(root):
    root = Path(root)
    old = root / 'legacy_generated'
    old.mkdir(exist_ok=False)
    original = legacy_generate(old)
    rows = []
    for x in original:
        if x['operation'] == 'load':
            continue
        y = dict(x, calibration_role='model')
        if x['operation'] == 'store':
            y.update(operation='store_repeat_diagnostic', calibration_role='diagnostic')
        if x['operation'] == 'empty':
            y['calibration_role'] = 'diagnostic'
        rows.append(y)
    text = [(old / 'kernels.c').read_text()]
    for precision, lane in [('fp32', 's'), ('fp64', 'd')]:
        for op in ['load1_seq', 'load2_seq', 'store1_seq', 'store2_seq']:
            pair = '2_' in op
            width = 128 if pair else 64
            is_load = op.startswith('load')
            for streams in (1, 2, 4, 8):
                name = f'cal_{op}_{precision}_{streams}'
                pre = ['ptrue p0.b'] + [f'fmov z{i}.{lane}, #'+('1.0' if pair and i % 2 else '0.5') for i in range(16)]
                insts = []
                for i in range(32):
                    d = (i % streams) * (2 if pair else 1)
                    regs = f'z{d}.{lane}, z{d+1}.{lane}' if pair else f'z{d}.{lane}'
                    mnemonic = ('ld' if is_load else 'st') + ('2' if pair else '1') + ('w' if lane == 's' else 'd')
                    insts.append(f'{mnemonic} {{{regs}}}, p0'+('/z' if is_load else '')+f', [%[q], #{(i%8)*(2 if pair else 1)}, mul vl]')
                    if i % 8 == 7:
                        insts.append(f'add %[q], %[q], #{8*width}')
                loop = pre + ['1:'] + insts + ['cmp %[q], %[end]', 'csel %[q], %[begin], %[q], hs', 'subs %[n], %[n], #1', 'b.ne 1b']
                if is_load:
                    loop.append('mov %[q], %[dump]')
                    for d in range(streams * (2 if pair else 1)):
                        loop += [f'st1'+('w' if lane == 's' else 'd')+f' {{z{d}.{lane}}}, p0, [%[q]]', 'add %[q], %[q], #64']
                text += [f'__attribute__((noinline,target("sme"))) __arm_new("za") __arm_locally_streaming\nvoid {name}(uint64_t n, void *p) {{',
                         '    void *q=p;', '    __asm__ volatile(',
                         '\n'.join('        '+json.dumps(v+'\n') for v in loop),
                         '        : [n] "+&r"(n), [q] "+&r"(q)',
                         '        : [begin] "r"(p), [end] "r"((char*)p+8192), [dump] "r"((char*)p+8192)',
                         '        : "cc", "memory", "p0", '+', '.join(f'"z{i}"' for i in range(16))+');', '}']
                rows.append(dict(name=name, operation=op, precision=precision, ilp=streams,
                                 instructions_per_loop=32, loop_instructions=insts,
                                 calibration_role='model', ring_bytes=8192, payload_bytes_per_instruction=width,
                                 source_or_destination_vectors=streams*(2 if pair else 1),
                                 dump_vectors=streams*(2 if pair else 1) if is_load else 0,
                                 loop_advance_bytes=32*width))
    assert len(rows) == 81
    (root/'kernels.c').write_text('\n'.join(text)+'\n')
    header=['#include <stdint.h>', 'typedef void (*cal_fn)(uint64_t,void*);']
    header += [f'void {x["name"]}(uint64_t,void*);' for x in rows]
    header += ['struct item {const char *name,*op,*precision; int ilp,count; cal_fn fn;};', 'static const struct item items[]={']
    header += [f'{{"{x["name"]}","{x["operation"]}","{x["precision"]}",{x["ilp"]},{x["instructions_per_loop"]},{x["name"]}}},' for x in rows]
    header += ['};']
    (root/'kernels.h').write_text('\n'.join(header)+'\n')
    (root/'kernel_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__ == '__main__':
    import sys
    print(json.dumps(dict(kernels=len(generate(sys.argv[1])))))
