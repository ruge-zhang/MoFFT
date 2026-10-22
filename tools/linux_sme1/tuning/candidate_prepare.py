"""Independent SME1 candidate corpus. Never execute or edit upstream sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def prepare(repo, profile, output, selected_radices=None):
    repo, profile, output = map(Path, (repo, profile, output))
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(repo/'compiler'))
    from mofft_compiler.backends import backend_for_architecture
    from mofft_compiler.emitter import RADICES, _compiler_digest, emit_candidate_source, profile_digest
    from mofft_compiler.patterns import enumerate_candidates
    from mofft_compiler.profile import MachineProfile
    from mofft_compiler.validate import _harness, _name, _wrapper
    model = MachineProfile.load(profile)
    if 'sme2' in model.architecture.lower():
        raise ValueError('SME1_REQUIRES_EXPLICIT_SME1_PROFILE')
    target = backend_for_architecture(model.architecture).target(model, sme1=True)
    rows, cases, sources = [], [], {}
    for radix in selected_radices or RADICES:
        if radix not in RADICES:
            raise ValueError('UNSUPPORTED_RADIX')
        chunks = []
        for precision in ('fp32', 'fp64'):
            for direction in ('forward', 'backward'):
                for stage in ('first', 'other'):
                    for c in enumerate_candidates(radix, precision, stage, direction, 'auto', target):
                        contexts = ('normal', 'transposed') if stage == 'first' else (
                            ('direct_broadcast',) if c.batch_pipeline_depth == 2 else
                            ('normal', 'broadcast', 'direct_broadcast'))
                        for context in contexts:
                            name = _name(c, len(cases), context)
                            cases.append((c, name, context))
                            code = emit_candidate_source(c, name, context)+_wrapper(c, name, context)
                            # Public non-streaming/private-ZA entry, typed direct call.
                            # No cast between incompatible streaming function types.
                            code += f'''
__attribute__((target("sme")))
__arm_new("za") __arm_locally_streaming
void {name}_entry(const void *input, size_t row_stride,
                 const void *twiddles, size_t twiddle_stride,
                 void *output, size_t batch) {{
  {name}_wrapper(input,row_stride,twiddles,twiddle_stride,output,batch);
}}
'''
                            chunks.append(code)
                            rows.append(dict(name=name,radix=radix,precision=precision,
                                direction=direction,stage=stage,context=context,
                                pattern=c.pattern.value,matrix_vg_width=c.matrix_vg_width,
                                rotate_temp_tiles=c.rotate_temp_tiles,
                                batch_pipeline_depth=c.batch_pipeline_depth,
                                grouped_vector_ops=c.target.grouped_vector_ops))
        sources[f'candidates_r{radix}.c'] = '\n'.join(chunks)
    harness = _harness(cases)
    old = 'extern void mofft_candidate_entry(void *, const void *, size_t, const void *,\n                                  size_t, void *, size_t);'
    assert harness.count(old) == 1
    decls = '\n'.join(f'extern void {n}_entry(const void *,size_t,const void *,size_t,void *,size_t);' for _,n,_ in cases)
    guard = (Path(__file__).parent/'candidate_guard.h').read_text()
    harness = harness.replace(old, guard+'\n'+decls+'''
typedef void (*candidate_entry_t)(const void *,size_t,const void *,size_t,void *,size_t);
static void mofft_candidate_entry(candidate_entry_t body,const void *in,size_t rs,
                                  const void *tw,size_t ts,void *out,size_t n) {
  unsigned long before,after;
  __asm__ volatile("mrs %0, S3_3_C4_C2_2" : "=r"(before));
  body(in,rs,tw,ts,out,n);
  __asm__ volatile("mrs %0, S3_3_C4_C2_2" : "=r"(after));
  if ((before & 3) != (after & 3)) { fputs("SVCR_LIFECYCLE_MISMATCH\\n",stderr); exit(1); }
}
''')
    harness = harness.replace('void *body;', 'candidate_entry_t body;')
    for _,name,_ in cases:
        harness = harness.replace('(void *)'+name+'_wrapper', name+'_entry')
    needle = '  int failures=0;'
    assert harness.count(needle) == 1
    harness = harness.replace(needle, '  if (!candidate_admission()) return 2;\n'+needle)
    needle='double e=hypot(out[q].real-rr,out[q].imag-ii); if(e>error) error=e;'
    assert harness.count(needle) == 2
    harness = harness.replace(needle, 'double e=hypot(out[q].real-rr,out[q].imag-ii); '
        'if(!isfinite(out[q].real)||!isfinite(out[q].imag)||!isfinite(e)) error=INFINITY; '
        'else if(e>error) error=e;')
    needle='  for(size_t q=0;q<count;++q)'
    assert harness.count(needle)==2
    harness=harness.replace(needle, '  if(!in||!tw||!out) { fputs("ALLOCATION_FAILED\\n",stderr); exit(1); }\n'+needle)
    needle='  qsort(samples,rounds,sizeof(double),compare_double);'
    assert harness.count(needle)==2
    harness=harness.replace(needle, '''  printf("raw_rounds,%s,%zu",tc->name,repeat);
  for(int q=0;q<rounds;++q) {
    if(!isfinite(samples[q])||samples[q]<=0) { fputs("INVALID_CLOCK_SAMPLE\\n",stderr); exit(1); }
    printf(",%.17g",samples[q]);
  }
  printf("\\n"); fflush(stdout);
'''+needle)
    for name, code in {**sources,'candidate_harness.c':harness}.items():
        (output/name).write_text(code, encoding='utf-8')
    manifest=dict(upstream_compiler_sha256=_compiler_digest(),profile_sha256=profile_digest(model),
        target='arm-sme1-port',streaming_vector_bits=512,cases=rows,
        original_harness_sha256=hashlib.sha256((repo/'compiler/mofft_compiler/validate.py').read_bytes()).hexdigest(),
        numeric_limits='original 64*FLT_EPSILON*radix / 192*DBL_EPSILON*radix; nonfinite rejected',
        actual_target_execution=False,kernel_wisdom_produced=False,
        source_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in sorted(output.glob('*.c'))])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo',type=Path);p.add_argument('profile',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();m=prepare(a.repo,a.profile,a.output)
    print(json.dumps(dict(cases=len(m['cases']),files=len(m['source_files']),target=m['target'])))
