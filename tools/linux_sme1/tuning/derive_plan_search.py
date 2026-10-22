"""Strict independent Linux derivation of the original complete-plan search."""
import argparse
import hashlib
import json
from pathlib import Path


def derive(repo, output):
    repo,output=Path(repo),Path(output);output.mkdir(exist_ok=False)
    source=repo/'tools/plan_search.cpp';text=source.read_text(encoding='utf-8');edits=[]
    def change(label,old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError(f'ANCHOR_CHANGED:{label}:{text.count(old)}!={count}')
        text=text.replace(old,new);edits.append(dict(label=label,count=count,old=old,new=new))
    change('linux_headers','#include <pthread/qos.h>','#include "linux_platform.h"\n#include <filesystem>')
    change('remove_sysctl','#include <sys/sysctl.h>\n','')
    a=text.index('static std::string sysstr(');b=text.index('static std::string json_string_field(',a)
    change('linux_identity',text[a:b],'')
    a=text.index('using Clock =');b=text.index('struct Candidate',a)
    helpers='''static std::ofstream raw_journal;
static void journal(size_t length,int precision,size_t index,const char *phase,
                    size_t repetitions,double seconds) {
  if(!std::isfinite(seconds)||seconds<=0) throw std::runtime_error("INVALID_CLOCK_SAMPLE");
  raw_journal << std::setprecision(17) << "{\\"kind\\":\\"clock\\",\\"length\\":" << length
    << ",\\"precision_bits\\":" << precision << ",\\"candidate_index\\":" << index
    << ",\\"phase\\":\\"" << phase << "\\",\\"repetitions\\":" << repetitions
    << ",\\"seconds\\":" << seconds << "}\\n";
  raw_journal.flush();
  if(!raw_journal) throw std::runtime_error("RAW_JOURNAL_WRITE_FAILED");
}
static void journal_plan(size_t length,int precision,size_t index,const char *phase,
                         const std::vector<int>& factors,uint64_t mask,bool direct,bool section) {
  raw_journal << "{\\"kind\\":\\"plan\\",\\"length\\":" << length << ",\\"precision_bits\\":" << precision
    << ",\\"candidate_index\\":" << index << ",\\"phase\\":\\"" << phase << "\\",\\"radices\\":[";
  for(size_t i=0;i<factors.size();++i) { if(i) raw_journal << ',';raw_journal << factors[i]; }
  raw_journal << "],\\"blocked_transpose_mask\\":" << mask << ",\\"direct_input\\":" << (direct?"true":"false")
    << ",\\"stage_layout\\":\\"" << (section?"section":"transpose") << "\\"}\\n";
  raw_journal.flush();if(!raw_journal) throw std::runtime_error("PLAN_DESCRIPTOR_WRITE_FAILED");
}
'''
    change('raw_journal_helpers',text[a:b],text[a:b]+helpers)
    change('linux_target_admission',
           '  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);',
           '''  if (compact) { std::cerr << "COMPACT_WOULD_DROP_RAW_SAMPLES\\n"; return 2; }
#ifdef MOFFT_LOGIN_MOCK
  if(formal) { std::cerr << "MOCK_CANNOT_PRODUCE_FORMAL_WISDOM\\n";return 2; }
#endif
  if(std::filesystem::exists(output)||std::filesystem::exists(output+".raw.jsonl")) {
    std::cerr << "OUTPUT_ALREADY_EXISTS\\n";return 2;
  }
  if(!linux_capture_admission()) return 2;
  raw_journal.open(output+".raw.jsonl");
  if(!raw_journal) { std::cerr << "RAW_JOURNAL_OPEN_FAILED\\n";return 1; }''')
    # Keep all plan factors, layouts, warmup counts and alternating search order.
    change('api_status_checked','Api<Complex>::run(', 'run_checked<Complex>(',6)
    marker='template <class Complex>\nstatic void measure_finalists'
    helper='''template <class Complex>
static void run_checked(typename Api<Complex>::Plan *p,const Complex *in,Complex *out) {
  if(Api<Complex>::run(p,in,out)!=MOFFT_SUCCESS) throw std::runtime_error("EXECUTE_STATUS_FAILED");
}
template <class Complex>
static void finite_output(const std::vector<Complex>& output) {
  for(const auto& x:output) if(!std::isfinite(x.real)||!std::isfinite(x.imag))
    throw std::runtime_error("NONFINITE_PLAN_OUTPUT");
}
'''
    change('checked_api_helper',marker,helper+marker)
    change('finalist_stable_identity_before_measurement',
        '    mofft_status status = Api<Complex>::create(\n        &plans[index], length, finalists[index]->factors);',
        '''    journal_plan(length,sizeof(Complex)==8?32:64,index,steady_state?"steady_finalist":"finalist",
        finalists[index]->factors,finalists[index]->blocked_transpose_mask,
        finalists[index]->direct_input,finalists[index]->section_layout);
    mofft_status status = Api<Complex>::create(
        &plans[index], length, finalists[index]->factors);''')
    change('coarse_stable_identity_before_measurement','      Candidate &candidate = candidates[index];',
        '''      Candidate &candidate = candidates[index];
      if(sample==0) journal_plan(length,sizeof(Complex)==8?32:64,index,"coarse",candidate.factors,
          candidate.blocked_transpose_mask,candidate.direct_input,candidate.section_layout);''')
    change('nonempty_median','  std::sort(values.begin(), values.end());',
           '  if(values.empty()) throw std::runtime_error("EMPTY_MEDIAN");\n  std::sort(values.begin(), values.end());')
    needle='double pilot = std::chrono::duration<double>(stop - start).count();'
    change('positive_pilot',needle,needle+'\n    if(!std::isfinite(pilot)||pilot<=0) throw std::runtime_error("INVALID_PILOT");',2)
    needle='    return std::chrono::duration<double>(stop - start).count() /\n           repetitions[index];'
    change('finalist_raw_journal',needle,'''    double value=std::chrono::duration<double>(stop-start).count()/repetitions[index];
    finite_output(output);
    journal(length,sizeof(Complex)==8?32:64,index,steady_state?"steady_finalist":"finalist",repetitions[index],value);
    return value;''')
    needle='      candidate.samples.push_back(\n          std::chrono::duration<double>(stop - start).count() /\n          repetitions[index]);'
    change('coarse_raw_journal',needle,'''      double value=std::chrono::duration<double>(stop-start).count()/repetitions[index];
      finite_output(output);
      journal(length,sizeof(Complex)==8?32:64,index,"coarse",repetitions[index],value);
      candidate.samples.push_back(value);''')
    # Preserve evidence first; quality failure still returns the original code 2.
    a=text.index('  if (formal) {\n    for (const auto &result : results)')
    b=text.index('  std::string manifest_text;',a)
    old=text[a:b];new=old.replace('  if (formal) {','  bool quality_rejected=false;\n  if (formal) {',1)
    new=new.replace('        return 2;','        quality_rejected=true;')
    new=new.replace('Discarding formal plan search:', 'Retaining rejected formal plan search:')
    change('retain_samples_before_quality_failure',old,new)
    change('explicit_evidence_identity',
        '      << "{\\n  \\"schema_version\\":1,\\n  \\"formal_measurement\\":"',
        '      << "{\\n  \\"evidence_class\\":\\"" << MOFFT_EVIDENCE_CLASS\n'
        '      << "\\",\\n  \\"node\\":\\"" << linux_hostname()\n'
        '      << "\\",\\n  \\"quality_rejected\\":" << (quality_rejected?"true":"false")\n'
        '      << ",\\n  \\"schema_version\\":1,\\n  \\"formal_measurement\\":"')
    needle='      out << "]}";'
    change('keep_relative_samples',needle,'''      out << "],\\"relative_samples\\":[";
      for(size_t j=0;j<candidate.relative_samples.size();++j) {
        if(j) out << ',';
        out << candidate.relative_samples[j];
      }
      out << "]}";''')
    change('checked_json_write','  { std::ofstream file(output); file << out.str(); }',
        '  { std::ofstream file(output); file << out.str(); file.flush();\n'
        '    if(!file) { std::cerr << "WISDOM_WRITE_FAILED\\n"; return 1; } }')
    # Only the final main return; algorithm returns elsewhere remain identical.
    assert text.endswith('  return 0;\n}\n')
    text=text[:-len('  return 0;\n}\n')]+'  return quality_rejected ? 2 : 0;\n}\n'
    edits.append(dict(label='quality_failure_exit_after_write',count=1))
    (output/'plan_search_linux.cpp').write_text(text,encoding='utf-8')
    from policy_patch import transform
    source_path = output / 'plan_search_linux.cpp'
    revised = output / 'plan_search_linux_load_warning.cpp'
    policy = transform(source_path, revised, 'search')
    source_path.write_bytes(revised.read_bytes())
    text = source_path.read_text(encoding='utf-8')
    (output/'derivation.json').write_text(json.dumps(dict(source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),edits=edits,
        derived_sha256=hashlib.sha256(text.encode()).hexdigest(),
        target_execution=False,formal_wisdom_produced=False),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('repo',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();derive(a.repo,a.output)
