"""Move unchanged upstream summary/bootstrap/CV code to a login-only mode."""
import hashlib
from pathlib import Path


HELPERS = r'''
static std::string captured_machine, captured_os, captured_node;
static std::string audited_sysstr(const char *key) {
  if(captured_machine.empty()) return sysstr(key);
  return !std::strcmp(key,"hw.model") ? captured_machine : captured_os;
}
static std::string audited_hostname() {
  return captured_node.empty() ? linux_hostname() : captured_node;
}
static void write_json(const std::string& path,const std::vector<Result>& results,
                       bool formal,double load[3],double ending_load[3]) {
  std::ofstream o(path);
  o.exceptions(std::ios::badbit|std::ios::failbit);
  o << std::setprecision(17) << "{\"schema_version\":3,\"evidence_class\":\""
    << MOFFT_EVIDENCE_CLASS << "\",\"raw_only\":true,\"formal_run\":"
    << (formal?"true":"false") << ",\"node\":\"" << linux_hostname()
    << "\",\"machine\":\"" << sysstr("hw.model") << "\",\"os\":\"" << sysstr("kern.osproductversion")
    << "\",\"wisdom_formal\":" << (mofft_wisdom_is_formal()?"true":"false")
    << ",\"load_average\":[" << load[0] << "," << load[1] << "," << load[2]
    << "],\"ending_load_average\":[" << ending_load[0] << "," << ending_load[1] << "," << ending_load[2]
    << "],\"results\":[";
  for(size_t i=0;i<results.size();++i) {
    const auto& r=results[i];if(i)o << ',';
    o << "{\"precision\":\"" << r.precision << "\",\"implementation\":\"" << r.implementation
      << "\",\"n\":" << r.n << ",\"warmup_executions\":" << r.warmup_executions
      << ",\"warmup_seconds\":" << r.warmup_seconds << ",\"repetitions_per_sample\":" << r.repetitions
      << ",\"raw_seconds\":[";
    for(size_t k=0;k<r.seconds.size();++k) {if(k)o << ',';o << r.seconds[k];}
    o << "]}";
  }
  o << "]}\n";o.flush();
}
static void write_csv(const std::string&,const std::vector<Result>&) {
  // Original CSV and all descriptive statistics are produced on login.
}
'''


LOGIN_MAIN = r'''
int main(int argc,char **argv) {
  if(argc==4 && !std::strcmp(argv[1],"--login-statistics")) {
    if(linux_hostname().rfind("login",0)!=0 || std::getenv("CCS_JOB_ID")) {
      std::cerr << "STATISTICS_REQUIRES_LOGIN_NODE\n";return 2;
    }
    std::ifstream in(argv[2]);
    std::getline(in,captured_machine);std::getline(in,captured_os);std::getline(in,captured_node);
    if(captured_machine.empty()||captured_os.empty()||captured_node.rfind("cn",0)!=0)return 2;
    double load[3],ending_load[3];size_t count;
    for(double& x:load)in>>x;
    for(double& x:ending_load)in>>x;
    in>>count;
    if(count!=108)return 2;
    std::vector<Result> results;
    for(size_t i=0;i<count;++i) {
      std::string precision,implementation;size_t n,repetitions,warmups,batches;double seconds;
      in>>precision>>implementation>>n>>repetitions>>warmups>>seconds>>batches;
      if(!in||batches!=30||!n||!repetitions)return 2;
      Result r{precision,implementation,n,repetitions,{},warmups,seconds};
      for(size_t k=0;k<batches;++k) {
        double v;in>>v;if(!in||!std::isfinite(v)||v<=0)return 2;r.seconds.push_back(v);
      }
      results.push_back(std::move(r));
    }
    const std::string output=argv[3];
    if(std::filesystem::exists(output)||std::filesystem::exists(
         std::filesystem::path(output).replace_extension(".csv")))return 2;
    frozen_write_json(output,results,true,load,ending_load);
    frozen_write_csv(output,results);
    const bool formal=true;
    QUALITY_GATE
    return 0;
  }
  return native_main(argc,argv);
}
'''


def derive(path):
    path = Path(path)
    text = path.read_text()
    original = text
    anchor = "static void write_json("
    assert text.count(anchor) == 1
    text = text.replace(anchor, "static void frozen_write_json(", 1)
    anchor = "static void write_csv("
    assert text.count(anchor) == 1
    text = text.replace(anchor, "static void frozen_write_csv(", 1)
    anchor = "int main(int argc, char **argv) {"
    assert text.count(anchor) == 1
    text = text.replace(anchor, "static int native_main(int argc, char **argv) {", 1)
    start = text.index("  if (formal) {\n    // The benchmark itself raises")
    end = text.index("  // Compact per-point GFLOPS", start)
    quality = text[start:end]
    assert "coefficient_of_variation(result.seconds) > .05" in quality
    # Keep native sampling completely unchanged; only post-capture work moves.
    text = text[:start] + '  std::cout << "RAW_CAPTURE_REQUIRES_LOGIN_STATISTICS\\n";\n  return 0;\n}\n'
    text = text.replace('sysstr("hw.model")', 'audited_sysstr("hw.model")')
    text = text.replace('sysstr("kern.osproductversion")', 'audited_sysstr("kern.osproductversion")')
    text = text.replace("<< linux_hostname()", "<< audited_hostname()")
    i = text.index("static void frozen_write_json(")
    text = text[:i] + HELPERS + "\n" + text[i:]
    text += LOGIN_MAIN.replace("QUALITY_GATE", quality)
    preserved_functions = {}
    import re
    for name in ("run_pair_f32", "run_pair_f64", "bootstrap_ci", "quantile"):
        pattern = r"^static [^\n]*\b" + name + r"\(.*?\n\}"
        a = re.search(pattern, original, re.M | re.S)
        b = re.search(pattern, text, re.M | re.S)
        assert a and b and a[0] == b[0], ("ORIGINAL_FUNCTION_CHANGED", name)
        preserved_functions[name] = hashlib.sha256(a[0].encode()).hexdigest()
    for invariant in (
        "std::mt19937_64 rng(0x4d6f464654ULL + x.size());",
        "std::vector<double> bs(2000)",
        "FFTW_FORWARD, FFTW_MEASURE",
        "static constexpr double target_batch_seconds = .002;",
    ):
        assert invariant in text
    path.write_text(text)
    return dict(
        input_sha256=hashlib.sha256(original.encode()).hexdigest(),
        output_sha256=hashlib.sha256(text.encode()).hexdigest(),
        execute_timing_unchanged=True, original_statistics_and_cv_gate_moved_to_login=True,
        upstream_median_bootstrap_seed_and_2000_repetitions_unchanged=True,
        preserved_sampling_and_statistics_functions=preserved_functions,
        raw_capture_precision=17)
