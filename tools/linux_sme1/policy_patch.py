"""Independent Linux load observation policy; never edits upstream sources."""
import hashlib
from pathlib import Path


ORIGINAL = {
    "search": '''  double load[3] = {};
  getloadavg(load, 3);
  if (formal && load[0] > 2.0) {
    std::cerr << "Refusing formal plan search: one-minute load average is "
              << load[0] << ".\\n";
    return 2;
  }
''',
    "benchmark": '''  double load[3] = {};
  getloadavg(load, 3);
  if (formal && load[0] > 2.0) {
    std::cerr << "Refusing formal run: one-minute load average is " << load[0]
              << ", so the machine is not in an exclusive window.\\n";
    return 2;
  }
''',
}

NOTICE = '''  if (formal) {
    std::cerr << "LINUX_LOAD_OBSERVATION policy=warning_only count="
              << initial_load_count << " load1=" << load[0]
              << " load5=" << load[1] << " load15=" << load[2] << "\\n";
    if (initial_load_count != 3) {
      std::cerr << "WARNING_LINUX_LOAD_UNAVAILABLE: continuing; "
                   "CPU identity, exclusivity and sample-quality checks remain required.\\n";
    } else if (load[0] > 2.0) {
      std::cerr << "WARNING_LINUX_SYSTEM_LOAD: one-minute load exceeds 2.0; "
                   "observational only, continuing under the revised Linux policy.\\n";
    }
  }
'''
REPLACEMENT = ("  double load[3] = {};\n"
               "  const int initial_load_count = getloadavg(load, 3);\n" + NOTICE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def transform(source, output, kind):
    source, output = Path(source), Path(output)
    before = source.read_bytes()
    text = before.decode("utf-8")
    original = ORIGINAL[kind]
    if text.count(original) != 1:
        raise ValueError("LOAD_POLICY_ANCHOR_CHANGED:" + kind)
    if "linux_capture_admission()" not in text:
        raise ValueError("LINUX_CAPTURE_ADMISSION_MISSING")
    after = text.replace(original, REPLACEMENT, 1)
    assert after.replace(REPLACEMENT, original, 1) == text
    assert after.count("WARNING_LINUX_SYSTEM_LOAD") == 1
    assert after.count("const int initial_load_count") == 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        f.write(after.encode("utf-8"))
    return dict(kind=kind, source=str(source), source_bytes=len(before),
                source_sha256=sha(before), output=str(output),
                output_sha256=sha(output.read_bytes()),
                only_initial_load_policy_changed=True,
                reversible_exactly=True, target_execution=False)


def guard_probe_source():
    """Compile the exact old/new branch blocks without any FFT dependency."""
    old = ORIGINAL["search"].split("  if (formal", 1)[1]
    old = "  if (formal" + old
    return ("#include <cassert>\n#include <iostream>\n"
            "int previous(double value, bool formal) {\n"
            "  double load[3] = {value, 1.0, 1.0};\n" + old +
            "  return 0;\n}\n"
            "int revised(double value, bool formal, int initial_load_count=3) {\n"
            "  double load[3] = {value, 1.0, 1.0};\n" + NOTICE +
            "  return 0;\n}\n"
            "int main() {\n"
            "  assert(previous(13.27,true)==2);\n"
            "  assert(revised(13.27,true)==0);\n"
            "  assert(revised(2.0,true)==0);\n"
            "  assert(revised(0.5,true)==0);\n"
            "  assert(revised(13.27,false)==0);\n"
            "  assert(revised(0.0,true,-1)==0);\n"
            "  std::cout << \"GUARD_CONTROL_6_CASES_PASS_NO_FFT_EXECUTION\\n\";\n"
            "}\n")
