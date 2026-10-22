"""Derive an independent Linux harness; do not edit the contribution checkout."""
import argparse
import hashlib
import json
from pathlib import Path


def derive(repo, output):
    repo, output = Path(repo), Path(output)
    output.mkdir(exist_ok=False)
    source = repo / 'tools/benchmark.cpp'
    original = source.read_text(encoding='utf-8')
    text = original
    edits = []

    def change(label, old, new, count=1):
        nonlocal text
        actual = text.count(old)
        if actual != count:
            raise RuntimeError(f'SOURCE_ANCHOR_CHANGED:{label}:{actual}!={count}')
        text = text.replace(old, new)
        edits.append(dict(label=label, occurrences=count, old=old, new=new))

    change('linux_headers', '#include <pthread/qos.h>',
           '#include "linux_platform.h"\n#include <limits>\n#include <filesystem>')
    change('remove_apple_sysctl', '#include <sys/sysctl.h>\n', '')
    start = text.index('static std::string sysstr(')
    end = text.index('template <class T> struct Api;', start)
    change('platform_identity_functions', text[start:end], '')
    change('runtime_platform_admission',
           '  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);',
           '  if (!linux_capture_admission()) return 2;')
    change('correctness_also_requires_target_admission',
           '  if (correctness)\n    return run_overall_correctness(sizes, precision);',
           '  if (correctness) {\n    if (!linux_capture_admission()) return 2;\n'
           '    return run_overall_correctness(sizes, precision);\n  }')
    change('honest_platform_metadata',
           '\\"qos\\":\\"user-interactive\\"',
           '\\"qos\\":\\"not_applied_on_linux\\"')
    # Add evidence identity before the existing JSON fields, retaining the
    # upstream sample, median, bootstrap and GFLOP/s calculations below.
    change('explicit_evidence_identity',
           '  o << std::setprecision(12) << "{\\n  \\"schema_version\\":2,\\n  \\"formal_run\\":"',
           '  o << std::setprecision(12) << "{\\n  \\"evidence_class\\":\\"" << MOFFT_EVIDENCE_CLASS\n'
           '    << "\\",\\n  \\"node\\":\\"" << linux_hostname()\n'
           '    << "\\",\\n  \\"schema_version\\":2,\\n  \\"formal_run\\":"')
    change('separate_mock_implementation', '"mofft"', 'MOFFT_RESULT_TAG', 5)
    change('validate_positive_batch_count',
           '  if (precision != "fp32" && precision != "fp64" && precision != "both") {',
           '  if (batches < 1) { std::cerr << "INVALID_BATCH_COUNT\\n"; return 2; }\n'
           '  if (precision != "fp32" && precision != "fp64" && precision != "both") {')
    change('no_overwriting_output', '  if (formal && !size_text.empty()) {',
           '  if (!correctness && (std::filesystem::exists(output) ||\n'
           '      std::filesystem::exists(std::filesystem::path(output).replace_extension(".csv")))) {\n'
           '    std::cerr << "OUTPUT_ALREADY_EXISTS\\n"; return 2;\n  }\n'
           '  if (formal && !size_text.empty()) {')
    # A NaN can be silently ignored by std::max. Check it before reduction.
    names = ['check_overall_roundtrip_f32', 'check_overall_roundtrip_f64',
             'check_overall_fftw_f32', 'check_overall_fftw_f64']
    for name in names:
        start = text.index('static int ' + name + '(')
        end = text.index('\n}', start) + 2
        old = text[start:end]
        output_name = 'recovered' if 'roundtrip' in name else 'output'
        predicate = (f'!std::isfinite({output_name}[i].real) || '
                     f'!std::isfinite({output_name}[i].imag)')
        if 'fftw' in name:
            predicate += ' || !std::isfinite(reference_output[i][0]) || !std::isfinite(reference_output[i][1])'
        needle = '    for (size_t i = 0; i < n; ++i) {\n'
        assert old.count(needle) == 1
        new = old.replace(needle, needle + f'      if ({predicate}) {{\n'
                          '        max_error = std::numeric_limits<double>::infinity(); break;\n      }\n')
        change('nonfinite_rejection_' + name, old, new)
    change('check_pair_allocations',
           '  auto fp = fftwf_plan_dft_1d',
           '  if (!fi || !fo) throw std::runtime_error("FFTW allocation failed");\n  auto fp = fftwf_plan_dft_1d')
    change('check_pair_allocations_f64',
           '  auto fp = fftw_plan_dft_1d',
           '  if (!fi || !fo) throw std::runtime_error("FFTW allocation failed");\n  auto fp = fftw_plan_dft_1d')
    for prefix in ('fftwf', 'fftw'):
        needle = f'  auto fp = {prefix}_plan_dft_1d((int)n, fi, fo, FFTW_FORWARD, FFTW_MEASURE);'
        change('check_pair_plan_' + prefix, needle,
               needle + '\n  if (!fp) throw std::runtime_error("FFTW plan failed");')
    helper = ('static void require_success(mofft_status status) {\n'
              '  if (status != MOFFT_SUCCESS) throw std::runtime_error(mofft_status_string(status));\n}\n\n')
    change('status_helper', 'using clock_type = std::chrono::steady_clock;',
           helper + 'using clock_type = std::chrono::steady_clock;')
    for precision in ('f32', 'f64'):
        call = f'mofft_execute_{precision}(mp, mi.data(), mo.data());'
        # Exclude the warmup assignment; check standalone timed executions.
        for indent, count in [('  ', 1), ('      ', 1)]:
            old = '\n' + indent + call
            change('timed_status_' + precision + '_' + str(len(indent)), old,
                   '\n' + indent + 'require_success(' + call[:-1] + ');', count)
    change('preserve_all_sample_output_before_quality_check',
           '  if (formal) {\n    // The benchmark itself raises',
           '  write_json(output, results, formal, load, ending_load);\n'
           '  write_csv(output, results);\n'
           '  if (formal) {\n    // The benchmark itself raises')
    change('do_not_write_output_twice',
           '  write_json(output, results, formal, load, ending_load);\n'
           '  write_csv(output, results);\n  // Compact',
           '  // Compact')
    change('quality_failure_is_not_deletion', 'Discarding formal round: unstable batch CV for ',
           'RETAINED_SAMPLES_QUALITY_REJECT: unstable batch CV for ')
    change('output_stream_errors_are_fatal', '  std::ofstream o(path);',
           '  std::ofstream o(path);\n  o.exceptions(std::ios::badbit | std::ios::failbit);', 2)
    change('preserve_completed_cells_after_capture_failure',
           '  } catch (const std::exception &e) {\n    std::cerr << e.what() << "\\n";\n    return 1;\n  }\n  double ending_load[3] = {};',
           '  } catch (const std::exception &e) {\n    std::cerr << e.what() << "\\n";\n'
           '    double failure_load[3] = {}; getloadavg(failure_load, 3);\n'
           '    write_json(output, results, false, load, failure_load);\n'
           '    write_csv(output, results);\n    return 1;\n  }\n  double ending_load[3] = {};')
    for invariant in ('static constexpr double target_batch_seconds = .002;',
                      'static constexpr double target_warmup_seconds = .250;',
                      'std::mt19937_64 rng(0x4d6f464654ULL + x.size());',
                      'double gflops = 5.0 * r.n * std::log2((double)r.n) / m / 1e9',
                      'return 256.0 * epsilon *', 'FFTW_FORWARD, FFTW_MEASURE'):
        assert invariant in text
    target = output / 'benchmark_linux.cpp'
    target.write_text(text, encoding='utf-8', newline='\n')
    from policy_patch import transform
    revised = output / 'benchmark_linux_load_warning.cpp'
    policy = transform(target, revised, 'benchmark')
    target.write_bytes(revised.read_bytes())
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  upstream_commit='006dd058570b460c22c0e6b43b57705a2615422b',
                  changes=edits, original_library_changed=False, target_execution=False)
    (output / 'derivation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = derive(a.repo, a.output)
    print(json.dumps(dict(status='DERIVATION_COMPLETE',changes=len(r['changes']),output_sha256=r['output_sha256'])))
