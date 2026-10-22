#!/usr/bin/env python3
"""Explicit Linux/Donau SME1 stages; no submission, retry, download or deadline."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import traceback

from common import Commands, check_manifest, file_manifest, login_only, now, put, read, sha

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SIZES = '256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,144,169,196,225,1728,2197,2744,3375,20736,28561,38416,50625,248832,371293,537824,759375'
FFTW_SHA256 = '5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1'


def compiler_paths(cc, cxx=None):
    # Keep the caller's executable name: clang may symlink to clang-N.
    cc = Path(cc).absolute()
    cxx = Path(cxx).absolute() if cxx else cc.with_name('clang++')
    if not cc.is_file() or not cxx.is_file():
        raise ValueError('EXPLICIT_C_AND_CXX_COMPILERS_REQUIRED')
    return str(cc), str(cxx)


def build_fftw(root, archive, cc, log):
    if sha(archive) != FFTW_SHA256:
        raise ValueError('FFTW_3_3_11_SOURCE_HASH_MISMATCH')
    source_root = root / 'fftw_source'
    source_root.mkdir()
    with tarfile.open(archive) as stream:
        for item in stream:
            relative = Path(item.name)
            if relative.is_absolute() or '..' in relative.parts or not (item.isdir() or item.isfile()):
                raise ValueError('UNEXPECTED_FFTW_ARCHIVE_MEMBER:'+item.name)
            path = source_root / relative
            if item.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as dest, stream.extractfile(item) as source:
                    shutil.copyfileobj(source, dest)
                path.chmod(item.mode & 0o777)
    prefix = root / 'fftw_install'
    env = dict(os.environ, CC=cc, CFLAGS='-O3 -fPIC',
               AR=str(Path(cc).with_name('llvm-ar')), RANLIB=str(Path(cc).with_name('llvm-ranlib')))
    for precision, options in [('fp64', ['--disable-neon']), ('fp32', ['--enable-float', '--enable-neon'])]:
        directory = root / ('fftw_build_'+precision)
        directory.mkdir()
        argv = [source_root/'fftw-3.3.11/configure', '--prefix='+str(prefix), '--disable-shared',
                '--enable-static', '--disable-fortran', '--disable-openmp', '--disable-mpi', *options]
        log.run('fftw_configure_'+precision, argv, cwd=directory, env=env)
        log.run('fftw_build_'+precision, ['make', '-j1'], cwd=directory, env=env)
        log.run('fftw_install_'+precision, ['make', 'install'], cwd=directory, env=env)
    put(root/'fftw_source_identity.json', dict(source_sha256=FFTW_SHA256, cc=cc,
        cflags=env['CFLAGS'], fp32_neon=True, fp64_neon=False,
        libraries=file_manifest([prefix/'lib/libfftw3.a', prefix/'lib/libfftw3f.a'])))
    return prefix


def configure_library(root, log, name, profile=None, kernel=None):
    cfg = read(root/'configuration.json')
    argv = [cfg['cmake'], '-S', root/'repo', '-B', root/name, '-G', 'Unix Makefiles',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER='+cfg['clang'],
            '-DCMAKE_CXX_COMPILER='+cfg['cxx'], '-DPython3_EXECUTABLE='+sys.executable,
            '-DMOFFT_SME1_PORT=ON', '-DMOFFT_BUILD_TOOLS=OFF', '-DMOFFT_BUILD_TESTS=OFF']
    if profile:
        argv += ['-DMOFFT_PROFILE='+str(profile), '-DMOFFT_KERNEL_WISDOM='+str(kernel)]
    log.run(name+'_configure', argv)
    log.run(name+'_build', [cfg['cmake'], '--build', root/name, '--parallel', '1'])


def write_capture_scripts(root):
    """Prepare immutable phase entry points on login; never submit them."""
    for phase in ('C0', 'SEARCH', 'CHECK', 'P'):
        argv = [sys.executable, '-B', str(root/'program/workflow.py'), 'capture', str(root), phase]
        with (root/('run_'+phase+'.sh')).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('#!/bin/bash\nset -euo pipefail\nexec '+shlex.join(argv)+'\n')


def prepare_calibration(root, cfg, log):
    cal = root/'calibration'
    (cal/'build').mkdir()
    contract = read(cal/'contract.json')
    contract.update(target=cfg['node'], preserved_probe_objects=False,
                    source_rebuild=True, reference='fresh same-job CPU identity')
    (cal/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    log.run('generate_calibration', [sys.executable, '-B', cal/'generate.py', cal])
    cc = cfg['clang']
    common = [cc, '-std=gnu11', '-O3', '-fno-lto']
    log.run('compile_calibration_kernels', [*common, '-march=armv9.2-a+sme+sme-f64f64',
        '-c', cal/'kernels.c', '-o', cal/'build/kernels.o'])
    log.run('compile_legacy_probe_driver', [*common, '-march=armv9.2-a+nosve',
        '-I'+str(cal), '-c', cal/'legacy_driver.c', '-o', cal/'build/legacy_driver.o'])
    log.run('expose_legacy_probes', [Path(cc).with_name('llvm-objcopy'), '--globalize-symbol=transpose',
        '--globalize-symbol=memory_op', '--redefine-sym=main=cal02_original_main_unused',
        cal/'build/legacy_driver.o', cal/'build/preserved_driver.o'])
    log.run('compile_calibration_driver', [*common, '-march=armv9.2-a+nosve',
        '-DMOFFT_CAL_TARGET_NODE="'+cfg['node']+'"', '-c', cal/'driver.c', '-o', cal/'build/driver.o'])
    log.run('link_calibration', [cc, cal/'build/driver.o', cal/'build/preserved_driver.o',
        cal/'build/kernels.o', '--rtlib=compiler-rt', '-lgcc_s', '-lm', '-o', cal/'calibrate'])
    assert json.loads(log.run('calibration_list', [cal/'calibrate', '--list']))['cases'] == 137
    log.run('calibration_login_refusal', [cal/'calibrate', '--capture'], expected=2)
    log.run('generate_calibration_scalar_mock', [sys.executable, '-B', cal/'mock.py', cal])
    log.run('build_calibration_scalar_mock', [cc, '-std=gnu11', '-O2', '-march=armv8-a', '-DCAL_SCALAR',
        cal/'driver.c', cal/'build/mock.c', '-o', cal/'build/scalar_check'])
    log.run('calibration_scalar_checks', [cal/'build/scalar_check', '--selfcheck'])
    log.run('build_calibration_probe_check', [cc, '-std=gnu11', '-O2', '-fno-lto', '-march=armv8-a',
        cal/'probe_check.c', cal/'build/preserved_driver.o', cal/'build/kernels.o',
        '--rtlib=compiler-rt', '-lgcc_s', '-lm', '-o', cal/'build/probe_check'])
    log.run('calibration_probe_checks', [cal/'build/probe_check', '--check'])


def prepare_candidates(root, cfg, log):
    log.run('generate_candidates', [sys.executable, '-B', root/'tuning/candidate_prepare.py',
        root/'repo', root/'repo/profiles/linux-sme1-512-bootstrap.json', root/'candidates'])
    assert len(read(root/'candidates/manifest.json')['cases']) == 1076
    objects = []
    for source in sorted((root/'candidates').glob('candidates_r*.c')):
        output = source.with_suffix('.o')
        log.run('compile_'+source.stem, [cfg['clang'], '-std=c11', '-D_GNU_SOURCE=1', '-O3',
            '-march=armv9.2-a+sme+sme-f64f64', '-I'+str(root/'repo/include'), '-c', source, '-o', output])
        objects.append(output)
    harness = root/'candidates/candidate_harness.c'
    output = harness.with_suffix('.o')
    log.run('compile_candidate_harness', [cfg['clang'], '-std=c11', '-D_GNU_SOURCE=1', '-O3',
        '-march=armv9.2-a+nosve', '-I'+str(root/'repo/include'), '-c', harness, '-o', output])
    log.run('link_candidates', [cfg['clang'], *objects, output, '--rtlib=compiler-rt', '-lgcc_s', '-lm',
        '-o', cfg['candidate_executable']])
    log.run('candidate_login_refusal', [cfg['candidate_executable']], expected=2)


def init(args):
    login_only()
    root = args.workspace.resolve()
    if root == REPO or root.is_relative_to(REPO) or REPO.is_relative_to(root):
        raise ValueError('WORKSPACE_MUST_BE_SEPARATE_FROM_CHECKOUT')
    if not re.fullmatch(r'cn[0-9]+', args.node):
        raise ValueError('THIS_DONAU_WORKFLOW_REQUIRES_EXPLICIT_CN_NODE')
    root.mkdir(parents=True, exist_ok=False)
    log = Commands(root/'init_logs')
    try:
        cc, cxx = compiler_paths(args.cc, args.cxx)
        log.run('compiler_identity', [cc, '--version'])
        log.run('cmake_identity', [args.cmake, '--version'])
        shutil.copytree(REPO, root/'repo', ignore=shutil.ignore_patterns('.git', 'build*', '__pycache__', '*.pyc', '.venv'))
        for directory in ('calibration', 'overall', 'tuning'):
            shutil.copytree(HERE/directory, root/directory)
        (root/'program').mkdir()
        for path in HERE.iterdir():
            if path.suffix in ('.py', '.cpp'):
                shutil.copyfile(path, root/'program'/path.name)
        prefix = args.fftw_prefix.resolve() if args.fftw_prefix else build_fftw(root, args.fftw_source.resolve(), cc, log)
        for path in (prefix/'include/fftw3.h', prefix/'lib/libfftw3.a', prefix/'lib/libfftw3f.a'):
            if not path.is_file():
                raise ValueError('MISSING_FFTW_INPUT:'+str(path))
        cfg = dict(node=args.node, clang=cc, cxx=cxx, cmake=str(Path(args.cmake).absolute()),
            original_repo=str(root/'repo'), fftw_prefix=str(prefix), extended_sizes=SIZES,
            candidate_executable=str(root/'candidate_validate_target'),
            candidate_configurations=[[32,32],[256,32],[512,32],[4096,4096],[8192,8192]],
            initial_submissions_per_stage=1, automatic_retry=0, runtime_limit=None,
            fftw_built_from_verified_source=not bool(args.fftw_prefix),
            reused_fftw_is_build_check_only=bool(args.fftw_prefix))
        put(root/'configuration.json', cfg)
        write_capture_scripts(root)
        prepare_calibration(root, cfg, log)
        prepare_candidates(root, cfg, log)
        configure_library(root, log, 'build_bootstrap')
        prefix = Path(cfg['fftw_prefix'])
        log.run('compile_fftw_direct_reference', [cxx, '-std=c++17', '-O3',
            '-I'+str(prefix/'include'), root/'overall/fftw_reference_check.cpp',
            prefix/'lib/libfftw3f.a', prefix/'lib/libfftw3.a', '--rtlib=compiler-rt', '-lgcc_s', '-lm',
            '-o', root/'fftw_direct_reference'])
        log.run('fftw_direct_reference', [root/'fftw_direct_reference'])
        # Seal only immutable source/build objects, never growing logs or captures.
        paths = [root/'configuration.json', root/'candidate_validate_target',
                 prefix/'lib/libfftw3.a', prefix/'lib/libfftw3f.a']
        paths += list(root.glob('run_*.sh'))
        for directory in ('repo', 'program', 'overall', 'tuning', 'calibration', 'candidates'):
            paths += [p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        put(root/'prepared_C0_seal.json', dict(status='LOGIN_PREPARATION_PASS_TARGET_REQUIRED', objects=file_manifest(paths)))
        import review_login
        review_login.main(root)
        verify_tools(root)
        put(root/'preparation_completion.json', dict(status='LINUX_WORKFLOW_LOGIN_BUILD_PASS', utc=now(),
            cases=137, candidates=1076, real_sme_executed=False, jobs_submitted=0, runtime_limit=None))
    except Exception:
        put(root/'preparation_failure.json', dict(utc=now(), traceback=traceback.format_exc(),
            last_command=log.rows[-1] if log.rows else None, automatic_retry=0))
        raise
    print(json.dumps(dict(status='LINUX_WORKFLOW_LOGIN_BUILD_PASS', workspace=str(root))))


def verify_tools(root):
    """Check real-entry refusal and mock protocols without executing SME."""
    login_only()
    cfg = read(root/'configuration.json')
    out = root/'login_tool_review_01'
    out.mkdir()
    log = Commands(out/'logs')
    sys.path.insert(0, str(root/'tuning'))
    from derive_plan_search import derive
    derive(root/'repo', out/'plan_derived')
    prefix = Path(cfg['fftw_prefix'])
    flags = ['-std=c++17', '-D_GNU_SOURCE=1', '-O3', '-I'+str(root/'repo/include'),
             '-I'+str(root/'overall'), '-I'+str(prefix/'include')]
    link = ['--rtlib=compiler-rt', '-lgcc_s', '-lm']
    libs = [prefix/'lib/libfftw3f.a', prefix/'lib/libfftw3.a']
    source = out/'plan_derived/plan_search_linux.cpp'
    real = out/'plan_search'
    mock = out/'mock_plan_search'
    log.run('link_real_search', [cfg['cxx'], *flags, source, root/'build_bootstrap/libmofft.a', *link, '-o', real])
    log.run('real_search_login_refusal', [real, '--sizes', '16', '--output', out/'must_not_exist.json'], expected=2)
    log.run('link_mock_search', [cfg['cxx'], *flags, '-DMOFFT_LOGIN_MOCK=1', source,
        root/'overall/mock_mofft.cpp', root/'tuning/mock_plan_api.cpp', *libs, *link, '-o', mock])
    args = ['--sizes', '16,169', '--precision', 'both', '--candidates', '2', '--samples', '3',
            '--steady-finalists', '1', '--steady-warmup', '2']
    result = out/'mock_plan.json'
    log.run('mock_plan_protocol', [mock, *args, '--output', result])
    data = read(result)
    assert len(data['entries']) == 4 and data['evidence_class'] == 'login_mock_not_target_evidence'
    assert not data['formal_measurement']
    assert all(c['raw_seconds'] and 'relative_samples' in c for e in data['entries'] for c in e['candidates'])
    original = sha(result)
    log.run('overwrite_refused', [mock, *args, '--output', result], expected=2)
    assert sha(result) == original
    log.run('mock_formal_refused', [mock, *args, '--formal', '--exclusive-confirmed', '--output', out/'formal.json'], expected=2)
    log.run('compact_refused', [mock, *args, '--compact', '--output', out/'compact.json'], expected=2)
    for fault in ('allocation', 'status', 'nan'):
        log.run('fault_'+fault, [mock, *args, '--output', out/(fault+'.json')], expected=1,
                env=dict(os.environ, MOFFT_MOCK_FAULT=fault))
    source = root/'login_protocol_review_01/derived/benchmark_linux.cpp'
    log.run('link_real_benchmark', [cfg['cxx'], *flags, '-DMOFFT_HAVE_FFTW=1', source,
        root/'build_bootstrap/libmofft.a', *libs, *link, '-o', out/'benchmark'])
    log.run('real_benchmark_login_refusal', [out/'benchmark', '--correctness', '--sizes', '16'], expected=2)
    put(out/'completion.json', dict(status='LINUX_TOOL_PROTOCOL_PASS_NO_SME', mock_plan_conditions=4,
        real_entry_refusals=2, injected_faults=3, new_target_qualification=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('init', help='Login: build from source in a new independent workspace')
    setup.add_argument('--workspace', required=True, type=Path)
    setup.add_argument('--node', required=True)
    setup.add_argument('--cc', required=True, type=Path)
    setup.add_argument('--cxx', type=Path, help='Explicit C++ compiler; default sibling clang++, not resolved C compiler plus ++')
    setup.add_argument('--cmake', required=True)
    fftw = setup.add_mutually_exclusive_group(required=True)
    fftw.add_argument('--fftw-source', type=Path, help='Verified official fftw-3.3.11.tar.gz')
    fftw.add_argument('--fftw-prefix', type=Path, help='Existing libraries for build checks only; formal capture disabled')
    for name in ('capture', 'accept', 'build', 'plot'):
        item = sub.add_parser(name)
        item.add_argument('workspace', type=Path)
        if name in ('capture', 'accept'):
            item.add_argument('phase', choices=('C0', 'SEARCH', 'CHECK', 'P'))
        elif name == 'build':
            item.add_argument('stage', choices=('search', 'final'))
    args = parser.parse_args()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.command == 'init':
        return init(args)
    root = args.workspace.resolve()
    if args.command != 'capture':
        login_only()
    if args.command == 'capture':
        cfg = read(root/'configuration.json')
        if cfg.get('reused_fftw_is_build_check_only'):
            raise ValueError('REUSED_FFTW_IS_BUILD_CHECK_ONLY_USE_VERIFIED_SOURCE_FOR_CAPTURE')
        if args.phase != 'C0':
            previous = {'SEARCH':'C0', 'CHECK':'SEARCH', 'P':'CHECK'}[args.phase]
            if not read(root/(previous+'_acceptance/completion.json'))['accepted']:
                raise ValueError('PREVIOUS_STAGE_NOT_ACCEPTED')
        import target
        return target.main(root, args.phase)
    if args.command == 'accept':
        if not (root/(args.phase+'_job.json')).exists():
            complete = read(root/(args.phase+'_capture/completion.json'))
            put(root/(args.phase+'_job.json'), dict(job=complete['job'], node=complete['node']))
        if args.phase == 'C0':
            import accept_c0
            return accept_c0.main(root)
        import accept_stage
        return accept_stage.main(root, args.phase)
    if args.command == 'build':
        check_manifest(read(root/'prepared_C0_seal.json')['objects'])
        import build_stage
        return build_stage.build(root, final=args.stage == 'final')
    if args.command == 'plot':
        import plot_results
        return plot_results.main(root)


if __name__ == '__main__':
    main()
