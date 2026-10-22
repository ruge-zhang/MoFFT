"""All generation, linking and sealing run on login nodes, never inside jobs."""
import os
from pathlib import Path
import sys
import traceback

from common import Commands, check_manifest, file_manifest, login_only, now, put, read, sha


def build(root, final=False):
    login_only()
    root = Path(root)
    cfg = read(root / "configuration.json")
    stage = "final" if final else "search"
    log = Commands(root / ("build_" + stage + "_logs"))
    try:
        assert read(root / "C0_acceptance/completion.json")["accepted"] is True
        if final:
            assert read(root / "SEARCH_acceptance/completion.json")["accepted"] is True
        check_manifest(read(root / "prepared_C0_seal.json")["objects"])
        repo = root / "repo"
        directory = root / ("build_" + stage)
        profile = root / "C0_acceptance/accepted_profile.json"
        wisdom = root / "C0_acceptance/kernel_wisdom.json"
        args = [
            cfg["cmake"], "-S", str(repo), "-B", str(directory),
            "-G", "Unix Makefiles", "-DCMAKE_MAKE_PROGRAM=/usr/bin/make",
            "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_COMPILER=" + cfg["clang"],
            "-DCMAKE_CXX_COMPILER=" + cfg["cxx"],
            "-DPython3_EXECUTABLE=" + sys.executable, "-DMOFFT_SME1_PORT=ON",
            "-DMOFFT_BUILD_TESTS=OFF", "-DMOFFT_BUILD_TOOLS=OFF",
            "-DMOFFT_PROFILE=" + str(profile), "-DMOFFT_KERNEL_WISDOM=" + str(wisdom),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]
        if final:
            plan = read(root / "SEARCH_capture/plan_wisdom.json")
            args += ["-DMOFFT_WISDOM=" + str(root / "SEARCH_capture/plan_wisdom.json"),
                     "-DMOFFT_WISDOM_MACHINE=" + plan["machine"]]
        log.run("configure", args)
        log.run("build", [cfg["cmake"], "--build", str(directory), "--parallel", "1"])
        programs = root / "programs"
        programs.mkdir(exist_ok=True)
        sys.path.insert(0, str(root / "overall"))
        from derive import derive
        derived = root / ("overall_derived_" + stage)
        derive(repo, derived)
        source = derived / "benchmark_linux.cpp"
        text = source.read_text()
        # Success traces are outside the execute timer and expose original error
        # and tolerance. Numerical comparisons and thresholds are unmodified.
        for bits, eps in ((32, "FLT_EPSILON"), (64, "DBL_EPSILON")):
            needle = f"  const double tolerance = correctness_reference_tolerance(n, {eps});"
            assert text.count(needle) == 1
            text = text.replace(needle, needle + "\n" +
                f'  std::cout << std::setprecision(17) << "correctness_case,fp{bits}," << n'
                ' << "," << relative_error << "," << tolerance << "," << (int)status << "\\n";')
        source.write_text(text)
        from defer_statistics import derive as defer_statistics
        put(derived / "statistics_boundary_extension.json", defer_statistics(source))
        put(derived / "correctness_trace_extension.json", dict(
            source_sha256=sha(source), trace_only=True, limits_changed=False,
            numerical_comparison_changed=False, timer_changed=False))
        cc = cfg["cxx"]
        prefix = Path(cfg["fftw_prefix"])
        flags = ["-std=c++17", "-D_GNU_SOURCE=1", "-O3",
                 "-I" + str(repo / "include"), "-I" + str(root / "overall"),
                 "-I" + str(prefix / "include")]
        lib = directory / "libmofft.a"
        link = ["--rtlib=compiler-rt", "-lgcc_s", "-lm"]
        log.run("benchmark_link", [
            cc, *flags, "-DMOFFT_HAVE_FFTW=1",
            '-DMOFFT_COMPILER_INFO="BiSheng5.2.0-Clang19.1.7"',
            str(source), str(lib),
            str(prefix / "lib/libfftw3f.a"), str(prefix / "lib/libfftw3.a"),
            *link, "-o", str(programs / ("benchmark_" + stage))])
        log.run("benchmark_login_refusal", [
            programs / ("benchmark_" + stage), "--correctness", "--sizes", "256"], expected=2)
        if not final:
            sys.path.insert(0, str(root / "tuning"))
            from derive_plan_search import derive as derive_plan
            plan_dir = root / "plan_derived"
            derive_plan(repo, plan_dir)
            log.run("plan_search_link", [
                cc, *flags, str(plan_dir / "plan_search_linux.cpp"), str(lib),
                *link, "-o", str(programs / "plan_search")])
        else:
            control = Path(__file__).parent
            log.run("lifecycle_link", [
                cc, *flags, "-march=armv9.2-a+nosve", str(control / "lifecycle.cpp"),
                str(lib), *link, "-o", str(programs / "lifecycle_final")])
            log.run("lifecycle_login_refusal", [programs / "lifecycle_final"], expected=2)
        objects = list(programs.glob("*")) + [
            lib, directory / "generated/manifest.json", profile, wisdom,
            root / "configuration.json"]
        objects += list((root / "program").glob("*"))
        if final:
            objects += [root / "SEARCH_capture/plan_wisdom.json"]
        seal = dict(utc=now(), objects=file_manifest(objects), stage=stage,
                    old_seal_sha256=sha(root / "prepared_C0_seal.json"))
        put(root / ("CHECK_seal.json" if final else "SEARCH_seal.json"), seal)
        if final:
            put(root / "P_seal.json", seal)
        put(root / ("build_" + stage + "_completion.json"),
            dict(status="LOGIN_BUILD_AND_SEAL_COMPLETE", utc=now(), commands=len(log.rows)))
    except Exception:
        put(root / ("build_" + stage + "_failure.json"),
            dict(utc=now(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    build(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "final")
