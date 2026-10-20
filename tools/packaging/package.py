#!/usr/bin/env python3
"""Package and verify an already-installed MoFFT static library.

This script deliberately does not build MoFFT or regenerate tuning data. The
caller chooses the profile and plan/kernel wisdom at configure time; packaging
copies only the public header, installed library, and relocatable CMake files.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


REQUIRED_FILES = (
    Path("include/mofft.h"),
    Path("lib/libmofft.a"),
    Path("lib/cmake/MoFFT/MoFFTConfig.cmake"),
    Path("lib/cmake/MoFFT/MoFFTConfigVersion.cmake"),
    Path("lib/cmake/MoFFT/MoFFTTargets.cmake"),
    Path("share/mofft/mofft-build-info.json"),
    Path("share/licenses/MoFFT/LICENSE"),
)
OPTIONAL_FILES = (Path("share/licenses/MoFFT/NOTICE"),)


def fail(message: str) -> "NoReturn":
    print(f"package.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def package_version(prefix: Path) -> str:
    text = (prefix / "lib/cmake/MoFFT/MoFFTConfigVersion.cmake").read_text()
    match = re.search(r'^set\(PACKAGE_VERSION "([0-9][A-Za-z0-9.+_-]*)"\)',
                      text, re.MULTILINE)
    if not match:
        fail("could not determine the installed MoFFT package version")
    return match.group(1)


def validate_prefix(prefix: Path) -> Path:
    prefix = prefix.resolve(strict=True)
    missing = [str(path) for path in REQUIRED_FILES if not (prefix / path).is_file()]
    if missing:
        fail("install prefix is incomplete (missing " + ", ".join(missing) + ")")
    for path in prefix.rglob("*"):
        if path.is_symlink():
            fail("install prefix contains a symlink; use a normal CMake install tree")
    return prefix


def stage_prefix(source: Path, destination: Path) -> Path:
    """Copy only documented public package files, never generated source data."""
    validate_prefix(source)
    destination.mkdir(parents=True, exist_ok=False)
    for relative in (
        Path("include/mofft.h"), Path("lib/libmofft.a"),
        Path("share/mofft/mofft-build-info.json"),
        Path("share/licenses/MoFFT/LICENSE"),
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    notice = source / "share/licenses/MoFFT/NOTICE"
    if notice.is_file():
        target = destination / "share/licenses/MoFFT/NOTICE"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notice, target)
    cmake_source = source / "lib/cmake/MoFFT"
    cmake_destination = destination / "lib/cmake/MoFFT"
    cmake_destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(cmake_source.glob("*.cmake")):
        shutil.copy2(path, cmake_destination / path.name)
    return destination


def privacy_audit(root: Path, *, enforce_install_surface: bool = True) -> None:
    patterns = [
        re.compile(rb"/Users/[^/\x00\s]+"),
        re.compile(rb"/private/tmp/[^/\x00\s]+"),
        re.compile(rb"(?<![A-Za-z0-9])/(?:tmp|var/folders)/[^/\x00\s]+"),
        re.compile(rb"OneDrive", re.IGNORECASE),
    ]
    home = Path.home().as_posix().encode()
    if home and home != b"/":
        patterns.append(re.compile(re.escape(home), re.IGNORECASE))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        result = subprocess.run(["strings", "-a", str(path)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, check=False)
        data = result.stdout if result.returncode == 0 else path.read_bytes()
        if any(pattern.search(data) for pattern in patterns):
            fail("privacy audit detected a machine-specific path or storage identifier")
    if enforce_install_surface:
        # Keep the CMake package surface intentionally small and auditable.
        allowed = {str(path) for path in REQUIRED_FILES + OPTIONAL_FILES} | {
            "README.txt",
            "bin/mofft-demo",
            "examples/consumer/CMakeLists.txt",
            "examples/consumer/main.c",
        }
        for path in root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative not in allowed and not relative.startswith("lib/cmake/MoFFT/"):
                    fail("package contains an undocumented file: " + relative)


def check_arm64_archive(path: Path) -> None:
    result = subprocess.run(["lipo", "-archs", str(path)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    if result.returncode != 0 or "arm64" not in result.stdout.split():
        fail("installed static library is not an inspectable arm64 archive")


def verify_macos(prefix: Path, project_root: Path,
                 require_wisdom_size: int = 0) -> bytes:
    with tempfile.TemporaryDirectory(prefix="mofft-package-verify-") as temp:
        temp_root = Path(temp)
        relocated = stage_prefix(prefix, temp_root / "relocated install")
        consumer_source = relocated / "examples/consumer"
        consumer_source.mkdir(parents=True, exist_ok=True)
        for name in ("CMakeLists.txt", "main.c"):
            shutil.copy2(project_root / "examples/consumer" / name,
                         consumer_source / name)
        check_arm64_archive(relocated / "lib/libmofft.a")
        privacy_audit(relocated)
        build = temp_root / "consumer build"
        run(["cmake", "-S", str(project_root / "examples/installed-consumer"),
             "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_PREFIX_PATH={relocated}",
             f"-DMOFFT_CONSUMER_REQUIRE_WISDOM_SIZE={require_wisdom_size}"])
        run(["cmake", "--build", str(build), "--config", "Release"])
        executable = build / "mofft-installed-consumer"
        if not executable.is_file():
            executable = build / "Release/mofft-installed-consumer"
        if not executable.is_file():
            fail("CMake consumer executable was not produced")
        run([str(executable)])

        demo_build = temp_root / "downloaded demo build"
        run(["cmake", "-S", str(consumer_source), "-B", str(demo_build),
             "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_PREFIX_PATH={relocated}"])
        run(["cmake", "--build", str(demo_build), "--config", "Release"])
        demo = demo_build / "mofft-demo"
        if not demo.is_file():
            demo = demo_build / "Release/mofft-demo"
        if not demo.is_file():
            fail("downloaded plain-C demo executable was not produced")
        recipe = temp_root / "consumer.plan"
        for precision in ("fp32", "fp64"):
            run([str(demo), "8", precision, "pretuned"])
            run([str(demo), "8", precision, "measure", "--save-plan",
                 str(recipe)])
            run([str(demo), "8", precision, "--load-plan", str(recipe)])
        return demo.read_bytes()


def verify_ios(prefix: Path, project_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="mofft-ios-package-verify-") as temp:
        temp_root = Path(temp)
        relocated = stage_prefix(prefix, temp_root / "relocated install")
        check_arm64_archive(relocated / "lib/libmofft.a")
        privacy_audit(relocated)
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "iphoneos", "--show-sdk-path"], text=True).strip()
        executable = temp_root / "mofft-installed-consumer-ios"
        run([
            "xcrun", "--sdk", "iphoneos", "clang",
            "-target", "arm64-apple-ios16.0", "-isysroot", sdk,
            "-I", str(relocated / "include"),
            str(project_root / "examples/installed-consumer/main.c"),
            str(relocated / "lib/libmofft.a"), "-lm", "-o", str(executable),
        ])
        if not executable.is_file():
            fail("iOS consumer link did not produce an executable")
        print("iOS arm64 installed-package consumer linked successfully (not run on host).")


def deterministic_tar_gz(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for path in sorted(root.rglob("*")):
                    relative = path.relative_to(root.parent).as_posix()
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def write_sha256_sidecar(output: Path) -> None:
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii")


def package_macos(prefix: Path, output: Path,
                  require_wisdom_size: int = 0) -> None:
    prefix = validate_prefix(prefix)
    version = package_version(prefix)
    project_root = Path(__file__).resolve().parents[2]
    verified_demo = verify_macos(prefix, project_root, require_wisdom_size)
    with tempfile.TemporaryDirectory(prefix="mofft-macos-package-") as temp:
        root = Path(temp) / f"MoFFT-{version}-macos-arm64"
        stage_prefix(prefix, root)
        demo = root / "bin/mofft-demo"
        demo.parent.mkdir(parents=True, exist_ok=True)
        demo.write_bytes(verified_demo)
        demo.chmod(0o755)
        run(["strip", "-S", "-x", str(demo)])
        example_dir = root / "examples/consumer"
        example_dir.mkdir(parents=True, exist_ok=True)
        for name in ("CMakeLists.txt", "main.c"):
            shutil.copy2(project_root / "examples/consumer" / name,
                         example_dir / name)
        write_package_readme(root / "README.txt", version, "macOS arm64")
        check_arm64_archive(root / "lib/libmofft.a")
        privacy_audit(root)
        output = output.resolve()
        deterministic_tar_gz(root, output)
        write_sha256_sidecar(output)
        print(f"Created {output}")


def package_ios(prefix: Path, output: Path) -> None:
    prefix = validate_prefix(prefix)
    version = package_version(prefix)
    project_root = Path(__file__).resolve().parents[2]
    verify_ios(prefix, project_root)
    with tempfile.TemporaryDirectory(prefix="mofft-ios-package-") as temp:
        temp_root = Path(temp)
        staged = stage_prefix(prefix, temp_root / "staged")
        check_arm64_archive(staged / "lib/libmofft.a")
        privacy_audit(staged)
        xcframework = temp_root / "MoFFT.xcframework"
        run([
            "xcodebuild", "-create-xcframework",
            "-library", str(staged / "lib/libmofft.a"),
            "-headers", str(staged / "include"),
            "-output", str(xcframework),
        ])
        privacy_audit(xcframework, enforce_install_surface=False)
        info = xcframework / "Info.plist"
        if not info.is_file():
            fail("xcodebuild did not produce XCFramework metadata")
        metadata = plistlib.loads(info.read_bytes())
        libraries = metadata.get("AvailableLibraries", [])
        if not any(item.get("SupportedPlatform") == "ios" and
                   not item.get("SupportedPlatformVariant") and
                   "arm64" in item.get("SupportedArchitectures", [])
                   for item in libraries):
            fail("XCFramework does not contain an iOS device arm64 slice")
        readme = package_readme(version, "iOS device arm64")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for path in sorted(xcframework.rglob("*")):
                relative = path.relative_to(temp_root).as_posix()
                if path.is_dir():
                    continue
                item = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(item, path.read_bytes())
            archive_files = {
                "README.txt": readme.encode(),
                "LICENSE": (staged / "share/licenses/MoFFT/LICENSE").read_bytes(),
                "mofft-build-info.json":
                    (staged / "share/mofft/mofft-build-info.json").read_bytes(),
            }
            notice = staged / "share/licenses/MoFFT/NOTICE"
            if notice.is_file():
                archive_files["NOTICE"] = notice.read_bytes()
            for name, data in archive_files.items():
                item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(item, data)
        write_sha256_sidecar(output)
        print(f"Created {output} (MoFFT {version}, iOS device arm64)")


def package_readme(version: str, target: str) -> str:
    if target.startswith("macOS"):
        contents = """Contents: public C header, static MoFFT library with generated
kernels already compiled in, a relocatable CMake package exporting
MoFFT::mofft, a runnable bin/mofft-demo, sanitized build metadata, and the
Apache-2.0 license.

Run immediately (no compiler, CMake, Python, or generator required):
  ./bin/mofft-demo 256 fp32 pretuned
  ./bin/mofft-demo 4096 fp64 measure --save-plan local.plan
  ./bin/mofft-demo 4096 fp64 --load-plan local.plan

CMake usage:
  find_package(MoFFT {version} CONFIG REQUIRED)
  target_link_libraries(my_target PRIVATE MoFFT::mofft)

The examples/consumer directory contains a buildable plain-C example. Build it
with:
  cmake -S examples/consumer -B build -DCMAKE_PREFIX_PATH="$PWD"
  cmake --build build
Then try a local search and save the chosen plan recipe:
  ./build/mofft-demo 4096 fp64 measure --save-plan local.plan
Reuse it on a later run of the same binary and CPU:
  ./build/mofft-demo 4096 fp64 --load-plan local.plan
""".format(version=version)
        platform_note = "This package targets macOS arm64."
    else:
        contents = """Contents: `MoFFT.xcframework` with a static iOS device arm64
library and public C header, sanitized build metadata, and the Apache-2.0
license. Add the XCFramework to the Xcode target's Frameworks, Libraries, and
Embedded Content, then include <mofft.h>.
"""
        platform_note = ("It is not an iOS simulator build. This cross-linked "
                         "package has not been validated by on-device execution.")
    return f"""MoFFT {version} prebuilt static package

Target: {target} (Apple SME2-capable hardware required).

{contents}
Planning policies:
  - The default/PRETUNED policy uses an exact plan case embedded in this
    binary when available, otherwise it uses the fast analytical model.
  - MOFFT_PLANNER_MEASURE measures a bounded set of plans using kernels that
    are already in this binary. It does not generate or compile new kernels,
    and measurements are not persisted automatically.
  - A measured plan selects one compiled kernel bank for the whole plan;
    bank selection is not independently mixed across stages.

The accompanying mofft-build-info.json reports only the sanitized target,
minimum OS and SDK version, profile name/status, whether plan wisdom is
present, and compiled bank count. The consuming target must use the package's
minimum OS version or newer.

This is a MoFFT C API, not a binary-compatible FFTW implementation.
{platform_note}
"""


def write_package_readme(path: Path, version: str, target: str) -> None:
    path.write_text(package_readme(version, target), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("package-macos", "package-ios", "verify-macos", "verify-ios"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--prefix", type=Path, required=True,
                         help="existing CMake install prefix; this command never builds")
        if command.startswith("package-"):
            suffix = ".tar.gz" if command == "package-macos" else ".xcframework.zip"
            sub.add_argument("--output", type=Path, required=True,
                             help=f"output archive (normally {suffix})")
        if command in ("package-macos", "verify-macos"):
            sub.add_argument(
                "--require-wisdom-size", type=int, default=0,
                help="also require strict PRETUNED FP32/FP64 hits at this size")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "package-macos":
        if args.require_wisdom_size < 0:
            parser.error("--require-wisdom-size must be nonnegative")
        package_macos(args.prefix, args.output, args.require_wisdom_size)
    elif args.command == "package-ios":
        package_ios(args.prefix, args.output)
    elif args.command == "verify-macos":
        if args.require_wisdom_size < 0:
            parser.error("--require-wisdom-size must be nonnegative")
        verify_macos(validate_prefix(args.prefix), root,
                     args.require_wisdom_size)
    else:
        verify_ios(validate_prefix(args.prefix), root)


if __name__ == "__main__":
    main()

