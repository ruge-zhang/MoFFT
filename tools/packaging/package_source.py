#!/usr/bin/env python3
"""Create a deterministic source-only MoFFT library distribution.

The archive contains a standalone CMake project and pre-generated kernels.
Packaging copies an explicit allowlist; it never runs the generator or reads
machine profiles, raw manifests, or calibration/wisdom inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from package import deterministic_tar_gz, privacy_audit, write_sha256_sidecar


DEFAULT_VERSION = "0.4.0-preview.1"
RELEASE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+-preview\.[0-9]+$")

TEMPLATE_FILES = (
    Path("CMakeLists.txt"),
    Path("README.md"),
    Path("cmake/MoFFTConfig.cmake.in"),
    Path("cmake/MoFFTReleaseVersion.cmake.in"),
    Path("cmake/mofft-build-info.json.in"),
    Path("tests/source_smoke.c"),
)
PROJECT_FILES = (
    Path("LICENSE"),
    Path("include/mofft.h"),
    Path("src/mofft.c"),
    Path("src/streaming_entry.S"),
    Path("examples/consumer/CMakeLists.txt"),
    Path("examples/consumer/main.c"),
)
GENERATED_FILES = (
    Path("mofft_generated_kernels.h"),
    Path("mofft_generated_wisdom.c"),
    Path("mofft_generated_wisdom.h"),
    Path("mofft_kernel_portfolio.c"),
    Path("mofft_kernel_portfolio.h"),
    Path("mofft_kernel_registry.c"),
    Path("mofft_kernels_fp32.c"),
    Path("mofft_kernels_fp64.c"),
)
BANK_FILES = (
    Path("mofft_generated_kernels.h"),
    Path("mofft_kernel_registry.c"),
    Path("mofft_kernels_fp32.c"),
    Path("mofft_kernels_fp64.c"),
)


def fail(message: str) -> "NoReturn":
    print(f"package_source.py: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        fail(f"required regular file is missing: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def copy_templates(source_root: Path, generated_root: Path, stage: Path,
                   version: str) -> None:
    for relative in PROJECT_FILES:
        copy_file(source_root / relative, stage / relative)
    for relative in TEMPLATE_FILES:
        copy_file(source_root / "tools/packaging/source-kit" / relative,
                  stage / relative)
    for relative in GENERATED_FILES:
        copy_file(generated_root / relative, stage / "generated" / relative)
    for bank in range(1, 4):
        for relative in BANK_FILES:
            copy_file(generated_root / f"bank{bank}" / relative,
                      stage / f"generated/bank{bank}" / relative)

    readme = stage / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("@MOFFT_RELEASE_VERSION@", version)
    if "@MOFFT_" in text:
        fail("unexpanded release placeholder in README")
    readme.write_text(text, encoding="utf-8")

    release_template = stage / "cmake/MoFFTReleaseVersion.cmake.in"
    release_text = release_template.read_text(encoding="utf-8")
    release_text = release_text.replace("@MOFFT_RELEASE_VERSION@", version)
    if "@MOFFT_" in release_text:
        fail("unexpanded release placeholder in CMake release metadata")
    release_path = stage / "cmake/MoFFTReleaseVersion.cmake"
    release_path.write_text(release_text, encoding="utf-8")
    release_path.chmod(0o644)
    release_template.unlink()


def allowed_paths() -> set[str]:
    allowed = {path.as_posix() for path in PROJECT_FILES}
    allowed.update(path.as_posix() for path in TEMPLATE_FILES
                   if path != Path("cmake/MoFFTReleaseVersion.cmake.in"))
    allowed.add("cmake/MoFFTReleaseVersion.cmake")
    allowed.update((Path("generated") / path).as_posix()
                   for path in GENERATED_FILES)
    for bank in range(1, 4):
        allowed.update((Path("generated") / f"bank{bank}" / path).as_posix()
                       for path in BANK_FILES)
    return allowed


def audit_stage(stage: Path) -> None:
    actual: set[str] = set()
    for path in stage.rglob("*"):
        if path.is_symlink():
            fail("source package contains a symbolic link")
        if path.is_file():
            actual.add(path.relative_to(stage).as_posix())
    expected = allowed_paths()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        fail("source package allowlist check failed (" + "; ".join(details) + ")")

    # Reuse the existing path/privacy scan, then enforce the source-specific
    # allowlist above rather than the prebuilt SDK's much smaller install list.
    privacy_audit(stage, enforce_install_surface=False)


def package_source(generated_dir: Path, output: Path, version: str,
                   *, force: bool = False) -> None:
    if not RELEASE_VERSION.fullmatch(version):
        fail("version must use the numeric 0.4.0-preview.N form")
    root = Path(__file__).resolve().parents[2]
    generated_root = generated_dir.resolve(strict=True)
    if not generated_root.is_dir():
        fail("--generated-dir must name an existing generated-source directory")
    if not (generated_root / "mofft_generated_wisdom.c").is_file():
        fail("generated directory is missing the embedded plan cases")
    if not (generated_root / "mofft_kernel_portfolio.c").is_file():
        fail("generated directory is missing the multi-bank dispatch table")
    if not all((generated_root / f"bank{bank}" / "mofft_kernel_registry.c").is_file()
               for bank in range(1, 4)):
        fail("generated directory must contain baseline plus banks 1 through 3")

    # Inspect only public shape/count fields for consistency. The original
    # generated metadata may identify its measurement profile, so it is never
    # copied into the release archive.
    build_info_path = generated_root / "mofft-build-info.json"
    if not build_info_path.is_file() or build_info_path.is_symlink():
        fail("generated directory is missing sanitized build metadata")
    try:
        generated_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        wisdom_info = generated_info["preset_plan_wisdom"]
        bank_count = generated_info["kernel_bank_count"]
        plan_case_count = wisdom_info["entry_count"]
        wisdom_source = (generated_root / "mofft_generated_wisdom.c").read_text(
            encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        fail(f"generated metadata is malformed ({exc})")
    actual_plan_case_count = len(re.findall(r"if \(length ==", wisdom_source))
    if (not isinstance(generated_info, dict) or
            not isinstance(wisdom_info, dict) or
            generated_info.get("target") != "apple-arm64-sme2" or bank_count != 4 or
            wisdom_info.get("available") is not True or
            wisdom_info.get("formal") is not False or
            plan_case_count != actual_plan_case_count or actual_plan_case_count == 0):
        fail("generated metadata does not match a four-bank SME2 preview with embedded plan cases")

    output = output.resolve()
    sidecar = output.with_name(output.name + ".sha256")
    if not force and (output.exists() or sidecar.exists()):
        fail("output or checksum already exists; choose another path or pass --force")

    with tempfile.TemporaryDirectory(prefix="mofft-source-kit-") as temp:
        staging_root = Path(temp)
        stage = staging_root / f"mofft-{version}"
        stage.mkdir()
        copy_templates(root, generated_root, stage, version)
        audit_stage(stage)
        output.parent.mkdir(parents=True, exist_ok=True)
        deterministic_tar_gz(stage, output)
        write_sha256_sidecar(output)

    print(f"Created {output}")
    print(f"Created {sidecar}")


def main() -> None:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-dir", type=Path,
        default=source_root / "build/m5-sdk/generated",
        help="existing frozen generated source tree (no generator is invoked)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="source archive path")
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help="preview version used in the archive name and metadata")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing archive and checksum")
    args = parser.parse_args()
    output = args.output or source_root / f"dist/mofft-{args.version}.tar.gz"
    package_source(args.generated_dir, output, args.version, force=args.force)


if __name__ == "__main__":
    main()
