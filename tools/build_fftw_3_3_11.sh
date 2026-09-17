#!/bin/sh
# Build the pinned FFTW 3.3.11 baseline for the host and iOS arm64.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=${1:-"$root/build/deps/fftw-3.3.11"}
archive="$work/fftw-3.3.11.tar.gz"
source_dir="$work/source"
prefix="$work/install"
ios_prefix="$work/install-ios"
expected_sha256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
ios_sdk=$(xcrun -sdk iphoneos --show-sdk-path)
ios_cc="xcrun -sdk iphoneos clang"
ios_cflags="-arch arm64 -isysroot $ios_sdk -miphoneos-version-min=16.0 -O3 -fno-common"
ios_ar="xcrun -sdk iphoneos ar"
ios_strip="xcrun -sdk iphoneos strip"
ios_conf="--host=arm-apple-darwin --disable-shared --enable-static \
--disable-fortran --disable-sse2 --disable-avx --disable-avx2 \
--disable-openmp --disable-mpi"

mkdir -p "$work"
if [ ! -f "$archive" ]; then
  curl --fail --location --proto '=https' \
    https://www.fftw.org/fftw-3.3.11.tar.gz --output "$archive"
fi
actual_sha256=$(shasum -a 256 "$archive" | awk '{print $1}')
if [ "$actual_sha256" != "$expected_sha256" ]; then
  echo "FFTW archive checksum mismatch: $actual_sha256" >&2
  exit 1
fi
echo "FFTW 3.3.11 sha256=$actual_sha256"

if [ ! -f "$source_dir/configure" ]; then
  mkdir -p "$source_dir"
  tar -xzf "$archive" -C "$source_dir" --strip-components=1
fi

build_host() {
  variant=$1; shift
  bd="$work/build-host-$variant"
  mkdir -p "$bd"
  (cd "$bd" && CC=clang "$source_dir/configure" --prefix="$prefix" \
    --disable-shared --enable-static "$@" &&
    make -j"$(sysctl -n hw.ncpu)" && make install)
}

build_ios() {
  variant=$1; shift
  bd="$work/build-ios-$variant"
  mkdir -p "$bd"
  (cd "$bd" && CC="$ios_cc" CFLAGS="$ios_cflags" AR="$ios_ar" STRIP="$ios_strip" \
    "$source_dir/configure" --prefix="$ios_prefix" $ios_conf "$@" &&
    make -j"$(sysctl -n hw.ncpu)" && make install)
}

# FFTW 3.3.11 provides NEON codelets only for single precision.
build_host double --disable-neon
build_host float --enable-float --enable-neon
build_ios double --disable-neon
build_ios float --enable-float --enable-neon
echo "host:  $prefix"
echo "ios:   $ios_prefix"
