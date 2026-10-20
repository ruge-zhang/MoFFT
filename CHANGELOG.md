# Changelog

## 0.4.0 Preview 1 — 2026-10-20

- Add FFTW-inspired bound-array plan/execute constructors for FP32 and FP64,
  preserving the explicit-buffer C API.
- Add PRETUNED and bounded MEASURE planning over precompiled expression-family
  banks, FFT decompositions and layouts, with selection diagnostics.
- Add explicit versioned plan recipe export/import, guarded by build identity
  and CPU family, without implicit cache files or runtime code generation.
- Add relocatable macOS arm64 static SDK packaging, privacy-filtered metadata,
  checksums and a standalone C consumer example.
- Extend correctness checks to kernel banks, planning policies, stored recipes
  and relocated binary-package consumers.

Prepared library source and prebuilt M5 SDK packages have functional validation.
Formal performance tuning and validation on other platforms are outside this
preview release.
