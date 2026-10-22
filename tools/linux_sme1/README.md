# Linux SME1 workflow tools

See the [from-source instructions](../../docs/linux-sme1-workflow.md). From the checkout root, run `python3 tools/linux_sme1/workflow.py --help`.

`init` builds an independent workspace from this repository and the pinned FFTW source: calibration probes, all 1,076 candidates, bootstrap library and protocol checks. It needs no historical experiment directory. `capture` performs native target execution and resource observations inside an allocation. `accept`, `build` and `plot` run on login. Tools never submit/retry jobs, set execution deadlines or download dependencies.

`calibration` contains source-rebuilt probes, retained CAL03 analysis and boundary checks. `tuning` generates SME1 candidates and derives complete plan search. `overall` derives the Linux benchmark and supplies admission, FFTW reference and explicitly marked mocks. Root modules implement identity/seal checks, lifecycle checks, original statistics and plotting.

`submit.py` is a read-only Donau query adapter retained for existing imports; it cannot submit/cancel jobs. Users explicitly invoke their scheduler after checking current resources and permissions. `registry_terminators.py` corrects SME1 generated locality-dispatch statement terminators; default SME2 generation does not invoke it.

Build success, a mock result or a replot is not a new target performance result. Complete CPU-specific calibration, plan search, correctness and capture acceptance on the actual target.

Source line counts and UTF-8/LF-normalized SHA256 values are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
