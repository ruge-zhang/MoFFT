# Linux SME1 from source

The opt-in workflow builds calibration probes and candidates from this repository, then performs CPU calibration, kernel selection, full FFT plan search, final-library validation, timing capture and login analysis. It needs no private experiment tree or historical object file. Build/copy/hash/statistics/plotting run on login; compute stages perform only native target execution and resource observations. Tools never submit or retry jobs, download dependencies or set execution deadlines. Failed and rejected samples are retained.

The current control adapter targets Linux AArch64/Donau: login hostnames begin with `login`, compute hostnames follow the cluster's `cn` convention, and the scheduler provides `CCS_JOB_ID` and `djob -J`. It requires CPU0 affinity, one thread, whole-machine exclusivity and 64-byte SVE/SME vector lengths. Other schedulers need a separate admission adapter; do not forge scheduler variables to execute on login.

## Prerequisites

Use an installed SME-capable Clang/BiSheng toolchain, matching `llvm-objcopy`, Unix Make, CMake 3.24 or newer, and Python 3.12 or newer. Initialize the installed compiler environment first. Plotting needs Matplotlib (validated with 3.9.2). Tools do not modify system settings or install packages.

Place the [official FFTW 3.3.11 source archive](https://www.fftw.org/fftw-3.3.11.tar.gz) on login. Its required SHA256 is `5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`. Initialization checks the digest before extraction and rejects paths outside the destination, links and special members. It builds static FP32 with NEON and FP64 without NEON, both with `-O3 -fPIC`, without Fortran/OpenMP/MPI.

In the Linux login shell, from the MoFFT checkout root, set these paths. Replace every placeholder; the workspace must be new, outside the checkout. Select the compute resource using current permissions, queue membership and labels, not a historical machine name.

```sh
export MOFFT_REPO=/absolute/path/to/MoFFT
export MOFFT_WORK=/absolute/path/to/new-linux-workspace
export MOFFT_PY=/absolute/path/to/python3.12
export MOFFT_CC=/absolute/path/to/clang
export MOFFT_CMAKE=/absolute/path/to/cmake
export MOFFT_FFTW_ARCHIVE=/absolute/path/to/fftw-3.3.11.tar.gz
export MOFFT_NODE=REPLACE_WITH_QUALIFIED_COMPUTE_RESOURCE
export MOFFT_QUEUE=REPLACE_WITH_AUTHORIZED_QUEUE
export MOFFT_LABELS=REPLACE_WITH_REQUIRED_LABELS
```

## Login preparation

In the login shell, working directory `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py init --workspace "$MOFFT_WORK" --node "$MOFFT_NODE" --cc "$MOFFT_CC" --cmake "$MOFFT_CMAKE" --fftw-source "$MOFFT_FFTW_ARCHIVE"
```

This saves source/toolchain identity and builds FFTW, 83 source-rebuilt probes, a 137-case calibration driver, all 1,076 SME1 candidates, the bootstrap library and checking tools. Source rebuilding is not claimed to reuse an old binary object byte-for-byte. The bootstrap remains `measured=false`.

Login checks include calibration buffer guards, ordinary-probe ABI, direct DFT reference, real-entry refusal on login, mock protocols/fault injection, fixed statistics and quality rejection. Mock output is not target qualification. Success is `preparation_completion.json`; failures retain command/stdout/stderr/exit code and `preparation_failure.json`.

`--fftw-prefix` permits existing FFTW libraries only for development build checks. Such a workspace is explicitly build-check-only and target capture is disabled. Formal reproduction uses the fixed source archive above.

## Allocation and explicit submission

Before each submission, check current account/queue permissions, effective labels from a successful receipt, queue membership, CPU0 availability, one CPU/4096 MB, whole-machine exclusivity, immutable source seal and independent output paths. Record costs when known; do not assume an unknown price is zero. Confirm that the scheduler itself sets no deadline and no automatic retry.

The examples use Donau queue/label syntax. Confirm those fields in your deployment. The user explicitly submits each phase from login after its prerequisites pass. An ambiguous submission result must be queried before another attempt. Generated `run_*.sh` files contain no submission loop or deadline.

## Calibration and all candidates

In the login shell, working directory `$MOFFT_REPO`, after resource and seal review:

```sh
dsub -n mofft-linux-C0 -q "$MOFFT_QUEUE" -R 'cpu=1,mem=4096MB' -a 'core[count=1]' -nl "$MOFFT_NODE" --label "$MOFFT_LABELS" -x job --cwd "$MOFFT_WORK" -oo "$MOFFT_WORK/C0_%J.out" -eo "$MOFFT_WORK/C0_%J.err" "bash '$MOFFT_WORK/run_C0.sh'"
```

Check actual allocation and labels immediately after obtaining the scheduler receipt. C0 captures 3,699 formal/411 warmup calibration values and five settings of 1,076 candidates, retaining 145,260 candidate round values. Calibration keeps IQR/median at most 0.15 and block-median ratio at most 1.15. Predefined diagnostic roles remain distinct. Candidate validation keeps the original numerical limits and forward/reverse/forward passes, nine rounds per pass and 100 executions per round.

After natural successful termination, on login in `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py accept "$MOFFT_WORK" C0
```

Acceptance checks actual scheduler state, before/after CPU identity, immutable inputs and complete outputs. It produces `C0_acceptance/accepted_profile.json` and `kernel_wisdom.json`. Candidates are regenerated under the accepted profile and their source identity is checked before binding new compiler/profile digests. Calibration rejection blocks later stages; do not substitute a rejected profile.

## Full plan search

On login in `$MOFFT_REPO`, after C0 acceptance:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py build "$MOFFT_WORK" search
```

In the same environment and directory, recheck resources/seal and submit once:

```sh
dsub -n mofft-linux-SEARCH -q "$MOFFT_QUEUE" -R 'cpu=1,mem=4096MB' -a 'core[count=1]' -nl "$MOFFT_NODE" --label "$MOFFT_LABELS" -x job --cwd "$MOFFT_WORK" -oo "$MOFFT_WORK/SEARCH_%J.out" -eo "$MOFFT_WORK/SEARCH_%J.err" "bash '$MOFFT_WORK/run_SEARCH.sh'"
```

The search first checks all 54 numerical conditions, then retains the original 32 candidates, 15 samples, four steady finalists and 1,000 steady warmup executions, including all descriptors/raw clocks. Whole-system load above 2.0 is warning-only; numerical, identity, exclusivity and 5% CV requirements remain unchanged.

After natural successful termination, on login in `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py accept "$MOFFT_WORK" SEARCH
```

Require `SEARCH_acceptance/completion.json`; plan wisdom is `SEARCH_capture/plan_wisdom.json`. Do not use incomplete/rejected wisdom for formal timing.

## Final library and correctness

On login in `$MOFFT_REPO`, after SEARCH acceptance:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py build "$MOFFT_WORK" final
```

In the same environment and directory, recheck resources/seal and submit CHECK:

```sh
dsub -n mofft-linux-CHECK -q "$MOFFT_QUEUE" -R 'cpu=1,mem=4096MB' -a 'core[count=1]' -nl "$MOFFT_NODE" --label "$MOFFT_LABELS" -x job --cwd "$MOFFT_WORK" -oo "$MOFFT_WORK/CHECK_%J.out" -eo "$MOFFT_WORK/CHECK_%J.err" "bash '$MOFFT_WORK/run_CHECK.sh'"
```

CHECK covers 54 final-library numerical conditions and 216 observed state transitions. It does not claim full active-ZA caller-content coverage. After natural successful termination, on login in `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py accept "$MOFFT_WORK" CHECK
```

## Timing, acceptance and plotting

On login in `$MOFFT_REPO`, after CHECK acceptance and another resource/seal review:

```sh
dsub -n mofft-linux-P -q "$MOFFT_QUEUE" -R 'cpu=1,mem=4096MB' -a 'core[count=1]' -nl "$MOFFT_NODE" --label "$MOFFT_LABELS" -x job --cwd "$MOFFT_WORK" -oo "$MOFFT_WORK/P_%J.out" -eo "$MOFFT_WORK/P_%J.err" "bash '$MOFFT_WORK/run_P.sh'"
```

The capture retains all 54 conditions, two implementations and 30 batches each: 3,240 samples. Input, alternating order, execute timer, warmup and repetition calibration remain unchanged. Plan/input/warmup/destruction stay outside the execute timer; MoFFT status checks stay inside. Compute performs no hashing, statistics, copying or plotting.

After natural successful termination, on login in `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py accept "$MOFFT_WORK" P
```

The login analysis checks complete samples, previous qualification, actual allocation/environment and source seals, then runs the original statistics and 5% CV gate. The original `run_pair_f32`, `run_pair_f64`, `bootstrap_ci` and `quantile` bodies are preserved. Bootstrap uses 2,000 MT19937_64 resamples with the upstream seed `0x4d6f464654 + sample count`; do not substitute a seed from another experiment. Rejection retains all samples without automatic supplementation/retry.

After `P_acceptance/completion.json` accepts, on login in `$MOFFT_REPO`:

```sh
"$MOFFT_PY" -B tools/linux_sme1/workflow.py plot "$MOFFT_WORK"
```

Outputs are PNG/SVG/PDF, the 104-point CSV and provenance under `figures`. The predefined 26 displayed lengths per precision remain; extra `N=144` data are retained but not plotted. Definitions are in the [benchmark notes](linux-sme1-benchmark.md). New environments need independent interpretation, not inherited historical speedups.

## Evidence and compatibility

Private workspaces retain commands/versions/stdout/stderr/exit codes, raw samples, failures, source/library/configuration identities and resource records. Tools do not automatically download or delete them. Export only a reviewed compact whitelist; do not add private execution records to a public repository.

`MOFFT_SME1_PORT=OFF` keeps the original Apple entry, public API, generation and tool settings. SME1 cost-key changes follow upstream compiler-digest rules, so regenerate matching wisdom. Only the SME1 CMake path applies registry syntax correction.

This workflow is based on the existing adaptation branch and its validated engineering chain. Later upstream M5 SDK, binary-distribution and plan-API changes have not been merged into it. The checks do not claim compatibility with that newer SDK. Default SME2 source/object equality, login/mock checks, historical SME1 measurements and new hardware runs are separate evidence classes. SME2 hardware correctness/performance requires an SME2 device.
