# Indirect memory guards

The personal manifest retains the existing int16 gather and adds 29
validation-derived configurations in `indirect_memory_guards.json`:

- SIMD-origin Load: unmasked, prefix mask, loaded Boolean mask.
- SIMD-origin Store: unmasked and prefix mask.
- SIMT-origin Load/Store: each mask family at source F1/F2/F4.
- Small-q, INT64-index Load/Store: source F1/F2/F4.

`simd_store_loaded_bool` (runtime K=0) was explicitly withdrawn. Its removal
does not mean that unknown-density Store prediction is accurate. Geometry and
validation-source IDs are preserved for the remaining 29 configurations.

## Run

From the repository root, with compatible Ascend950PR/CANN/BiShengIR and
Triton-Ascend, NumPy, PyTorch and torch_npu:

```bash
python costmodel_tests/run.py list --category personal
python costmodel_tests/run.py --category personal \
  --case maxh2018_simt_store_none_f1 --device 0 --retries 0
python costmodel_tests/run.py --category personal --device 0 --retries 0
```

Auto is free to choose SIMD, SIMT or Mixed. No compiler subprocess, guard,
core-count or predicted-time override is used. Explicit routes/factors are
available through the runner's `--validation-scope all` option.

The JSON `mode` preserves the source kernel's mask/reshape expression; it does
not force the selected implementation. `F` preserves the logical program
count; Auto may select a different superblock factor. Physical core count is
read from the actual device. Consequently these runs are not identical to
the archived, forced-single-AIV calibration experiments.

Each configuration checks eight deterministic inputs (seeds 90521000..7),
including untouched inputs and output canaries. Profiler timing uses seed
90521000 only. Store uses a separate unfilled destination per timed launch;
canaries and written values are checked after completion. Loaded masks do not
provide their population count to the compiler.

Correctness, successful execution, legal Auto selection and whole-kernel
prediction error are separate checks. Uploading these cases is not a claim
that every route or prediction passes 10%; failures must remain visible.
There is no independent stage-timing label in this portable benchmark.

## Validation status (2026-10-22)

Pre-commit and all three new offline guard checks pass. The complete offline
suite passes 44/45 checks; the remaining CI-manifest regeneration mismatch
also fails on the unchanged base branch (41/42), so it is not a guard-upload
regression. No CI manifest or unrelated test is changed here.

Representative NPU smoke runs use the existing installed compiler, not a new
build of the local CostModel refactor. These checks do not certify all 29
Auto predictions; the runner intentionally reports prediction failures.
