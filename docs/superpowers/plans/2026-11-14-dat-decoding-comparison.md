# DAT decoding comparison implementation plan

**Goal:** Enable real DAT sampling and run greedy 256 followed by non-thinking sampling 256 using the existing checkpoint and LR640 + crop400.

**Spec:** User-approved experiment in this chat: update and push existing main branches; no new branches. Preserve the repaired single-instruction prompt and all visual settings.

**Architecture:** Keep DAT preparation and visual features unchanged. Normalize and validate generation options at the adapter boundary, apply penalties and sampling inside the existing autoregressive decoder, and record actual token counts and stopping reasons. A separate lmms-eval launcher reads the verified rerun configuration and executes two evaluations sequentially.

**Constraints:** Python >=3.10; existing transformers/torch only; no changes to checkpoint weights, geometry, or legacy thinking behavior; preserve default greedy outputs; leave unrelated working files alone.

## Tasks

- [ ] Write failing behavioral tests for sampling, penalties, EOS/length, invalid configuration, and adapter forwarding.
- [ ] Add generation options in `ldat_eval/decoding.py`; pass them through `LDatRunner.generate`, `finish`, and `_decode` and the lmms adapter.
- [ ] Add per-sample counts and optional rank-local trace files with raw answers and finish reasons.
- [ ] Independently add `lmms-eval/tools/vstar_dat_decode_compare.py` with tests: inherited checkpoint and seeds, two precise configs, sequential verification of 191 samples, fresh outputs, writable cache.
- [ ] Run DAT CPU integration tests including a real tiny model; run evaluator regression tests and launcher dry-run. Review both diffs, commit and push main, then verify remote hashes.

## Review focus

- Penalties must not mutate model logits; presence penalizes generated tokens only.
- `do_sample=False` stays deterministic and consumes no RNG; string false is not true.
- EOS and length limits report real generated token counts; missing images do not reuse stale counters.
- Unsupported decoding options fail visibly rather than silently changing the experiment.
- Cached preparation and visual tensors must remain identical; second trial starts only after the first has valid complete artifacts.

GPU accuracy and runtime validation require the user's cluster; CPU tests do not establish a V* score improvement.

## Existing extended-test failures

The following nine failures also reproduce on an unmodified `git archive HEAD`
snapshot before this change (19 other extended tests and six subtests pass).
They do not import the modified decoder. No training code is changed here.

- `scripts/test_dat_experiments.py::ExperimentTests::test_fixed_coarse_zero_origin_and_reader_freeze_after_step`: stale fixture missing `glob_grid`.
- `scripts/test_dat_experiments.py::ExperimentTests::test_multi_image_multi_answer_readers_have_distinct_keys_and_400_tokens`: same fixture.
- `scripts/test_dat_experiments.py::ExperimentTests::test_per_group_override_ce_gradient_and_finite_difference`: same fixture.
- `scripts/test_dat_experiments.py::ExperimentTests::test_trainer_replays_reference_through_checkpoint_backward_then_cleans_up`: same fixture.
- `scripts/test_dat_loading_geometry.py::LoadingTests::test_auto_and_bf16_preserve_all_66_original_tensors`: fixture expects 66 tensors; existing code enumerates 72.
- `scripts/test_probe_report.py::ProbeReportTest::test_constant_coordinates_have_undefined_correlation`: stale arguments missing `dump_ground`.
- `scripts/test_probe_report.py::ProbeReportTest::test_full_report_keeps_accuracy_and_serializes`: same fixture.
- `scripts/test_probe_report.py::ProbeReportTest::test_hd_diagnostics_without_global_offset`: same fixture.
- `scripts/test_probe_report.py::ProbeReportTest::test_small_hd_sample_skips_correlation`: same fixture.
