# Changelog

## v1.1.1 — 2026-09-06

Fixes the WebAssembly build, which v1.1.0 shipped broken.

- **`load_reranker` was cfg-gated off for wasm32.** Removing the corpus lexicon
  in v1.1.0 deleted `load_lexicon`'s body but left its
  `#[cfg(not(target_arch = "wasm32"))]` attribute orphaned, so it silently
  attached to the next item — `load_reranker` — making it unavailable on the
  browser target.
- **`wasm.rs` still referenced the removed lexicon** in `reset_learning`, and
  called `ImeEngine::from_model` and `Reranker::new` with their old arities.
- Three wasm-only clippy lints fixed (`div_ceil`, `is_multiple_of`, a `return`
  inside a cfg block that is now two cfg-gated functions).

**Root cause: `make check` only compiled the native target.** The wasm target
compiles a different set of cfg branches, so a change can pass every native
check and still break the browser build. `make check` now runs `check-native`
*and* `check-wasm` (compile plus clippy under `-D warnings` for
`wasm32-unknown-unknown`), so this class of breakage cannot ship again.

No functional change: native accuracy is unchanged at AK-Freq 81.83% / 92.22%.

The JS API is unaffected — `createEngine(model, lexicon, weights)` still
accepts its lexicon argument and ignores it.

## v1.1.0 — 2026-09-06

A correctness and performance release. Every figure below was measured on the
4,101-case Aksharantar Nepali test split; see `docs/MANUAL.md` for method.

### Fixed — accuracy

- **Recovered 30.79pp of native top-1** (50.95% → 81.83%). The corpus-wide
  SymSpell source scored at 850,000, above the decoder band (`FRESH_SCALE`
  800,000), so any fuzzy hit displaced the reranker's top-1 — and it verified no
  edit distances. Given a fair test (distances verified, per-edit penalty) it
  still cost 9.6pp at a competing band and was byte-identical to being disabled
  at any safe band. Removed.
  Named entities recovered likewise: AK-NEI 25.68 → 47.79%, AK-NEF 20.81 → 31.21%.

### Fixed — training mathematics

- **Modified Kneser-Ney discounts** were clamped to ≤ 0.9/0.9/0.95. Chen-Goodman
  requires `0 ≤ D_i ≤ i`; this corpus yields `D₂ = 1.036`, `D₃ = 1.444`, so both
  were pinned at the ceiling and MKN had degenerated into absolute discounting.
- **Scaled forward-backward in EM.** The old `z < 1e-12` floor sat 296 orders of
  magnitude above the f64 subnormal limit and silently discarded long or
  flat-emission words from training. Now Rabiner-scaled with a per-column
  `1/c_j` posterior correction, covered by three tests including equivalence to
  an unscaled reference.
- **Learning-rate collapse.** The schedule compounded `0.8^(epochs-1) × 0.995`
  *per batch* — 4.4e-9 by batch 36, so `train-full` trained on roughly the first
  500k pairs and no-opped the rest. Now anneals to 10% of the initial rate
  across the whole run. Validated on a 5-batch run.
- **Stale normalisation statistics.** `MEAN_DENSE`/`STD_DENSE` were compiled-in
  constants no training stage refreshed; a retrain moves `lm` by 0.49σ. Container
  **v5** now carries `dense_mean`/`dense_std`. v1–v4 still load.
- Continuation counts key off first insertion, not a non-zero weight.
- Sparse-table quantisation scale from the 99.9th percentile, not a lone outlier.

### Performance

- **3.5 ms → 0.67 ms per query (5.2×), output byte-identical.** `bigram_weight`
  and `trigram_weight` scanned successor rows linearly, ~50,000 scans per query.
  Rows are stored ascending, so this is a binary search.
- O(n) beam pruning via `select_nth_unstable_by`; arena cells materialised only
  for pruning survivors.
- **Cold start 4.35 s → 1.55 s** — `akshara_id` was a linear scan called ~500k
  times at start-up.

### Removed

~1,862 lines, no measurable accuracy change:

- `crf.rs` (702) — never constructed; `DecoderConfig::crf` was always `None`.
- `pair_model.rs` + `finalize_pair` + `pair_transitions` (816) — zero callers,
  never packed into the container.
- `lexicon.rs` + `build_lexicon` (344) — dead by construction; the shipped path
  never loaded one. 0.00pp on every split.

`ImeEngine::from_bytes` and the WASM `createEngine` keep their `lexicon`
parameter as an ignored no-op, so existing JS callers still work.

### Added

- **`docs/MANUAL.md` / `AksharIME-Manual.pdf`** — a 42-page source manual:
  mathematics, architecture, training, evaluation method, ablations with
  McNemar significance, performance, deployment, defect register, roadmap.
- `tests/accuracy_regression.rs` — a 30.8pp regression once shipped through 92
  green unit tests. Now guarded.
- `tests/fuzzy_behavior.rs` — pins fuzzy behaviour and records defect D18.
- Held-out dev monitoring during reranker training, with best-by-dev packing.
- Runtime ablation switches (`core::ablation`) so the ablation table is
  reproducible from the shipped binary.
- `make manual`, `make check`, `make release-check`, `make ablate`,
  `make profile`, `make train-mid`.

### Measured, and honest about it

- The discriminative reranker used alone is **worse** than the 3-parameter
  heuristic (76.47% vs 81.02%). Net contribution +0.81pp.
- 5× more reranker training data changed nothing. Cause: `W_DENSE` has never
  been refit by this pipeline (manual §12.3).
- Modified Kneser-Ney is not measurably better than a single δ = 0.75.
- The 2²⁰ sparse table does nothing on native words (p = 0.851); its value is
  named entities only.
- Defect **D18**: the user fuzzy path is structurally unreachable.

### Known limitations

One language, one test set. Named entities well behind the neural baseline.
Browser profile not re-measured since this release.

## v1.0.2 and earlier

See git history.
