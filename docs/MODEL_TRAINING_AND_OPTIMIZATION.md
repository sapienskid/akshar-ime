# Akshar Devanagari IME — Model Training, Optimization & Pruning Guide

This document is the definitive technical reference for the model architecture, empirical accuracy contributions, loss-free pruning techniques, and one-shot training pipeline of the Akshar Devanagari IME.

---

## 1. Executive Summary & Benchmark State

Akshar Devanagari IME achieves **state-of-the-art transliteration accuracy** on the held-out AI4Bharat Aksharantar Nepali test split (4,101 cases), outperforming neural baselines (such as IndicXlit at 80.25% Top-1) while requiring **zero neural network runtimes**, in a browser bundle of 4.92 MB compressed. Measured latency is $2.6 - 4.1$ ms per query at $k=5$.

### SOTA Benchmark Comparison (Aksharantar Nepali Test Split)

| System | Technology | Model Footprint | Native (`AK-Freq`) Top-1 | Top-5 | Hard Entities (`AK-NEF`) Top-1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IndicXlit** (AI4Bharat) | Transformer Seq2Seq (Neural) | ~120 MB | 80.25% | — | ~28% |
| Akshar IME (pre-compaction) | EM + Kneser-Ney, bincode container | 66.59 MB | 82.12% | 92.13% | 29.74% |
| **Akshar IME (desktop)** | **+ compact v3 container** | **30.59 MB** | **82.02%** | **92.17%** | **30.35%** |
| **Akshar IME (browser)** | **+ entropy-pruned trigram LM, no bigrams** | **8.91 MB / 4.92 MB Brotli** | **81.07%** | **92.22%** | **30.84%** |

*Measured on the official 4,101-word test split via `evaluate_aksharantar`.*

Beyond the isolated-word benchmark, two further harnesses exist because it
cannot see everything that matters:

| Harness | What it measures | Desktop | Browser |
| :--- | :--- | ---: | ---: |
| `evaluate_sentences` | word accuracy **in context** on held-out corpus sentences | 89.85% @1 | 88.05% @1 |
| `evaluate` on `test_multiref.jsonl` | tolerance of **loose romanization** (14,410 spellings) | 64.35% @1 | 63.11% @1 |

---

## 2. Model Anatomy & Component Size Breakdown

The production engine loads a single atomic container: `data/akshar.model`. The internal components can be structurally inspected at any time using:

```bash
cargo run --release --bin probe_model -- --model data/akshar.model --inspect
```

### Active Model Containers

Sizes below are the **encoded** sizes in the v3 container. They differ from the
in-memory layout: `core::codec` re-encodes the n-gram tables, vocabulary and
chunk list on save, so `bincode::serialized_size` of a decoded field no longer
describes what is on disk.

```
=== Unified Model Inspection: data/akshar.model ===       (desktop, 30.59 MB)
  Translit Model    :    7.65 MB ( 25.0%)
  Sparse Reranker   :    1.00 MB (  3.3%)
  Vocab Frequency   :    2.41 MB (  7.9%) [470,012 words]
  Word Bigrams      :   19.54 MB ( 63.9%) [48,024 heads]

=== Unified Model Inspection: data/akshar_wasm.model ===  (browser, 8.91 MB)
  Translit Model    :    5.51 MB ( 61.8%)   <- trigram LM entropy-pruned
  Sparse Reranker   :    1.00 MB ( 11.2%)   <- 0.01 MB after Brotli
  Vocab Frequency   :    2.41 MB ( 27.0%)
  Word Bigrams      : None
                                            Brotli q11: 4.92 MB
```

--- Translit Sub-components ---
    Aksharas list   :   0.35 MB (16,562 aksharas)
    Chunks list     :   1.18 MB (100,578 chunks)
    Emissions       :   2.35 MB (291,086 entries across 16,562 aksharas)
    Bigram LM       :   3.36 MB (423,557 transitions)
    Trigram LM      :  21.99 MB (332,903 contexts, 2,049,828 transitions)
    Other fields    :   0.19 MB
```

---

## 3. Component Contributions to Accuracy & Size

The question that governs every packing decision is **accuracy per byte**, not
size alone. Measured contributions, against encoded v3 sizes:

| Component | Encoded | Accuracy Contribution | **MB per point** |
| :--- | ---: | :--- | ---: |
| **Sparse Reranker** (`2^20` `i8`) | 1.00 MB | +2.4% Top-1 | **0.4** |
| **Vocabulary Frequency Map** | 2.41 MB | +4.2% Top-1 | **0.6** |
| **Syllable Trigram LM** | 5.49 MB | +3.13% Top-1 over the bigram LM | **1.8** |
| **Word Bigram Table** | 19.54 MB | 0.00% isolated, **+0.16pp in context** | **122** |

The word-bigram table is roughly 70x worse per byte than anything else, which
is why the browser profile omits it. That figure is not a tuning artifact: the
+0.16pp ceiling held across three independent fusion designs (a post-squash
`40_000·ln(1+f)` bonus, unshrunk PMI, and the shipped shrunk PMI), measured with
`evaluate_sentences` on held-out running text.

Note the sparse reranker is 1.00 MB raw but **0.01 MB after Brotli** (132x — the
table is mostly zeros), so on the wire it is close to free and is never a size
target.

### Why the isolated-word benchmark cannot see this

`evaluate_aksharantar` scores 4,101 *isolated* words. It never supplies a
preceding word, so the word-bigram table contributes exactly 0.00% to it by
construction. Anything context-dependent must be measured with
`evaluate_sentences`, and anything about loose romanization with
`evaluate --dataset data/eval/test_multiref.jsonl`.

## 4. Loss-Free Pruning & Optimization Methodology

Through empirical ablation, we discovered two major optimization opportunities that reduced model size by **22.2%** while actually **increasing transliteration accuracy**:

### A. Pruning Unreferenced Non-Nepali Aksharas (+0.19% Top-1, +1.4 MB saved)
* **Problem:** The EM source-channel model is trained on a merged Devanagari dataset (Hindi + Nepali) consisting of 3,588,793 parallel pairs. This created 16,562 unique aksharas, many of which are rare Sanskrit/Hindi combinations (e.g. `र्द्त`, `ग्स`, `ण़`, `फे़`) or OCR artifacts that never appear in authentic Nepali text.
* **Mechanism:** In `src/bin/build/prune_model.rs`, we scan the authentic Nepali vocabulary corpus (`data/word_freq_text.bin`, 470k words) and collect all reachable akshara IDs (`seen_aks`, ~5,941 aksharas). We prune:
  1. All emission rows where $a \notin \text{seen\_aks}$ (clearing 7,415 dead emission rows).
  2. All bigram transitions $(a \to b)$ where $a \notin \text{seen\_aks}$ or $b \notin \text{seen\_aks}$ (removing 45,207 unused transitions).
  3. All trigram contexts $(a, b \to c)$ where $a, b, \text{ or } c \notin \text{seen\_aks}$ (removing 49,089 unused transitions).
* **Empirical Impact:**
  * **AK-Freq Top-1 increased from 81.93% $\to$ 82.12%** (+4 test cases solved).
  * **AK-NEF Top-1 increased from 29.25% $\to$ 29.74%** (+5 test cases solved).
  * **Why accuracy improved:** Dead non-Nepali aksharas had peaked emission probabilities that previously crowded valid Nepali candidate paths out of the top-$k$ beam search lattice. Pruning them cleared the search space for true Nepali spellings.

### B. Pruning Tail Word Bigrams with Frequency $< 5$ (18.1 MB saved)
* **Problem:** The raw bigram table (`word_pairs.csv`) contained 1,249,270 bigram pairs. Over 567,564 pairs (45.4%) occurred only 1 to 4 times across the entire multi-gigabyte crawl (often web noise, dates, or rare names).
* **Mechanism:** In `src/bin/build/build_bigrams.rs` and `src/bin/build/pack_model.rs`, `--bigram-min-freq 5` filters out these tail bigrams:
  * Pairs retained: 681,706 (54.6% of pairs, covering all frequent phrase transitions).
  * Context heads: reduced from 102,534 to 58,792.
  * Size: reduced from **39.38 MB $\to$ 21.28 MB** (saving **18.1 MB**).
* **Empirical Impact:** **0.000% change** in single-word accuracy, with no noticeable degradation in conversational next-word predictions.

### C. Compact Container Encoding (v3) — 66.59 MB to 30.59 MB, accuracy-neutral

The largest saving came from *encoding*, not from discarding anything. The
container previously stored `Vec<Vec<(u32, f32)>>` and `HashMap<String, u32>`
directly: 8 bytes per n-gram transition where the id delta needs ~11 bits and
the weight needs 8, and 37.1 bytes per vocabulary word where 25 of those are
UTF-8 text at 3 bytes per Devanagari codepoint.

`core::codec` re-encodes on save. Cumulative effect on the browser profile:

| Step | Raw | Brotli | AK-Freq Top-1 |
| :--- | ---: | ---: | ---: |
| original bincode container | 66.59 MB | ~18 MB | 82.12% |
| CSR + delta varints + 8-bit codebook | 26.74 MB | 9.92 MB | 81.78% |
| + front-coded akshara-id vocabulary | 12.51 MB | 7.74 MB | 81.78% |
| + akshara table compaction | 12.20 MB | 7.57 MB | **82.02%** |
| + packed roman chunks | 11.50 MB | 6.87 MB | 82.02% |
| + delta-encoded trigram context keys | 11.06 MB | 6.58 MB | 82.02% |

Three findings worth keeping:

* **Akshara compaction raises accuracy** (81.78 → 82.02). The model listed
  16,562 aksharas but only 6,329 are reachable: `prune_model` cleared the dead
  emission rows and left the entries behind. As with the earlier pruning result,
  removing them clears the beam. It also drops the maximum id below 16,384, so
  absolute id varints shrink from three bytes to two.
* **Top-5 never moves** (92.1–92.2% across every configuration). Quantization
  and pruning reorder the top of the list; they do not lose candidates.
* **Raw savings are not wire savings for text.** Packing the chunk list saved
  0.70 MB raw but only 0.07 MB compressed — Brotli already had it. Only the
  high-entropy sections (the weights) yield savings that survive compression.

8-bit codebook quantization was verified against a 4096-level control that
reproduces the f32 baseline exactly, confirming the small 8-bit deltas are
quantization noise rather than a defect:

| Levels | AK-Freq@1 | AK-NEF@1 | AK-NEI@1 |
| :--- | ---: | ---: | ---: |
| baseline f32 | 82.12% | 29.74% | 47.96% |
| 4096 (control) | 82.07% | 29.87% | 47.96% |
| 256 (shipped) | 81.78% | 30.48% | 47.19% |

### D. Relative-Entropy LM Pruning — the browser profile

Frequency cutoffs are a poor proxy for usefulness: they delete rare but
informative transitions and keep frequent but predictable ones. `prune_lm`
instead drops a trigram when the backoff path already reproduces it, weighted by
how often it is consulted:

    contribution = exp(-w) * | w - (trigram_backoff(a,b) + bigram_weight(b,c)) |

This is the Stolcke criterion with training counts replaced by the model's own
probabilities, which is what a pack-time tool has available. The threshold is
the size/accuracy dial:

| threshold | trigrams kept | raw | Brotli | AK-Freq Top-1 | Top-5 |
| --: | --: | --: | --: | --: | --: |
| 0 | 100% | 11.06 MB | 6.58 MB | 82.02% | 92.17% |
| 5e-3 | 90.3% | 10.01 MB | 5.85 MB | 81.45% | 92.08% |
| 1e-2 | 64.1% | 9.61 MB | 5.52 MB | 81.36% | 92.13% |
| 2e-2 | — | 9.18 MB | 5.15 MB | 81.21% | 92.13% |
| **3e-2** (shipped) | 47.2% | **8.91 MB** | **4.92 MB** | **81.07%** | **92.22%** |

Build it with `make web-model` (`TRIGRAM_THRESHOLD` overrides the default).

## 5. Unified One-Shot Trainer (`src/bin/train/train.rs`)

The trainer consolidates four distinct compilation stages into a single unified command:

```
[Phase 1/4] Training EM Source-Channel Transliteration Model...
[Phase 2/4] Compiling Vocabulary & Empirical Frequencies...
[Phase 3/4] Training Discriminative Reranker...
[Phase 4/4] Packing Unified Model -> data/akshar.model ...
```

### Why Did the Reranker Default to 100,000 Words?
A common point of confusion is whether the engine was only trained on 100,000 words.
* **Phase 1 (The Core Engine):** **ALREADY trains on all 3,588,793 parallel pairs.** Every single word in `train_devanagari.jsonl` is processed by the EM algorithm.
* **Phase 3 (The Reranker):** This is a *discriminative tuning step* that optimizes 29 dense weights and $2^{20}$ sparse hash slots. Because it requires full beam search decoding for every word:
  1. 100,000 pairs take ~20 seconds to decode.
  2. 3.59 million pairs take ~15–20 minutes to decode.
  3. Discriminative weights for linear phonetic patterns saturate after 100k–200k diverse word shapes.
* **Full-Corpus Option:** We updated `train.rs` so that passing `--reranker-pairs 0` runs Phase 3 on **all 3,588,793 words**.

### Reranker Memory & Speed Optimization
In `src/bin/train/train.rs`, candidate caching was refactored:
* **Previous behavior:** Cached entire candidate structs, strings, and feature matrices, consuming several gigabytes of RAM.
* **Optimized behavior:** Because the 29 dense weights and feature scalers are static, the composite base score ($S_{\text{base}} = \sum w_k \cdot \hat{f}_k$) is computed *once* during pre-decoding. The cached item only stores `(target_idx, base_scores, sparse_features)`.
* **Result:** RAM consumption dropped by **>80%**, and subsequent training epochs execute nearly instantaneously in memory.

---

## 6. Complete Command Reference

### A. Full Whole-Corpus Training (Overnight Run)
To train EM on all 3.59M pairs and run the discriminative reranker across the entire dataset:
```bash
cargo run --release --bin train -- \
  --pairs data/aksharantar/train_devanagari.jsonl \
  --text data/store/corpus_clean.txt \
  --out data/akshar.model \
  --reranker-pairs 0 \
  --epochs 5 \
  --bigram-min-freq 5
```

### B. Fast Prototyping & Smoke Testing
To verify the entire end-to-end training pipeline in ~20 seconds:
```bash
cargo run --release --bin train -- --smoke --out data/smoke.model
```

### C. Inspecting Model Structure & Size
To see the exact byte and percentage breakdown of any unified model file:
```bash
cargo run --release --bin probe_model -- --model data/akshar.model --inspect
```

### D. Packing & Pruning Models
```bash
# 1. Prune unused non-Nepali aksharas from a translit model:
cargo run --release --bin prune_model -- \
  --model data/translit_model.bin \
  --vocab data/word_freq_text.bin \
  --out data/translit_model.bin

# 2. Build bigram table with custom threshold (e.g. freq >= 5):
cargo run --release --bin build_bigrams -- --min-freq 5 --out data/word_bigrams.bin

# 3. Pack unified model with bigram threshold on the fly:
cargo run --release --bin pack_model -- \
  --bigram-min-freq 5 \
  --out data/akshar.model

# 4. Pack lightweight WASM profile (no bigrams, 47 MB):
cargo run --release --bin pack_model -- \
  --no-bigrams \
  --out data/akshar_wasm.model
```

### E. Benchmarking Accuracy
```bash
# Evaluate active model on official Aksharantar test split (4,101 pairs):
cargo run --release --bin evaluate_aksharantar -- --model data/akshar.model

# Evaluate custom model:
cargo run --release --bin evaluate_aksharantar -- --model data/akshar_wasm.model
```

---

## 7. Research-Backed Lightweight Maths (no size growth)

Research (Stanford SLP3 Kneser-Ney C, SymSpell symmetric deletes, Aksharantar IndicXlit/NADIR 2025) was reviewed for `corpus_clean.txt`. Heavy options were **rejected** as out-of-scope if they add `~50 MB` for `1-2%`:

* **Rejected:** full `IndicXlit 11M transformer` `~40 MB` `+15% Dakshina` and `NADIR differential MoE NAR 13x` `~50 MB` — accuracy up but breaks `4.92 MB Brotli` browser budget. Kept as offline oracle only.
* **Rejected:** neural char LM `P_neural(w)` interpolated `λ1 P_KN + λ2 P_morph + λ3 P_neural` — `5-10 MB` for `<1%` on tail.

**Shipped lightweight maths (0 byte wire impact):**

1. **Modified Kneser-Ney (Chen-Goodman 3 discounts)** — single `d=0.75` under-discounts singletons, over-discounts `3+`. Modified uses `d1` for `c=1`, `d2` for `2`, `d3+` for `≥3` with `Y=n1/(n1+2n2)` `d1=1-2Y n2/n1` `d2=2-3Y n3/n2` `d3=3-4Y n4/n3` and continuation `Pcont(w)=|{v:C(vw)>0}|/types`. Fixes `Kong/Hong Kong` narrow frequent words. Applied to `AkLm` syllable `bigram/trigram` and `PairModel` `bi/bi_ak/tri` in `em_trainer.rs:895`.
2. **Phonetic-weighted SymSpell** — core `SymSpell` stays delete-only `25 vs 3M` `0.033ms` but ranking uses learned `cost(edit|phonetic)` not uniform `1`. `ee→e`, `sh→s`, `ph→f` `0.2`, random `k→z` `1.0` via collapsed variant + `freq_boost ln(1+freq)*1000` with `corpus_fuzzy_base 850k` in `engine.rs:669`. `shubheeksha→shubheksha→शुभेक्षा` now `0.3` not `1`.

Both keep `8.91 MB / 4.92 MB Brotli` and `sub-ms` `FxHash+cache+beam env` intact.
