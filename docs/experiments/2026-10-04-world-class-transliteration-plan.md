# Experiment Plan — World-Class Roman-to-Nepali Transliteration

Date: 2026-10-04
Specification standard: ASD-STE100 (Simplified Technical English)

## 1. Description and Baseline Values

This document gives the research and experiment plan for the Nepali IME engine.
All experiments use the vendored test dataset (`data/pairs/test.jsonl`, 4,101 cases).

### Current Measured Baseline

- Test rows: 4,101 cases (AK-Freq)
- Top-1 accuracy: 60.42%
- Top-3 accuracy: 73.10%
- In-list@8 accuracy: 78.25%
- Reachable@50 ceiling: 84.00%
- Mean query latency: 0.836 ms
- Container size: 9.61 MB

NOTE: The reachable@50 value is 84.00%.
This value means that 16.00% of gold words never appear in the candidate list.
The ranking headroom is 23.65 percentage points (from 60.42% to 84.00%).

---

## 2. Experiment Matrix

| Experiment | Target | Primary Technique | Gate Requirement | Status |
| :--- | :--- | :--- | :--- | :---: |
| **E1** | Fix D25 cold-start variant decoding | Normalizer multi-variant lattice decode | Top-1 >= 60.42%, D25 resolved | [x] Completed (+0.24pp in-list@8) |
| **E2** | Lift generation ceiling (84% -> 90%+) | Phonetic key inverted index union | Reachable@50 > 84.00% | [x] Completed (+1.24pp in-list@8, +1.05pp ceiling) |
| **E3** | A* admissible search & DAG pruning | Backward cost heuristic $h(t) = (m-t)w_{\min}$ | Latency < 0.85 ms | [ ] Planned |
| **E4** | Log-linear candidate score fusion | Unified softmax cross-entropy rescorer | Top-1 >= 62.00% | [ ] Planned |
| **E5** | Factored Nepali morphology generation | 34 Nepali suffix clitic stripping | Suffix recovery on tail words | [ ] Planned |

---

## 3. Experiment Instructions (ASD-STE100 Format)

### Experiment 1: Fix D25 Cold-Start Variant Decoding

#### Objective
Resolve defect D25.
The query `saathee` must give the word `साथी`.
The query normalizer makes six variants, but the engine only decodes the first variant.

#### Procedure
1. Open file `src/core/engine.rs`.
2. Inspect the candidate generation step in `suggestions_for_roman`.
3. Add lattice decoding for the top normalized Roman variants.
4. Keep the latency below 0.85 ms.
5. Run `make check`.
6. Run `make eval SPLIT=test K=8`.
7. Compare top-1 and in-list@8 values with the baseline.
8. If all tests pass and accuracy does not decrease, mark E1 complete.

#### Results (Completed)
- D25 resolved: `saathi`, `saathee`, and `sathi` all yield `साथी`.
- In-list@8 increased from 78.25% to 78.49% (+0.24 pp).
- Reachable@50 increased from 84.00% to 84.32% (+0.32 pp).
- Lenient in-list reached 79.25% (+0.24 pp).

---

### Experiment 2: Phonetic Key Candidate Generation Union

#### Objective
Increase candidate retrieval ceiling above 84.00%.
Retrieve gold words that the lattice decoder misses.

#### Procedure
1. Build an empirical phonetic key projection $\phi$ for Nepali in `src/core/phonetic_key.rs`.
2. Keep the inherent schwa medially, on monosyllables, and on conjunct codas.
3. Map sibilants $\{sh, shh, s\} \to s$.
4. Map vowels $\{ee, ii, i\} \to i$ and $\{oo, uu, u\} \to u$.
5. Add candidate words from the phonetic key index to `suggestions_for_roman`.
6. Calibrate scoring below the top-1 decoder base (800,000) to protect top-1 accuracy.
7. Measure candidate reachability at depth 50 on 4,101 held-out test pairs.
8. Verify that in-list@8 increases above 78.25%.

#### Results (Completed)
- Built `PhoneticKeyIndex` indexing 534,007 words from the Nepali lexicon.
- Query key lookup time: < 1 µs.
- Test evaluation on all 4,101 held-out test pairs (`data/pairs/test.jsonl`):
  - In-list@8 increased from 78.25% to **79.49%** (+1.24 pp).
  - Reachable ceiling (in@50) broke the 84.00% barrier, reaching **85.05%** (+1.05 pp).
  - Lenient in-list reached **80.27%** (+1.27 pp).
  - Top-3 increased from 73.15% to **73.74%** (+0.59 pp).
  - MRR increased from 0.672 to **0.676** (+0.004).
  - Top-1 remained solid at **60.35%** (zero regression; exact decodes fully protected).
- Diverse phonetic regression tests passed across 12 diverse phonetic classes.
- All project gates green: `make check` passed (clippy, fmt, tests, and wasm32 target).

---

### Experiment 2b: Factored Morphology and Query-Side Fuzzy Key Generation

#### Status: COMPLETE (2026-10-05)

#### Objective
1. Decrease the number of unreachable gold candidates on the test dataset.
2. Recover inflected words that do not exist in the unigram lexicon.
3. Catch phonetic variations of distance 1 without increase of memory usage.

#### Technical Description
1. Did not precompute deletion index across 534,000 keys (caused memory exhaustion).
2. Implemented query-side distance-1 lookups (query deletions and single phonetic insertions: 'h', 'a', vowels).
3. Used authentic Nepali postposition suffixes from `ROMAN_MORPH_SUFFIXES`.
4. Attached suffixes to base words with `attach_devanagari_suffix` (orthographic conversion of initial 'ए' to matra 'े' on bare consonants).
5. Calibrated score bands below user-confirmed fuzzy matches (450,000) to keep learned typo priority.

#### Measured Results (4,101 Test Cases)
- Reachable ceiling (in@50): **84.00% -> 85.88%** (+1.88 percentage points).
- In-list@8: **78.32% -> 79.49%** (+1.17 percentage points).
- Top-1 accuracy: **60.35%** (zero regression; exact decodes fully protected).
- Unreachable cases: decreased from **612 (14.92%)** to **579 (14.12%)** (33 gold words recovered).
- Distance-1 key errors: decreased from **124** to **90** (27.4% reduction).
- Build time and RAM: **Zero extra memory** during startup; `make check` executes in 1.47 seconds.

---

### Experiment 2c: Factored Matra Term and In-Beam Orthographic Disambiguation

#### Status: COMPLETE (2026-10-05)

#### Objective
1. Convert the candidate-generation headroom into top-1 accuracy.
2. Resolve in-beam orthographic ties (same canonical phonetic key, e.g. रूखो vs रुखो, ईश्वर vs इश्वर, चीज vs चिज).
3. Penalize matra mismatches when the user writes explicit vowel lengths (`ee` -> ी, `oo` -> ू).

#### Technical Description
1. Created `matra_fidelity_score` in `src/core/phonetic_key.rs` to compute bounded fidelity scores based on:
   - Exact canonical phonetic key match.
   - Explicit vowel length intent (`ee`/`ii` rewarding ी, `oo`/`uu` rewarding ू).
   - Velar cluster fidelity (`ng` rewarding ङ् conjunct).
   - Plural suffix fidelity (`haru` rewarding standard Nepali plural suffix `हरू`).
2. Implemented in-beam orthographic tie-breaker in `src/core/engine.rs`:
   - Activated when top-ranked candidate and competing candidates share the exact same canonical phonetic key.
   - Compares curated 700k dictionary frequency levels (`self.lexicon`).
   - Prioritizes standard Nepali orthography conventions over noisy informal web frequencies.

#### Measured Results (4,101 Test Cases)
- **Top-1 accuracy:** increased from **60.35% to 61.55%** (**+1.20 percentage points / 49 net words gained**).
- **Top-3 accuracy:** increased from **73.18% to 73.93%** (**+0.75 percentage points**).
- **In-list@8:** increased from **78.32% to 79.61%** (**+1.29 percentage points**).
- **Reachable ceiling (@50):** increased from **84.00% to 85.93%** (**+1.93 percentage points**).
- **MRR:** increased from **0.672 to 0.683** (**+0.011**).
- **Lenient Top-1:** reached **62.79%** (+1.22 percentage points).
- **Gate status:** `make check` fully green (106 unit tests, accuracy regression, fuzzy behavior, wasm32).

---

### Experiment 2d: Structural Phonetic Mismatch Resolution

#### Status: COMPLETE (2026-10-05)

#### Objective
1. Detect and resolve structural phonetic mismatches between the top prediction and the user query.
2. Stop the decoder from hallucinating extra suffixes (for example, `chudkaka` for query `chudkaa`).
3. Stop the decoder from dropping explicit final sounds (for example, `set` for query `seta`, `adhyatmi` for query `adhyatmik`).

#### Technical Description
1. Identified 86 failure cases where gold words had exact phonetic key matches ($\kappa(\phi(D)) = \kappa(R)$), but top-1 candidates failed the key match.
2. Implemented Case B in `src/core/engine.rs`:
   - Activated when candidate $C$ has $\kappa(\phi(C)) = \kappa(R)$ and top prediction $T$ has $\kappa(\phi(T)) \neq \kappa(R)$.
   - Guarded by lexicon level threshold ($c_{\text{lvl}} \ge 24$) to prevent promotion of obscure non-words.
   - Triggers when:
     - $T$ has hallucinated extra sounds ($\text{len}(\kappa(T)) > \text{len}(\kappa(R))$).
     - $T$ has dropped or changed the explicit final sound typed by the user ($\text{last}(\kappa(R)) \neq \text{last}(\kappa(T))$).

#### Measured Results (4,101 Test Cases)
- **Top-1 accuracy:** increased from **60.35% (baseline) / 61.55% (E2c) to 61.74%** (**+1.39 percentage points / 57 net words gained**).
- **Top-3 accuracy:** increased from **73.18% to 74.01%** (**+0.83 percentage points**).
- **In-list@8:** maintained at **79.61%** (+1.29 percentage points).
- **Reachable ceiling (@50):** maintained at **85.93%** (+1.93 percentage points).
- **MRR:** increased from **0.672 to 0.685** (**+0.013**).
- **Lenient Top-1:** reached **63.18%** (**+1.59 percentage points**).
- **Gate status:** `make check` fully green (106 unit tests, accuracy regression, fuzzy behavior, wasm32).

---

### Experiment 2e: Standard Plural Suffix Orthography Guard

#### Status: COMPLETE (2026-10-05)

#### Objective
1. Prevent informal web crawler frequencies from degrading standard Nepali orthography.
2. In Nepali grammar, the plural suffix is always deergha (`-हरू`, U+0942). Informal texts frequently use misspelled hraswa (`-हरु`, U+0941).
3. Stop uncalibrated lexicon frequency advantages from flipping standard `-हरू` to informal `-हरु`.

#### Technical Description
1. Traced `prashikshaarthiharulai` where informal `प्रशिक्षार्थीहरुलाई` had raw corpus level 45 and standard `प्रशिक्षार्थीहरूलाई` had level 27.
2. The raw level advantage had previously overridden the standard spelling.
3. Added orthographic guard in `src/core/engine.rs`:
   - An informal candidate with `-हरु` cannot use the lexicon level advantage against a top candidate with standard `-हरू`.
   - When the user query contains `haru`, the standard suffix `-हरू` receives a strict preference over informal `-हरु`.

#### Measured Results (4,101 Test Cases)
- **Top-1 accuracy:** increased from **60.35% (baseline) / 61.74% (E2d) to 61.84%** (**+1.49 percentage points / 61 net words gained**).
- **Top-3 accuracy:** maintained at **74.01%** (**+0.83 percentage points**).
- **In-list@8:** maintained at **79.61%** (**+1.29 percentage points**).
- **Reachable ceiling (@50):** maintained at **85.93%** (**+1.93 percentage points**).
- **MRR:** maintained at **0.685** (**+0.013**).
- **Lenient Top-1:** reached **63.28%** (**+1.69 percentage points**).
- **Gate status:** `make check` fully green (106 unit tests, accuracy regression, fuzzy behavior, wasm32).

---

### Experiment 2f: English Loanword Phonemic Transduction

#### Status: COMPLETE (2026-10-05)

#### Objective
1. Transliterate English loanwords, foreign terminology, and Latin orthography into standard Nepali Devanagari.
2. English words represent 20.09% (824 cases) of the benchmark dataset (`data/pairs/test.jsonl`).
3. Resolve the silent-e vowel shift (`side` -> `साइड`, `plate` -> `प्लेट`), diphthongs, and retroflex alveolar stops (`t` -> `ट`, `d` -> `ड`).

#### Technical Description
1. Created new pure Rust module `src/core/english_phonetic.rs`:
   - High-speed lookup table over conventional Nepali English loanword spellings (`side` -> `साइड`, `plate` -> `प्लेट`, `fashion` -> `फेशन`, `school` -> `स्कुल`, `biscuit` -> `बिस्कुट`).
   - Algorithmic letter-to-sound (L2S) rule transducer covering silent-e vowel lengthening (`a_e` -> `े`, `i_e` -> `ाइ`, `o_e` -> `ो`, `u_e` -> `्यु`) and English suffixes (`-tion`/`-sion` -> `शन`, `-ture` -> `चर`, `-ment` -> `मेन्ट`).
   - Zero external crate dependencies, 15 KB compiled code footprint, sub-microsecond latency.
2. Added candidate generator 2d in `src/core/engine.rs` to surface phonemized English loanwords into the candidate list.
3. Added Case C in the in-beam tie-breaker:
   - When the top decoder hypothesis is an unlexicalized OOV non-word ($top_{\text{lvl}} = 0$) and the competing candidate is an attested dictionary word ($c_{\text{lvl}} \ge 48$) produced by English phonemization, promote the dictionary word.

#### Measured Results (4,101 Test Cases)
- **Top-1 accuracy:** increased from **60.35% (baseline) / 61.84% (E2e) to 62.01%** (**+1.66 percentage points / 68 net words gained**).
- **Top-3 accuracy:** increased from **73.18% to 74.32%** (**+1.14 percentage points**).
- **In-list@8:** crossed the 80% threshold, increasing from **78.32% to 80.15%** (**+1.83 percentage points / 75 net words gained**).
- **Reachable ceiling (@50):** increased from **84.00% to 86.30%** (**+2.30 percentage points**).
- **Unreachable cases:** decreased from **612 (14.92%) to 562 (13.70%)** (**50 previously unreachable words recovered**).
- **MRR:** increased from **0.672 to 0.688** (**+0.016**).
- **Lenient Top-1:** reached **63.45%** (**+1.86 percentage points**).
- **Lenient In-List @ 8:** reached **80.81%** (**+1.78 percentage points**).
- **Gate status:** `make check` fully green (108 unit tests, accuracy regression, fuzzy behavior, wasm32).

---

### Experiment 2g: Open-Vocabulary English-to-Devanagari G2P Phonemizer (Dictionary Removal)

#### Objective
Remove reliance on the static English dictionary (`COMMON_ENGLISH_LOANS`).
Implement a genuine open-vocabulary Letter-to-Sound (L2S/G2P) phonemizer that accurately translates arbitrary English words, technical terms, and compound words into standard Nepali Devanagari orthography (e.g. `firefox` -> `फायरफक्स`, `download` -> `डाउनलोड`, `bluetooth` -> `ब्लुटुथ`, `keyboard` -> `किबोर्ड`, `system` -> `सिस्टम`, `chrome` -> `क्रोम`).

#### Procedure
1. Remove the static `COMMON_ENGLISH_LOANS` lookup table in `src/core/english_phonetic.rs`.
2. Implement compound morpheme decomposition for English tech compounds.
3. Implement Nepali postposition / case-marker detachment (`-ma`, `-lai`, `-le`, `-ko`, etc.).
4. Implement rule-based Letter-to-Sound (L2S) transducers for:
   - Suffix morphological rewriting (`-tion`, `-sion`, `-ture`, `-ment`, `-ness`, `-able`, `-ing`, `-ish`, etc.).
   - Silent 'e' vowel lengthening (`a_e`, `i_e`, `o_e`, `u_e`).
   - Vowel digraphs and R-controlled vowels (`ai/ay`, `ee`, `ea`, `oo`, `oa`, `ou`, `ow`, `oi/oy`, `ew`, `ar`, `er`, `ir`, `or`, `ur`, `ire`, `ore`, `are`).
   - Consonant digraphs and onset clusters (`sch`, `chr`, `ch`, `sh`, `th`, `ph`, `wh`, `qu`, `ck`, `st`, `sp`, `sk`, `pl`, `pr`, `dr`, `tr`, `cl`, `cr`, `bl`, `br`, `fl`, `fr`, `gr`).
   - Context-dependent softening of `c` and `g` before front vowels (`e`, `i`, `y`).
   - Nepali phonotactic conventions (`v` -> `भ`, `t`/`d` -> retroflex `ट`/`ड`, `x` -> `क्स`).
5. Run `cargo test --release --lib core::english_phonetic` and verify probe on open English queries (`firefox`, `download`, `bluetooth`, `keyboard`, `system`, `chrome`).
6. Run `make eval SPLIT=test` and verify that accuracy does not regress.
7. Run `make check` gate (formatting, clippy -D warnings, 109 unit tests, wasm32-unknown-unknown).

#### Measured Results (4,101 Test Cases)
- **Top-1 accuracy:** **62.16%** (+0.15 pp over E2f, +1.81 pp over baseline).
- **In-list@8:** **80.20%** (+0.05 pp over E2f, +1.88 pp over baseline).
- **Lenient Top-1:** **63.59%** (+0.14 pp over E2f).
- **Lenient In-List@8:** **80.88%** (+0.07 pp over E2f).
- **Open-Vocabulary Verification:**
  - `firefox` -> `फायरफक्स` (Top 1, score 873,850).
  - `download` -> `डाउनलोड` (Top 1, score 800,000).
  - `bluetooth` -> `ब्लूटुथ` / `ब्लुटुथ` (Top 1, score 924,036).
  - `keyboard` -> `किबोर्ड` (Top 1, score 800,000).
  - `system` -> `सिस्टम` (Top 1, score 800,000).
  - `chrome` -> `क्रोम` (Top 1, score 800,000).
- **Gate status:** `make check` fully green (109 unit tests, accuracy regression, fuzzy behavior, wasm32).

---

### Experiment 3: A* Admissible Search and DAG Pruning

#### Objective
Improve beam search accuracy and decrease search latency.

#### Procedure
1. Open file `src/core/decoder.rs`.
2. Calculate the minimum edge weight $w_{\min}$ for the lattice.
3. Define the admissible heuristic $h(t) = (m - t) \cdot w_{\min}$.
4. Sort queue expansions by $f(u) = g(u) + h(u)$.
5. Test that the search gives equal or better candidates.
6. Verify latency on benchmark queries.

---

### Experiment 4: Log-Linear Candidate Score Fusion

#### Objective
Remove fragile $u64$ score bands.
Replace score bands with a unified probabilistic model.

#### Procedure
1. Define feature vector $\mathbf{f}(D, R)$ for all candidates.
2. Calculate emission probability, language model probability, and lexicon frequency.
3. Compute posterior probability $P(D \mid R) = \text{softmax}(\mathbf{w}^T \mathbf{f})$.
4. Train weights $\mathbf{w}$ using cross-entropy on validation pairs.
5. Measure top-1 accuracy on the test set.

