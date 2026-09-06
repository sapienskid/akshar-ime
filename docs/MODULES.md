# Akshar Devanagari IME — Comprehensive Module Reference & Mathematics

This document provides complete implementation specifications, mathematical derivations, data structures, and algorithmic flows for every module in Akshar Devanagari IME.

---

## 1. `src/core/akshara.rs` — Devanagari Syllable Segmentation

### Purpose & Linguistics
Devanagari is an abugida script: consonants carry an inherent vowel (the schwa `/a/`), modified by matras (dependent vowel signs) or suppressed by a virama/halanta (`्`, `U+094D`). Segmentation cannot occur at Unicode codepoint boundaries; it must group characters into linguistic orthographic syllables (**aksharas**).

### Formal Grammar (EBNF)
```ebnf
Akshara       ::= Vowel_Syllable | Consonant_Syllable | Non_Devanagari
Vowel_Syllable ::= Independent_Vowel [Anusvara | Visarga | Chandrabindu]*
Consonant_Syllable ::= Consonant [Nukta]? (Halanta Consonant [Nukta]?)* [Matra]? [Anusvara | Visarga | Chandrabindu]*
```

### Unicode Ranges Handled:
- **Consonants:** `U+0915..=U+0939`, `U+0958..=U+095F` (q, kh, g, z, r, rh, f, y).
- **Independent Vowels:** `U+0904..=U+0914`, `U+0960..=U+0961`.
- **Matras (Dependent Vowels):** `U+093E..=U+094C`, `U+0962..=U+0963`.
- **Halanta (Virama):** `U+094D`.
- **Diacritics:** Anusvara (`U+0902`), Visarga (`U+0903`), Chandrabindu (`U+0901`), Nukta (`U+093C`).

### Implementation Details:
- Implemented as a single-pass streaming finite state machine in `akshara::segment(s: &str) -> Vec<String>`.
- Preserves zero-width joiners (ZWJ `U+200D` and ZWNJ `U+200C`) when attached to viramas to maintain conjunct visual forms (e.g., eyelash-ra in Nepali: `र्‍`).
- Complexity: $O(N)$ time, $O(N)$ space where $N$ is string byte length.

---

## 2. `src/core/alignment.rs` — Emissive Forward-Backward Alignment

### Purpose & Mathematics
Given a roman string $R = r_1 r_2 \dots r_m$ and a gold Devanagari word segmented into aksharas $D = a_1 a_2 \dots a_n$, the alignment problem finds which contiguous Roman substring $s_j$ generated each Devanagari akshara $a_j$.

### Dynamic Programming Formulation:
Let $C(i, j)$ be the minimal alignment cost between prefix $R_{1..i}$ and akshara prefix $D_{1..j}$:
$$
C(i, j) = \min \begin{cases}
C(i, j - 1) + \text{cost}_{\text{del}}, & \text{(Deletion: akshara receives empty Roman chunk)} \\
\min_{1 \le l \le L} \big[ C(i - l, j - 1) + \text{dist}(R_{i-l+1..i}, a_j) \big], & \text{(Emission match: chunk of length } l \le 5 \text{)}
\end{cases}
$$

### Features:
- `align_emissive(roman, dev_aks, current_emissions)` returns `Vec<(Vec<u8>, u32)>` pairing each roman slice with its corresponding akshara ID.
- Resolves digraphs (e.g., `kh` $\to$ `ख`, `chh` $\to$ `छ`, `sh` $\to$ `श`/`ष`) and silent schwas automatically without hardcoded rule lists.

---

## 3. `src/core/translit_model.rs` — Syllable Transliteration Model

### Purpose & Compact Representation
Encapsulates all generative parameters for the source-channel model. Held in memory as directly indexed `Vec`s for decode speed; written to disk through `core::codec`, which re-encodes the n-gram tables compactly (see `core/codec.rs`). Loading fully deserializes the file — it is not memory-mapped.

### Bitwise Chunk Packing:
Roman chunks (up to length $L = 5$) are packed into single `u32` integers to eliminate heap allocations:
```
Byte 0: length (1..=5)
Byte 1..=4: ASCII character bytes
```
Implemented via `pack_chunk(bytes: &[u8]) -> u32` and `unpack_chunk(val: u32) -> String`.

### Mathematical Formulation:
1. **Emission Weights ($-\ln P(s \mid a)$):**
   $$
   w_{\text{emit}}(s \mid a) = -\ln \frac{C(a, s) + \alpha}{\sum_{s'} C(a, s') + \alpha \cdot |\mathcal{S}|}
   $$
2. **Kneser-Ney Syllable Language Model ($P_{\text{KN}}(a_i \mid a_{i-2}, a_{i-1})$):**
   Interpolated Kneser-Ney smoothing with absolute discount $\delta = 0.75$:
   $$
   P_{\text{KN}}(a_i \mid a_{i-1}) = \frac{\max(C(a_{i-1}, a_i) - \delta, 0)}{C(a_{i-1})} + \lambda(a_{i-1}) P_{\text{continuation}}(a_i)
   $$
   $$
   \lambda(a_{i-1}) = \frac{\delta}{C(a_{i-1})} \cdot |\{a : C(a_{i-1}, a) > 0\}|
   $$
   $$
   P_{\text{continuation}}(a_i) = \frac{|\{a' : C(a', a_i) > 0\}|}{\sum_{a''} |\{a' : C(a', a'') > 0\}|}
   $$

---

## 4. `src/core/em_trainer.rs` — Baum-Welch EM Training

### Purpose & Algorithmic Design
Trains the emission distribution $P(s \mid a)$ over 3,588,793 unaligned word pairs from the Aksharantar corpus.

### Algorithm (Expectation-Maximization):
1. **Initialization:**
   Seed emissions using character-level phonetic priors and deterministic heuristic alignments.
2. **E-Step (Expectation over Segmentations):**
   For each training pair $(R, D)$, compute forward trellis $\alpha(i, j)$ and backward trellis $\beta(i, j)$ in log-space:
   $$
   \alpha(i, j) = \sum_{l=1}^{\min(i, 5)} \alpha(i - l, j - 1) \cdot P(R_{i-l+1..i} \mid a_j)
   $$
   $$
   \beta(i, j) = \sum_{l=1}^{\min(m - i, 5)} \beta(i + l, j + 1) \cdot P(R_{i+1..i+l} \mid a_{j+1})
   $$
   The posterior probability that akshara $a_j$ consumed chunk $s = R_{i-l+1..i}$ is:
   $$
   \gamma(i, j, l) = \frac{\alpha(i-l, j-1) \cdot P(s \mid a_j) \cdot \beta(i, j)}{\alpha(m, n)}
   $$
3. **M-Step (Maximization & Normalization):**
   Accumulate expected counts across all pairs using thread-safe parallel accumulators:
   $$
   C(a, s) = \sum_{\text{pairs}} \sum_{j : D_j = a} \sum_{i, l} \gamma(i, j, l)
   $$
   Re-normalize with Dirichlet smoothing parameter $\alpha = 0.01$. Run for 12 iterations until convergence.

---

## 5. `src/core/decoder.rs` — Tropical Semiring Viterbi Lattice Beam Search

### Purpose & Complexity
Finds the top-$K$ Devanagari transliterations for a given Roman query prefix $R = r_1 \dots r_m$.

### Semiring Definition:
Operates over the **tropical semiring** $(\mathbb{R}^+ \cup \{\infty\}, \min, +)$, where additive costs represent negative log-probabilities. Shortest path corresponds to Maximum A Posteriori (MAP) transliteration.

### Beam Search Architecture:
- State Representation: `BeamState { pos, prev, prev2, emit, lm, phash, path }`
- At each position $i \in [1, m]$:
  1. Inspect all chunks $s = R_{i-l..i}$ of length $l \in [1, \min(i, 5)]$.
  2. For every akshara $a$ that can emit chunk $s$, compute:
     $$
     \text{cost} = \text{cost}_{\text{prev}} + w_{\text{emit}}(s \mid a) + \lambda_{\text{lm}} \cdot w_{\text{lm}}(a \mid \text{prev}, \text{prev2})
     $$
  3. Deduplicate states by context `(pos, prev2, prev, phash)` keeping only the minimal cost path.
  4. Truncate beam to width $B = 64$ (or 256 for offline evaluation).
- **Candidate Union (`decode_union`):**
  Intersects the generative beam with the vocabulary `WordTrie`. Any valid vocabulary word matching the Roman prefix that fell out of the beam is injected with its exact emission and LM score.

---

## 6. `src/core/wordtrie.rs` — Lexical Vocabulary Trie

### Purpose & Layout
An in-memory trie over the 470,012 cleaned Devanagari vocabulary words, keyed by **akshara id** (`HashMap<u32, usize>` per node), not by `char`.

### Data Structures:
```rust
pub struct WordTrieNode {
    pub is_terminal: bool,
    pub freq: u32,
    pub akshara_id: Option<u32>,
    pub children: HashMap<char, usize>,
}

pub struct WordTrie {
    pub nodes: Vec<WordTrieNode>,
}
```
### Algorithmic Features:
- Enables prefix lookups in $O(L)$ where $L$ is word length in characters.
- Injects full words directly into decoder beams without searching the entire generative state space.
- Memory footprint: $\approx 18$ MB heap for 470k words.

---

## 7. `src/core/reranker.rs` & `src/core/reranker_weights.rs` — Discriminative Log-Linear Softmax Reranker

### Mathematical Formulation
Given an input Roman string $x$ and a set of candidates $\mathcal{C}(x) = \{y_1, y_2, \dots, y_K\}$ produced by the decoder, the reranker models the conditional distribution:
$$
P(y \mid x) = \frac{\exp\big( \mathbf{w} \cdot \mathbf{\phi}(x, y) \big)}{\sum_{y' \in \mathcal{C}(x)} \exp\big( \mathbf{w} \cdot \mathbf{\phi}(x, y') \big)}
$$

### 29 Dense Features ($\mathbf{\phi}_{\text{dense}}$):
| Index | Feature Name | Description |
| :--- | :--- | :--- |
| 0 | `emission` | Decoder negative log emission score |
| 1 | `lm` | Kneser-Ney syllable trigram negative log score |
| 2 | `akshara_count` | Number of Devanagari aksharas |
| 3 | `decoder_rank` | Ordinal rank in generative decoder output |
| 4 | `heuristic` | Baseline heuristic: $\text{emit} + 0.85 \cdot \text{lm} - 0.75 \ln(1 + \text{freq})$ |
| 5 | `heuristic_rank` | Ordinal rank under the baseline heuristic |
| 6 | `log1p_freq` | $\ln(1 + \text{corpus\_freq})$ |
| 7 | `freq_rank_pct` | Percentile rank in vocabulary frequency table |
| 8 | `in_vocab` | Binary indicator ($1.0$ if word $\in$ vocabulary, else $0.0$) |
| 9 | `len_dev` | UTF-8 byte length of Devanagari string |
| 10 | `matra_total` | Total count of dependent vowel signs (matras) |
| 11..20 | `matra_profile` | Vector of 10 binary flags for individual matras (ा, ि, ी, ु, ू, े, ै, ो, ौ, ृ) |
| 21 | `nasals` | Count of anusvara (`ं`) and chandrabindu (`ँ`) |
| 22 | `visarga` | Count of visarga (`ः`) |
| 23 | `halants` | Count of viramas / half-letters (`्`) |
| 24 | `vowel_initial` | Binary flag indicating if word starts with independent vowel |
| 25 | `ends_matra` | Binary flag indicating if word ends in a matra |
| 26 | `ends_nasal_visarga`| Binary flag indicating if word ends in nasal or visarga |
| 27 | `len_roman` | Character length of input Roman string |
| 28 | `morph_log_freq` | Effective log-frequency derived from stem + suffix MDL morphology |

### Sparse Lexicalized Hash Table ($\mathbf{\phi}_{\text{sparse}}$):
Sparse interaction templates are hashed into a fixed $2^{20}$-slot table ($1,048,576$ entries) using 64-bit SplitMix hashing:
1. `LengthDelta`: Bucketed $(|D| - |R|)$.
2. `FinalAkshara x FinalRoman`: Captures final vowel/consonant spelling habits.
3. `FirstAkshara x FirstRoman`: Captures word-initial digraph mappings.
4. `Matra x PrecedingConsonant`: Direct confusion discriminator (e.g., `की` vs `कि`).
5. `MorphSuffix x RomanTail`: Postposition agreement (e.g., `लाई` $\times$ `lai`).
6. `FinalMatra x FinalRoman`: End-of-word vowel disambiguation.
7. `PenultimateConsonant x FinalMatra`: Sub-word morphological shape harmony.

### Quantization & Inference Optimization:
- Static sparse weights are quantized to signed 8-bit integers (`i8`):
  $$
  w_{\text{quantized}} = \text{clamp}\left( \text{round}\left( w \cdot \frac{127.0}{\max |w|} \right), -128, 127 \right)
  $$
- Inference cost evaluates $O(1)$ direct byte array lookups without string hashing allocations.
- Final candidate score is a variance-normalized Z-score blend ($\gamma = 0.3$) against the baseline heuristic.

---

## 8. `src/core/context.rs` — Corpus Bigram Context Layer

### Purpose & Mathematics
Re-ranks suggestions based on the previously committed word $w_{t-1}$.

### Mathematical Formulation:
Given committed word $w_{t-1}$, each candidate $w_t$ receives an additive evidence boost:
$$
\text{Score}_{\text{final}}(w_t \mid w_{t-1}) = \text{Score}(w_t) + \beta \cdot \ln(1 + C(w_{t-1}, w_t))
$$
Where $\beta = 40,000.0$ (calibrated via `evaluate_context`).

### Status: not shipped

The corpus word-bigram table described above was removed in `UnifiedModel` v4.
It cost 19.5 MB of the container for +0.16pp and is no longer built, packed or
loaded; `data/word_bigrams.bin` is not produced by the training pipeline.
Phrase-level context now comes from user-learned bigrams only
(`src/core/context.rs`). The formulation is kept here because the same additive
form is used for the user-learned counts.

---

## 9. `src/core/normalizer.rs` — Orthographic Normalization

### Purpose & Transformations
Handles typing variants and common Roman transliteration ambiguity prior to decoding:
- Collapses duplicate vowels: `kaal` $\to$ variants `kal` and `kaal`.
- Digraphs and phonetic alternations: `v` $\leftrightarrow$ `w`, `ee` $\to$ `i`/`ii`, `oo` $\to$ `u`/`uu`.
- Generates up to 6 candidate query variants ordered by edit cost.

---

## 10. `src/core/pair_model.rs` — Joint Syllable-Roman Transition Grammar

### Purpose & Architecture
Implements a Google Gboard-style Weighted Finite State Transducer (WFST) pair grammar (Hellsten et al., FSMNLP 2017):
- Models joint transitions over aligned pairs: $P(c_t, a_t \mid c_{t-1}, a_{t-1})$.
- Pairs are packed into 64-bit keys via `pair_key(akshara_id, chunk_key)`.
- Smooths sparse pair transitions with an underlying dense syllable trigram backoff model.

---

## 11. `src/core/engine.rs` — Unified IME Orchestrator

### Evidence Aggregation Scale
The engine combines candidate scores across all disparate modules onto a coherent $0 \dots 800,000$ scale:

| Source | Score Contribution | Purpose |
| :--- | :--- | :--- |
| **Fresh Generative Transliteration** | $0 \dots 800,000$ | Ranked candidates from the discriminative reranker |
| **Corpus Bigram Match** | Additive $+40,000 \cdot \ln(1 + C)$ | Context-conditioned boost from previously typed word |
| **User Learned Dictionary** | Base $+500,000$ + frequency | Guarantees previously confirmed words win top-1 |
| **SymSpell Fuzzy Match** | Base $+50,000 - 12,000 \cdot d$ | Recovers user typos up to Levenshtein distance 2 |

---

## 12. `src/fuzzy/` — SymSpell & Orthographic Skeletal Invariance

### Modules:
- **`symspell.rs`:** Symmetric Delete spelling correction. Precomputes deletions up to distance 2 for $O(1)$ typo recovery without full Levenshtein matrix computations.
- **`grammar.rs`:** Computes orthographic invariant skeletons:
  Maps confusable character sets (e.g., `श`/`ष`/`स`, `ब`/`व`, `ऋ`/`रि`, `ज्ञ`/`ग्य`, short/long vowels `ि`/`ी`) to identical canonical keys for typo-tolerant lookup.

---

## 13. `src/learning.rs` & `src/persistence.rs` — Local Adaptive Learning

### Capabilities:
- Records user-confirmed word selections in an in-memory prefix trie.
- Thread-safe serialization to `~/.config/akshar-devanagari/user_dictionary.bin`.
- Fully isolated and zero-telemetry.

---

## 14. `src/c_api.rs` & `src/ibus_engine.c` — Linux IBus Integration

### Foreign Function Interface (C-ABI):
- `akshar_ime_new() -> *mut ImeEngine`
- `akshar_ime_get_suggestions(engine, roman, limit, out_json) -> c_int`
- `akshar_ime_commit_word(engine, word) -> c_int`
- `akshar_ime_free(engine)`
Integrates with IBus event loops without blocking main thread X11/Wayland input pipelines.

---

## 15. `src/wasm.rs` — WebAssembly Interface

### Browser Integration:
- Exposes `WasmEngine` with drop-in JavaScript helper `js/akshar-ime.js`.
- Automatically persists user-learned vocabulary to browser `localStorage` under `akshar-ime-state-v1`.
- Fully offline, 0 HTTP network requests after initial model load.

---

## 16. `src/core/unified.rs` — Unified Model Container (`akshar.model`)

### Purpose & Single-Artifact Container Architecture
Instead of distributing 4 disparate binary artifacts (`translit_model.bin`, `word_freq_text.bin`, `reranker_weights_sparse.bin`, and `word_bigrams.bin`), `UnifiedModel` packages the entire engine into a single atomic binary file: `akshar.model`.

### Binary Wire Format:
```
+-------------------------------------------------------------------------+
| Magic Bytes: [0x41, 0x4B, 0x53, 0x48] ("AKSH")  (4 bytes)               |
+-------------------------------------------------------------------------+
| Version: u32 = 1                                (4 bytes)               |
+-------------------------------------------------------------------------+
| Payload: bincode-serialized UnifiedModel:                               |
|   1. translit: TranslitModel (EM emission tables + syllable LM)        |
|   2. sparse_reranker_table: Vec<f32> (2^20 table slots, 4 MB)           |
|   3. vocab_freq: HashMap<String, u32> (470k unigram words)              |
|   4. bigrams: Option<HashMap<(u32, u32), u32>> (phrase context)         |
+-------------------------------------------------------------------------+
```

### Loading & Fallback Semantics:
- `ImeEngine::new()` inspects `data/akshar.model` first (or `/usr/share/akshar-ime/akshar.model`).
- If present, it loads all model components in a single atomic I/O operation.
- If missing, it falls back seamlessly to multi-file legacy loading for backward compatibility.
- Streamlined training (`cargo run --release --bin train` or `make train`) produces `data/akshar.model` directly in one command.

---

## `core/codec.rs` — Compact Wire Encoding

The translation layer between the runtime model layout and its on-disk form.
Nothing here runs during decoding; it executes once on save and once on load.

The in-memory model is tuned for decode speed — `Vec<Vec<(u32, f32)>>` with
direct indexing — which is a poor thing to *store*: 8 bytes per transition,
where the id delta needs ~11 bits and the weight needs 8. Three techniques
close that gap, each measured before adoption:

| Technique | Applied to | Effect |
| :--- | :--- | :--- |
| CSR + delta varints | emissions, bigram LM, trigram LM | successor ids sorted ascending, stored as first differences |
| 256-entry Lloyd-Max codebook | all n-gram weights | 1 byte per weight; mean abs. error 0.011-0.031 nats |
| Front coding over akshara ids | 470k-word vocabulary | 37.1 B/word → ~5.4 B/word |
| `pack_chunk_bytes` into `u32` | 100,578 roman chunks | 1.18 MB → 0.48 MB |
| Delta-encoded sorted pairs + permutation | trigram context keys | chosen only when smaller than the plain layout |

**Every lossy-capable encoding verifies its own round trip and falls back to a
literal.** The vocabulary checks that `akshara::segment` rejoins to the original
string; chunk packing checks that `pack_chunk_bytes` reproduces it, since it
silently mangles anything outside `a-z`. A corrupted vocabulary is worse than a
large one, so these are checked rather than assumed.

Key API: `encode_adjacency` / `decode_adjacency`, `encode_weights` /
`decode_weights`, `encode_vocab` / `decode_vocab`, `encode_chunks` /
`decode_chunks`, `encode_pairs` / `decode_pairs`, `fit_codebook`.

---

## `core/holdout.rs` — Deterministic Corpus Holdout

A stable 1-in-N split shared by the trainer and `evaluate_sentences`, so
held-out text cannot leak into vocabulary counts, word bigrams or the EM model.

The split hashes the **sentence text** (FNV-1a 64), not the line number:
`corpus_clean.txt` concatenates Wikipedia, CC100 and a news crawl, so a tail
slice would sample a single source. FNV-1a rather than `DefaultHasher` because
the latter is explicitly not stable across Rust releases, and this decision has
to be reproducible across rebuilds.

Key API: `is_holdout(line, denom)`, `stable_hash`, `DEFAULT_HOLDOUT_DENOM` (200).
