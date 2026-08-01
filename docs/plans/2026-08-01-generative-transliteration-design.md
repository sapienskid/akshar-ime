# Generative Transliteration Core — Design

Date: 2026-08-01
Status: Implemented

## Overview

The engine is a pure-Rust generative transliteration model trained by
expectation-maximisation over the Aksharantar Nepali corpus (2.4M roman →
Devanagari pairs). Devanagari is a regular abugida; the roman side carries the
ambiguity, so the model is built as a source-channel model:

```
P(D | R) ∝ P(R | D) · P(D)
```

* `P(R | D)` — a transliteration table `P(roman_chunk | akshara)` trained by EM.
* `P(D)` — a Kneser-Ney akshara n-gram language model.

Decoding is beam search over an akshara lattice. There is no neural network:
all parameters are counts converted to probabilities (EM, maximum likelihood,
absolute-discount smoothing) and the runtime is a shortest-path search over
negative-log weights.

## The transliteration model

Following Li, Zhang & Su (ACL-04, *A Joint Source-Channel Model for Machine
Transliteration*):

```
P(R | D) = sum over segmentations of R into chunks s_1..s_n aligned to the
           aksharas a_1..a_n of D, of  prod_j P(s_j | a_j)
```

### EM alignment
Each Devanagari word is segmented into aksharas (`akshara.rs`). The roman word
is a sequence of characters; a *segmentation* assigns each akshara a contiguous
chunk of 0..=MAX_CHUNK (5) roman characters. Forward-backward EM over the
alignment space maximises the corpus likelihood. Emissions are seeded from a
deterministic codepoint aligner and **schwa-augmented**: every bare-consonant
akshara also seeds its `chunk + "a"` (and schwa-dropped) forms, so EM can
discover the medial-schwa convention (`cha → च`).

The E-step is embarrassingly parallel over pairs and runs on `std::thread`
scoped threads.

### Kneser-Ney akshara LM
- `word_start`: `P(a | word start)` from corpus word-initial frequencies.
- `bigram`: `P_KN(a_j | a_{j-1})` with absolute-discount backoff.
- `trigram`: `P_KN(a_k | a_{j-1}, a_j)` — disambiguates word patterns such as
  the trigram एउटा resolving `eutako`.

### Decoder
Beam search over the akshara lattice. Each edge consumes 1..=MAX_CHUNK roman
chars and emits one akshara, weighted by emission + LM (tunable `lm_weight`,
default 1.0). Results are the k lowest-total-weight complete paths.

## Architecture

```
ImeEngine (src/core/engine.rs)          — one decoder, one evidence score
  ├─ ModelDecoder (decoder.rs)          — beam search over the akshara lattice
  ├─ TranslitModel (translit_model.rs)  — emissions + KN bigram/trigram LM
  ├─ RomanLexicon (lexicon.rs)          — corpus roman→devanagari (exact match)
  ├─ Trie / SymSpell / ContextModel     — user learning + typo tolerance
  └─ LearningEngine / persistence       — user-confirmation feedback
```

Offline tools:
- `train_model` — builds `data/translit_model.bin` (EM + LM), threaded.
- `build_lexicon` — builds `data/roman_lexicon.bin` from the corpus.
- `evaluate_model` / `evaluate_aksharantar` / `evaluate_nepali_transliteration`
  — benchmarks.

## Results (Aksharantar Nepali test, 4,101 words)

| Metric | ImeEngine | IndicXlit (neural, reference) |
|---|---|---|
| Native top-1 | 73.7% | 80.3% |
| Native top-5 | 90.4% | — |
| Named-entity top-1 | 28–39% | 52.7% |
| Keystroke latency | 2–4 ms | — |

Full held-out IME eval (`data/eval/aksharantar_test.tsv`, 4,101 real cases from
the Aksharantar test split): 54.5% top-1 / 74.9% top-5 overall.

## Rebuilding

```bash
cargo run --release --bin train_model    # ~2-3 min threaded, 12 EM iterations
cargo run --release --bin build_lexicon  # ~4 s
cargo run --release --bin evaluate_model -- --beam 128     # decoder-only, source-split
cargo run --release --bin evaluate_aksharantar             # IME engine, source-split
cargo run --release --bin evaluate_nepali_transliteration  # IME engine, full real TSV
```

## Known limitations

- **Vowel length is not encoded in the roman corpus** (`la` = both ल and ला).
  The model disambiguates with the LM; a real Nepali word-frequency corpus would
  close most of the remaining gap.
- Named entities and some >5-char conjuncts (e.g. `न्थ्यौं`) remain hard.
- The Aksharantar lexicon is news-mined compounds; prefix-completion floods
  with compounds, so the IME uses exact-match lexicon evidence only.
