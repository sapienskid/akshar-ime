# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Nepali-native details:** digits map to Devanagari numerals (123 → १२३)
  and a trailing `.` offers purnabiram (namaste. → नमस्ते।).
- **Fast:** Sub-millisecond keystroke latency (0.4–0.8 ms), single generative decoder.
- **SOTA Transliteration Core:** Outperforms neural baselines (IndicXlit top-1: 80.25% vs AksharIME top-1: **81.93%**, top-5: **91.94%** on held-out Aksharantar native test). Combines an EM-trained source-channel model (`P(roman | akshara)` over 3.59M pairs) with a Kneser-Ney syllable trigram LM, candidate union decoding, and a canonical discriminative log-linear reranker (29 dense shape/frequency/morphology features + $2^{20}$-slot sparse lexicalized table). Zero neural network runtime dependencies, 100% classical and memory-safe.
- **Adaptive Learning:** the IME learns your vocabulary and spelling variants
  in real time; the words you use most frequently appear first.
- **Fuzzy Search:** finds the correct words even with spelling mistakes in
  Roman script.
- **Context-Aware:** suggestions are re-ranked based on the words you've just
  typed.

## Architectural Overview

The engine is a modular, pure-Rust core with a C-API for integration with the
IBus input framework on Linux, and a WebAssembly (WASM) interface for zero-latency in-browser typing.

```
+-------------------------------------------------------------------+
|                        IBus Engine (C Layer)                      |
| (Handles key events, UI updates, communication with the OS)       |
+---------------------------------^---------------------------------+
                                  | (FFI: C-API / WebAssembly)
+---------------------------------v---------------------------------+
|                        IME Engine (Rust Core)                     |
|  engine.rs       — candidate union & suggestion orchestrator      |
|  decoder.rs      — persistent-path beam search over akshara lattice|
|  reranker.rs     — discriminative log-linear k-best reranker      |
|  normalizer.rs   — phonetic Roman input normalizer & skeletonizer |
|  translit_model  — EM emissions + Kneser-Ney syllable LM          |
|  wordtrie.rs     — compressed prefix Trie dictionary              |
|  lexicon.rs      — corpus roman→devanagari dictionary             |
|  trie/symspell   — user learning + typo tolerance                 |
|  context.rs      — phrase-level bigram re-ranking                 |
+-------------------------------------------------------------------+
```

For complete technical and mathematical details, see:
- [**System Architecture (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md): Complete system design, Mermaid data-flow sequence diagrams, and memory/latency benchmarks.
- [**Module Specifications (`docs/MODULES.md`)**](docs/MODULES.md): Mathematical formulations, derivations, and algorithmic implementations for all 15 core modules.
- [**WebAssembly & Browser Guide (`docs/WASM.md`)**](docs/WASM.md): Browser integration, zero-server deployment, and performance specs.

## Building and Installation

The engine is designed for Linux systems using the IBus input framework.

### Prerequisites

- A Rust toolchain (`rustc`, `cargo`)
- A C compiler (`gcc`)
- `ibus-1.0` and `jansson` development libraries.
- The Aksharantar Nepali corpus (`data/aksharantar/nep_{train,valid,test}.json`).

**On Debian/Ubuntu:**
```bash
sudo apt-get update
sudo apt-get install build-essential rustc cargo libibus-1.0-dev libjansson-dev
```

**On Fedora/CentOS:**
```bash
sudo dnf groupinstall "Development Tools" "Development Libraries"
sudo dnf install rust cargo ibus-devel jansson-devel
```

### Step 1 — Clone and get the data

```bash
git clone https://github.com/sapienskid/akshar-ime.git
cd akshar-ime
```

All data lives under `data/` (never committed to the repo). Because no data
is in the repository, there are no `make` targets for it — the download and
cleaning pipeline is private and documented with exact commands in
[data/README.md](data/README.md). For users, Option A below needs nothing
more.

(The word-pair corpus is the Aksharantar dataset, published by AI4Bharat on
[Hugging Face](https://huggingface.co/datasets/ai4bharat/Aksharantar);
running text comes from Nepali Wikipedia, CC100, and an akshar-ime news crawl.)

### Step 2 — Get the model artifacts

Two artifacts power the engine (both are gitignored):

- `translit_model.bin` (~32 MB) — the EM-trained transliteration table and
  syllable language model (built from the merged Devanagari word-pair set).
- `word_freq_text.bin` (~17 MB) — the vocabulary: 470k clean Devanagari words
  with usage frequencies (counted from the cleaned Wikipedia + CC100 + news
  corpus).

**Option A — download prebuilt artifacts** from the
[GitHub Releases](https://github.com/sapienskid/akshar-ime/releases) page into
`data/` (recommended; no training needed).

**Option B — build them locally:** follow the recipes in
[data/README.md](data/README.md) (download sources → `clean_aksharantar.py` →
`build_corpus.py` → `build_wordfreq_text` → `train_model`).

### Step 3 — Build and install

```bash
make
sudo make install
make restart-ibus
```

`make` compiles the Rust core and the C engine. `sudo make install` copies the
engine binary + library + IBus component + model artifacts into the system
directories (it only re-runs `make` if the artifacts aren't built, so you don't
need a Rust toolchain under `sudo`). `make restart-ibus` (no sudo) reloads your
IBus session.

> If `make install` ever needs to build under `sudo` on a rustup-managed
> system, pass the rustup home explicitly:
> `sudo env RUSTUP_HOME=$HOME/.rustup make install`

### Step 4 — Enable the input source

1. Open `Settings` → `Keyboard` → `Input Sources`.
2. Click `+`, search for **"Devanagari (Akshar)"**, and add it.
3. (Optional) Log out and back in so the input source list refreshes.

### Resetting the learned dictionary

```bash
make reset-learning
```

Removes `~/.config/akshar-devanagari/user_dictionary.bin` so the engine starts
with a clean learning history.

## WASM / Browser (any website, any `<input>`)

The engine also compiles to **WebAssembly** for use on any website — no server, fully offline, ~3 ms per keystroke.

```html
<input data-akshar placeholder="type: namaste" />
<script type="module">
  import { AksharIME } from './js/akshar-ime.js';
  await AksharIME.init({ modelUrl: '/data/translit_model.bin' });
  AksharIME.autoAttach(); // enhances all [data-akshar]
</script>
```

Type `namaste` → popup `नमस्ते` → `Enter`/`Tab`/`1`. Learned words persist in `localStorage`.

- **Build:** `make wasm` (or `./wasm/build.sh`) — outputs `wasm/pkg/` (377 KB wasm, 139 KB gzip / 111 KB brotli; +39 KB JS glue).
- **Demo:** `make wasm-serve` then open `http://localhost:8000/web/` (auto-picks free port if 8000 busy; override with `PORT=9000 make wasm-serve`).
- **Host:** serve `translit_model.bin` (21 MB → 11.1 MB gzip → 9.1 MB brotli) with `Cache-Control: immutable` + `Content-Encoding: br`. Lexicon is optional (118 MB → 24.4 MB gzip; lite mode saves transfer and is still 90%+ accurate).
- **Docs:** [`docs/WASM.md`](docs/WASM.md) (API, React/Vue examples, CDN, performance) · [`web/index.html`](web/index.html) live demo · [`js/akshar-ime.js`](js/akshar-ime.js) drop-in wrapper.

## Project Structure

```text
akshar-ime/
├── docs/                               # Comprehensive engineering & mathematical documentation
│   ├── ARCHITECTURE.md                 # System architecture, runtime sequence diagrams & memory specs
│   ├── MODULES.md                      # Complete algorithmic & mathematical spec for all 15 modules
│   ├── WASM.md                         # WebAssembly architecture, performance & browser integration
│   └── plans/                          # Historical design RFCs, math notes & milestone roadmaps
├── src/                                # Core Rust engine & platform bindings
│   ├── core/                           # Classical SOTA transliteration core (zero neural deps)
│   │   ├── akshara.rs                  # Devanagari syllable segmentation & boundary detection
│   │   ├── alignment.rs                # Dynamic programming char-level alignment (seeds EM)
│   │   ├── context.rs                  # Phrase-level bigram language model & re-ranking
│   │   ├── crf.rs                      # Conditional Random Field sequence model
│   │   ├── decoder.rs                  # Persistent-path beam search over akshara lattice
│   │   ├── em_trainer.rs               # Expectation-Maximization source-channel trainer
│   │   ├── engine.rs                   # IME coordinator, candidate union & suggestion lifecycle
│   │   ├── lexicon.rs                  # Exact binary roman-to-Devanagari corpus dictionary
│   │   ├── normalizer.rs               # Phonetic Roman input normalizer & skeletonizer
│   │   ├── pair_model.rs               # Joint pair sequence transliteration model
│   │   ├── reranker.rs                 # Discriminative log-linear k-best reranker (29 dense features)
│   │   ├── reranker_weights.rs         # Statically baked dense feature weights
│   │   ├── translit_model.rs           # EM emissions table + Kneser-Ney syllable LM
│   │   ├── trie.rs                     # Dynamic Trie for user-learned vocabulary & Roman variants
│   │   ├── types.rs                    # Core type definitions (WordId, WordMetadata, TranslitModel)
│   │   └── wordtrie.rs                 # Compressed prefix Trie over Devanagari vocabulary
│   ├── fuzzy/                          # Typo-tolerant candidate generation (SymSpell & orthography)
│   │   ├── grammar.rs                  # Phonetic Roman canonicalization & skeletonization
│   │   ├── mod.rs
│   │   └── symspell.rs                 # Symmetric delete spelling correction for Devanagari & Roman
│   ├── learning.rs                     # Real-time adaptive user dictionary learning
│   ├── persistence.rs                  # Memory-mapped user dictionary serialization
│   ├── c_api.rs                        # Foreign Function Interface (FFI) for C / IBus
│   ├── wasm.rs                         # WebAssembly FFI bindings & localStorage persistence
│   ├── lib.rs                          # Root crate library definition
│   ├── ibus_engine.c                   # Native Linux IBus engine integration (C layer)
│   └── bin/                            # Command-line tools organized by functional domain
│       ├── train/                      # Model training pipelines
│       │   ├── train_model.rs          # EM source-channel transliteration trainer
│       │   ├── train_reranker.rs       # Streaming discriminative log-linear reranker trainer
│       │   ├── train_matra.rs          # Factored matra confusion transition probability trainer
│       │   ├── train_pair_model.rs     # Aligned subword joint pair transliteration trainer
│       │   └── train_crf.rs            # CRF sequence model trainer
│       ├── evaluate/                   # Benchmarks, ablation harnesses & validation
│       │   ├── evaluate_aksharantar.rs # Held-out Aksharantar test suite evaluation
│       │   ├── evaluate_model.rs       # Core decoder accuracy & candidate dump harness
│       │   ├── evaluate_pair_model.rs  # Joint pair model benchmark harness
│       │   ├── evaluate_context.rs     # Phrase-level bigram context re-ranking harness
│       │   ├── evaluate_matra.rs       # Factored matra accuracy benchmark
│       │   ├── evaluate_candidate_union.rs # Multi-source candidate union smoke test
│       │   ├── evaluate_shrinkage.rs   # Empirical-Bayes frequency shrinkage benchmark
│       │   ├── evaluate_nepali_transliteration.rs # Out-of-domain Nepali evaluation
│       │   ├── analyze_errors.rs       # Error taxonomy analyzer & oracle bounds
│       │   ├── verify_parity.rs        # Cross-pipeline golden test parity verification
│       │   └── evaluate.rs             # Bootstrap confidence interval evaluation harness
│       └── build/                      # Data preprocessing & artifact generators
│           ├── build_wordfreq_text.rs  # Vocabulary frequency builder from 75M+ text tokens
│           ├── build_wordfreq.rs       # Frequency counter from parallel corpus pairs
│           ├── build_lexicon.rs        # Roman-to-Devanagari binary lexicon builder
│           ├── build_bigrams.rs        # Word bigram frequency builder from text
│           ├── build_lm_from_text.rs   # Syllable / word LM builder from running text
│           ├── build_morph.rs          # MDL morphological prefix/suffix table builder
│           ├── romanize.rs             # Backward transliteration / romanization utility
│           ├── probe_model.rs          # Interactive model emission & prediction inspector
│           └── prune_model.rs          # Model parameter pruning utility
├── data/                               # Dataset directory (gitignored; see data/README.md)
│   ├── aksharantar/                    # AI4Bharat Aksharantar parallel word-pairs
│   ├── raw/                            # Raw text dumps (Wikipedia, CC100, news crawl)
│   └── pipeline/                       # Data cleaning, deduplication & splitting scripts
├── js/akshar-ime.js                    # Zero-dependency browser helper for WASM integration
├── wasm/                               # WebAssembly packaging scripts & manifest
├── web/                                # Local interactive web demo
├── Makefile                            # Top-level build, test, install & WASM automation
└── devanagari-smart.xml                # Linux IBus component descriptor
```

## Training, Evaluation & Build Recipes

All binaries are mapped cleanly and can be executed with `cargo run --release --bin <name>`:

### 1. Training Workflows (`src/bin/train/`)
```bash
# Train the generative EM transliteration model & syllable LM:
cargo run --release --bin train_model -- --train data/corpus_clean.json --output data/translit_model.bin

# Train the discriminative log-linear reranker (dense weights + sparse hash table):
cargo run --release --bin train_reranker -- --train data/aksharantar/nep_train.json --epochs 5

# Train the factored vowel/matra confusion transition model:
cargo run --release --bin train_matra -- --train data/aksharantar/nep_train.json --output data/matra_transitions.bin
```

### 2. Evaluation & Benchmarking Workflows (`src/bin/evaluate/`)
```bash
# Run the official Aksharantar test benchmark (reports Top-1 and Top-5):
cargo run --release --bin evaluate_aksharantar

# Evaluate the multi-source candidate union oracle coverage:
cargo run --release --bin evaluate_candidate_union

# Evaluate the multi-core empirical-Bayes frequency shrinkage:
cargo run --release --bin evaluate_shrinkage

# Run error taxonomy and failure analysis:
cargo run --release --bin analyze_errors -- --dataset data/aksharantar/nep_test.json
```

### 3. Data & Artifact Construction (`src/bin/build/`)
```bash
# Build word frequency table from 75M+ cleaned running text tokens:
cargo run --release --bin build_wordfreq_text -- --input data/raw/nepali_text.txt --output data/word_freq_text.bin

# Build the exact Roman-to-Devanagari corpus lexicon:
cargo run --release --bin build_lexicon -- --corpus data/aksharantar/nep_train.json --output data/roman_lexicon.bin

# Build MDL morphological prefix/suffix segmenter tables:
cargo run --release --bin build_morph -- --vocab data/word_freq_text.bin --output data/morph_tables.bin
```

## Data & Attribution

The engine is built from three open datasets:

- **[Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar)** (AI4Bharat,
  IIT Madras; arXiv:2205.03018) — 3.59M roman→Devanagari word pairs (Hindi + Nepali,
  merged and strict-cleaned) that train the
  transliteration model (CC0, some portions CC-BY). The corpus is **not** included in this
  repository — download it separately (Step 1).
- **Nepali Wikipedia** (CC-BY-SA) and **CC100 Nepali** (CC0) — 75M tokens of running text
  providing the 570k-word frequency vocabulary. Only the derived word-frequency counts are
  shipped (`word_freq_text.bin`).
- **Your own typing** — the engine's adaptive learning happens entirely on-device
  (`~/.config/akshar-devanagari/user_dictionary.bin`); it never leaves your machine.

- The corpus itself is **not** included in this repository — data provenance,
  cleaning rules, and rebuild recipes are documented in [data/README.md](data/README.md).
- The 75M-token text vocabulary now also includes an akshar-ime news crawl
  (~34M tokens of current Nepali news) merged on top of Wikipedia + CC100 —
  see `data/README.md` and the scraping pipeline in `data/pipeline/` (private,
  untracked).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.