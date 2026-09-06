# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Nepali-native details:** digits map to Devanagari numerals (123 → १२३)
  and a trailing `.` offers purnabiram (namaste. → नमस्ते।).
- **Small:** 8.91 MB browser model (4.94 MB Brotli), 11.37 MB on desktop, from a single unified container.
- **Transliteration core:** an EM-trained source-channel model (`P(roman | akshara)`, 3.59M pairs) with a modified Kneser-Ney syllable trigram LM, candidate-union decoding, and a discriminative log-linear reranker (29 dense shape/frequency/morphology features + a $2^{20}$-slot sparse lexicalized table). No neural network at runtime; pure safe Rust.
- **Adaptive Learning:** the IME learns your vocabulary and spelling variants
  in real time; the words you use most frequently appear first.
- **Fuzzy Search:** tolerates Roman spelling mistakes within edit distance 2
  over words you have confirmed before. Matches are distance-verified. A
  corpus-wide fuzzy source was removed on 2026-09-06: it cost 30.8pp of native
  top-1 and contributed no measured recall.
- **Context (opt-in):** phrase-level bigrams were removed (`19.5 MB` for `+0.16pp`, see `docs/MODEL_TRAINING_AND_OPTIMIZATION.md`). Context now comes from user-learned bigrams only.

## Measured performance

On the held-out AI4Bharat Aksharantar Nepali test split (4,101 cases), via
`evaluate_aksharantar` and `evaluate`. Measured 2026-09-06 on `data/akshar.model`
at default settings.

| Split | top-1 | top-5 |
| :--- | ---: | ---: |
| `AK-Freq` (native words, n=2,108) | 81.83% | 92.22% |
| `AK-NEI` (named entities, n=1,176) | 47.5% | 69.6% |
| `AK-NEF` (named entities, n=817) | 31.2% | 53.0% |
| All 4,101 cases | 62.0% | 78.0% |

Character error rate on `AK-Freq` top-1 is 3.90%; MRR over all cases is 0.690.
For reference, IndicXlit (an ~11M-parameter transformer) reports 80.25% top-1 on
the native split and 52.67% on named entities — so the native figure here is
comparable and the named-entity figure is well behind.

Query latency is **3.2–3.6 ms** at k=5–10 with beam 64, and cold start is ~2.4 s.
Neither is sub-millisecond; see `docs/plans/2026-09-06-repair-and-path-to-90.md`
for what stands between the current numbers and that target.

## Model profiles

| Profile | File | Size | Brotli |
| :--- | :--- | ---: | ---: |
| Desktop / IBus | `data/akshar.model` | 11.37 MB | 6.72 MB |
| Browser | `data/akshar_wasm.model` | 8.91 MB | 4.94 MB |

The browser profile's accuracy has not been re-measured since the 2026-09-06
engine changes; the desktop figures above should not be assumed to carry over.
Build it with `make web-model`; `TRIGRAM_THRESHOLD` trades size against accuracy
along a measured curve (see `docs/WASM.md`).

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
|  context.rs      — user history re-ranking                      |
+-------------------------------------------------------------------+
```

For complete technical and mathematical details, see:
- [**Model Training & Optimization Guide (`docs/MODEL_TRAINING_AND_OPTIMIZATION.md`)**](docs/MODEL_TRAINING_AND_OPTIMIZATION.md): End-to-end training pipeline, loss-free pruning methodology, and component ablation study.
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

### Step 2 — Get the model artifact

A single unified model container powers the engine:

- `akshar.model` (11.1 MB, 8.9 MB pruned for browser) — bundles the EM-trained transliteration model, Kneser-Ney syllable LM, 470k-word Devanagari vocabulary frequency distribution, and $2^{20}$-entry discriminative reranker weights into a single atomic binary.

**Option A — download prebuilt model:**
Download `akshar.model` from the [GitHub Releases](https://github.com/sapienskid/akshar-ime/releases) page into `data/akshar.model` (recommended; no training needed).

**Option B — train end-to-end with one command:**
```bash
cargo run --release --bin train
# or simply:
make train
```
This ingests the cleaned corpus, trains the EM transliteration model, builds the vocabulary, trains the reranker, packages `data/akshar.model`, and runs a self-verifying smoke test.

### Step 3 — Build and install

```bash
make
sudo make install
make restart-ibus
```

`make` compiles the Rust core and the C engine. `sudo make install` copies the
engine binary + library + IBus component + `akshar.model` into the system
directories (and automatically cleans up any legacy multi-file binaries from `/usr/share/akshar-ime/`). `make restart-ibus` (no sudo) reloads your
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

The engine also compiles to **WebAssembly** for use on any website — no server, fully offline. Browser latency has not been re-measured since 2026-09-06; the last figure was ~3–7 ms per `getSuggestions()` (see `docs/WASM.md`).

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
│   ├── MODULES.md                      # Complete algorithmic & mathematical spec for all 16 modules
│   ├── WASM.md                         # WebAssembly architecture, performance & browser integration
│   └── plans/                          # Historical design RFCs, math notes & milestone roadmaps
├── src/                                # Core Rust engine & platform bindings
│   ├── core/                           # Classical SOTA transliteration core (zero neural deps)
│   │   ├── akshara.rs                  # Devanagari syllable segmentation & boundary detection
│   │   ├── alignment.rs                # Dynamic programming char-level alignment (seeds EM)
│   │   ├── context.rs                  # User history language model & re-ranking
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
│   │   ├── unified.rs                  # Atomic single-file model container (akshar.model)
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
│   └── bin/                            # Streamlined command-line tools
│       ├── train/                      # End-to-end model training
│       │   └── train.rs                # One-shot training & packaging pipeline with smoke test
│       ├── evaluate/                   # Benchmarks, ablation harnesses & validation
│       │   ├── evaluate_aksharantar.rs # Held-out Aksharantar test split evaluation
│       │   ├── evaluate.rs             # Bootstrap confidence intervals & latency benchmark
│       │   └── analyze_errors.rs       # Error taxonomy analyzer & oracle bounds
│       └── build/                      # Data preprocessing & artifact packaging
│           ├── pack_model.rs           # Package binary tables into unified akshar.model container
│           ├── build_wordfreq_text.rs  # Vocabulary frequency builder from text corpus
│           ├── build_lexicon.rs        # Roman-to-Devanagari binary lexicon builder
│           ├── romanize.rs             # Backward transliteration / romanization utility
│           ├── probe_model.rs          # Interactive model emission & prediction inspector
│           └── prune_model.rs          # Model parameter pruning utility
├── data/                               # Dataset directory (gitignored; see data/README.md)
│   ├── akshar.model                    # Unified production model container (all components bundled)
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

All tools can be executed directly with `cargo run --release --bin <name>` or via `make`:

### 1. Unified One-Shot Training (`src/bin/train/`)
```bash
# Complete end-to-end training + packaging in a single command:
cargo run --release --bin train
# or:
make train

# Run a self-verification smoke test:
cargo run --release --bin train -- --smoke
```

### 2. Evaluation & Benchmarks (`src/bin/evaluate/`)
```bash
# Run the official Aksharantar test benchmark (reports Top-1 and Top-5):
cargo run --release --bin evaluate_aksharantar

# Run full evaluation with bootstrap 95% confidence intervals and latency:
cargo run --release --bin evaluate

# Run linguistic error taxonomy and oracle bounds analysis:
cargo run --release --bin analyze_errors
```

### 3. Packaging & Utilities (`src/bin/build/`)
```bash
# Package loose model binaries into a unified data/akshar.model container:
cargo run --release --bin pack_model
# or:
make pack

# Interactive inspector for syllable emissions and candidate completions:
cargo run --release --bin probe_model -- --roman namaste

# Build vocabulary frequencies from cleaned text:
cargo run --release --bin build_wordfreq_text -- data/store/corpus_clean.txt
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