# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Nepali-native details:** digits map to Devanagari numerals (123 → १२३)
  and a trailing `.` offers purnabiram (namaste. → नमस्ते।).
- **Fast:** Sub-millisecond keystroke latency (0.4–0.8 ms), single generative decoder.
- **SOTA Transliteration Core:** Outperforms neural baselines (IndicXlit top-1: 80.25% vs AksharIME top-1: **81.02%**, top-5: **91.75%** on held-out Aksharantar native test). Combines an EM-trained source-channel model (`P(roman | akshara)` over 3.59M pairs) with a Kneser-Ney syllable trigram LM, candidate union decoding, and a canonical discriminative log-linear reranker (29 dense shape/frequency/morphology features + $2^{20}$-slot sparse lexicalized table). Zero neural network runtime dependencies, 100% classical and memory-safe.
- **Adaptive Learning:** the IME learns your vocabulary and spelling variants
  in real time; the words you use most frequently appear first.
- **Fuzzy Search:** finds the correct words even with spelling mistakes in
  Roman script.
- **Context-Aware:** suggestions are re-ranked based on the words you've just
  typed.

## Architectural Overview

The engine is a modular, pure-Rust core with a C-API for integration with the
IBus input framework on Linux.

```
+-------------------------------------------------------------------+
|                        IBus Engine (C Layer)                      |
| (Handles key events, UI updates, communication with the OS)       |
+---------------------------------^---------------------------------+
                                  | (FFI: C-API)
+---------------------------------v---------------------------------+
|                        IME Engine (Rust Core)                     |
|  decoder.rs      — persistent-path beam search over akshara lattice|
|  reranker.rs     — discriminative reranking of the k-best list    |
|  translit_model  — EM emissions + Kneser-Ney LM                   |
|  lexicon.rs      — corpus roman→devanagari dictionary             |
|  trie/symspell   — user learning + typo tolerance                 |
|  context.rs      — phrase-level re-ranking                        |
+-------------------------------------------------------------------+
```

For a deep dive into the mathematics and measured results, see
[docs/plans/2026-08-01-generative-transliteration-design.md](docs/plans/2026-08-01-generative-transliteration-design.md).

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

- `src/`: The Rust source code for the core IME.
  - `core/`: The generative transliteration core.
    - `engine.rs`: The IME engine (decoder + lexicon + learning + context).
    - `decoder.rs`: Persistent-path beam search over the akshara lattice.
    - `reranker.rs`: Discriminative reranking of the decoder's k-best list.
    - `em_trainer.rs`: EM alignment trainer over the Aksharantar corpus.
    - `translit_model.rs`: Learned emissions + Kneser-Ney bigram/trigram LM.
    - `lexicon.rs`: Roman → Devanagari dictionary from the corpus.
    - `akshara.rs`: Devanagari syllable segmenter.
  - `fuzzy/`: Fuzzy search implementation (SymSpell).
  - `learning/`: The real-time learning module.
  - `persistence/`: Logic for saving/loading the user dictionary.
  - `c_api.rs`: The Foreign Function Interface (FFI) for the C layer (native only).
  - `wasm.rs`: WASM bindings (`WasmEngine`, `createEngine`, localStorage persistence).
- `js/akshar-ime.js`: Drop-in browser helper — `AksharIME.attach(input)` / `autoAttach()`.
- `wasm/`: WASM package build (`wasm/build.sh` → `wasm/pkg/`, `wasm/package.json` for npm).
- `web/index.html`: Local demo page for the WASM IME.
- `src/bin/`: Training and evaluation tools (`train_model`, `build_lexicon`,
  `train_reranker`, `evaluate`, `evaluate_model`, `evaluate_aksharantar`,
  `evaluate_nepali_transliteration`, `probe_model`).
- `data/`: The single data root (gitignored; see [data/README.md](data/README.md)) —
  Aksharantar corpus (`aksharantar/`), raw text dumps (`raw/`), scraping-pipeline
  store (`store/`), and built artifacts (`translit_model.bin`, `word_freq_text.bin`).
- `src/ibus_engine.c`: The C code that integrates the Rust library with IBus.
- `Makefile`: The build and installation script.
- `devanagari-smart.xml`: The IBus component registration file.

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