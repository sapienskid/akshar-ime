# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Fast:** ~3 ms keystroke latency (7 ms worst case), single generative decoder.
- **Generative transliteration core:** a source-channel model (EM-trained
  `P(roman | akshara)` over a 2.4M-pair corpus) with a Kneser-Ney akshara
  bigram/trigram language model. All probabilities are estimated by EM from the
  corpus — no neural network, no ML dependencies.
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

### Step 1 — Clone and get the corpus

```bash
git clone https://github.com/sapienskid/akshar-ime.git
cd akshar-ime
```

Download the Aksharantar Nepali split (train/valid/test) into `data/aksharantar/`.
The dataset is published by AI4Bharat on
[Hugging Face](https://huggingface.co/datasets/ai4bharat/Aksharantar) (Nepali
files: `nep_train.json`, `nep_valid.json`, `nep_test.json`).

### Step 2 — Build the model artifacts

The transliteration model and lexicon are generated from the corpus (they are
gitignored), so build them once:

```bash
cargo run --release --bin train_model      # ~2-3 min (threaded), EM + Kneser-Ney LM
cargo run --release --bin build_lexicon    # ~5 s, roman → Devanagari dictionary
```

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
  - `c_api.rs`: The Foreign Function Interface (FFI) for the C layer.
- `src/bin/`: Training and evaluation tools (`train_model`, `build_lexicon`,
  `train_reranker`, `evaluate_model`, `evaluate_aksharantar`,
  `evaluate_nepali_transliteration`, `probe_model`).
- `data/`: Aksharantar corpus (`aksharantar/`) and built artifacts.
- `src/ibus_engine.c`: The C code that integrates the Rust library with IBus.
- `Makefile`: The build and installation script.
- `devanagari-smart.xml`: The IBus component registration file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.