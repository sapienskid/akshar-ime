# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Fast:** ~2-4ms keystroke latency, single generative decoder.
- **Generative transliteration core:** a source-channel model (EM-trained
  `P(roman | akshara)` over a 2.4M-pair corpus) with a Kneser-Ney akshara
  bigram/trigram language model. All probabilities are estimated by EM from the
  corpus.
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
|  decoder.rs      — beam search over the akshara lattice           |
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

### Installation

Clone the repository and use the provided `Makefile`:

```bash
git clone https://github.com/sapienskid/akshar-devanagari-ime.git
cd akshar-devanagari-ime
make install
```

The `make install` command will compile the Rust core, build the C engine, and install all necessary files into the system directories. It will also restart the IBus daemon to load the new engine.

After installation, you need to add the IME to your system's input sources:
1. Go to `Settings` > `Keyboard` > `Input Sources`.
2. Click `+` to add a new source.
3. Search for "Devanagari (Akshar)" and add it.
4. (Optional) Log out and log back in to ensure all changes are applied.

## Project Structure

- `src/`: The Rust source code for the core IME.
  - `core/`: The generative transliteration core.
    - `engine.rs`: The IME engine (decoder + lexicon + learning + context).
    - `decoder.rs`: Beam search over the akshara lattice.
    - `em_trainer.rs`: EM alignment trainer over the Aksharantar corpus.
    - `translit_model.rs`: Learned emissions + Kneser-Ney bigram/trigram LM.
    - `lexicon.rs`: Roman → Devanagari dictionary from the corpus.
    - `akshara.rs`: Devanagari syllable segmenter.
  - `fuzzy/`: Fuzzy search implementation (SymSpell).
  - `learning/`: The real-time learning module.
  - `persistence/`: Logic for saving/loading the user dictionary.
  - `c_api.rs`: The Foreign Function Interface (FFI) for the C layer.
- `src/bin/`: Training and evaluation tools (`train_model`, `build_lexicon`,
  `evaluate_model`, `evaluate_aksharantar`, `evaluate_nepali_transliteration`).
- `data/`: Aksharantar corpus (`aksharantar/`) and built artifacts.
- `src/ibus_engine.c`: The C code that integrates the Rust library with IBus.
- `Makefile`: The build and installation script.
- `devanagari-smart.xml`: The IBus component registration file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.