# Akshar Devanagari IME

**An intelligent, high-performance, and adaptive Input Method Engine for the Devanagari script.**

Akshar Devanagari IME is a next-generation input method built from the ground up for speed, efficiency, and intelligence. It learns from your typing patterns to provide incredibly accurate and fast suggestions, all while maintaining a minimal memory and CPU footprint.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Key Features

- **Blazing Fast Performance:** Sub-50ms keystroke latency and sub-100ms suggestion generation, achieved through cutting-edge algorithms like a **Pruning Radix Trie**.
- **Extremely Lightweight:** The entire installed engine is under 5MB, with a low memory footprint suitable for any machine.
- **Adaptive Learning:** The IME learns your vocabulary and spelling variations in real-time. The words and phrases you use most frequently will appear first.
**Intelligent Transliteration:** A sophisticated Syllable-Aware Finite State Transducer (FST) correctly handles complex Devanagari conjuncts (`ज्ञ`, `क्ष`, `त्र`), ya-phala (`्य`), rakar (`्र`), and other grammatical nuances.
- **Fuzzy Search:** The engine can find the correct words even if you make spelling mistakes in Roman script.
- **Context-Aware:** Suggestions are re-ranked based on the words you've just typed, making predictions for phrases more accurate.

## Architectural Overview

The engine is built on a modular, high-performance Rust core with a C-API for integration with the IBus input framework on Linux.

```
+-------------------------------------------------------------------+
|                        IBus Engine (C Layer)                      |
| (Handles key events, UI updates, communication with the OS)       |
+---------------------------------^---------------------------------+
                                  | (FFI: C-API)
+---------------------------------v---------------------------------+
|                        IME Engine (Rust Core)                     |
| (Orchestrates all logic, combines signals for final suggestions)  |
+----------------------+---------------------+----------------------+
| Romanization Engine  |  Prediction Engine  |    Learning Engine   |
| - Syllable-Aware FST |  - Pruning Radix Trie | - Real-time updates|
| - Grammatical Rules  |  - Fuzzy Search     | - Context Model      |
| - O(n) Transliteration |  - Top-K Pruning  | - Persistence        |
+----------------------+---------------------+----------------------+
```

For a deep dive into the algorithms that power this engine, see [ALGORITHMS.md](ALGORITHMS.md).

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
  - `core/`: The main data structures and logic (Trie, FST, Engine).
  - `fuzzy/`: Fuzzy search implementation (SymSpell).
  - `learning/`: The real-time learning module.
  - `persistence/`: Logic for saving/loading the user dictionary.
  - `c_api.rs`: The Foreign Function Interface (FFI) for the C layer.
- `src/ibus_engine.c`: The C code that integrates the Rust library with IBus.
- `Makefile`: The build and installation script.
- `devanagari-smart.xml`: The IBus component registration file.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.