# Akshar Devanagari IME — System Architecture & Data Flow

## 1. System Overview & Engineering Principles

Akshar Devanagari IME is an intelligent, high-performance input method engine for the Devanagari script (specifically optimized for Nepali and Hindi orthography). It achieves **state-of-the-art transliteration accuracy (81.93% top-1, 91.94% top-5)** on the standard AI4Bharat Aksharantar test benchmark, outperforming neural transliteration baselines (such as IndicXlit at 80.25% top-1) while requiring **zero neural network runtimes** and operating strictly within sub-millisecond latency budgets.

### Core Architectural Principles:
1. **Classical Statistical Transduction over Neural Dependencies:**
   All inference relies on exact shortest-path dynamic programming (Viterbi beam search in the tropical semiring), n-gram language models with Kneser-Ney smoothing, and a log-linear discriminative reranker with hash-table lexicalization. No PyTorch, ONNX, or C++ neural runtimes are needed.
2. **Strict WebAssembly Compatibility & Memory Budget:**
   The entire engine compiles to native Linux C-ABI binaries and WebAssembly (`wasm32-unknown-unknown`). Total compressed runtime footprint is $\le 9.1$ MB (Brotli), with memory consumption capped at $\approx 25$ MB on browser runtimes and $\approx 40$ MB on native systems.
3. **Sub-Millisecond Keystroke Latency:**
   Every keystroke is processed in $0.40 - 0.80$ ms on standard consumer CPUs, delivering immediate, flicker-free typing without background thread lag.
4. **Local On-Device Adaptive Learning:**
   User selections update a local, persistent prefix trie and frequency table without any telemetry or cloud round-trips.

---

## 2. High-Level Architecture

The architecture is divided into three primary tiers: Platform Integration, IME Orchestration, and the Core Statistical Engine.

```mermaid
graph TD
    subgraph UI_Platform ["Platform Integration Layer"]
        IBus["Linux IBus Engine (ibus_engine.c + c_api.rs)"]
        WASM["WebAssembly / Browser (wasm.rs + akshar-ime.js)"]
    end

    subgraph Orchestration ["IME Orchestration (engine.rs)"]
        ImeEngine["ImeEngine: Unified Evidence Scoring & Aggregation"]
        Learning["LearningEngine: On-Device Frequency & SymSpell (learning.rs)"]
        Context["ContextModel: Sentence & Bigram Context (context.rs)"]
    end

    subgraph Core_Engine ["Statistical Transliteration Core"]
        Normalizer["Orthographic Normalizer & Query Variant Expander (normalizer.rs)"]
        Decoder["ModelDecoder: Lattice Beam Search & Candidate Union (decoder.rs)"]
        WordTrie["WordTrie: 470k Vocabulary Lexical Trie (wordtrie.rs)"]
        TranslitModel["TranslitModel: Syllable Emissions + Kneser-Ney LM (translit_model.rs)"]
        Reranker["Discriminative Reranker: 29 Dense + 2^20 Sparse Hash (reranker.rs)"]
        RerankerWeights["Quantized Static Weights & Table (reranker_weights.rs)"]
    end

    IBus --> ImeEngine
    WASM --> ImeEngine
    ImeEngine --> Normalizer
    ImeEngine --> Decoder
    ImeEngine --> Learning
    ImeEngine --> Context
    Decoder --> TranslitModel
    Decoder --> WordTrie
    ImeEngine --> Reranker
    Reranker --> RerankerWeights
```

---

## 3. End-to-End Runtime Keystroke Data Flow

When a user types a keystroke in any text field, the data flows synchronously through deterministic transformation, generative decoding, candidate union, discriminative reranking, and multi-source evidence fusion.

```mermaid
sequenceDiagram
    autonumber
    actor User as User Keyboard
    participant FFI as Platform Interface (IBus / WASM)
    participant Engine as ImeEngine (engine.rs)
    participant Norm as Normalizer (normalizer.rs)
    participant Dec as Decoder & WordTrie (decoder.rs, wordtrie.rs)
    participant Rerank as Discriminative Reranker (reranker.rs)
    participant Ctx as Bigram Context (context.rs)
    participant Learn as Learning Engine (learning.rs)

    User->>FFI: Types keystroke (e.g. 'k', 'a', 'l')
    FFI->>Engine: get_suggestions("kal", count=5)
    
    rect rgb(240, 248, 255)
        note over Engine,Norm: Step 1: Normalization & Digits
        Engine->>Norm: expand_query_variants("kal")
        Norm-->>Engine: [QueryVariant("kal", cost=0.0)]
    end

    rect rgb(255, 250, 240)
        note over Engine,Dec: Step 2: Multi-Source Candidate Generation
        Engine->>Dec: decode_union("kal", k=50, word_trie)
        Dec->>Dec: Lattice Viterbi Beam Search (TranslitModel)
        Dec->>Dec: WordTrie Lexical Intersection
        Dec-->>Engine: 50 DecodedCandidate structs (emit, lm, aks_count, dev)
    end

    rect rgb(240, 255, 240)
        note over Engine,Rerank: Step 3: Feature Extraction & Discriminative Softmax
        Engine->>Rerank: rerank("kal", candidates, freq, ranks)
        Rerank->>Rerank: Extract 29 dense features (emission, LM, shape, morphology)
        Rerank->>Rerank: Hash 7 lexical templates into 2^20 table
        Rerank->>Rerank: Z-score blend: (1-γ)*(-heur) + γ*score (γ=0.3)
        Rerank-->>Engine: Ranked Devanagari candidates
    end

    rect rgb(255, 245, 245)
        note over Engine,Learn: Step 4: Evidence Scoring & Multi-Source Fusion
        Engine->>Engine: Convert rerank scores to FRESH_SCALE u64
        Engine->>Ctx: Apply bigram context boost (if last_word committed)
        Engine->>Learn: Check local user trie (USER_TRIE_BASE bonus)
        Engine->>Learn: Check SymSpell fuzzy edit distance
        Engine->>Engine: Sort by max evidence across sources
    end

    Engine-->>FFI: Final ordered candidate list: ["कल", "काल", "कला", ...]
    FFI-->>User: Render candidates in popup UI
```

### Detailed Runtime Steps:

1. **Query Ingestion & Normalization:**
   - Detects special cases (e.g., ASCII digits `0..=9` mapped to Devanagari numerals `०..=९`, trailing `.` mapped to purnabiram `।`).
   - Normalizes unicode sequences and generates canonical roman phonetic variants (handling `v`/`w` and digraphs).

2. **Lattice Beam Search & Candidate Union:**
   - Runs persistent-path Viterbi beam search across the akshara lattice using chunk emission tables $-\ln P(r \mid a)$ and Kneser-Ney syllable trigram LM probabilities.
   - Concurrently searches the 470,012-word `WordTrie` to recover real vocabulary words whose raw syllable probability fell below the beam threshold (recovers $+52$ gold test words).
   - Deduplicates candidates and outputs up to 50 `DecodedCandidate` items with exact split scores `(emit, lm, akshara_count)`.

3. **Discriminative Log-Linear Reranking:**
   - Computes 29 dense shape, frequency, and morphology features for each candidate.
   - Hashes 7 sparse lexical interaction templates into a static $2^{20}$-slot table (`i8` quantized, 1 MB).
   - Combines dense dot-product with sparse lookups and computes a variance-normalized Z-score blend ($\gamma = 0.3$) against the baseline heuristic.

4. **Multi-Source Evidence Fusion:**
   - Converts the reranker log-score into an unsigned 64-bit integer scale ($0 \dots 800,000$).
   - Injects contextual bonuses:
     - **Corpus Bigrams ($+40,000$ per ln unit):** Boosts candidates that form valid word bigrams with the previously committed word.
     - **User-Confirmed Words ($+500,000$ base):** User-confirmed words from previous sessions immediately outrank unseen transliterations.
     - **SymSpell Fuzzy Search ($+50,000$ base):** Tolerates roman typing mistakes within Levenshtein distance 2.
   - Deduplicates across all candidate generators and returns the top-$K$ suggestions to the UI.

---

## 4. Offline Training Data Flow Pipeline

The training architecture transforms raw multilingual corpora into ultra-compact, runtime-efficient binaries.

```mermaid
graph TD
    subgraph Data_Sources ["Raw Data Sources"]
        Aksharantar["Aksharantar Corpus: 3.59M pairs (Hindi + Nepali)"]
        Wikipedia["Nepali Wikipedia Dump (75M tokens)"]
        CC100["CC100 Clean Nepali CommonCrawl"]
        NewsCrawl["Nepali News Crawl: 18,190 articles (34M tokens)"]
    end

    subgraph Data_Cleaning ["Strict Cleaning & Normalization"]
        Dedup["Deduplication & Orthographic Filter (pipeline/clean_aksharantar.py)"]
        CorpusText["data/store/corpus_clean.txt: 2.89M clean sentences, 86.1M tokens"]
    end

    subgraph Artifact_Generators ["Core Training Stage (One-Shot train / Pack)"]
        TrainOneShot["train (One-Shot Pipeline) -> data/akshar.model"]
        BuildVocab["build_wordfreq_text -> vocab frequencies"]
        BuildBigrams["build_bigrams -> word bigrams"]
        PackModel["pack_model -> data/akshar.model"]
    end

    subgraph Output_Artifacts ["Production Runtime Artifact"]
        UnifiedModel["data/akshar.model (Bundled container: translit + vocab + reranker + bigrams)"]
        LegacyBins["Legacy Fallbacks (optional): translit_model.bin, word_freq_text.bin, etc."]
    end

    Aksharantar --> Dedup
    Wikipedia --> CorpusText
    CC100 --> CorpusText
    NewsCrawl --> CorpusText
    Dedup --> TrainOneShot
    CorpusText --> TrainOneShot
    TrainOneShot --> UnifiedModel
    UnifiedModel --> LegacyBins
```

### Training Pipeline Phases:
1. **Corpus Ingestion & Normalization (`data/pipeline/`):**
   - Filters text strictly to valid Devanagari Unicode sequences (`U+0900..=U+0963`). Strips punctuation, non-Devanagari scripts, digits, and control characters.
   - Removes cross-language duplicate pairs between Hindi and Nepali splits (107,758 duplicates eliminated).
2. **Unified One-Shot Training (`src/bin/train/train.rs`):**
   - A single unified CLI: `cargo run --release --bin train` (or `make train`).
   - Runs Baum-Welch expectation-maximization over 3.59M pairs to learn emission probabilities $P(\text{roman chunk} \mid \text{akshara})$ and syllable Kneser-Ney trigrams.
   - Computes Devanagari word frequency distribution from text corpus.
   - Fits discriminative reranker features and packs everything directly into `data/akshar.model`.
   - Supports self-contained smoke tests via `--smoke`.
3. **Model Packaging (`src/bin/build/pack_model.rs`):**
   - Allows bundling or re-packing disparate binaries into `data/akshar.model` with optional bigram inclusion and parameter pruning.

---

## 5. Performance, Memory & Binary Specifications

| Component | Disk Footprint | Memory at Runtime | Algorithmic Complexity | Keystroke Latency |
| :--- | :--- | :--- | :--- | :--- |
| **Unified Container (`akshar.model`)** | 48 MB (no bigrams) / 88 MB (with bigrams) | $\approx 45$ MB heap (atomic load) | Single read | N/A (load time < 0.6s) |
| **Generative Decoder** | Included in container | $\approx 32$ MB (or 9.1 MB Brotli in WASM) | $O(M \cdot B \cdot L)$ | $0.25 - 0.40$ ms |
| **Vocabulary WordTrie** | Included in container | $\approx 22$ MB heap | $O(M \cdot \Sigma)$ prefix walk | $0.05 - 0.10$ ms |
| **Discriminative Reranker**| Included in container (4 MB) | 4 MB sparse table | $O(K \cdot (D + S))$ | $0.08 - 0.15$ ms |
| **Bigram Context Layer** | Included in container (41 MB) | $\approx 15$ MB (native only) | $O(K \log \text{deg})$ | $0.01 - 0.03$ ms |
| **SymSpell & User Trie** | $\approx 50$ KB (`user_dictionary.bin`) | $< 2$ MB heap | $O(1)$ hash lookup | $0.01 - 0.02$ ms |
| **Total Runtime Engine** | **$\approx 15$ MB (Brotli)** | **$\approx 25 - 45$ MB** | **Strictly sub-linear** | **$0.40 - 0.80$ ms** |

