# Data root — everything the IME learns from lives here

This directory is **fully gitignored** (only this README is tracked): corpora,
the cleaned corpus, pipeline code, and built artifacts never enter the
repository. Everything is rebuildable with the `make data*` targets below.

**The single source of text is `store/corpus_clean.txt`** — 2.89M clean,
deduplicated Devanagari sentences (86.1M word tokens) compiled from Nepali
Wikipedia, CC100, and the news crawl. Nothing raw is stored: the download and
cleaning pipeline consumes the raw dumps and deletes them. Word frequencies
are *counted* from this one file; word bigrams can be re-derived from it.

```
data/
  aksharantar/        AI4Bharat Aksharantar Nepali splits (download, ~560 MB)
                      nep_{train,valid,test}.json — EM training + benchmark
  pipeline/           the data pipeline (private, untracked):
                      pipeline.py        crawl / count / export / romanize
                      build_corpus.py    compile the cleaned corpus (single feed file)
                      fetch_corpus.py    download Aksharantar from Hugging Face
                      extract_wiki.py    Wikipedia dump -> text lines
                      filter_cc100.py    CC100 -> Devanagari lines
                      make_eval_tsv.py   regenerate data/eval/aksharantar_test.tsv
                      .venv/             its Python environment
  store/
                      corpus_clean.txt   THE single cleaned corpus (~1.5 GB):
                                         one sentence per line, pure Devanagari
                                         words only, exact duplicates removed
                      word_pairs.csv     word bigrams (freq>=3) — E6 context
                                         layer data; re-derivable from corpus_clean.txt
  eval/               derived evaluation subsets
                      aksharantar_test.tsv  nep_test.json flattened (regenerate:
                                            python3 data/pipeline/make_eval_tsv.py)
  translit_model.bin  BUILT artifact — EM emissions + akshara KN LM (~22 MB)
  word_freq_text.bin  BUILT artifact — 470k-word frequency vocabulary (~17 MB),
                      counted from corpus_clean.txt
```

## Build commands

| Target | What it does |
|---|---|
| `make data-raw` | download sources (Aksharantar, Wikipedia, CC100) into transient `data/raw/` |
| `make data-clean` | compile everything into `store/corpus_clean.txt`, then delete the raw inputs |
| `make data-vocab` | count `corpus_clean.txt` → `data/word_freq_text.bin` |
| `make data-model` | train EM model (`nep_train` + `nep_valid`) → `data/translit_model.bin` |
| `make data` | the whole chain: raw → clean → vocab → model |
| `make data-store` | incremental news crawl, then re-clean + re-count |
| `make release-upload TAG=vX.Y.Z` | upload built artifacts to a GitHub release |

Cleaning rules (enforced in `build_corpus.py`, mirrored in
`build_wordfreq_text.rs`): a word is a maximal run of Devanagari *letters*
(U+0900..=U+0963) — danda, digits (Devanagari and ASCII), punctuation,
ZWJ/ZWNJ, and Latin are stripped or dropped; lines need ≥4 surviving words;
exact duplicate lines are removed (syndication/boilerplate otherwise inflates
counts).

The engine loads exactly four files at runtime (resolved via `$AKSHAR_DATA_DIR`,
`data/`, `~/.local/share/akshar-ime/`, then `/usr/share/akshar-ime/`):
`translit_model.bin`, `word_freq_text.bin`, `roman_lexicon.bin` (optional),
`reranker_weights.json` (optional).

**Do not write into the top level from other tools** — the top-level `.bin`
files are owned by `build_wordfreq_text` and `train_model`. Verified result of
this exact chain: **80.98% native top-1** on the Aksharantar Nepali benchmark.
