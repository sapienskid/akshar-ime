# Data root — everything the IME learns from lives here

This directory is **fully gitignored** (only this README is tracked): corpora,
raw text, the pipeline store, and built artifacts never enter the repository.
Everything here is rebuildable — either by downloading or from the commands
below. `make data` runs the whole chain.

```
data/
  aksharantar/        AI4Bharat Aksharantar Nepali splits (download, ~560 MB)
                      nep_{train,valid,test}.json — EM training + benchmark
  raw/                raw Devanagari text dumps (never tracked, ~1.9 GB)
                      newiki.txt   Nepali Wikipedia (CC-BY-SA)
                      cc100ne.txt  CC100 Nepali (CC0)
                      news.txt     akshar-ime news-crawl extraction
  store/              the scraping pipeline's database + derived products
                      nepali_text.db        articles + word/pair counts (SQLite, ~1.4 GB)
                      word_freq.csv         word,freq (all words)
                      word_pairs.csv        w1,w2,freq (freq>=3) — E6 context layer
                      synthetic*.jsonl      synthetic pairs (experimental; see
                                            docs/plans/2026-09-03-accuracy-experiments.md —
                                            self-training measured NEGATIVE, kept for reference)
                      word_freq_news_only.bin / word_freq_merged.csv  intermediate vocab files
  eval/               derived evaluation subsets
  translit_model.bin  BUILT artifact — EM emissions + akshara KN LM (~22 MB)
  word_freq_text.bin  BUILT artifact — word-frequency vocabulary (~26 MB)
```

## Build commands

| Target | What it does |
|---|---|
| `make data-raw` | download Aksharantar (HuggingFace), Wikipedia dump, CC100 → `data/raw/` |
| `make data-vocab` | count `data/raw/*` → `data/word_freq_text.bin` (576k+ words) |
| `make data-model` | train EM model from `nep_train` + `nep_valid` → `data/translit_model.bin` |
| `make data-store` | (optional) scraping pipeline: crawl news, count, export → `data/store/` |
| `make data` | all of the above in dependency order |

The engine loads exactly four files at runtime (resolved via `$AKSHAR_DATA_DIR`,
`data/`, `~/.local/share/akshar-ime/`, then `/usr/share/akshar-ime/`):
`translit_model.bin`, `word_freq_text.bin`, `roman_lexicon.bin` (optional),
`reranker_weights.json` (optional).

**Do not write into the top level from other tools** — the top-level `.bin`
files are owned by `build_wordfreq_text` and `train_model`. The scraping
pipeline exports into `data/store/` only; its `export --merge-base` flag
explicitly merges news counts into `data/word_freq_text.bin` (this separation
exists because a news-only vocabulary once silently overwrote the real one and
cost 1.2 accuracy points — see the experiment log).
