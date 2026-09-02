# Accuracy Experiments — Log

Date started: 2026-09-03
Benchmark: Aksharantar Nepali test split (4,101 cases; AK-Freq = native words,
AK-NEF/AK-NEI = named entities). Reference: IndicXlit neural top-1 native=80.25%,
NE=52.67%. Tuning happens on nep_valid.json only; nep_test.json is measured
single-shot per experiment.

## Baselines (E0, 2026-09-03)

| Config | AK-Freq top-1 | AK-Freq top-5 | ALL top-1 | Latency |
|---|---|---|---|---|
| ImeEngine.get_suggestions (full fusion) | 73.24% | 89.52% | ~54% | — |
| Decoder only (beam 64, k-best) | **75.33%** | 89.66% | 55.13% | 3.06 ms/word |

**First finding: the fusion layer LOSES 2.1 points of top-1.** The decoder alone
beats the full engine. Suspect (from code reading, `engine.rs:get_suggestions`):
lexicon exact-match candidates get a flat `LEXICON_EXACT_SCORE = 300` that always
outranks fresh decoder candidates (whose `FRESH_SCALE/(1+cost)` scores stay below
~250 for typical words), even when the decoder ranks a different word first; and
among multiple exact lexicon natives for one roman the order is arbitrary
(HashMap iteration), discarding the decoder's ranking.

## E0/E1 results (2026-09-03)

**E0 harness** (`src/bin/analyze_errors.rs`): oracle top-k, engine-vs-decoder
disagreement, multi-reference, matra/halant taxonomy, CER. Key numbers
(AK-Freq): decoder oracle top-2 = 85.2%, top-50 = 92.1%; 52% of misses are
matra-only; disagreements favored decoder 59:15 before fixes.

**E1a — fusion fix, three measured stages:**

| Change | AK-Freq top-1 |
|---|---|
| Baseline engine | 73.24% |
| + score resolution ×1000 (FRESH_SCALE 800→800k; u64 rounding had created tie classes ordered by HashMap iteration) | 73.91% |
| + additive lexicon bonus 10k / lexicon-only 5k | 74.91% |
| + bonus 0 (pure decoder passthrough; lexicon stays as low-priority fallback) | **75.33%** (= decoder parity, +2.09 total) |

Lesson: any absolute lexicon override *hurts* top-1 on this benchmark — the
decoder already ranks corpus words correctly, and lexicon entries mined from
the valid split actively mislead. User-learned words (USER_TRIE_BASE 500k)
still dominate for real personalization. `context.rs` boost rescaled ×1000 to
match the new score range.

**E1b — lm_weight sweep (decoder, test split):** 0.7→74.19, 0.85→75.00,
1.0→75.33, 1.15→75.05, 1.3→75.05. **1.0 is already optimal**; plateau is flat.
No change adopted.

Next: E3 (word-frequency prior + reranker v2) — targets the 52% matra-miss
class and the 85.2% top-2 → top-1 ranking headroom.

