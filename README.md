# Akshar Devanagari IME

Roman-script input method for Devanagari, targeted at Nepali. You type
`namaste`, it offers `नमस्ते`.

![CI](https://github.com/sapienskid/akshar-ime/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

**No neural network at runtime.** An EM-trained source-channel model over
orthographic syllables, a modified Kneser-Ney syllable language model, and a
linear discriminative reranker. Inference is beam search plus dot products —
no tensor library, no GPU.

**11.37 MB desktop, 4.94 MB Brotli in the browser. 0.67 ms per query.**

## Measured performance

Held-out AI4Bharat Aksharantar Nepali test split (4,101 cases), measured
2026-09-06 on `data/akshar.model` at default settings.

| Split | n | top-1 | top-5 |
| :--- | ---: | ---: | ---: |
| `AK-Freq` (native words) | 2,108 | **81.83%** | 92.22% |
| `AK-NEI` (named entities) | 1,176 | 47.79% | 69.81% |
| `AK-NEF` (named entities) | 817 | 31.21% | 53.00% |
| All cases | 4,101 | 61.98% | 77.98% |

Pooled top-1 bootstrap 95% CI [60.61%, 63.36%]; MRR 0.6906; CER on `AK-Freq`
top-1 is 3.90%. For reference, IndicXlit (an ~11M-parameter transformer) reports
80.25% top-1 on the native split and 52.67% on named entities — native accuracy
here is comparable, named-entity accuracy is well behind.

| | Desktop | Browser |
| :--- | ---: | ---: |
| Container | 11.37 MB | 8.91 MB |
| Brotli | 6.72 MB | 4.94 MB |
| Query latency | 0.63–0.82 ms | not re-measured since 2026-09-06 |
| Cold start | ~1.5 s | — |

## Documentation

**[`docs/MANUAL.md`](docs/MANUAL.md)** is the complete reference — mathematics,
architecture, training, evaluation methodology, ablations with significance
tests, performance engineering, deployment, and a register of known defects.
Build a PDF with `make manual`.

`docs/plans/` holds the live defect register and roadmap.

## Quick start

```sh
make release            # build
sudo make install       # install to IBus
make restart-ibus       # NOT as root
```

```rust
use akshar_ime::ImeEngine;
let mut engine = ImeEngine::new();
let suggestions = engine.get_suggestions("namaste", 5);
engine.user_confirms("namaste", "नमस्ते");   // adaptive learning
```

Browser: `make wasm-serve`, then see the manual's deployment chapter.

## Reproducing the numbers

```sh
make test          # tests, including the accuracy regression guard
make eval          # accuracy by split
make eval-full     # bootstrap CIs, MRR, per-query latency
make eval-errors   # oracle curves, error taxonomy, CER, collision bound
make ablate        # component contributions
```

Training (`make train-mid`, `make train-full`) is documented in the manual.
Note that `--reranker-pairs` sizes the reranker's training set only — the EM
model and language model always use all 3.59M pairs.

## Honest limitations

- One language (Nepali), one test set (Aksharantar).
- Named entities are well behind the neural baseline.
- The discriminative reranker is *worse* than the 3-parameter heuristic when
  used alone; its net contribution is +0.81pp on native words.
- 5x more reranker training data was tested and changed nothing; the cause is
  that `W_DENSE` has never been refit by this pipeline (manual §12.3).

Manual chapter 11 lists every known defect.

## License

MIT.
