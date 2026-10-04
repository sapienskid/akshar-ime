# Experiment archive

Working documents from the development of Akshar IME. These are **snapshots,
kept for provenance only** — nothing here is a live plan, and nothing here
describes the shipped system. Figures are from the configurations of their
dates and are frequently superseded.

Read `docs/MANUAL.md` for the current source of truth and
`docs/experiments/` for the live experiment log.

## Archived by reason

### Superseded by the Nepali-only scope (2026-09-22 multi-language era)

The engine shipped eight Devanagari languages until the per-language corpora
were dropped. These were written for that edition and cannot be re-run:

| File | Why archived |
| :--- | :--- |
| `2026-08-01-generative-transliteration-design.md` | Generative core design; superseded by the lattice + reranker actually built |
| `2026-09-05-mathematics.md` | Mathematical treatment of the 8-language formulation |
| `2026-09-05-research-agenda.md` | Ranked bets across 8 languages |
| `2026-09-05-data-flow.md` | Data flow including the removed IndicCorp path |
| `2026-09-05-data-research.md` | Corpus survey for 8 languages |
| `devanagari-script-plan.md` | Script-general research program |
| `research/indic-morphology.md` | Morphological analyzers for Hindi/Marathi/Sanskrit/Nepali |
| `research/sota-signals.md` | SOTA signals, framed for a multi-language engine |
| `research/rerank-beyond-linear.md` | Reranking survey, multi-language framing |

`research/indic-morphology.md` is still the citation for the morphological
suffix union table in `src/core/reranker.rs` — its *content* survives even
though its framing does not.

### Measured and rejected

Kept because the numbers are the reason the approach is not in the product:

| File | Outcome |
| :--- | :--- |
| `2026-09-03-transliteration-accuracy-research.md` | Feasibility findings that redirected the work |
| `2026-09-03-accuracy-experiments.md` | Early accuracy log |
| `2026-09-06-repair-and-path-to-90.md` | Audit and 90% plan; the 90% target was not reached by this route |
| `2026-09-05-path-past-90.md` | Four levers, none of which closed the gap |
| `2026-09-05-roadmap-to-90.md` | Roadmap that the measurement later invalidated |
| `2026-09-08-experiment-report.md` | Which ideas carried the accuracy, and by how much |
| `2026-09-22-attestation-training.md` | Attestation-weighted training: measured, not adopted |
| `2026-09-22-factored-matra-model.md` | Factored matra model: measured, folded into the reranker |
| `2026-09-22-mbr-selection.md` | MBR consensus selection: measured, no gain |
| `2026-09-22-ne-routing.md` | Named-entity routing: no gain |
| `2026-09-22-top2-discriminator.md` | Top-2 discriminator: implemented, no gain |

### Process

| File | Outcome |
| :--- | :--- |
| `playground-extension.md` | Superseded by section B of `docs/plans/browser-extension.md` |
| `correctness-audit-2026-09-10.md` | Audit done; findings folded into the defect register, MANUAL §17 |