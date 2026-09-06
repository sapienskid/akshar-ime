# Experiment archive

Working documents from the development of Akshar IME, kept because they record
**what was tried, measured and rejected** — which the manual's final numbers
cannot show on their own. Superseded as plans; preserved as evidence.

Read in this order:

| Document | Date | What it records |
| :--- | :--- | :--- |
| `2026-08-01-generative-transliteration-design.md` | Aug 1 | Original design of the generative core: the source-channel decision, akshara units, first results. |
| `2026-09-03-transliteration-accuracy-research.md` | Sep 3 | Error analysis and a ranked technique shortlist (E0–E7) with expected gains. The "95% question". |
| `2026-09-03-accuracy-experiments.md` | Sep 3 | **The experiment log.** E0–E3 with measured deltas, the v2 WFST core, depth-2 pair context. |
| `2026-09-05-data-flow.md` | Sep 5 | How raw text becomes the artefacts a keystroke touches. |
| `2026-09-05-data-research.md` | Sep 5 | Literature review: how IndicXlit uses data, context in production IMEs (Kirov et al. 2024), larger Nepali corpora, prior art on the blocking ambiguity classes. |
| `2026-09-05-research-agenda.md` | Sep 5 | Mathematics considered but not executed: context-tree weighting, A* anytime decoding, the entropy harness, incremental decoding. |
| `2026-09-05-roadmap-to-90.md` | Sep 5 | First plan to 90%: audit of how every byte of data is used. |
| `2026-09-05-path-past-90.md` | Sep 5 | Revision of the above, with the W0 measurement-gate results. |

Live documents are one level up in `docs/plans/`. The mathematics and the final
measurements are in `docs/MANUAL.md`.

**Caution: the accuracy figures in these documents are historical.** They were
measured before the 2026-09-06 defect fixes and do not describe the shipped
system. The manual is authoritative.
