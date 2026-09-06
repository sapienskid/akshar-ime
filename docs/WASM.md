# Akshar IME — WASM / Browser Guide

Use the Devanagari IME on **any website** via WebAssembly. The engine runs entirely in the browser (no server, offline-capable) and is ~3 ms per keystroke.

---

## Building the web model

The browser loads the unified container, not the loose `.bin` files. Build the
web profile from a trained model with:

```bash
# Relative-entropy prune the syllable trigram LM
cargo run --release --bin prune_lm -- \
  --model data/akshar.model --trigram-threshold 3e-2 --out data/akshar_wasm.model

# Serve it Brotli-compressed.
brotli -q 11 data/akshar_wasm.model
```

Size and accuracy are a tunable curve, measured on the 4,101-case Aksharantar
Nepali test split. `--trigram-threshold` is the knob:

| threshold | raw | Brotli | AK-Freq top-1 | top-5 |
| --: | --: | --: | --: | --: |
| 0 (no pruning) | 11.06 MB | 6.58 MB | 82.02% | 92.17% |
| 5e-3 | 10.01 MB | 5.85 MB | 81.45% | 92.08% |
| 1e-2 | 9.61 MB | 5.52 MB | 81.36% | 92.13% |
| 2e-2 | 9.18 MB | 5.15 MB | 81.21% | 92.13% |
| **3e-2** (shipped) | **8.91 MB** | **4.92 MB** | **81.07%** | **92.22%** |

Top-5 is flat across the whole range: pruning reorders the top of the list, it
does not lose candidates.

## Quick start (any website, 3 lines)

```html
<!-- any <input>, <textarea>, or contenteditable with data-akshar -->
<input data-akshar placeholder="type: namaste" />

<script type="module">
  import { AksharIME } from 'https://cdn.example.com/akshar-ime/js/akshar-ime.js';
  await AksharIME.init({ modelUrl: 'https://cdn.example.com/akshar-ime/akshar_wasm.model' });
  AksharIME.autoAttach(); // done — all [data-akshar] now transliterate
</script>
```

Type `namaste` → suggestion popup shows `नमस्ते` → `Enter`/`Tab`/`1` to commit. Learned words persist in `localStorage`.

---

## Installation

### Option A — copy files (self-host)

```bash
git clone https://github.com/sapienskid/akshar-ime
cargo build --release --bin train --bin repack_model --bin prune_lm
cargo run --release --bin train              # builds data/akshar.model
cargo run --release --bin build_lexicon      # builds data/roman_lexicon.bin (optional)
./wasm/build.sh                               # builds wasm/pkg/
# serve repo root: python -m http.server 8000
# demo at http://localhost:8000/web/
```

Copy to your site:

```
your-site/
  js/akshar-ime.js
  wasm/pkg/akshar_ime.js
  wasm/pkg/akshar_ime_bg.wasm
  data/akshar_wasm.model   (or CDN URL)
```

### Option B — npm (when published)

```bash
npm install akshar-ime
# node_modules/akshar-ime/wasm/pkg/  +  js/akshar-ime.js
```

Vite / Next.js: add `akshar-ime` to `optimizeDeps` exclude if needed and copy `wasm/pkg/*.wasm` to `public/`.

---

## API

### `AksharIME.init(opts)`

```js
await AksharIME.init({
  modelUrl: '/data/akshar_wasm.model',   // required (8.91 MB, 4.92 MB with br)
  lexiconUrl: '/data/roman_lexicon.bin',  // optional (adds ~25 MB gzip) — boosts exact-word ranking
  rerankerUrl: '/data/reranker_weights.json', // optional
  wasmUrl: '/wasm/pkg/akshar_ime_bg.wasm', // optional override
});
```

Returns `WasmEngine`. Must be called once before `attach()`. Subsequent calls are no-ops.

### `AksharIME.attach(element, opts?)`

Enhances one element. Returns `{ destroy() }`.

```js
const h = AksharIME.attach(document.querySelector('#my-input'), {
  suggestions: 5,
  autoConfirmOnSpace: false,
  onCommit: (roman, devanagari) => console.log(roman, '→', devanagari)
});
// later: h.destroy();
```

Works with:

- `<input type="text">`
- `<textarea>`
- any `contenteditable` element

### `AksharIME.autoAttach(selector = '[data-akshar]', opts?)`

Attaches to all current + future elements matching `selector`. Ideal for forms.

### Low-level engine

```js
AksharIME.engine.getSuggestions('namaste', 8) // → ["नमस्ते", ...]
AksharIME.engine.transliterate('nepal')       // → "नेपाल"
AksharIME.engine.confirm('namaste', 'नमस्ते') // teach + persist
AksharIME.engine.learnedCount()               // number of learned words
AksharIME.engine.exportState()                // base64 backup
AksharIME.engine.importState(b64)
AksharIME.engine.resetLearning()
AksharIME.engine.vocabSize()
AksharIME.engine.isReady()
```

#### Raw instantiation (no fetch)

```js
import init, { WasmEngine } from './wasm/pkg/akshar_ime.js';
await init();
const modelBytes = await fetch('/data/akshar_wasm.model').then(r => r.arrayBuffer());
const engine = new WasmEngine(new Uint8Array(modelBytes));
engine.getSuggestions('namaste')
```

#### Fetch helper

```js
import { createEngine } from './wasm/pkg/akshar_ime.js';
const engine = await createEngine(modelUrl, lexiconUrl, rerankerUrl);
```

---

## How it works in the browser

```
[input keystroke] → last Roman word (/\b[a-zA-Z]+$/) → WasmEngine.getSuggestions()
  → floating popup (keyboard: Tab/Enter/↑↓/1-5/Esc)
  → on commit: replace Roman word with Devanagari + engine.confirm() + localStorage persist
```

All computation is local:

- **Model:** EM-trained `P(roman|akshara)` + Kneser-Ney trigram — same as Linux IBus engine.
- **Lexicon:** optional roman→devanagari dictionary (118 MB). Lite mode (no lexicon) is faster to load and still 90%+ accurate via decoder alone.
- **Learning:** trie + SymSpell (edit-distance-2) + context reranker, saved to `localStorage` key `akshar-ime-state-v1` (base64 bincode, ~KBs).

---

## Hosting & performance (measured `v1.0.0`, `wasm-bindgen 0.2.100`)

| Artifact | Raw | gzip | brotli | Notes |
|---|---|---|---|---|
| `akshar_ime_bg.wasm` | 377 KB | 139 KB | 111 KB | long-cache, immutable |
| `wasm/pkg/akshar_ime.js` (glue) | 26 KB | 5.7 KB | — | ESM glue |
| `js/akshar-ime.js` (wrapper) | 13 KB | 4.1 KB | — | `AksharIME.attach` helper |
| **Total WASM + JS** | **416 KB** | **~149 KB** | **~121 KB** | without model |
| `akshar_wasm.model` | 8.91 MB | 5.6 MB | **4.92 MB** | **required**, cache 1y |
| `roman_lexicon.bin` | 118 MB | 24.4 MB | ~19 MB* | optional, skip for lite |
| `reranker_weights.json` | 72 B | — | — | optional |

\* brotli lexicon ~19–20 MB (slow to compress, not measured in CI). Lite mode (`modelUrl` only) saves ~24 MB gzip transfer and is still 90%+ accurate — decoder alone handles most words.

**Recommendations:**

- Serve with `Content-Encoding: br` and `Cache-Control: public, max-age=31536000, immutable`.
- Use lite mode (`modelUrl` only) for fast first paint. Add lexicon later if you need exact corpus-word boosting.
- Lazy-load: call `AksharIME.init()` on first focus, not on page load, to avoid blocking.
- For offline/PWA, precache `*.wasm` + `*.bin` via Service Worker.

**Benchmarks (browser, M1):** ~3–7 ms per `getSuggestions()` (beam 64), <1 ms for learned-word hits.

---

## Framework examples

### React / Next.js

```jsx
import { useEffect, useRef } from 'react';
import { AksharIME } from '@/lib/akshar-ime';

export function DevanagariInput(props) {
  const ref = useRef(null);
  useEffect(() => {
    let handle;
    (async () => {
      await AksharIME.init({ modelUrl: '/data/akshar_wasm.model' });
      handle = AksharIME.attach(ref.current);
    })();
    return () => handle?.destroy();
  }, []);
  return <input ref={ref} {...props} />;
}
```

### Vue

```js
onMounted(async () => {
  await AksharIME.init({ modelUrl: '/data/akshar_wasm.model' });
  AksharIME.attach(inputEl.value);
});
```

### Plain HTML + CDN (no build)

```html
<script type="importmap">
{ "imports": { "akshar-ime": "https://cdn.jsdelivr.net/npm/akshar-ime@1.0.0/js/akshar-ime.js" } }
</script>
<script type="module">
  import { AksharIME } from 'akshar-ime';
  await AksharIME.init({ modelUrl: 'https://cdn.jsdelivr.net/npm/akshar-ime@1.0.0/data/akshar_wasm.model' });
  AksharIME.autoAttach();
</script>
```

---

## Troubleshooting

- **Popup not showing:** check `await AksharIME.init()` completed and element has `data-akshar` or was passed to `attach()`. Open console for `[AksharIME] ready` log.
- **CORS / file://:** model must be served via HTTP(S). `file://` cannot `fetch()` the `.bin`. Run `python -m http.server`.
- **Large download:** use lite mode (no lexicon). Consider splitting model if you self-train a smaller one (fewer aksharas).
- **localStorage quota:** learned state is tiny (<50 KB for 1000 words). If full, `resetLearning()` or `exportState()` then clear.
- **Version mismatch `wasm-bindgen`:** rebuild with `./wasm/build.sh` after `cargo update`.

---

## Releasing

```bash
./wasm/build.sh
npm --prefix wasm publish    # publishes wasm/pkg as npm package
# or: npm publish ./wasm/pkg
# Tag git release and attach wasm/pkg + data/*.bin (gzipped) as artifacts
```

See `Makefile` target `wasm` and GitHub workflow `.github/workflows/wasm.yml`.

---

## License

MIT — same as the main crate. Model artifacts retain the CC0 portions of the Aksharantar training data (see README).
