# Akshar IME — WASM

Browser package for the Akshar Devanagari IME.

See [`docs/WASM.md`](../docs/WASM.md) for full integration docs and [`web/index.html`](../web/index.html) for a live demo.

## Build

```bash
./wasm/build.sh
# outputs to wasm/pkg/ (akshar_ime.js + akshar_ime_bg.wasm + d.ts)
```

Requires `wasm-bindgen-cli` (installed automatically if missing) and `wasm32-unknown-unknown` target.

## Use

```html
<input data-akshar />
<script type="module">
  import { AksharIME } from '../js/akshar-ime.js';
  await AksharIME.init({ modelUrl: '../data/translit_model.bin' });
  AksharIME.autoAttach();
</script>
```

## Publish

```bash
npm publish ./wasm/pkg
# or: npm --prefix wasm publish
```
