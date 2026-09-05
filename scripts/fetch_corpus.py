#!/usr/bin/env python3
# Download the Aksharantar Nepali split from Hugging Face into data/aksharantar/.
# Usage: fetch_corpus.py [out_dir]
import os
import pathlib
import sys
import zipfile

out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/aksharantar")
out_dir.mkdir(parents=True, exist_ok=True)
if (out_dir / "nep_train.json").exists():
    print("corpus already present:", out_dir / "nep_train.json")
    sys.exit(0)

repo = "ai4bharat/Aksharantar"
zip_path = None
try:
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(repo_id=repo, filename="nep.zip", repo_type="dataset")
    print(f"hf_hub_download nep.zip -> {zip_path} ({os.path.getsize(zip_path)} bytes)")
except Exception as e:
    print(f"hf_hub_download failed ({e}); falling back to HTTPS", file=sys.stderr)
    import subprocess
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/nep.zip"
    tmp = "/tmp/akshar-nep.zip"
    rc = subprocess.call(["curl", "-sL", "-o", tmp, url])
    if rc != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        sys.exit(f"download failed: {url}")
    zip_path = tmp

with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        base = os.path.basename(name)
        if base.startswith("nep_") and base.endswith(".json"):
            with z.open(name) as src, open(out_dir / base, "wb") as dst:
                dst.write(src.read())
            print("extracted", base)

assert (out_dir / "nep_train.json").exists(), "nep_train.json missing after extraction"
print("corpus ready in", out_dir)
