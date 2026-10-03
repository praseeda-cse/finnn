"""Zip the code + data into dist/finsight_bundle.zip for upload to Google Drive
(My Drive/finsight/finsight_bundle.zip). Re-run and re-upload after changing code.

    python scripts/make_bundle.py
"""
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "dist", "finsight_bundle.zip")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for folder in ("finsight", "data", "scripts"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, folder)):
            for name in files:
                if "__pycache__" in dirpath:
                    continue
                path = os.path.join(dirpath, name)
                z.write(path, os.path.relpath(path, ROOT))
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
