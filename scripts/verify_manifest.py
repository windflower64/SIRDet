"""Verify release inputs without importing the training runtime."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8-sig"))
errors = []
for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(entry["path"])
        continue
    data = path.read_bytes()
    if path.suffix.lower() in {".py", ".md", ".sh", ".ps1", ".yaml", ".json", ".csv", ".txt", ".cpp", ".hpp", ".h"} or path.name in {"LICENSE", ".gitignore", ".gitattributes"}:
        data = data.replace(b"\r\n", b"\n")
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        errors.append(entry["path"])
if errors:
    raise SystemExit("Manifest mismatches: " + ", ".join(errors))
print(f"PASS: {len(manifest['files'])} release files verified.")
